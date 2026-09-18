import pandas as pd
import torch
from transformers import BertTokenizer, BertForTokenClassification,pipeline, AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os
import glob
import re
import json
import joblib
from pathlib import Path

LAST_EXAM = None   # Conversation state for resolving omitted exam references.
warnings.filterwarnings("ignore")
# 1. Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIO_MODEL_PATH = os.getenv(
    "BIO_MODEL_PATH", str(PROJECT_ROOT / "exam_bio_final_model")
)
EXCEL_FOLDER = os.getenv("FAQ_DATA_DIR", str(PROJECT_ROOT / "faq_data"))
INTENT_MODEL_PATH = os.getenv(
    "INTENT_MODEL_PATH", str(PROJECT_ROOT / "experiments" / "intent_model.pkl")
)
QWEN_MODEL_PATH = os.getenv("QWEN_MODEL_PATH", str(PROJECT_ROOT / "Qwen"))

# Shared inference device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Unanswered-query behavior:
# True returns a fixed response; False delegates to the local Qwen model.
USE_FIXED_REPLY_FOR_UNANSWERED = False


# 2.1. BIO entity-recognition model
def load_bio_model():
    labels = ['O', 'B-EXAM', 'I-EXAM', 'B-SUBJECT', 'I-SUBJECT',
        'B-TIME', 'I-TIME', 'B-ACTION', 'I-ACTION',
        'B-CONSTRAINT', 'I-CONSTRAINT', 'B-CITY', 'I-CITY']
    id_to_label = {i: label for i, label in enumerate(labels)}
    
    # Load the locally trained tokenizer and token classifier.
    tokenizer = BertTokenizer.from_pretrained(BIO_MODEL_PATH)
    model = BertForTokenClassification.from_pretrained(BIO_MODEL_PATH)
    model = model.to(DEVICE)
    model.eval()
    
    return tokenizer, model, id_to_label

bio_tokenizer, bio_model, bio_id_to_label = load_bio_model()

# 2.2. Intent classifier
def load_intent_model():
    try:
        print(f"Loading the intent model: {INTENT_MODEL_PATH}")
        model = joblib.load(INTENT_MODEL_PATH)
        print("Intent model loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load the intent model: {e}")
        return None


# Initialize the intent classifier once at import time.

intent_classifier = load_intent_model()



def predict_intent(text):
    if intent_classifier:
        try:
            # Labels are defined by the intent-training dataset.
            label = intent_classifier.predict([text])[0]
            return label
        except Exception as e:
            print(f"Intent prediction failed: {e}")
            return None
    return None



# 2.3. Local Qwen model
def load_local_qwen():
    # QWEN_MODEL_PATH can override the project-relative default.
    local_qwen_path = QWEN_MODEL_PATH
    
    # Load the tokenizer from local files without downloading.
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        local_qwen_path,
        trust_remote_code=True,
        padding_side="right",
        local_files_only=True
    )
    qwen_tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}"

    
    # Load the 1.8B model directly in FP16; quantization is not required here.
    qwen_model = AutoModelForCausalLM.from_pretrained(
        local_qwen_path,
        torch_dtype=torch.float16,  # Approximately 3-4 GB of VRAM for this model.
        device_map="auto",  # Select available compute devices automatically.
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True 
    ).eval()  # Inference mode
    qwen_model.generation_config.stream_generator = False #
    print("Local Qwen-1.8B model loaded successfully.")
    return qwen_model, qwen_tokenizer
# Initialize the local language model once at import time.
qwen_model, qwen_tokenizer = load_local_qwen()


# 2.3.1. Qwen generation helper
def qwen_chat(prompt, max_new_tokens=200):
    messages = [{"role": "user", "content": prompt}]
    text = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = qwen_tokenizer([text], return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        generated_ids = qwen_model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,  # Keep generation conservative.
            top_p=0.85,
            eos_token_id=qwen_tokenizer.eos_token_id,
            pad_token_id=qwen_tokenizer.pad_token_id, 
            num_return_sequences=1, 
            repetition_penalty=1.2,
            early_stopping=True,  # Stop when the EOS token is generated.
            stopping_criteria=[torch.nn.CrossEntropyLoss()]
        )
    
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = qwen_tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()

    irrelevant_keywords = [
    "知识库", "合作关系", "服务", "优质", "生活", "变化", "慢慢", "长期",
    "更新", "优化", "需求", "期待", "建立", "感受", "意识到"
    ]
    for kw in irrelevant_keywords:
        if kw in response:
            response = response.split(kw)[0].strip("。；; ")
            break

    # Bound fallback responses to keep them concise.
    response = response[:100]
    # Collapse repeated blank lines.
    response = re.sub(r"\n+", "\n", response)

    return response

def handle_unanswered_question(user_input, matched_exam="", context=""):
    """Return an exam introduction and context-aware guidance for unanswered queries."""
    try:
        # Normalize invalid exam values before checking the knowledge base.
        matched_exam = matched_exam.strip() if matched_exam else ""
        matched_exam = "" if str(matched_exam).lower() in ["nan", "none", ""] else matched_exam
        # Determine whether the exam has curated FAQ coverage.
        is_in_db = matched_exam and (matched_exam in exam_question_map)
        
        # Generate a short introduction, with a deterministic fallback.
        introduction = ""
        if matched_exam:
            try:
                # Ask for a bounded overview that includes the assessment scope.
                intro_prompt = f"""请用80-100字详细介绍【{matched_exam}】，说明其性质、报考意义、行业价值和核心考核方向，语言正式专业。
                ⚠️ 严格禁止：
                1. 添加知识库更新、合作关系、服务承诺等客服套话；
                2. 出现个人生活、感受、变化等无关内容；
                3. 生成完考试介绍后立即停止，不额外延伸。"""
                # Limit output length to reduce generation drift.
                introduction = qwen_chat(intro_prompt, max_new_tokens=100)
                # Reject empty or implausibly short generations.
                if not introduction or len(introduction.strip()) < 30:
                    raise ValueError("Generated content is empty or too short")
            except Exception as e:
                print(f"Qwen failed to generate an exam introduction: {e}")
                # Use a stable user-facing introduction if generation fails.
                introduction = f"【{matched_exam}】是国家统一组织的执业资格考试，旨在考核专业知识和实践能力，是担任项目经理的必备条件，具有较高的行业认可度和职业价值。"
        else:
            introduction = "您好！我是人事考试智能客服。"
        
        # Exclude invalid values from the displayed exam list.
        available_exams = [
            exam for exam in list(exam_question_map.keys()) 
            if str(exam).lower() not in ["nan", "none", ""]
        ][:5]
        exam_list = "\n".join([f"- {exam}" for exam in available_exams])
        
        # Tailor guidance to the available knowledge-base coverage.
        if not matched_exam:
            # No exam name was identified.
            return f"""{introduction}

请告诉我您想咨询哪个考试？或者从以下已收录考试中选择：
{exam_list}"""
        
        elif is_in_db:
            # Show examples for an exam covered by the knowledge base.
            example_questions = exam_question_map.get(matched_exam, [])[:3]
            examples = ""
            if example_questions:
                # Truncate non-empty examples for concise display.
                examples = "\n例如：\n" + "\n".join([f"- {q[:30]}..." if q and len(q)>30 else f"- {q}" for q in example_questions])
            
            # Generate a brief follow-up suggestion with a fixed fallback.
            try:
                guide_prompt = f"""用户询问了【{matched_exam}】这项考试。
                
已生成考试介绍：{introduction}

现在需要给用户一个引导，建议他们可以询问哪些具体问题。
请生成一个简短、友好的引导语，包含2-3个示例问题，如报名条件、考试时间、报名地点等。
要求：不超过40字，语气亲切。"""
                guide_text = qwen_chat(guide_prompt, max_new_tokens=40)
                # Discard empty or implausibly short guidance.
                guide_text = guide_text if guide_text and len(guide_text.strip())>5 else ""
            except Exception as e:      
                print(f"Qwen failed to generate follow-up guidance: {e}")
                # Deterministic fallback guidance.
                guide_text = "您可以直接问我具体问题，比如报名条件、考试时间、在哪里报名等。"
            
            return f"""{introduction}

{guide_text if guide_text else '您可以告诉我具体想了解的问题，我会为您详细解答~如报名条件、考试时间、报名地点等'}"""
        
        else:
            # The exam is not covered by the knowledge base.
            try:
                guide_prompt = f"""用户询问了【{matched_exam}】这项考试，但系统暂未收录该考试的信息。
                
已生成考试介绍：{introduction}

系统已收录的考试包括：{', '.join(available_exams[:3]) if available_exams else '二级建造师考试、计算机技术与软件专业技术资格考试'}

请生成一个简短、友好的引导语，建议用户从已收录的考试中选择，或者告诉用户如何提问。
要求：不超过50字，语气礼貌。"""
                guide_text = qwen_chat(guide_prompt, max_new_tokens=70)
                guide_text = guide_text if guide_text and len(guide_text.strip())>5 else ""
            except Exception as e:
                print(f"Qwen failed to generate follow-up guidance: {e}")
                # Deterministic fallback guidance.
                guide_text = f"目前系统暂未收录【{matched_exam}】的详细问答信息。您可以了解以上已收录的考试，或者告诉我您的专业方向。"
            
            return f"""{introduction}

{guide_text if guide_text else f"目前系统暂未收录【{matched_exam}】的信息，您可从已收录考试中选择咨询~"}"""
            
    except Exception as e:
        print(f"handle_unanswered_question failed: {e}")
        # Last-resort response that remains safe when the exam name is absent.
        if matched_exam:
            return f"关于【{matched_exam}】这项考试，我可以为您解答报名条件、考试时间、报名地点等问题，请具体提问。"
        else:
            return "请告诉我您想咨询哪个考试？"





# 3. Excel knowledge-base loader
def load_exam_database_from_folder(folder_path):
    """Load all .xlsx files in a folder into a unified FAQ knowledge base."""
    # Maps use the original Chinese exam, question, and answer text.
    exam_question_map = {}      # {exam_name: [questions]}
    question_answer_map = {}    # {question: answer}
    exam_keywords = set()       # Unique exam names

    # Discover all FAQ workbooks in the configured folder.
    excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    
    if not excel_files:
        print(f"Warning: no .xlsx files were found in {folder_path}.")
        return {}, {}, []

    print(f"Loading {len(excel_files)} Excel file(s)...")

    for file_path in excel_files:
        try:
            print(f"  Loading: {os.path.basename(file_path)}")
            df = pd.read_excel(file_path, sheet_name=0, engine="openpyxl")
            
            for _, row in df.iterrows():
                exam_name_raw = str(row.get("分类一", "")).strip().replace("\u3000", "").replace("　", "").replace(" ", "")
                if exam_name_raw.lower() == "nan" or exam_name_raw == "":
                    continue

                # Normalize known aliases to a canonical exam name.
                if exam_name_raw in ["软件水平考试", "软考"]:
                    exam_name = "计算机技术与软件专业技术资格考试"
                else:
                    exam_name = exam_name_raw
                if exam_name.lower() == "nan" or exam_name == "":
                    continue
                question = str(row.get("问题", "")).strip()
                answer = str(row.get("答案", "")).strip()

                if not exam_name or not question or not answer:
                    continue  # Skip incomplete rows.

                # Populate the lookup maps.
                if exam_name not in exam_question_map:
                    exam_question_map[exam_name] = []
                exam_question_map[exam_name].append(question)
                question_answer_map[question] = answer
                if exam_name and exam_name not in ("", "nan", "NaN"):
                    exam_keywords.add(exam_name)

                if not exam_name or exam_name.lower() == "nan":
                    continue

        except Exception as e:
            print(f"  Skipping {file_path} because it could not be loaded: {e}")

    return exam_question_map, question_answer_map, sorted(list(exam_keywords))

exam_question_map, question_answer_map, exam_keywords = load_exam_database_from_folder(EXCEL_FOLDER)

# 4. Fuzzy matching
def fuzzy_match(query, candidates, threshold=0.1):
    if not query: return ""
    query = query.strip()
    
    # 1. Prefer an exact match.
    if query in candidates:
        return query
    
    # 2. Check bidirectional containment for long formal names.
    for candidate in candidates:
        candidate_clean = candidate.strip()
        if query in candidate_clean or candidate_clean in query:
            return candidate
    
    # 3. Fall back to character-set similarity for minor typos.
    max_sim = 0
    best_candidate = ""
    for candidate in candidates:
        common = len(set(query) & set(candidate))
        sim = common / max(len(query), len(candidate))
        if sim > max_sim and sim >= threshold:
            max_sim = sim
            best_candidate = candidate
    
    return best_candidate
    


# Canonical exam-name aliases
EXAM_ALIAS = {
    "二级建造师考试": [
        "二级建造师",
        "建造师",
        "二建"
    ], 
    "计算机技术与软件专业技术资格考试": [
        "软考",
        "软件考试",
        "软件水平考试",
        "计算机软考",
        "软件资格考试"
    ],
    "咨询工程师": [
        "咨询工程师",
        "咨询工程师考试"
    ]
}



# Conversation state used for omitted exam references.
LAST_EXAM = None

# Normalize aliases and long-form exam names.
def normalize_exam(exam_query):
    if not exam_query:
        return ""
    
    # Lock common abbreviations to their canonical long-form names.
    query_strip = exam_query.strip()
    if any(kw in query_strip for kw in ["软考", "软件"]):
        return "计算机技术与软件专业技术资格考试"
    if any(kw in query_strip for kw in ["二建", "二级建造"]):
        return "二级建造师考试"
    if any(kw in query_strip for kw in ["咨询工程师"]):
        return "咨询工程师"
    
    for canonical_exam, aliases in EXAM_ALIAS.items():
        if query_strip == canonical_exam:
            return canonical_exam
        for alias in aliases:
            if alias in query_strip or query_strip in alias:
                return canonical_exam
                
    return exam_query

def extract_entities(user_input):
    """Extract and merge rule-based and BIO-model entities from a query."""
    entities = {"EXAM": "", "ACTION": "", "CONSTRAINT": "", "CITY": "", "TIME": ""}
    
    # 1. Rule-based city matching
    known_cities = {"遂宁", "成都", "绵阳", "德阳", "泸州", "宜宾", "南充", "达州", "资阳", "自贡", "内江", "乐山", "眉山", "广安", "雅安", "广元", "巴中", "攀枝花"}
    rule_city = ""
    for city in known_cities:
        if city in user_input:
            rule_city = city
            break

    # 2. Rule-based exam matching
    rule_exam = ""
    for canonical, aliases in EXAM_ALIAS.items():
        sorted_aliases = sorted(aliases, key=lambda x: len(x), reverse=True)
        for alias in sorted_aliases:
            # Direct substring matching is more reliable than word boundaries for Chinese.
            if alias in user_input:
                rule_exam = canonical
                break
        if rule_exam:
            break

    # 3. BERT token classification
    text_chars = list(user_input)
    encoded = bio_tokenizer(text_chars, is_split_into_words=True, return_tensors='pt', padding='max_length', truncation=True, max_length=510).to(DEVICE)
    with torch.no_grad():
        outputs = bio_model(**encoded)
        pred_ids = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().tolist()
        bio_labels = [bio_id_to_label[id] for id in pred_ids][1:len(text_chars)+1]
    
    # 4. Decode BIO tags
    current_type, current_val = None, ""
    model_entities = {}
    for char, label in zip(text_chars, bio_labels):
        if label.startswith("B-"):
            if current_type: model_entities[current_type] = current_val
            current_type, current_val = label.split("-")[1], char
        elif label.startswith("I-") and current_type == label.split("-")[1]:
            current_val += char
        else:
            if current_type: model_entities[current_type] = current_val
            current_type, current_val = None, ""

    # 5. Merge rule-based and model entities
    entities["EXAM"] = rule_exam if rule_exam else model_entities.get("EXAM", "")
    entities["CITY"] = rule_city if rule_city else model_entities.get("CITY", "")
    entities["ACTION"] = model_entities.get("ACTION", "")
    entities["CONSTRAINT"] = model_entities.get("CONSTRAINT", "")
    entities["TIME"] = model_entities.get("TIME", "")

    if len(entities["EXAM"]) < 2 and not rule_exam: entities["EXAM"] = ""
    
    # 6. Preserve common teacher-qualification aliases missed by the model.
    if not entities["EXAM"]:
        fallback_exam_keywords = ["教资", "教师资格", "教师资格证"]
        for kw in fallback_exam_keywords:
            if kw in user_input:
                entities["EXAM"] = kw   # Preserve the user's original wording.
                break
    
    return entities



# 5.1. Registration-information lookup
def search_registration_info(matched_exam, question_answer_map):
    """Find registration, address, or official-site information for an exam."""
    for q, a in question_answer_map.items():
        place_keywords = ["在哪", "哪里", "地址", "官网", "入口"]
        if any(kw in q for kw in place_keywords) and ("报名" in q):
        # Prevent a registration answer from leaking across exam scopes.
            if any(alias in q for alias in ["二建", "建造师", "二级建造", matched_exam]):
                return q, a
    return None, None



# 5.2. URL extraction
def extract_urls_from_exam(matched_exam, question_answer_map):
    urls = set()
    for q, a in question_answer_map.items():
        if matched_exam in q:
            found = re.findall(r"https?://\S+", a)
            urls.update(found)
    return list(urls)

# 5.3. Character-overlap similarity
def similarity(s1, s2):
    set1, set2 = set(s1), set(s2)
    return len(set1 & set2) / max(len(set1 | set2), 1)


# 5.4. Exam-introduction generation
def generate_exam_introduction(exam_name, is_in_database=False):
    """Generate a short exam overview, noting whether curated FAQ data exists."""
    if is_in_database:
        # Covered exams may use the curated context supplied later in the pipeline.
        intro_prompt = f"""
请为【{exam_name}】生成一个简要的专业介绍，要求：
1. 说明考试的性质和目的
2. 简单介绍考试的价值和用途
3. 用专业、官方的语言
4. 字数控制在100字以内
5. 不要包含具体的报名条件、时间等细节（这些会由后续问答提供）
"""
    else:
        # For uncovered exams, request only a general and clearly bounded overview.
        intro_prompt = f"""
请为【{exam_name}】生成一个简要的专业介绍，要求：
1. 说明这可能是哪种类型的考试
2. 解释其一般性质和目的
3. 用客观、专业的语言
4. 字数控制在80字以内
5. 说明这是通用信息
"""
    
    try:
        intro = qwen_chat(intro_prompt, max_new_tokens=150)
        return intro
    except Exception as e:
        print(f"Failed to generate an exam introduction: {e}")
        if is_in_database:
            return f"{exam_name}是一项重要的职业资格考试，具有较高的行业认可度和专业价值。"
        else:
            return f"{exam_name}是一项职业资格考试，具体信息建议咨询官方机构。"


# 6. Core retrieval and response pipeline
def get_answer_from_exam_db(user_input):
    global LAST_EXAM

    
    
    # 1. Handle account and login issues before exam routing.
    system_keywords = ["忘记", "密码", "登录", "登陆", "账号", "无法登录"]
    if any(k in user_input for k in system_keywords):
        for q, a in question_answer_map.items():
            if any(k in q for k in ["密码", "登录", "账号"]):
                return f"【系统问题】\n问题：{q}\n答案：{a}"
        return qwen_chat(f"用户忘记考试系统密码，给出清晰的4步找回建议，每步简洁明了，不超过50字。")

    recommend_directions = ["计算机类", "建筑类", "工程类"]
    user_text = user_input.strip()
    # Trigger category-based recommendations when a direction is explicit.
    matched_direction = None
    for d in recommend_directions:
        if d in user_text:
            matched_direction = d
            break

    if matched_direction:
        direction_map = {
            "计算机类": ["计算机技术与软件专业技术资格考试"],
            "建筑类": ["二级建造师考试"],
            "工程类": ["咨询工程师", "二级建造师考试"]
        }
        exams = direction_map.get(matched_direction, [])
        if exams:
            exam_list_text = "\n".join([f"- {e}" for e in exams])
            return f"""根据你的方向【{matched_direction}】，目前系统内可推荐的考试有：
{exam_list_text}

你想了解哪一个？比如：报名条件、报名时间或考试内容？"""
        else:
            return f"当前方向【{matched_direction}】暂无可推荐的考试。你可以换个方向试试。"


    # 2. Parse entities and intent.
    entities = extract_entities(user_input)
    intent = predict_intent(user_input)
    
    # Preserve the extracted exam text for user-facing fallback messages.
    raw_exam = entities.get("EXAM", "").strip()
    current_exam = raw_exam
    
    # Inherit the prior exam only when the current query omits one.
    exam = current_exam
    if not exam and LAST_EXAM:
        common_queries = ["在哪", "怎么", "条件", "时间", "入口", "满足", "免试"]
        if any(kw in user_input for kw in common_queries):
            exam = LAST_EXAM
            print(f"[DEBUG] Inherited exam from conversation context: {exam}")



    # 3. Normalize the exam name and match it to a knowledge-base scope.
    norm_exam = normalize_exam(exam)
    # Prefer the exact canonical names loaded from Excel.
    if norm_exam in exam_question_map:
        matched_exam = norm_exam
    else:
        # Fall back to fuzzy matching over available exam names.
        matched_exam = fuzzy_match(norm_exam, list(exam_question_map.keys()), threshold=0.1)
    
    # Case A: the knowledge base has no matching exam.
    if not matched_exam:
        # Route recommendation questions to a guided exam selection.
        recommend_keywords = ["推荐", "适合", "选哪个", "哪个好", "考哪个", "可以报名的考试", "报什么"]
        if any(k in user_input for k in recommend_keywords):
            # Build the options from exams actually loaded from Excel.
            available_exams = list(exam_question_map.keys())

            guide_text = "目前系统已收录的考试包括：\n"
            for i, exam_name in enumerate(available_exams[:5], 1):
                guide_text += f"{i}. {exam_name}\n"

            guide_text += "\n你更偏向哪个方向呢？比如：建筑类、计算机类、工程类？我可以继续帮你筛选。"
            return guide_text

        # The user named an exam that is not covered.
        display_name = raw_exam.strip() if raw_exam else ""
        display_name = "" if display_name.lower() in ["", "nan", "none"] else display_name

        return f"抱歉，本系统目前暂未收录【{display_name}】的相关信息，建议您关注官方公告或咨询对应主管部门。"

    # Case B: remember the matched exam and search its FAQ scope.
    LAST_EXAM = matched_exam
    print(f"[DEBUG] Selected FAQ scope: {matched_exam}")


    # 4. Apply deterministic keyword rules in priority order.
    candidate_questions = exam_question_map.get(matched_exam, [])
    action = entities.get("ACTION", "")
    # Refine the extracted action with the intent classifier.
    if intent == "报名意图": 
        action = "报名"
    if intent == "报名时间": 
        action = "报名"  
    if intent == "证书发放":
        action = "证书"
    
    # Give registration-location queries their own action.
    if entities.get("ACTION") == "报名" and any(kw in user_input for kw in ["在哪", "哪里", "地址", "官网", "在哪里"]):
        action = "报名地址"

    # Map query terms to required FAQ terms, ordered by priority.
    match_rules = [
    # Eligibility and requirements
    (["我想", "我适合", "条件", "要求", "资格", "能不能报", "可以报名吗","我要","我可以报考","我可以报名"], 
    ["报考条件", "报名条件", "报考资格"]),

    # Registration location and entry point
    (["在哪", "哪里", "地址", "官网", "入口", "网站", "在哪里"],
    ["报名入口", "官网", "报名地址", "在哪里报名", "报名网站"]),

    # Required documents
    (["材料", "证明", "上传什么", "需要什么材料"], 
    ["材料", "证明", "上传"]),

    # Registration dates
    (["什么时候报名", "几号报名", "哪天报名", "报名时间", "报名啥时候", "还能报名吗"], 
    ["报名时间"]),

    # Exam dates
    (["什么时候考", "几号考", "考试时间", "哪天考试"], 
    ["考试时间"]),

    # Registration process
    (["流程", "步骤", "怎么报", "如何报名"], 
    ["报名流程", "报考流程", "报名步骤"]),

    # Registration errors and exceptional states
    (["怎么办", "未审核", "缴费失败", "状态异常", "提交成功"], 
    ["未审核", "缴费", "状态", "提交"]),

    # Exemptions and additional specializations
    (["增项", "相应专业", "免试", "减免"], 
    ["增项", "相应专业", "免试"]),

    # Certificate delivery
    (["证书", "发放", "怎么领取", "发证"], 
    ["证书", "发放", "领取"]),
]

    # Select the first matching FAQ according to rule priority.
    best_q = ""
    user_input_lower = user_input.lower()
    # Step 1: apply exact keyword rules.
    for user_kw, db_kw in match_rules:
        # Enter a rule when any of its user-query terms is present.
        if any(kw in user_input_lower for kw in user_kw):
            # Find the first scoped FAQ containing a required term.
            for q in candidate_questions:
                q_lower = q.lower()
                if any(kw in q_lower for kw in db_kw):
                    best_q = q
                    break
            # Stop after the highest-priority match.
            if best_q:
                break
    # Step 2: use character similarity for less common questions.
    if not best_q and candidate_questions:
        if any(k in user_input for k in ["报名", "报考", "考试"]):
            max_sim = 0
            for q in candidate_questions:
                sim = similarity(user_input, q)
                if sim > max_sim:
                    max_sim = sim
                    best_q = q

    # 5. Return the curated answer when a question matched.
    if best_q:
        ans = question_answer_map[best_q].strip()
        # Append a consistent verification notice to eligibility answers.
        if any(kw in best_q.lower() for kw in ["报考条件", "报名条件", "报考资格"]):
            return f"【{matched_exam}】\n问题：{best_q}\n答案：{ans}\n\n💡 助手提示：请您对照上述要求自行核实学历、年限及属地等条件。若不确定，请以官方审核结果为准。"
        else:
            return f"【{matched_exam}】\n问题：{best_q}\n答案：{ans}"

    # Final fallback: pass bounded context to Qwen.
    urls = extract_urls_from_exam(matched_exam, question_answer_map)
    city = entities.get("CITY", "全国")
    prompt = f"""
你现在是【政务咨询机器人】。
只能基于【已知信息】回答。如果信息不足，请引导用户查看官网。
请根据已知信息给予回复。如果无法直接回答，请告知用户库中暂无此细节，并引导用户查看官网。
回复要求：简洁、专业，末尾可以加一句温馨提示。

【禁止行为】：
- 严禁输出任何感性鼓励的话（如：加油、祝你成功、坚持等）。
- 严禁编造日期和链接。



【用户提问】：
{user_input}

【政务格式回答】：
- 回答必须简洁专业，避免冗余。

【考试】
{matched_exam}

【地区】
{city or "全国"}

【已知信息】
"""
    for q in candidate_questions[:8]:
        a = question_answer_map.get(q, "暂无").replace("\n", " ")
        prompt += f"- {q} → {a}\n"

    if urls:
        prompt += "已知官网：\n" + "\n".join(urls) + "\n"

    prompt += f"""
【用户问题】
{user_input}

【要求】
- 只基于已知信息
- 无法确定请明确说明
- 不新增网址、不猜时间
"""

    return handle_unanswered_question(user_input, matched_exam, prompt)


# 7. Interactive command-line demo
if __name__ == "__main__":
    print("=== 考试问答系统 ===")
    while True:
        user_input = input("你：")
        if user_input.lower() in ["退出", "quit"]:
            print("系统：再见！")
            break
        answer = get_answer_from_exam_db(user_input)
        print(f"系统：{answer}\n")
