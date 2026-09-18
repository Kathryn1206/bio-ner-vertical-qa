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
    
    # Load the tokenizer from the configured local directory.
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        local_qwen_path,
        trust_remote_code=True,
        padding_side="right"
    )
    qwen_tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}"

    
    # Load the 1.8B model directly in FP16; quantization is not required here.
    qwen_model = AutoModelForCausalLM.from_pretrained(
        local_qwen_path,
        torch_dtype=torch.float16,  # Approximately 3-4 GB of VRAM for this model.
        device_map="auto",  # Select available compute devices automatically.
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval()  # Inference mode
    qwen_model.generation_config.stream_generator = False #
    print("Local Qwen-1.8B model loaded successfully.")
    return qwen_model, qwen_tokenizer
# Initialize the language model once at import time.
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
            temperature=0.3,
            top_p=0.85,
            eos_token_id=qwen_tokenizer.eos_token_id,
            pad_token_id=qwen_tokenizer.pad_token_id, 
            num_return_sequences=1, 
            repetition_penalty=1.1  
        )
    
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = qwen_tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()

    # Stop before common filler phrases.
    for stop_word in ["不断", "尝试", "探索", "提升", "能力", "学习"]:
        if stop_word in response:
            response = response.split(stop_word)[0].strip("。；; ")

    # Collapse repeated blank lines.
    response = re.sub(r"\n+", "\n", response)

    # Enforce a short fallback response for this public-service use case.
    response = response[:50]

    return response

def handle_unanswered_question(user_input, context=""):
    """Return either a fixed fallback or a constrained Qwen response."""
    if USE_FIXED_REPLY_FOR_UNANSWERED:
        return "抱歉，该问题暂无相关信息，建议您关注官方公告或联系客服咨询。"
    else:
        prompt = f"""
你是一个专业、谨慎的考试问答助手，只能基于已知信息回答，不允许编造事实。

【用户问题】
{user_input}

{context}

【回答规则】
1. 若问题涉及报名、考试时间、地点等，请说明"请以当地人社局或住建局通知为准"；
2. 若问题涉及具体操作流程，请给出通用建议；
3. 若无法从已知信息推断，请明确回答"该问题暂无明确官方信息，建议关注官方公告或联系客服"。

请直接给出简洁、专业的回答：
"""
        return qwen_chat(prompt)


# 3. Excel knowledge-base loader
def load_exam_database_from_folder(folder_path):
    """Load all .xlsx files in a folder into a unified FAQ knowledge base."""
    # Maps retain the original Chinese exam, question, and answer text.
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
                exam_name = str(row.get("分类一", "")).strip()
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
        if any(alias in user_input for alias in aliases):
            rule_exam = canonical
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


# 6. Core retrieval and response pipeline
def get_answer_from_exam_db(user_input):
    global LAST_EXAM

    # 1. Handle account and login issues before exam routing.
    system_keywords = ["忘记", "密码", "登录", "登陆", "账号", "无法登录"]
    if any(k in user_input for k in system_keywords):
        for q, a in question_answer_map.items():
            if any(k in q for k in ["密码", "登录", "账号"]):
                return f"【系统问题】\n问题：{q}\n答案：{a}"
        return qwen_chat(f"用户遇到登录问题：{user_input}\n请给出通用找回建议，不超过30字。")

    # 2. Parse entities and intent.
    entities = extract_entities(user_input)
    intent = predict_intent(user_input)
    
    # Read the exam extracted from the current query.
    current_exam = entities.get("EXAM", "")
    
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
    if norm_exam not in list(exam_question_map.keys()):
        # Fall back to fuzzy matching for small naming differences.
        norm_exam = fuzzy_match(norm_exam, list(exam_question_map.keys()))

    # Lock the final value to an exam name available in the knowledge base.
    if norm_exam in exam_keywords:
        matched_exam = norm_exam
    else:
        matched_exam = fuzzy_match(norm_exam, exam_keywords, threshold=0.1)

    # Case A: the knowledge base has no matching exam.
    if not matched_exam:
        # Clean the extracted name before displaying it.
        display_name = str(exam).strip() if exam else ""
        # Do not surface invalid placeholder values.
        display_name = "" if display_name.lower() in ["", "nan", "none"] else display_name
        
        if display_name:
            # The user named an exam that is not covered.
            return f"抱歉，本系统目前暂未收录【{display_name}】的相关信息，建议您关注官方公告或咨询对应主管部门。"
        else:
            # Ask for an exam name when none was identified.
            return "您好！本系统主营政务及专业技术资格考试咨询，请告知您想咨询的具体考试名称~"

    # Case B: remember the matched exam and search its FAQ scope.
    LAST_EXAM = matched_exam
    print(f"[DEBUG] Selected FAQ scope: {matched_exam}")

    # 4. Score FAQ candidates within the selected exam scope.
    candidate_questions = exam_question_map.get(matched_exam, [])
    action = entities.get("ACTION", "")
    # Refine the extracted action with the intent classifier.
    if intent == "报名意图": 
        action = "报名"
    elif intent == "报名时间": 
        action = "报名"  
    elif intent == "证书发放":
        action = "证书"

    def score_question(q):
        score = 0
        q_l, u_l = q.lower(), user_input.lower()
        
        # 1. Reward exam-name agreement to prevent cross-exam matches.
        if matched_exam.lower() in q_l: 
            score += 5 
        
        # Priority 1: eligibility and requirements.
        condition_keywords = ["我想", "我适合", "我可以", "能不能", "符合吗", "条件", "要求", "资格"]
        if any(kw in u_l for kw in condition_keywords):
            if any(target in q_l for target in ["报考条件", "报名条件", "报名要求", "报考资格"]):
                score += 35
                return score
        
        # Priority 2: registration location and entry point.
        # Exclude residence-policy and document questions from this route.
        address_keywords = ["在哪报名", "哪里报", "报名入口", "官网", "网站", "报名地址"]
        if any(kw in u_l for kw in address_keywords) or (any(k in u_l for k in ["在哪", "哪里"]) and "报名" in u_l):
            # Require explicit site or address terms in the FAQ candidate.
            if any(target in q_l for target in ["报名入口", "官网", "报名地址", "报名网站"]) and not any(exclude in q_l for exclude in ["属地", "证明", "材料"]):
                score += 30
                return score
        
        # Priority 3: required documents and supporting evidence.
        elif any(k in u_l for k in ["材料", "证明", "上传什么"]):
            if any(target in q_l for target in ["材料", "证明", "上传"]):
                score += 25
                return score
        
        # Priority 4: distinguish registration dates from exam dates.
        elif any(k in u_l for k in ["时间", "几号", "日期", "什么时候"]):
            if "报名时间" in q_l:
                score += 20
            elif "考试时间" in q_l:
                score += 18
            return score
        
        # Other supported categories, such as exemptions.
        elif any(k in u_l for k in ["免试", "减免"]):
            if "免试" in q_l: 
                score += 20
        
        # Add a small character-overlap score.
        overlap = len(set(user_input) & set(q_l)) / max(len(set(user_input)), 1)
        score += overlap * 5
        
        return score


        
    best_q, best_score = "", 0
    used_questions = set()
    for q in candidate_questions:
        if q in used_questions:
            continue
        s = score_question(q)
        if s > best_score:
            best_q, best_score = q, s
    used_questions.add(best_q)

    # 5. Return the best curated answer above the threshold.
    if best_score >= 3:
        ans = question_answer_map[best_q].strip()
        # Append a consistent verification notice to eligibility answers.
        if any(kw in best_q.lower() for kw in ["报考条件", "报名条件", "报考资格"]):
            return f"【{matched_exam}】\n问题：{best_q}\n答案：{ans}\n\n💡 助手提示：请您对照上述要求自行核实学历、年限及属地等条件。若不确定，请以官方审核结果为准。"
        
        # Preserve the standard response format for other answer types.
        elif any(kw in user_input for kw in ["推荐", "适合问"]):
            return f"根据您的需求，为您找到【{matched_exam}】的相关信息：\n{ans}"
        
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

    return qwen_chat(prompt)


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
