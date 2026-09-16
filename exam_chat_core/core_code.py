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

LAST_EXAM = None   # 全局变量：用于指代消解
warnings.filterwarnings("ignore")
# ===================== 1. 配置参数 =====================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIO_MODEL_PATH = os.getenv(
    "BIO_MODEL_PATH", str(PROJECT_ROOT / "exam_bio_final_model")
)
EXCEL_FOLDER = os.getenv("FAQ_DATA_DIR", str(PROJECT_ROOT / "faq_data"))
INTENT_MODEL_PATH = os.getenv(
    "INTENT_MODEL_PATH", str(PROJECT_ROOT / "核心模型算法" / "intent_model.pkl")
)
QWEN_MODEL_PATH = os.getenv("QWEN_MODEL_PATH", str(PROJECT_ROOT / "Qwen"))

# 全局设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 配置：无法回答问题的处理方式 =====================
# True: 使用固定回复（更快，但回复固定）
# False: 接入Qwen生成回复（更灵活，但需要调用模型）
USE_FIXED_REPLY_FOR_UNANSWERED = False


# ===================== 2.1. 加载BIO模型（适配你的考试实体识别） =====================
def load_bio_model():
    labels = ['O', 'B-EXAM', 'I-EXAM', 'B-SUBJECT', 'I-SUBJECT',
        'B-TIME', 'I-TIME', 'B-ACTION', 'I-ACTION',
        'B-CONSTRAINT', 'I-CONSTRAINT', 'B-CITY', 'I-CITY']
    id_to_label = {i: label for i, label in enumerate(labels)}
    
    # 加载模型和分词器
    tokenizer = BertTokenizer.from_pretrained(BIO_MODEL_PATH)
    model = BertForTokenClassification.from_pretrained(BIO_MODEL_PATH)
    model = model.to(DEVICE)
    model.eval()
    
    return tokenizer, model, id_to_label

bio_tokenizer, bio_model, bio_id_to_label = load_bio_model()

# ======================= 2.1 加载意图识别模型 =======================
def load_intent_model():
    try:
        print(f"📂 正在加载意图模型: {INTENT_MODEL_PATH}")
        model = joblib.load(INTENT_MODEL_PATH)
        print("✅ 意图模型加载成功")
        return model
    except Exception as e:
        print(f"❌ 加载意图模型失败: {e}")
        return None


# ======================= 初始化意图模型=======================

intent_classifier = load_intent_model()



#======================= 2.2 加载意图识别模型=======================
def predict_intent(text):
    if intent_classifier:
        try:
            # 模型预测返回的是你在 train_intent.py 里定义的 label
            label = intent_classifier.predict([text])[0]
            return label
        except Exception as e:
            print(f"意图预测出错: {e}")
            return None
    return None



#====================== 2.3 加载千问模型======================
def load_local_qwen():
    # 本地Qwen-1.8B路径，可通过环境变量 QWEN_MODEL_PATH 覆盖
    local_qwen_path = QWEN_MODEL_PATH
    
    # 加载千问分词器（用本地路径，不重新下载）
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        local_qwen_path,
        trust_remote_code=True,
        padding_side="right",
        local_files_only=True
    )
    qwen_tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}"

    
    # 加载千问模型（1.8B显存占用低，不用量化，直接跑）
    qwen_model = AutoModelForCausalLM.from_pretrained(
        local_qwen_path,
        torch_dtype=torch.float16,  # 1.8B用float16足够，显存约3-4GB
        device_map="auto",  # 自动使用可用的计算设备
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True 
    ).eval()  # 预测模式，不训练
    qwen_model.generation_config.stream_generator = False #
    print("✅ 本地Qwen-1.8B加载完成！")
    return qwen_model, qwen_tokenizer
# 调用函数加载本地千问（只运行一次）
qwen_model, qwen_tokenizer = load_local_qwen()


# ===================== 2.3.1 新增：Qwen对话函数（缺失会导致报错） =====================
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
            temperature=0.2 ,  # 控制生成文本的随机性
            top_p=0.85,
            eos_token_id=qwen_tokenizer.eos_token_id,
            pad_token_id=qwen_tokenizer.pad_token_id, 
            num_return_sequences=1, 
            repetition_penalty=1.2,
            early_stopping=True,  # 启用早停，触发EOS立即停止
            stopping_criteria=[torch.nn.CrossEntropyLoss()]  # 辅助早停  
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

    # 删掉stop_word提前截断逻辑（原核心问题）
    # 50字硬截改100字，符合你的要求
    response = response[:100]
    # 保留去空行逻辑
    response = re.sub(r"\n+", "\n", response)

    return response

def handle_unanswered_question(user_input, matched_exam="", context=""):
    """处理无法回答的问题，先给考试介绍，再根据是否收录给出不同引导"""
    try:
        # 过滤考试名无效值（彻底解决nan/None导致的判断错误）
        matched_exam = matched_exam.strip() if matched_exam else ""
        matched_exam = "" if str(matched_exam).lower() in ["nan", "none", ""] else matched_exam
        # 判断考试是否在数据库中
        is_in_db = matched_exam and (matched_exam in exam_question_map)
        
        # 生成考试介绍（必须调用Qwen，失败兜底）
        introduction = ""
        if matched_exam:
            try:  # 这里缩进和if同级，统一4个空格
                # 明确要求80-100字，补充考核方向，让内容更完整
                intro_prompt = f"""请用80-100字详细介绍【{matched_exam}】，说明其性质、报考意义、行业价值和核心考核方向，语言正式专业。
                ⚠️ 严格禁止：
                1. 添加知识库更新、合作关系、服务承诺等客服套话；
                2. 出现个人生活、感受、变化等无关内容；
                3. 生成完考试介绍后立即停止，不额外延伸。"""
                # 调低max_new_tokens，避免模型“凑字数”发散
                introduction = qwen_chat(intro_prompt, max_new_tokens=100)
                # 放宽有效判断，避免误判
                if not introduction or len(introduction.strip()) < 30:
                    raise ValueError("生成内容过短或为空")
            except Exception as e:  # 这一行缩进和上面的try完全匹配（4个空格），解决第166行错误
                print(f"⚠️ Qwen生成介绍失败: {e}")
                # 加长兜底文本，确保字数
                introduction = f"【{matched_exam}】是国家统一组织的执业资格考试，旨在考核专业知识和实践能力，是担任项目经理的必备条件，具有较高的行业认可度和职业价值。"
        else:
            introduction = "您好！我是人事考试智能客服。"
        
        # 过滤数据库无效考试名（避免展示nan/空值）
        available_exams = [
            exam for exam in list(exam_question_map.keys()) 
            if str(exam).lower() not in ["nan", "none", ""]
        ][:5]
        exam_list = "\n".join([f"- {exam}" for exam in available_exams])
        
        # 根据是否收录给出不同的引导信息
        if not matched_exam:
            # 没有匹配到考试名称
            return f"""{introduction}

请告诉我您想咨询哪个考试？或者从以下已收录考试中选择：
{exam_list}"""
        
        elif is_in_db:
            # 已收录的考试：截取问题时做非空判断，避免索引错误
            example_questions = exam_question_map.get(matched_exam, [])[:3]
            examples = ""
            if example_questions:
                # 对每个问题做非空/截取处理，避免空字符串报错
                examples = "\n例如：\n" + "\n".join([f"- {q[:30]}..." if q and len(q)>30 else f"- {q}" for q in example_questions])
            
            # 生成更详细的引导（再次调用Qwen，失败兜底）
            try:
                guide_prompt = f"""用户询问了【{matched_exam}】这项考试。
                
已生成考试介绍：{introduction}

现在需要给用户一个引导，建议他们可以询问哪些具体问题。
请生成一个简短、友好的引导语，包含2-3个示例问题，如报名条件、考试时间、报名地点等。
要求：不超过40字，语气亲切。"""
                guide_text = qwen_chat(guide_prompt, max_new_tokens=40)
                # 过滤Qwen生成的无效引导语
                guide_text = guide_text if guide_text and len(guide_text.strip())>5 else ""
            except Exception as e:      
                print(f"⚠️ Qwen生成引导失败: {e}")
                # 备选引导
                guide_text = "您可以直接问我具体问题，比如报名条件、考试时间、在哪里报名等。"
            
            return f"""{introduction}

{guide_text if guide_text else '您可以告诉我具体想了解的问题，我会为您详细解答~如报名条件、考试时间、报名地点等'}"""
        
        else:
            # 未收录的考试
            try:
                guide_prompt = f"""用户询问了【{matched_exam}】这项考试，但系统暂未收录该考试的信息。
                
已生成考试介绍：{introduction}

系统已收录的考试包括：{', '.join(available_exams[:3]) if available_exams else '二级建造师考试、计算机技术与软件专业技术资格考试'}

请生成一个简短、友好的引导语，建议用户从已收录的考试中选择，或者告诉用户如何提问。
要求：不超过50字，语气礼貌。"""
                guide_text = qwen_chat(guide_prompt, max_new_tokens=70)
                guide_text = guide_text if guide_text and len(guide_text.strip())>5 else ""
            except Exception as e:
                print(f"⚠️ Qwen生成引导失败: {e}")
                # 备选引导
                guide_text = f"目前系统暂未收录【{matched_exam}】的详细问答信息。您可以了解以上已收录的考试，或者告诉我您的专业方向。"
            
            return f"""{introduction}

{guide_text if guide_text else f"目前系统暂未收录【{matched_exam}】的信息，您可从已收录考试中选择咨询~"}"""
            
    except Exception as e:
        print(f"❌ handle_unanswered_question函数出错: {e}")
        # 终极备选方案：做考试名非空判断，避免拼接报错
        if matched_exam:
            return f"关于【{matched_exam}】这项考试，我可以为您解答报名条件、考试时间、报名地点等问题，请具体提问。"
        else:
            return "请告诉我您想咨询哪个考试？"





# ===================== 3. 加载Excel数据库（适配你的列：考试名称、分类、问题、答案） =====================
def load_exam_database_from_folder(folder_path):
    """
    从指定文件夹加载所有 .xlsx 文件，合并为统一的知识库
    """
    # 初始化全局映射
    exam_question_map = {}      # {考试名称: [问题列表]}
    question_answer_map = {}    # {问题: 答案}
    exam_keywords = set()       # 所有考试名称（用 set 避免重复）

    # 获取文件夹中所有 .xlsx 文件
    excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    
    if not excel_files:
        print(f"⚠️ 警告：文件夹 {folder_path} 中没有找到 .xlsx 文件！")
        return {}, {}, []

    print(f"📂 正在加载 {len(excel_files)} 个 Excel 文件...")

    for file_path in excel_files:
        try:
            print(f"  → 加载: {os.path.basename(file_path)}")
            df = pd.read_excel(file_path, sheet_name=0, engine="openpyxl")
            
            for _, row in df.iterrows():
                exam_name_raw = str(row.get("分类一", "")).strip().replace("\u3000", "").replace("　", "").replace(" ", "")
                if exam_name_raw.lower() == "nan" or exam_name_raw == "":
                    continue

                # ✅ 统一映射
                if exam_name_raw in ["软件水平考试", "软考"]:
                    exam_name = "计算机技术与软件专业技术资格考试"
                else:
                    exam_name = exam_name_raw
                if exam_name.lower() == "nan" or exam_name == "":
                    continue
                question = str(row.get("问题", "")).strip()
                answer = str(row.get("答案", "")).strip()

                if not exam_name or not question or not answer:
                    continue  # 跳过空行

                # 填充映射
                if exam_name not in exam_question_map:
                    exam_question_map[exam_name] = []
                exam_question_map[exam_name].append(question)
                question_answer_map[question] = answer
                if exam_name and exam_name not in ("", "nan", "NaN"):
                    exam_keywords.add(exam_name)

                if not exam_name or exam_name.lower() == "nan":
                    continue

        except Exception as e:
            print(f"  ❌ 跳过文件 {file_path}（错误：{e}）")

    return exam_question_map, question_answer_map, sorted(list(exam_keywords))

exam_question_map, question_answer_map, exam_keywords = load_exam_database_from_folder(EXCEL_FOLDER)

# ===================== 4. 模糊匹配函数 =====================
def fuzzy_match(query, candidates, threshold=0.1): # 降低默认阈值
    if not query: return ""
    query = query.strip()
    
    # 1. 第一优先级：全匹配
    if query in candidates:
        return query
    
    # 2. 第二优先级：互相包含判断 (针对长名字最有效)
    for candidate in candidates:
        candidate_clean = candidate.strip()
        if query in candidate_clean or candidate_clean in query:
            return candidate
    
    # 3. 第三优先级：模糊相似度 (处理错别字)
    max_sim = 0
    best_candidate = ""
    for candidate in candidates:
        common = len(set(query) & set(candidate))
        sim = common / max(len(query), len(candidate))
        if sim > max_sim and sim >= threshold:
            max_sim = sim
            best_candidate = candidate
    
    return best_candidate
    


#        ============考试归一化函数        ================
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



# ===================== 全局变量：用于指代消解 =====================
LAST_EXAM = None 

# ===================== 修改：归一化函数（增加长名称强制锁定） =====================
def normalize_exam(exam_query):
    if not exam_query:
        return ""
    
    # 强制补丁：针对长名称和常见简称的硬锁定
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
    """封装原本散落在外的实体识别逻辑"""
    entities = {"EXAM": "", "ACTION": "", "CONSTRAINT": "", "CITY": "", "TIME": ""}
    
    # --- 1. 规则匹配城市 ---
    known_cities = {"遂宁", "成都", "绵阳", "德阳", "泸州", "宜宾", "南充", "达州", "资阳", "自贡", "内江", "乐山", "眉山", "广安", "雅安", "广元", "巴中", "攀枝花"}
    rule_city = ""
    for city in known_cities:
        if city in user_input:
            rule_city = city
            break

# --- 2. 规则匹配考试 ---
    rule_exam = ""
    for canonical, aliases in EXAM_ALIAS.items():
        sorted_aliases = sorted(aliases, key=lambda x: len(x), reverse=True)
        for alias in sorted_aliases:
            # 去掉\b，直接匹配字符串包含关系（适配中文）
            if alias in user_input:
                rule_exam = canonical
                break
        if rule_exam:
            break

    # --- 3. BERT 模型预测 ---
    text_chars = list(user_input)
    encoded = bio_tokenizer(text_chars, is_split_into_words=True, return_tensors='pt', padding='max_length', truncation=True, max_length=510).to(DEVICE)
    with torch.no_grad():
        outputs = bio_model(**encoded)
        pred_ids = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().tolist()
        bio_labels = [bio_id_to_label[id] for id in pred_ids][1:len(text_chars)+1]
    
    # --- 4. 解析 BIO 标签 ---
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

    # --- 5. 逻辑融合 ---
    entities["EXAM"] = rule_exam if rule_exam else model_entities.get("EXAM", "")
    entities["CITY"] = rule_city if rule_city else model_entities.get("CITY", "")
    entities["ACTION"] = model_entities.get("ACTION", "")
    entities["CONSTRAINT"] = model_entities.get("CONSTRAINT", "")
    entities["TIME"] = model_entities.get("TIME", "")

    if len(entities["EXAM"]) < 2 and not rule_exam: entities["EXAM"] = ""
    
    # --- 6. 修复：针对“教资”类考试的补充规则 ---
    if not entities["EXAM"]:
        fallback_exam_keywords = ["教资", "教师资格", "教师资格证"]
        for kw in fallback_exam_keywords:
            if kw in user_input:
                entities["EXAM"] = kw   # 保留用户原话
                break
    
    # 【修复：补充函数返回值，必须有这行】
    return entities



#======================= 5.2.4 新增：报名类信息查找函数 =====================
def search_registration_info(matched_exam, question_answer_map):
    """
    在数据库中查找某考试的报名 / 地址 / 官网相关问题
    """
    for q, a in question_answer_map.items():
        place_keywords = ["在哪", "哪里", "地址", "官网", "入口"]
        if any(kw in q for kw in place_keywords) and ("报名" in q):
        # 再粗略判断是否属于该考试（避免跨考试匹配）
            if any(alias in q for alias in ["二建", "建造师", "二级建造", matched_exam]):
                return q, a
    return None, None



# ======================== 5.3 新增：从答案中提取URL函数 =====================
def extract_urls_from_exam(matched_exam, question_answer_map):
    urls = set()
    for q, a in question_answer_map.items():
        if matched_exam in q:
            found = re.findall(r"https?://\S+", a)
            urls.update(found)
    return list(urls)

# ======================== 5.4 新增：字符串相似度函数 =====================
def similarity(s1, s2):
    set1, set2 = set(s1), set(s2)
    return len(set1 & set2) / max(len(set1 | set2), 1)


# ===================== 5.5 新增：生成考试介绍函数 =====================
def generate_exam_introduction(exam_name, is_in_database=False):
    """
    生成考试介绍信息
    exam_name: 考试名称
    is_in_database: 该考试是否在数据库中收录
    """
    if is_in_database:
        # 对于已收录的考试，我们可以从数据库中提取一些基本信息
        intro_prompt = f"""
请为【{exam_name}】生成一个简要的专业介绍，要求：
1. 说明考试的性质和目的
2. 简单介绍考试的价值和用途
3. 用专业、官方的语言
4. 字数控制在100字以内
5. 不要包含具体的报名条件、时间等细节（这些会由后续问答提供）
"""
    else:
        # 对于未收录的考试，生成通用介绍
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
        print(f"❌ 生成考试介绍失败: {e}")
        if is_in_database:
            return f"{exam_name}是一项重要的职业资格考试，具有较高的行业认可度和专业价值。"
        else:
            return f"{exam_name}是一项职业资格考试，具体信息建议咨询官方机构。"


# ===================== 6. 核心：实体匹配Excel，返回答案 =====================
def get_answer_from_exam_db(user_input):
    global LAST_EXAM  # 声明引用全局变量

    
    
# 1. 系统级拦截（密码/登录问题）
    system_keywords = ["忘记", "密码", "登录", "登陆", "账号", "无法登录"]
    if any(k in user_input for k in system_keywords):
        for q, a in question_answer_map.items():
            if any(k in q for k in ["密码", "登录", "账号"]):
                return f"【系统问题】\n问题：{q}\n答案：{a}"
        return qwen_chat(f"用户忘记考试系统密码，给出清晰的4步找回建议，每步简洁明了，不超过50字。")

    recommend_directions = ["计算机类", "建筑类", "工程类"]
    user_text = user_input.strip()
    # 只要用户输入里包含这些方向词，就触发推荐
    matched_direction = None
    for d in recommend_directions:
        if d in user_text:
            matched_direction = d
            break

    if matched_direction:
        direction_map = {
            "计算机类": ["计算机技术与软件专业技术资格考试"],  # 改用完整考试名，避免匹配不到
            "建筑类": ["二级建造师考试"],
            "工程类": ["咨询工程师", "二级建造师考试"]
        }
        exams = direction_map.get(matched_direction, [])
        if exams:
            exam_list_text = "\n".join([f"- {e}" for e in exams])
            # 修复返回文本的缩进（去掉多余空格）
            return f"""根据你的方向【{matched_direction}】，目前系统内可推荐的考试有：
{exam_list_text}

你想了解哪一个？比如：报名条件、报名时间或考试内容？"""
        else:
            return f"当前方向【{matched_direction}】暂无可推荐的考试。你可以换个方向试试。"


# 2. 实体与意图解析
    entities = extract_entities(user_input)
    intent = predict_intent(user_input)
    
    # 获取本次提取到的考试
    raw_exam = entities.get("EXAM", "").strip()   # ✅ 用户原始输入
    current_exam = raw_exam
    
    # --- 指代消解逻辑 (逻辑优化：仅在当前没提考试时才继承) ---
    exam = current_exam
    if not exam and LAST_EXAM:
        common_queries = ["在哪", "怎么", "条件", "时间", "入口", "满足", "免试"]
        if any(kw in user_input for kw in common_queries):
            exam = LAST_EXAM
            print(f"【DEBUG】指代消解：继承上下文考试: {exam}")



# 3. 规范化与库匹配
    norm_exam = normalize_exam(exam)
    # 强制用Excel实际加载的考试名列表做匹配
    if norm_exam in exam_question_map:
        matched_exam = norm_exam
    else:
        # 调用现有模糊匹配函数，匹配考试名列表
        matched_exam = fuzzy_match(norm_exam, list(exam_question_map.keys()), threshold=0.1)
    
# --- 情况 A: 彻底没这个考试 (例如用户提了“教资”，但库里没匹配到) ---
    if not matched_exam:
        # 🎯 1️⃣ 推荐类 / 选择类问题 → 进入引导模式
        recommend_keywords = ["推荐", "适合", "选哪个", "哪个好", "考哪个", "可以报名的考试", "报什么"]
        if any(k in user_input for k in recommend_keywords):
            # 从你已加载的 Excel 里拿考试列表
            available_exams = list(exam_question_map.keys())

            # 可以简单按方向分类（可选）
            guide_text = "目前系统已收录的考试包括：\n"
            for i, exam_name in enumerate(available_exams[:5], 1):
                guide_text += f"{i}. {exam_name}\n"

            guide_text += "\n你更偏向哪个方向呢？比如：建筑类、计算机类、工程类？我可以继续帮你筛选。"
            return guide_text

        # 🎯 2️⃣ 用户明确提了考试名，但库里没有
        display_name = raw_exam.strip() if raw_exam else ""
        display_name = "" if display_name.lower() in ["", "nan", "none"] else display_name

        #if display_name:
        return f"抱歉，本系统目前暂未收录【{display_name}】的相关信息，建议您关注官方公告或咨询对应主管部门。"

        # 🎯 3️⃣ 啥也没说清楚 → 普通兜底
        #return "您好！本系统目前收录部分职业资格考试信息。请告知您想咨询哪一类考试？如：建造类、计算机类。"

    # --- 情况 B: 匹配成功，更新记忆并进入知识库检索 ---
    LAST_EXAM = matched_exam
    print(f"【DEBUG】最终锁定题库: {matched_exam}")


# 4. 抛弃打分！改用【关键词精准提取匹配】（替换原有所有score_question相关代码）
    candidate_questions = exam_question_map.get(matched_exam, [])
    action = entities.get("ACTION", "")
    # 结合意图识别结果调整 action（原有逻辑保留）
    if intent == "报名意图": 
        action = "报名"
    if intent == "报名时间": 
        action = "报名"  
    if intent == "证书发放":
        action = "证书"
    
    # 针对“报名+地址”类问题的特殊处理
    if entities.get("ACTION") == "报名" and any(kw in user_input for kw in ["在哪", "哪里", "地址", "官网", "在哪里"]):
        action = "报名地址"

    # ✅ 核心：关键词映射规则（用户提问关键词 → 数据库问题需包含的关键词，按优先级排序）
    # 优先级从上到下，匹配到即锁定答案，不再继续匹配
    match_rules = [
    # 【条件类】
    (["我想", "我适合", "条件", "要求", "资格", "能不能报", "可以报名吗","我要","我可以报考","我可以报名"], 
    ["报考条件", "报名条件", "报考资格"]),

   # 【地址入口类】✅ 仅修改这部分：拆分关键词，覆盖“在哪里”
    (["在哪", "哪里", "地址", "官网", "入口", "网站", "在哪里"],  # 新增“在哪里”，拆分“报名”前缀
    ["报名入口", "官网", "报名地址", "在哪里报名", "报名网站"]),

    # 【材料类】
    (["材料", "证明", "上传什么", "需要什么材料"], 
    ["材料", "证明", "上传"]),

    # 【报名时间类】✅ 关键修复
    (["什么时候报名", "几号报名", "哪天报名", "报名时间", "报名啥时候", "还能报名吗"], 
    ["报名时间"]),

    # 【考试时间类】
    (["什么时候考", "几号考", "考试时间", "哪天考试"], 
    ["考试时间"]),

    # 【报名流程类】
    (["流程", "步骤", "怎么报", "如何报名"], 
    ["报名流程", "报考流程", "报名步骤"]),

    # 【异常处理类】
    (["怎么办", "未审核", "缴费失败", "状态异常", "提交成功"], 
    ["未审核", "缴费", "状态", "提交"]),

    # 【免试/增项类】
    (["增项", "相应专业", "免试", "减免"], 
    ["增项", "相应专业", "免试"]),

    # 【证书类】
    (["证书", "发放", "怎么领取", "发证"], 
    ["证书", "发放", "领取"]),
]

    # ✅ 核心匹配逻辑：遍历规则，按优先级提取第一个匹配的问题
    best_q = ""
    user_input_lower = user_input.lower()
    # 第一步：按优先级匹配精准规则
    for user_kw, db_kw in match_rules:
        # 只要用户提问含任意一个当前规则关键词
        if any(kw in user_input_lower for kw in user_kw):
            # 遍历数据库候选问题，找第一个含数据库关键词的
            for q in candidate_questions:
                q_lower = q.lower()
                if any(kw in q_lower for kw in db_kw):
                    best_q = q
                    break
            # 匹配到后立即退出，按优先级锁定答案
            if best_q:
                break
    # 第二步：兜底匹配（无精准规则时，按字符相似度取最相似的，兼容小众问题）
    if not best_q and candidate_questions:
        if any(k in user_input for k in ["报名", "报考", "考试"]):
            max_sim = 0
            for q in candidate_questions:
                sim = similarity(user_input, q)
                if sim > max_sim:
                    max_sim = sim
                    best_q = q

    # 5. 结果处理（原有逻辑基本保留，仅去掉分数判断）
    if best_q:  # 只要匹配到问题就返回，无需分数阈值
        ans = question_answer_map[best_q].strip()
        # 🌟 核心优化：所有条件类回答强制加统一提示（原有逻辑保留）
        if any(kw in best_q.lower() for kw in ["报考条件", "报名条件", "报考资格"]):
            return f"【{matched_exam}】\n问题：{best_q}\n答案：{ans}\n\n💡 助手提示：请您对照上述要求自行核实学历、年限及属地等条件。若不确定，请以官方审核结果为准。"
        else:
            return f"【{matched_exam}】\n问题：{best_q}\n答案：{ans}"

    # 最终兜底：交给 Qwen，但带上约束信息（原有逻辑保留）
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


# ===================== 7. 测试对话 =====================
if __name__ == "__main__":
    print("=== 考试问答系统 ===")
    while True:
        user_input = input("你：")
        if user_input.lower() in ["退出", "quit"]:
            print("系统：再见！")
            break
        answer = get_answer_from_exam_db(user_input)
        print(f"系统：{answer}\n")
