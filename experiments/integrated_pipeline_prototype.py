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
    "INTENT_MODEL_PATH", str(PROJECT_ROOT / "experiments" / "intent_model.pkl")
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
        padding_side="right"
    )
    qwen_tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}"

    
    # 加载千问模型（1.8B显存占用低，不用量化，直接跑）
    qwen_model = AutoModelForCausalLM.from_pretrained(
        local_qwen_path,
        torch_dtype=torch.float16,  # 1.8B用float16足够，显存约3-4GB
        device_map="auto",  # 自动使用可用的计算设备
        low_cpu_mem_usage=True,
        trust_remote_code=True
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

    # 1️⃣ 如果有明显“废话模板”，只截到第一句
    for stop_word in ["不断", "尝试", "探索", "提升", "能力", "学习"]:
        if stop_word in response:
            response = response.split(stop_word)[0].strip("。；; ")

    # 2️⃣ 去掉多余空行
    response = re.sub(r"\n+", "\n", response)

    # 3️⃣ 最终兜底：只保留前 50 字（政务安全阈值）
    response = response[:50]

    return response

def handle_unanswered_question(user_input, context=""):
    """处理无法回答的问题，根据配置选择固定回复或Qwen生成"""
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
                exam_name = str(row.get("分类一", "")).strip()
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
        if any(alias in user_input for alias in aliases):
            rule_exam = canonical
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


# ===================== 6. 核心：实体匹配Excel，返回答案 =====================
def get_answer_from_exam_db(user_input):
    global LAST_EXAM  # 声明引用全局变量
    
    # 1. 系统级拦截（密码/登录问题）
    system_keywords = ["忘记", "密码", "登录", "登陆", "账号", "无法登录"]
    if any(k in user_input for k in system_keywords):
        for q, a in question_answer_map.items():
            if any(k in q for k in ["密码", "登录", "账号"]):
                return f"【系统问题】\n问题：{q}\n答案：{a}"
        return qwen_chat(f"用户遇到登录问题：{user_input}\n请给出通用找回建议，不超过30字。")

    # 2. 实体与意图解析
    entities = extract_entities(user_input)
    intent = predict_intent(user_input)
    
    # 获取本次提取到的考试
    current_exam = entities.get("EXAM", "")
    
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
    if norm_exam not in list(exam_question_map.keys()):
        # 额外加一次模糊匹配，防止名称细微差异
        norm_exam = fuzzy_match(norm_exam, list(exam_question_map.keys()))

    #===== 强制锁定长名称 =====
    if norm_exam in exam_keywords:
        matched_exam = norm_exam
    else:
        matched_exam = fuzzy_match(norm_exam, exam_keywords, threshold=0.1)

 # --- 情况 A: 彻底没这个考试 (例如用户提了“教资”，但库里没匹配到) ---
    if not matched_exam:
        # 提取并清洗要展示的考试名，空值/无效值置空
        display_name = str(exam).strip() if exam else ""
        # 过滤无效值（避免显示nan/None等异常字符）
        display_name = "" if display_name.lower() in ["", "nan", "none"] else display_name
        
        if display_name:
            # 有明确考试名（如教资/教师资格考试）→ 直接提示未收录
            return f"抱歉，本系统目前暂未收录【{display_name}】的相关信息，建议您关注官方公告或咨询对应主管部门。"
        else:
            # 无明确考试名 → 简洁提示并引导，不重复冗余
            return "您好！本系统主营政务及专业技术资格考试咨询，请告知您想咨询的具体考试名称~"

    # --- 情况 B: 匹配成功，更新记忆并进入知识库检索 ---
    LAST_EXAM = matched_exam
    print(f"【DEBUG】最终锁定题库: {matched_exam}")

    # 4. 知识库匹配得分逻辑
    candidate_questions = exam_question_map.get(matched_exam, [])
    action = entities.get("ACTION", "")
    # 结合意图识别结果调整 action
    if intent == "报名意图": 
        action = "报名"
    elif intent == "报名时间": 
        action = "报名"  
    elif intent == "证书发放":
        action = "证书"

    def score_question(q):
        score = 0
        q_l, u_l = q.lower(), user_input.lower()
        
        # 1. 考试名称匹配（基础分，防止跨考试）
        if matched_exam.lower() in q_l: 
            score += 5 
        
        # 🌟 优先级1：条件类提问（35分，覆盖“我想/我适合”）
        condition_keywords = ["我想", "我适合", "我可以", "能不能", "符合吗", "条件", "要求", "资格"]
        if any(kw in u_l for kw in condition_keywords):
            if any(target in q_l for target in ["报考条件", "报名条件", "报名要求", "报考资格"]):
                score += 35
                return score
        
        # 🌟 优先级2：地址/入口类提问（30分，明确关键词，排除“属地/材料”）
        # 核心：只匹配“在哪报名/入口/官网/网站”，不包含“属地/证明/材料”
        address_keywords = ["在哪报名", "哪里报", "报名入口", "官网", "网站", "报名地址"]
        if any(kw in u_l for kw in address_keywords) or (any(k in u_l for k in ["在哪", "哪里"]) and "报名" in u_l):
            # 只匹配含“入口/官网/地址/网站”的问题，排除“属地/证明/材料”
            if any(target in q_l for target in ["报名入口", "官网", "报名地址", "报名网站"]) and not any(exclude in q_l for exclude in ["属地", "证明", "材料"]):
                score += 30
                return score
        
        # 🌟 优先级3：材料/证明类提问（25分，单独分类，不干扰地址）
        elif any(k in u_l for k in ["材料", "证明", "上传什么"]):
            if any(target in q_l for target in ["材料", "证明", "上传"]):
                score += 25
                return score
        
        # 🌟 优先级4：时间类提问（20分，区分报名/考试时间）
        elif any(k in u_l for k in ["时间", "几号", "日期", "什么时候"]):
            if "报名时间" in q_l:
                score += 20
            elif "考试时间" in q_l:
                score += 18
            return score
        
        # 其他类型（免试等，20分）
        elif any(k in u_l for k in ["免试", "减免"]):
            if "免试" in q_l: 
                score += 20
        
        # 字符重叠度（补充分）
        overlap = len(set(user_input) & set(q_l)) / max(len(set(user_input)), 1)
        score += overlap * 5
        
        return score


        
    best_q, best_score = "", 0
    used_questions = set() # 新增：记录已返回的问题
    for q in candidate_questions:
        if q in used_questions:
            continue
        s = score_question(q)
        if s > best_score:
            best_q, best_score = q, s
    used_questions.add(best_q) # 新增：标记已使用

    # 5. 结果处理
    if best_score >= 3:
        ans = question_answer_map[best_q].strip()
        # 🌟 核心优化：所有条件类回答强制加统一提示
        # 只要问题含“条件/资格”，无论用户提问是否带“适合”，都加提示
        if any(kw in best_q.lower() for kw in ["报考条件", "报名条件", "报考资格"]):
            return f"【{matched_exam}】\n问题：{best_q}\n答案：{ans}\n\n💡 助手提示：请您对照上述要求自行核实学历、年限及属地等条件。若不确定，请以官方审核结果为准。"
        
        # 其他类型回答保持原有逻辑
        elif any(kw in user_input for kw in ["推荐", "适合问"]):
            return f"根据您的需求，为您找到【{matched_exam}】的相关信息：\n{ans}"
        
        else:
            return f"【{matched_exam}】\n问题：{best_q}\n答案：{ans}"

    # 最终兜底：交给 Qwen，但带上约束信息
    urls = extract_urls_from_exam(matched_exam, question_answer_map)
    city = entities.get("CITY", "全国") # 修正变量名
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
