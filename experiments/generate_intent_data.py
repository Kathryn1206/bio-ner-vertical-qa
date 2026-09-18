# generate_intent_data.py
# ===============================
# 自动生成更真实的 Intent 训练数据（text,label）
# ===============================

import csv
import random

OUTPUT_FILE = "intent_data.csv"
SAMPLES_PER_INTENT = 120   # ⭐ 每个 intent 生成多少条（可调）

# ========= 1️⃣ 基础词库 =========

EXAMS = [
    "二建", "二级建造师",
    "软件考试", "软考"
]

PREFIXES = [
    "", "我想问一下", "请问", "想咨询一下", "麻烦问下"
]

SUFFIXES = [
    "", "啊", "呀", "呢", "可以吗", "怎么办"
]

# ========= 2️⃣ Intent 模板 =========

INTENT_PATTERNS = {
    "报名时间": [
        "{exam}什么时候报名",
        "{exam}报名时间",
        "什么时候可以报名{exam}",
        "{exam}几号开始报名",
        "{exam}报名是几号",
        "报名{exam}要等到什么时候"
    ],

    "报名入口": [
        "{exam}在哪报名",
        "{exam}报名入口",
        "{exam}报名官网吗",
        "报{exam}要去哪里",
        "哪里可以报名{exam}",
        "{exam}报名网址是什么"
    ],

    "报名意图": [
        "我想报名{exam}",
        "我想报考{exam}",
        "准备考{exam}",
        "想考一个{exam}",
        "打算报{exam}"
    ],

    "报名异常": [
        "报名失败怎么办",
        "报名系统打不开",
        "报不了名",
        "报名页面进不去",
        "报名一直失败"
    ],

    "登录问题": [
        "忘记密码怎么办",
        "忘记用户名",
        "登录失败",
        "账号登不上",
        "系统登录不了"
    ]
}

# ========= 3️⃣ 生成函数 =========

def generate_sentence(pattern, intent):
    exam = random.choice(EXAMS) if "{exam}" in pattern else ""
    prefix = random.choice(PREFIXES)
    suffix = random.choice(SUFFIXES)

    sentence = pattern.format(exam=exam)

    # 前后随机拼接
    if prefix:
        sentence = prefix + sentence
    if suffix:
        sentence = sentence + suffix

    return sentence.strip()


# ========= 4️⃣ 主生成逻辑 =========

rows = []

for intent, patterns in INTENT_PATTERNS.items():
    for _ in range(SAMPLES_PER_INTENT):
        pattern = random.choice(patterns)
        text = generate_sentence(pattern, intent)
        rows.append((text, intent))

# 打乱顺序
random.shuffle(rows)

# ========= 5️⃣ 写入 CSV =========

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(rows)

print(f"✅ 已生成 {len(rows)} 条 Intent 数据 → {OUTPUT_FILE}")
