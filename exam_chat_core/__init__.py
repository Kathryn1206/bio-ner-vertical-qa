# exam_chat_core/__init__.py 完整代码（直接覆盖原文件）
from .core_code import (
    # 核心问答函数
    get_answer_from_exam_db,
    # 初始化函数
    load_bio_model,
    load_intent_model,
    load_local_qwen,
    load_exam_database_from_folder,
    # 配置路径（与core_code.py保持一致）
    BIO_MODEL_PATH,
    EXCEL_FOLDER,
    INTENT_MODEL_PATH
)

# 全局初始化函数（加载所有模型/Excel，仅启动时执行一次）
def init_exam_chat():
    import sys
    from .core_code import (
        exam_question_map, question_answer_map, exam_keywords,
        intent_classifier, bio_tokenizer, bio_model, bio_id_to_label,
        qwen_model, qwen_tokenizer
    )
    # 声明全局变量，确保后续调用可使用
    global exam_question_map, question_answer_map, exam_keywords
    global intent_classifier, bio_tokenizer, bio_model, bio_id_to_label
    global qwen_model, qwen_tokenizer

    # 按顺序加载资源
    print("📂 加载Excel数据库...")
    exam_question_map, question_answer_map, exam_keywords = load_exam_database_from_folder(EXCEL_FOLDER)
    print("🤖 加载意图模型...")
    intent_classifier = load_intent_model()
    print("🔍 加载BIO实体识别模型...")
    bio_tokenizer, bio_model, bio_id_to_label = load_bio_model()
    print("💬 加载Qwen大模型...")
    qwen_model, qwen_tokenizer = load_local_qwen()
    print("✅ 核心系统初始化完成！")

# 对外暴露的接口（仅保留这2个，简化调用）
__all__ = ["init_exam_chat", "get_answer_from_exam_db"]