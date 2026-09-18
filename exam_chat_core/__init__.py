from .core_code import (
    # Core question-answering interface
    get_answer_from_exam_db,
    # Resource loaders
    load_bio_model,
    load_intent_model,
    load_local_qwen,
    load_exam_database_from_folder,
    # Shared configuration paths
    BIO_MODEL_PATH,
    EXCEL_FOLDER,
    INTENT_MODEL_PATH
)

# Load all models and FAQ files once during application startup.
def init_exam_chat():
    import sys
    from .core_code import (
        exam_question_map, question_answer_map, exam_keywords,
        intent_classifier, bio_tokenizer, bio_model, bio_id_to_label,
        qwen_model, qwen_tokenizer
    )
    # Expose the initialized resources to subsequent calls.
    global exam_question_map, question_answer_map, exam_keywords
    global intent_classifier, bio_tokenizer, bio_model, bio_id_to_label
    global qwen_model, qwen_tokenizer

    # Load resources in dependency order.
    print("Loading the Excel knowledge base...")
    exam_question_map, question_answer_map, exam_keywords = load_exam_database_from_folder(EXCEL_FOLDER)
    print("Loading the intent classifier...")
    intent_classifier = load_intent_model()
    print("Loading the BIO entity-recognition model...")
    bio_tokenizer, bio_model, bio_id_to_label = load_bio_model()
    print("Loading the local Qwen model...")
    qwen_model, qwen_tokenizer = load_local_qwen()
    print("Core system initialization complete.")

# Keep the package-level public API intentionally small.
__all__ = ["init_exam_chat", "get_answer_from_exam_db"]
