import sys
import os

# 1. Resolve the project root from this file's location.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 2. Import application dependencies after configuring the path.
from flask import Flask, request, render_template_string
from exam_chat_core import get_answer_from_exam_db, get_backend_mode, init_exam_chat

app = Flask(__name__)
BACKEND_MODE = get_backend_mode()

# 3. Embedded HTML template for the prototype UI.

HTML_TPL = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>四川省人事考试智能客服</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: "微软雅黑", sans-serif; }
        body { background: #f0f2f5; padding: 20px; max-width: 800px; margin: 0 auto; }
        .box { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
        h1 { color: #1f2937; font-size: 24px; text-align: center; margin-bottom: 30px; font-weight: 600; }
        .mode-banner { margin: -12px 0 20px; padding: 10px 14px; border: 1px solid #fde68a; border-radius: 8px; background: #fffbeb; color: #92400e; font-size: 13px; line-height: 1.5; }
        /* Conversation history */
        .chat-history { height: 400px; overflow-y: auto; padding: 20px; border: 1px solid #e5e7eb; border-radius: 8px; margin-bottom: 20px; background: #f9fafb; }
        .chat-history::-webkit-scrollbar { width: 6px; }
        .chat-history::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }
        /* User messages */
        .user-message { display: flex; justify-content: flex-end; margin-bottom: 15px; }
        .user-message .content { background: #4f46e5; color: white; padding: 12px 16px; border-radius: 12px 12px 0 12px; max-width: 70%; }
        /* Assistant messages */
        .assistant-message { display: flex; justify-content: flex-start; margin-bottom: 15px; }
        .assistant-message .content { background: #ffffff; color: #1f2937; padding: 12px 16px; border-radius: 12px 12px 12px 0; max-width: 70%; border: 1px solid #e5e7eb; }
        /* Input and submit controls */
        .input-group { display: flex; gap: 10px; align-items: flex-end; }
        textarea { width: 100%; height: 100px; padding: 12px 16px; border: 1px solid #e5e6eb; border-radius: 8px; font-size: 15px; outline: none; resize: none; }
        textarea:focus { border-color: #4f46e5; box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.1); }
        button { padding: 12px 30px; background: #4f46e5; color: white; border: none; border-radius: 8px; font-size: 16px; cursor: pointer; flex-shrink: 0; }
        button:hover { background: #4338ca; }
        button:disabled { background: #9ca3af; cursor: not-allowed; }
        /* Loading indicator */
        .loading { color: #6b7280; text-align: center; padding: 10px; font-size: 14px; }
        .answer { margin-top: 20px; padding: 15px; border-radius: 8px; background: #f9fafb; border: 1px solid #e5e7eb; }
    </style>
</head>
<body>
    <div class="box">
        <h1>四川省人事考试智能客服</h1>
        {% if demo_mode %}
        <div class="mode-banner">
            可复现演示模式：当前回答来自仓库内的合成 FAQ，仅用于验证路由和检索流程，不代表真实考试政策。
        </div>
        {% endif %}
        <!-- Conversation history -->
        <div class="chat-history" id="chatHistory">
            <!-- JavaScript appends new messages here. -->
            <div class="assistant-message">
                <div class="content">您好！我是人事考试智能客服，请问您想咨询哪项考试的相关问题？</div>
            </div>
        </div> 
        <!-- Question input and submit button -->
        <div class="input-group">
            <textarea id="questionInput" placeholder="请输入您的问题（如：我想报名二建需要什么条件？）" required></textarea>
            <button id="submitBtn">提交咨询</button>
        </div>
    </div>

    <script>
        document.addEventListener("DOMContentLoaded", function() {
            const submitButton = document.getElementById("submitBtn");
            const questionInput = document.getElementById("questionInput");
            const chatHistory = document.getElementById("chatHistory");

            submitButton.addEventListener("click", submitQuestion);
            questionInput.addEventListener("keydown", function(event) {
                if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    submitQuestion();
                }
            });

            async function submitQuestion() {
                const question = questionInput.value.trim();
                if (!question) {
                    alert("请输入您要咨询的问题！");
                    return;
                }

                appendMessage("user-message", question);
                questionInput.value = "";
                const loading = document.createElement("div");
                loading.className = "loading";
                loading.id = "loading";
                loading.textContent = "正在为您查询，请稍候...";
                chatHistory.appendChild(loading);
                scrollToBottom();
                submitButton.disabled = true;

                try {
                    const response = await fetch("/api/chat", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({question: question})
                    });
                    const payload = await response.json();
                    document.getElementById("loading")?.remove();
                    appendMessage("assistant-message", payload.answer || "请求未返回答案。");
                } catch (error) {
                    document.getElementById("loading")?.remove();
                    appendMessage("assistant-message", "网络异常，请求失败，请稍后再试！");
                } finally {
                    submitButton.disabled = false;
                }
            }

            function appendMessage(className, text) {
                const wrapper = document.createElement("div");
                wrapper.className = className;
                const content = document.createElement("div");
                content.className = "content";
                content.textContent = text;
                content.style.whiteSpace = "pre-wrap";
                wrapper.appendChild(content);
                chatHistory.appendChild(wrapper);
                scrollToBottom();
            }

            function scrollToBottom() {
                chatHistory.scrollTop = chatHistory.scrollHeight;
            }
        });
    </script>
</body>
</html>
"""


# 4. Flask routes.
@app.route('/', methods=['GET', 'POST'])
def index():
    ans = ""
    if request.method == 'POST':
        user_q = request.form.get('q', '').strip()
        if user_q:
            ans = get_answer_from_exam_db(user_q)
    return render_template_string(HTML_TPL, ans=ans, demo_mode=BACKEND_MODE == "demo")

@app.route('/api/chat', methods=['POST'])
def chat_api():
    # Read the question submitted by the AJAX client.
    payload = request.get_json(silent=True) or {}
    user_q = str(payload.get('question', '')).strip()
    if not user_q:
        return {"success": False, "answer": "请输入有效问题！", "mode": BACKEND_MODE}
    # Delegate to the core QA pipeline.
    answer = get_answer_from_exam_db(user_q)
    # Return a JSON response for the client.
    return {"success": True, "answer": answer, "mode": BACKEND_MODE}

# 5. Local application entry point.
if __name__ == "__main__":
    import webbrowser
    # Initialize the QA resources once before serving requests.
    init_exam_chat()
    app_host = os.getenv("APP_HOST", "127.0.0.1")
    app_port = int(os.getenv("APP_PORT", "5000"))
    flask_debug = os.getenv("FLASK_DEBUG", "false").lower() in {"1", "true", "yes", "on"}
    auto_open_browser = os.getenv("AUTO_OPEN_BROWSER", "true").lower() in {"1", "true", "yes", "on"}
    # Optionally open the local UI before starting Flask.
    browser_host = "127.0.0.1" if app_host == "0.0.0.0" else app_host
    web_url = f"http://{browser_host}:{app_port}"
    print(f"Starting the service at {web_url}")
    if auto_open_browser:
        print("Opening the browser...")
        webbrowser.open(web_url)
    # Disable the reloader so large models are not initialized twice.
    app.run(
        debug=flask_debug,
        host=app_host,
        port=app_port,
        use_reloader=False
    )
