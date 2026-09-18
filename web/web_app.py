import sys
import os

# 1. Resolve the project root from this file's location.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
# Keep path diagnostics visible during local startup.
print(f"Added project root to the Python path: {BASE_DIR}")
print(f"Current Python search path: {sys.path[:3]}")

# 2. Import application dependencies after configuring the path.
from flask import Flask, request, render_template_string
from exam_chat_core import init_exam_chat, get_answer_from_exam_db

app = Flask(__name__)

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
    <!-- jQuery keeps the prototype's AJAX code compact. -->
    <script src="https://cdn.bootcdn.net/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
</head>
<body>
    <div class="box">
        <h1>四川省人事考试智能客服</h1>
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
        $(function() {
            // Submit when the button is clicked.
            $("#submitBtn").click(submitQuestion);
            // Enter submits; Shift+Enter inserts a newline.
            $("#questionInput").keydown(function(e) {
                if (e.keyCode === 13 && !e.shiftKey) {
                    e.preventDefault();
                    submitQuestion();
                }
            });

            // Submit the question asynchronously.
            function submitQuestion() {
                const question = $.trim($("#questionInput").val());
                if (!question) {
                    alert("请输入您要咨询的问题！");
                    return;
                }

                // 1. Append the user message and clear the input.
                const userHtml = `
                    <div class="user-message">
                        <div class="content">${escapeHtml(question)}</div>
                    </div>
                `;
                $("#chatHistory").append(userHtml);
                $("#questionInput").val("");
                // Keep the newest message visible.
                scrollToBottom();

                // 2. Show the loading state.
                const loadingHtml = '<div class="loading" id="loading">正在为您查询，请稍候...</div>';
                $("#chatHistory").append(loadingHtml);
                scrollToBottom();
                $("#submitBtn").prop("disabled", true); // Prevent duplicate submissions.

                // 3. Call the backend API.
                $.ajax({
                    url: "/api/chat", // AJAX endpoint
                    type: "POST",
                    contentType: "application/json; charset=utf-8",
                    data: JSON.stringify({ question: question }), // User question payload
                    success: function(res) {
                        // 4. Replace the loading state with the response.
                        $("#loading").remove();
                        $("#submitBtn").prop("disabled", false);
                        if (res.success) {
                            const assistantHtml = `
                                <div class="assistant-message">
                                    <div class="content">${escapeHtml(res.answer).replace(/\\n/g, "<br>")}</div>
                                </div>
                            `;
                            $("#chatHistory").append(assistantHtml);
                        } else {
                            const errorHtml = `
                                <div class="assistant-message">
                                    <div class="content">${escapeHtml(res.answer)}</div>
                                </div>
                            `;
                            $("#chatHistory").append(errorHtml);
                        }
                        scrollToBottom();
                    },
                    error: function() {
                        // 5. Show a user-facing network error.
                        $("#loading").remove();
                        $("#submitBtn").prop("disabled", false);
                        const errorHtml = `
                            <div class="assistant-message">
                                <div class="content">网络异常，请求失败，请稍后再试！</div>
                            </div>
                        `;
                        $("#chatHistory").append(errorHtml);
                        scrollToBottom();
                    }
                });
            }

            // Escape HTML to prevent XSS injection.
            function escapeHtml(str) {
                if (!str) return "";
                return str
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;")
                    .replace(/"/g, "&quot;")
                    .replace(/'/g, "&#039;");
            }

            // Scroll to the bottom of the conversation.
            function scrollToBottom() {
                const chatHistory = $("#chatHistory");
                chatHistory.scrollTop(chatHistory[0].scrollHeight);
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
    return render_template_string(HTML_TPL, ans=ans)

@app.route('/api/chat', methods=['POST'])
def chat_api():
    # Read the question submitted by the AJAX client.
    user_q = request.json.get('question', '').strip()
    if not user_q:
        return {"success": False, "answer": "请输入有效问题！"}
    # Delegate to the core QA pipeline.
    answer = get_answer_from_exam_db(user_q)
    # Return a JSON response for the client.
    return {"success": True, "answer": answer}

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
