# web_app.py 完整代码（直接覆盖原文件）
import sys
import os

# 1. 根据当前文件位置确定项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
# 路径验证（方便排查，保留）
print(f"✅ 已添加主目录到Python路径：{BASE_DIR}")
print(f"✅ Python当前搜索路径：{sys.path[:3]}")

# 2. 路径配置后，再导入所有依赖
from flask import Flask, request, render_template_string
from exam_chat_core import init_exam_chat, get_answer_from_exam_db

app = Flask(__name__)

# 3. 前端HTML模板（无修改，保留原有样式）

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
        /* 对话历史容器 */
        .chat-history { height: 400px; overflow-y: auto; padding: 20px; border: 1px solid #e5e7eb; border-radius: 8px; margin-bottom: 20px; background: #f9fafb; }
        .chat-history::-webkit-scrollbar { width: 6px; }
        .chat-history::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }
        /* 用户消息样式 */
        .user-message { display: flex; justify-content: flex-end; margin-bottom: 15px; }
        .user-message .content { background: #4f46e5; color: white; padding: 12px 16px; border-radius: 12px 12px 0 12px; max-width: 70%; }
        /* 助手消息样式 */
        .assistant-message { display: flex; justify-content: flex-start; margin-bottom: 15px; }
        .assistant-message .content { background: #ffffff; color: #1f2937; padding: 12px 16px; border-radius: 12px 12px 12px 0; max-width: 70%; border: 1px solid #e5e7eb; }
        /* 输入框+按钮容器 */
        .input-group { display: flex; gap: 10px; align-items: flex-end; }
        textarea { width: 100%; height: 100px; padding: 12px 16px; border: 1px solid #e5e6eb; border-radius: 8px; font-size: 15px; outline: none; resize: none; }
        textarea:focus { border-color: #4f46e5; box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.1); }
        button { padding: 12px 30px; background: #4f46e5; color: white; border: none; border-radius: 8px; font-size: 16px; cursor: pointer; flex-shrink: 0; }
        button:hover { background: #4338ca; }
        button:disabled { background: #9ca3af; cursor: not-allowed; }
        /* 加载状态提示 */
        .loading { color: #6b7280; text-align: center; padding: 10px; font-size: 14px; }
        .answer { margin-top: 20px; padding: 15px; border-radius: 8px; background: #f9fafb; border: 1px solid #e5e7eb; }
    </style>
    <!-- 引入jQuery（简化AJAX请求，无需手动写原生JS） -->
    <script src="https://cdn.bootcdn.net/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
</head>
<body>
    <div class="box">
        <h1>四川省人事考试智能客服</h1>
        <!-- 对话历史展示区 -->
        <div class="chat-history" id="chatHistory">
            <!-- 历史对话会通过JS动态追加到这里 -->
            <div class="assistant-message">
                <div class="content">您好！我是人事考试智能客服，请问您想咨询哪项考试的相关问题？</div>
            </div>
        </div> 
        <!-- 输入框+提交按钮 -->
        <div class="input-group">
            <textarea id="questionInput" placeholder="请输入您的问题（如：我想报名二建需要什么条件？）" required></textarea>
            <button id="submitBtn">提交咨询</button>
        </div>
    </div>

    <script>
        $(function() {
            // 绑定提交按钮点击事件
            $("#submitBtn").click(submitQuestion);
            // 绑定回车键提交（按Enter提交，按Shift+Enter换行）
            $("#questionInput").keydown(function(e) {
                if (e.keyCode === 13 && !e.shiftKey) {
                    e.preventDefault();
                    submitQuestion();
                }
            });

            // 核心：提交问题+AJAX异步请求
            function submitQuestion() {
                const question = $.trim($("#questionInput").val());
                if (!question) {
                    alert("请输入您要咨询的问题！");
                    return;
                }

                // 1. 将用户问题追加到历史对话，清空输入框
                const userHtml = `
                    <div class="user-message">
                        <div class="content">${escapeHtml(question)}</div>
                    </div>
                `;
                $("#chatHistory").append(userHtml);
                $("#questionInput").val("");
                // 滚动到对话底部，显示最新消息
                scrollToBottom();

                // 2. 显示加载状态
                const loadingHtml = '<div class="loading" id="loading">正在为您查询，请稍候...</div>';
                $("#chatHistory").append(loadingHtml);
                scrollToBottom();
                $("#submitBtn").prop("disabled", true); // 禁用按钮，防止重复提交

                // 3. AJAX异步请求后端API
                $.ajax({
                    url: "/api/chat", // 后端新增的AJAX路由
                    type: "POST",
                    contentType: "application/json; charset=utf-8",
                    data: JSON.stringify({ question: question }), // 传递用户问题
                    success: function(res) {
                        // 4. 请求成功：移除加载状态，追加助手回答
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
                        // 5. 请求失败：提示错误
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

            // 工具函数：HTML转义，防止XSS注入
            function escapeHtml(str) {
                if (!str) return "";
                return str
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;")
                    .replace(/"/g, "&quot;")
                    .replace(/'/g, "&#039;");
            }

            // 工具函数：滚动到对话历史底部
            function scrollToBottom() {
                const chatHistory = $("#chatHistory");
                chatHistory.scrollTop(chatHistory[0].scrollHeight);
            }
        });
    </script>
</body>
</html>
"""


# 4. Flask路由（无修改，保留原有逻辑）
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
    # 获取前端AJAX提交的用户问题
    user_q = request.json.get('question', '').strip()
    if not user_q:
        return {"success": False, "answer": "请输入有效问题！"}
    # 调用原有核心问答逻辑，无任何修改
    answer = get_answer_from_exam_db(user_q)
    # 返回JSON格式结果，供前端解析
    return {"success": True, "answer": answer}

# 5. 项目启动入口（强制关闭重载器，避免重复加载）
if __name__ == "__main__":
    import webbrowser  # 新增：导入浏览器控制库
    # 初始化核心问答系统
    init_exam_chat()
    # 启动Flask服务前，自动打开浏览器网页（核心新增代码）
    web_url = "http://127.0.0.1:5000"
    print(f"🌐 服务启动：{web_url}，正在自动打开浏览器...")
    webbrowser.open(web_url)  # 新增：自动打开指定网址
    # 启动Flask服务（原有代码不变，保留use_reloader=False）
    app.run(
        debug=True,        # 开发模式，报错显示详情
        host='0.0.0.0',    # 局域网可访问
        port=5000,         # 服务端口
        use_reloader=False # 强制关闭重载器，避免重复加载/重复打开网页
    )
