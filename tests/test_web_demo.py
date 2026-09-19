import os
import unittest


os.environ["EXAM_CHAT_MODE"] = "demo"
os.environ["AUTO_OPEN_BROWSER"] = "false"

from web.web_app import app  # noqa: E402


class WebDemoTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_page_identifies_demo_mode(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("可复现演示模式".encode("utf-8"), response.data)

    def test_chat_api_returns_mode_and_answer(self):
        response = self.client.post("/api/chat", json={"question": "二建什么时候报名？"})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["mode"], "demo")
        self.assertIn("可复现演示", payload["answer"])

    def test_chat_api_rejects_empty_payload(self):
        response = self.client.post("/api/chat", json={})
        payload = response.get_json()
        self.assertFalse(payload["success"])
        self.assertEqual(payload["mode"], "demo")


if __name__ == "__main__":
    unittest.main()
