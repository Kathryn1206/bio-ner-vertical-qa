import json
from pathlib import Path
import tempfile
import unittest

from exam_chat_core.demo import DEFAULT_DATA_PATH, DemoPipeline


class DemoPipelineTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = DemoPipeline()

    def test_fixture_is_explicitly_synthetic(self):
        payload = json.loads(DEFAULT_DATA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["data_type"], "synthetic_demo_fixture")
        self.assertIn("not current exam policy", payload["notice"])

    def test_alias_and_registration_time_route(self):
        response = self.pipeline.ask("二建什么时候报名？")
        self.assertEqual(response.exam, "二级建造师考试")
        self.assertEqual(response.intent, "registration_time")
        self.assertIn("演示数据", response.answer)

    def test_context_carry_over(self):
        self.pipeline.ask("我想问二建的报名条件")
        response = self.pipeline.ask("那在哪里报名？")
        self.assertEqual(response.exam, "二级建造师考试")
        self.assertEqual(response.intent, "registration_entry")
        self.assertTrue(response.context_reused)

    def test_exam_scope_is_not_crossed(self):
        response = self.pipeline.ask("软考什么时候报名？")
        self.assertEqual(response.exam, "计算机技术与软件专业技术资格考试")
        self.assertIn("计算机技术与软件", response.matched_question)

    def test_unsupported_query_refuses_to_invent(self):
        response = self.pipeline.ask("二建的命题老师是谁？")
        self.assertIsNone(response.matched_question)
        self.assertIn("不生成答案", response.answer)

    def test_login_problem_does_not_require_exam(self):
        response = self.pipeline.ask("忘记密码怎么办？")
        self.assertEqual(response.intent, "login_problem")
        self.assertIsNone(response.exam)
        self.assertIn("官方登录页", response.answer)

    def test_invalid_fixture_fails_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid_path = Path(directory) / "invalid.json"
            invalid_path.write_text('{"schema_version": 1}', encoding="utf-8")
            with self.assertRaises(ValueError):
                DemoPipeline(invalid_path)


if __name__ == "__main__":
    unittest.main()
