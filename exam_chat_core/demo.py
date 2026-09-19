"""Deterministic, artifact-free reproduction backend.

This module mirrors the public pipeline stages with synthetic FAQ data:
alias normalization, intent routing, conversational exam carry-over, scoped
retrieval, and a bounded unsupported-query response. It intentionally does not
claim to reproduce the private internship data, trained NER model, or Qwen
outputs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "sample_faq.json"


INTENT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("login_problem", ("忘记密码", "登录不了", "无法登录", "账号登不上", "登录失败")),
    ("registration_entry", ("在哪报名", "哪里报名", "报名入口", "报名官网", "报名网站")),
    ("registration_time", ("什么时候报名", "几号报名", "报名时间", "报名啥时候", "还能报名吗")),
    ("eligibility", ("报名条件", "报考条件", "资格", "能不能报", "可以报名吗", "需要什么条件")),
    ("materials", ("报名材料", "需要什么材料", "上传什么", "证明")),
    ("registration_process", ("报名流程", "怎么报名", "如何报名", "报名步骤")),
    ("exam_time", ("什么时候考", "几号考", "考试时间", "哪天考试")),
    ("certificate", ("证书", "怎么领取", "发证", "发放")),
)

FOLLOW_UP_TERMS = ("在哪", "哪里", "怎么", "条件", "时间", "入口", "材料", "流程", "考试")


@dataclass(frozen=True)
class DemoResponse:
    """Structured routing result exposed by the JSON CLI and tests."""

    mode: str
    query: str
    exam: str | None
    intent: str | None
    matched_question: str | None
    answer: str
    context_reused: bool = False

    def render(self) -> str:
        if self.exam and self.matched_question:
            return (
                f"【可复现演示｜{self.exam}】\n"
                f"问题：{self.matched_question}\n"
                f"答案：{self.answer}"
            )
        return self.answer


class DemoPipeline:
    """Small deterministic analogue of the model-backed routing pipeline."""

    def __init__(self, data_path: str | Path = DEFAULT_DATA_PATH) -> None:
        self.data_path = Path(data_path)
        self.last_exam: str | None = None
        self._payload = self._load_payload(self.data_path)
        self._exams = self._index_exams(self._payload)
        self._aliases = self._index_aliases(self._exams)

    @staticmethod
    def _load_payload(data_path: Path) -> dict[str, Any]:
        if not data_path.is_file():
            raise FileNotFoundError(f"Demo FAQ fixture not found: {data_path}")
        with data_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != 1 or not payload.get("exams"):
            raise ValueError("Demo FAQ fixture must use schema_version 1 and include exams.")
        return payload

    @staticmethod
    def _index_exams(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        exams: dict[str, dict[str, Any]] = {}
        for exam in payload["exams"]:
            name = str(exam.get("name", "")).strip()
            faqs = exam.get("faqs", [])
            if not name or not faqs:
                raise ValueError("Every demo exam must include a name and at least one FAQ.")
            exams[name] = exam
        return exams

    @staticmethod
    def _index_aliases(exams: dict[str, dict[str, Any]]) -> tuple[tuple[str, str], ...]:
        aliases: list[tuple[str, str]] = []
        for canonical, exam in exams.items():
            for alias in {canonical, *exam.get("aliases", [])}:
                alias = str(alias).strip()
                if alias:
                    aliases.append((alias, canonical))
        return tuple(sorted(aliases, key=lambda item: len(item[0]), reverse=True))

    @property
    def available_exams(self) -> tuple[str, ...]:
        return tuple(self._exams)

    def reset_context(self) -> None:
        self.last_exam = None

    def _match_exam(self, query: str) -> tuple[str | None, bool]:
        for alias, canonical in self._aliases:
            if alias in query:
                return canonical, False
        if self.last_exam and any(term in query for term in FOLLOW_UP_TERMS):
            return self.last_exam, True
        return None, False

    @staticmethod
    def _classify_intent(query: str) -> str | None:
        for intent, patterns in INTENT_PATTERNS:
            if any(pattern in query for pattern in patterns):
                return intent
        return None

    def ask(self, query: str) -> DemoResponse:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")

        intent = self._classify_intent(query)
        if intent == "login_problem":
            system_faq = self._payload["system_faq"]["login_problem"]
            return DemoResponse(
                mode="demo",
                query=query,
                exam=None,
                intent=intent,
                matched_question=system_faq["question"],
                answer=system_faq["answer"],
            )

        exam, context_reused = self._match_exam(query)
        if exam is None:
            choices = "、".join(self.available_exams)
            return DemoResponse(
                mode="demo",
                query=query,
                exam=None,
                intent=intent,
                matched_question=None,
                answer=f"这是合成数据演示。请先说明考试名称；当前样例包括：{choices}。",
            )

        self.last_exam = exam
        if intent is None:
            return DemoResponse(
                mode="demo",
                query=query,
                exam=exam,
                intent=None,
                matched_question=None,
                answer=(
                    f"【可复现演示｜{exam}】\n"
                    "样例知识库未覆盖这个问题。为避免编造政策、日期或链接，演示系统不生成答案。"
                ),
                context_reused=context_reused,
            )

        faq = next(
            (item for item in self._exams[exam]["faqs"] if item["intent"] == intent),
            None,
        )
        if faq is None:
            return DemoResponse(
                mode="demo",
                query=query,
                exam=exam,
                intent=intent,
                matched_question=None,
                answer=(
                    f"【可复现演示｜{exam}】\n"
                    "样例知识库没有该意图的答案；演示系统按设计拒绝自由生成。"
                ),
                context_reused=context_reused,
            )

        return DemoResponse(
            mode="demo",
            query=query,
            exam=exam,
            intent=intent,
            matched_question=faq["question"],
            answer=faq["answer"],
            context_reused=context_reused,
        )


_PIPELINE: DemoPipeline | None = None


def init_demo_chat() -> DemoPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        configured_path = os.getenv("DEMO_FAQ_PATH", str(DEFAULT_DATA_PATH))
        _PIPELINE = DemoPipeline(configured_path)
    return _PIPELINE


def get_answer_from_exam_db(user_input: str) -> str:
    return init_demo_chat().ask(user_input).render()


def _run_self_test() -> None:
    pipeline = DemoPipeline()
    direct = pipeline.ask("二建什么时候报名？")
    assert direct.exam == "二级建造师考试"
    assert direct.intent == "registration_time"
    follow_up = pipeline.ask("那在哪里报名？")
    assert follow_up.exam == "二级建造师考试"
    assert follow_up.intent == "registration_entry"
    assert follow_up.context_reused is True
    unsupported = pipeline.ask("二建的命题老师是谁？")
    assert unsupported.matched_question is None
    print("Demo self-test passed: direct routing, context carry-over, and safe fallback.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="Run one Chinese query and exit.")
    parser.add_argument("--json", action="store_true", help="Emit a structured JSON trace.")
    parser.add_argument("--self-test", action="store_true", help="Run deterministic smoke checks.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Demo FAQ JSON path.")
    args = parser.parse_args()

    if args.self_test:
        _run_self_test()
        return

    pipeline = DemoPipeline(args.data)
    if args.query:
        response = pipeline.ask(args.query)
        if args.json:
            print(json.dumps(asdict(response), ensure_ascii=False, indent=2))
        else:
            print(response.render())
        return

    print("可复现演示已启动。输入“退出”结束。")
    while True:
        query = input("你：").strip()
        if query.lower() in {"退出", "quit", "exit"}:
            break
        if query:
            print(f"系统：{pipeline.ask(query).render()}\n")


if __name__ == "__main__":
    main()
