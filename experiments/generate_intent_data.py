"""Generate deterministic synthetic Chinese intent-classification examples."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random


DEFAULT_OUTPUT = Path(__file__).with_name("intent_data.csv")
DEFAULT_SEED = 42
DEFAULT_SAMPLES_PER_INTENT = 120

EXAMS = ["二建", "二级建造师", "软件考试", "软考"]
PREFIXES = ["", "我想问一下", "请问", "想咨询一下", "麻烦问下"]
SUFFIXES = ["", "啊", "呀", "呢", "可以吗", "怎么办"]

INTENT_PATTERNS = {
    "报名时间": [
        "{exam}什么时候报名",
        "{exam}报名时间",
        "什么时候可以报名{exam}",
        "{exam}几号开始报名",
        "{exam}报名是几号",
        "报名{exam}要等到什么时候",
    ],
    "报名入口": [
        "{exam}在哪报名",
        "{exam}报名入口",
        "{exam}报名官网吗",
        "报{exam}要去哪里",
        "哪里可以报名{exam}",
        "{exam}报名网址是什么",
    ],
    "报名意图": [
        "我想报名{exam}",
        "我想报考{exam}",
        "准备考{exam}",
        "想考一个{exam}",
        "打算报{exam}",
    ],
    "报名异常": [
        "报名失败怎么办",
        "报名系统打不开",
        "报不了名",
        "报名页面进不去",
        "报名一直失败",
    ],
    "登录问题": [
        "忘记密码怎么办",
        "忘记用户名",
        "登录失败",
        "账号登不上",
        "系统登录不了",
    ],
}


def generate_sentence(pattern: str, rng: random.Random) -> str:
    exam = rng.choice(EXAMS) if "{exam}" in pattern else ""
    sentence = pattern.format(exam=exam)
    prefix = rng.choice(PREFIXES)
    suffix = rng.choice(SUFFIXES)
    return f"{prefix}{sentence}{suffix}".strip()


def generate_rows(samples_per_intent: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []
    for intent, patterns in INTENT_PATTERNS.items():
        for _ in range(samples_per_intent):
            rows.append((generate_sentence(rng.choice(patterns), rng), intent))
    rng.shuffle(rows)
    return rows


def write_dataset(output_path: Path, samples_per_intent: int, seed: int) -> None:
    rows = generate_rows(samples_per_intent, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"Generated {len(rows)} deterministic intent examples -> {output_path}")
    print(f"Seed: {seed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-intent", type=int, default=DEFAULT_SAMPLES_PER_INTENT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.samples_per_intent < 1:
        parser.error("--samples-per-intent must be positive")
    write_dataset(args.output, args.samples_per_intent, args.seed)


if __name__ == "__main__":
    main()
