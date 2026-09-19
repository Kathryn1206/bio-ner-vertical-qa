import unittest

from experiments.generate_intent_data import INTENT_PATTERNS, generate_rows


class SyntheticIntentDataTests(unittest.TestCase):
    def test_seed_reproduces_identical_rows(self):
        first = generate_rows(samples_per_intent=20, seed=42)
        second = generate_rows(samples_per_intent=20, seed=42)
        self.assertEqual(first, second)

    def test_each_intent_has_requested_sample_count(self):
        samples_per_intent = 12
        rows = generate_rows(samples_per_intent=samples_per_intent, seed=42)
        counts = {intent: 0 for intent in INTENT_PATTERNS}
        for _, intent in rows:
            counts[intent] += 1
        self.assertEqual(set(counts.values()), {samples_per_intent})


if __name__ == "__main__":
    unittest.main()
