import unittest

from core.api.eval_by_masked_str_match import format_score, summarize_by_label
from core.service.privacy_eval import format_metric


class ScoreFormattingTest(unittest.TestCase):
    def test_formats_score_with_three_decimal_places(self) -> None:
        self.assertEqual(format_score(1 / 3), "0.333")
        self.assertEqual(format_score(0.2), "0.200")

    def test_formats_metric_with_three_decimal_places(self) -> None:
        self.assertEqual(format_metric(0.919469), "0.919")
        self.assertEqual(format_metric(1.0), "1.000")

    def test_masked_match_summary_scores_use_three_decimal_places(self) -> None:
        rows = summarize_by_label(
            [
                {"schema_name": "private_email", "score": 1},
                {"schema_name": "private_email", "score": 0},
                {"schema_name": "private_email", "score": 0},
            ]
        )

        self.assertEqual(rows[0]["score"], "0.333")


if __name__ == "__main__":
    unittest.main()
