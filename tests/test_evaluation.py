import unittest

from model_as_a_paper.evaluation import VerifiedOutcome, summarize


class EvaluationTests(unittest.TestCase):
    def test_before_after_by_category(self):
        rows = [
            VerifiedOutcome("c1", "correction", False, True),
            VerifiedOutcome("c2", "correction", False, False),
            VerifiedOutcome("r1", "retention", True, True),
            VerifiedOutcome("t1", "transfer", False, True),
        ]
        scores = summarize(rows)
        self.assertEqual(scores["correction"]["n"], 2)
        self.assertEqual(scores["correction"]["change_pp"], 50.0)
        self.assertEqual(scores["retention"]["after_accuracy"], 1.0)

    def test_duplicate_ids_rejected(self):
        rows = [
            VerifiedOutcome("same", "correction", False, True),
            VerifiedOutcome("same", "retention", True, True),
        ]
        with self.assertRaises(ValueError):
            summarize(rows)

    def test_invalid_labels_rejected(self):
        with self.assertRaises(TypeError):
            VerifiedOutcome("c1", "correction", 1, True)


if __name__ == "__main__":
    unittest.main()
