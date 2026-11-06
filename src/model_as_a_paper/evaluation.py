"""Evaluate independently labeled outcomes before and after a knowledge update.

This module does NOT decide whether scientific claims are true. It only aggregates
binary judgments obtained by independent verification.
"""
from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable

VALID_CATEGORIES = frozenset({"correction", "retention", "transfer"})


@dataclass(frozen=True)
class VerifiedOutcome:
    question_id: str
    category: str
    before_correct: bool
    after_correct: bool

    def __post_init__(self):
        if not self.question_id:
            raise ValueError("question_id cannot be empty")
        if self.category not in VALID_CATEGORIES:
            raise ValueError(f"Unknown category: {self.category}")
        if not isinstance(self.before_correct, bool) or not isinstance(self.after_correct, bool):
            raise TypeError("Correctness fields must be bool; labels require independent verification")


def summarize(outcomes: Iterable[VerifiedOutcome]) -> dict[str, dict[str, float | int | None]]:
    """Return counts and accuracy, checking duplicate item identifiers.

    Accuracy is an observed proportion, not evidence that a model's explanation
    is physically correct. Interpretation needs domain-specific verification.
    """
    rows: dict[str, list[VerifiedOutcome]] = defaultdict(list)
    seen: set[str] = set()
    for outcome in outcomes:
        if outcome.question_id in seen:
            raise ValueError(f"Duplicate question_id: {outcome.question_id}")
        seen.add(outcome.question_id)
        rows[outcome.category].append(outcome)

    result: dict[str, dict[str, float | int | None]] = {}
    for category in sorted(VALID_CATEGORIES):
        group = rows[category]
        n = len(group)
        if not n:
            continue
        before = sum(item.before_correct for item in group) / n
        after = sum(item.after_correct for item in group) / n
        known = [item for item in group if item.before_correct]
        result[category] = {
            "n": n,
            "previously_correct_n": len(known),
            "retention_among_previously_correct": sum(item.after_correct for item in known) / len(known) if known else None,
            "before_accuracy": before,
            "after_accuracy": after,
            "change_pp": (after - before) * 100.0,
        }
    return result
