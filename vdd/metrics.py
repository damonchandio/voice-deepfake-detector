"""Scores for a detector that outputs P(fake): how many fakes it catches and how many real voices it wrongly flags."""
from __future__ import annotations


def rates(scores: list[float], labels: list[str], threshold: float = 0.5) -> dict:
    """Catch rate on fakes, false-alarm rate on real clips, accuracy, at one threshold."""
    fakes = [s for s, l in zip(scores, labels) if l == "fake"]
    real = [s for s, l in zip(scores, labels) if l == "real"]
    caught = sum(s >= threshold for s in fakes)
    false_alarm = sum(s >= threshold for s in real)
    return {"caught": caught / len(fakes) if fakes else None,
            "false_alarm": false_alarm / len(real) if real else None,
            "accuracy": (caught + len(real) - false_alarm) / len(scores)}


def auc(scores: list[float], labels: list[str]) -> float:
    """Chance a random fake scores higher than a random real clip (ties count half). 0.5 = coin toss."""
    fakes = [s for s, l in zip(scores, labels) if l == "fake"]
    real = [s for s, l in zip(scores, labels) if l == "real"]
    wins = sum((f > r) + 0.5 * (f == r) for f in fakes for r in real)
    return wins / (len(fakes) * len(real))
