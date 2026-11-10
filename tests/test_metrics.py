from vdd.metrics import auc, rates


def test_rates_and_auc():
    scores = [0.9, 0.8, 0.2, 0.1, 0.6]
    labels = ["fake", "fake", "fake", "real", "real"]
    r = rates(scores, labels)
    assert r["caught"] == 2 / 3 and r["false_alarm"] == 1 / 2 and r["accuracy"] == 3 / 5
    assert auc([0.9, 0.1], ["fake", "real"]) == 1.0
    assert auc([0.5, 0.5], ["fake", "real"]) == 0.5
