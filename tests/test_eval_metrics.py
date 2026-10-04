from eval.metrics import accuracy, per_skill_errors, set_scores


def test_set_scores_micro_average():
    gold = [{"Python", "SQL"}, {"Kafka"}]
    pred = [{"Python", "Go"}, {"Kafka"}]
    s = set_scores(gold, pred)
    assert (s["tp"], s["fp"], s["fn"]) == (2, 1, 1)
    assert s["precision"] == 0.667 and s["recall"] == 0.667 and s["f1"] == 0.667


def test_set_scores_empty_is_zero_not_error():
    assert set_scores([set()], [set()])["f1"] == 0.0


def test_accuracy_skips_unlabelled():
    assert accuracy(["senior", "", "junior"], ["senior", "mid", "mid"]) == 0.5


def test_per_skill_errors():
    e = per_skill_errors([{"Python"}, {"Python"}], [{"Go"}, {"Python", "Go"}])
    assert e["false_positives"] == [("Go", 2)]
    assert e["false_negatives"] == [("Python", 1)]
