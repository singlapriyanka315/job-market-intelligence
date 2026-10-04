"""Scoring helpers for the extraction evaluation (pure functions, unit-tested)."""


def set_scores(gold_sets, pred_sets):
    """Micro-averaged precision / recall / F1 over a list of (gold, predicted) skill sets."""
    tp = fp = fn = 0
    for gold, pred in zip(gold_sets, pred_sets, strict=True):
        gold, pred = set(gold), set(pred)
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}


def accuracy(gold, pred):
    pairs = [(g, p) for g, p in zip(gold, pred, strict=True) if g]  # skip unlabelled fields
    return round(sum(g == p for g, p in pairs) / len(pairs), 3) if pairs else None


def per_skill_errors(gold_sets, pred_sets, top=10):
    """Which skills each extractor most often invents (fp) or misses (fn)."""
    fp, fn = {}, {}
    for gold, pred in zip(gold_sets, pred_sets, strict=True):
        for s in set(pred) - set(gold):
            fp[s] = fp.get(s, 0) + 1
        for s in set(gold) - set(pred):
            fn[s] = fn.get(s, 0) + 1
    order = lambda d: sorted(d.items(), key=lambda x: (-x[1], x[0]))[:top]
    return {"false_positives": order(fp), "false_negatives": order(fn)}
