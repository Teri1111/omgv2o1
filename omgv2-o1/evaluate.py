"""Local F1 / Precision / Recall evaluation (replaces external evaluate_v4 dependency)."""


def compute_f1(predicted_answers: set, golden_answers: set) -> dict:
    """Compute set-level precision, recall, and F1.

    Args:
        predicted_answers: set of predicted answer IDs.
        golden_answers: set of golden answer IDs.

    Returns:
        dict with keys "precision", "recall", "f1" (all floats in [0, 1]).
    """
    if not predicted_answers and not golden_answers:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted_answers or not golden_answers:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(predicted_answers & golden_answers)
    precision = tp / len(predicted_answers)
    recall = tp / len(golden_answers)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
