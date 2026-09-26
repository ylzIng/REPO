"""Read-only diagnostics for cached teacher judgments.

These metrics must be called AFTER target selection. They return only scalar
logging data, never rewards, labels, routes, or advantage weights.
"""


def teacher_judgment_metrics(judgments, mappings, groups, candidate_is_correct=None):
    """Measure the unfiltered judgment cache, including abstained questions.

    ``judgments`` contains exactly True / False / None (invalid). Confusion
    matrix rates use ``gt_valid_count``: judgments that have both a parsed
    verdict and a successfully scored reference answer. Invalid judgments are
    reported separately, as are missing references and scoring errors. If a
    denominator is zero, its rates are omitted; counts still explicitly report
    zero. Missing ground truth must never masquerade as a measured 0% accuracy.
    ``candidate_is_correct(candidate, reference)`` is diagnostic-only and is
    evaluated once per distinct question/candidate, never once per judgment.
    """
    if len(judgments) != len(mappings):
        raise ValueError("Teacher diagnostic judgments and mappings are misaligned")
    if any(value is not True and value is not False and value is not None for value in judgments):
        raise ValueError("Teacher diagnostics require parsed True / False / invalid judgments")
    group_by_id = {group["prompt_group_idx"]: group for group in groups}
    total = len(judgments)
    metrics = {
        "teacher/judgment_count": total,
        "teacher/judgment_true_count": sum(value is True for value in judgments),
        "teacher/judgment_false_count": sum(value is False for value in judgments),
        "teacher/judgment_invalid_count": sum(value is None for value in judgments),
    }
    counts = dict(tp=0, tn=0, fp=0, fn=0, known=0, valid=0, invalid=0,
                  missing=0, scoring_error=0)
    score_cache = {}
    for verdict, mapping in zip(judgments, mappings):
        group_id = mapping["prompt_group_idx"]
        group = group_by_id.get(group_id, {})
        reference = group.get("ground_truth")
        if reference is None or str(reference).strip() == "" or candidate_is_correct is None:
            counts["missing"] += 1
            continue
        key = (group_id, str(mapping["candidate_answer"]))
        if key not in score_cache:
            try:
                value = candidate_is_correct(mapping["candidate_answer"], reference)
                score_cache[key] = None if value is None else bool(value)
            except Exception:
                # A diagnostic scoring failure must not alter training or be
                # silently counted as an incorrect candidate.
                score_cache[key] = None
        correct = score_cache[key]
        if correct is None:
            counts["scoring_error"] += 1
            continue
        counts["known"] += 1
        if verdict is None:
            counts["invalid"] += 1
            continue
        counts["valid"] += 1
        counts[("tp" if correct else "fp") if verdict else ("fn" if correct else "tn")] += 1
    for key, value in counts.items():
        metrics[f"teacher/gt_{key}_count"] = value
    metrics["teacher/gt_unknown_count"] = counts["missing"] + counts["scoring_error"]
    if total:
        metrics["teacher/judgment_invalid_rate"] = metrics["teacher/judgment_invalid_count"] / total
        metrics["teacher/gt_known_fraction"] = counts["known"] / total
    if counts["valid"]:
        for cell in ("tp", "tn", "fp", "fn"):
            # Preserve the old plot keys with an explicit, correct denominator.
            metrics[f"train/gt_{cell}_rate"] = counts[cell] / counts["valid"]
        metrics["teacher/gt_valid_accuracy"] = (counts["tp"] + counts["tn"]) / counts["valid"]
    if counts["tp"] + counts["fn"]:
        metrics["teacher/gt_false_negative_rate"] = counts["fn"] / (counts["tp"] + counts["fn"])
    if counts["fp"] + counts["tn"]:
        metrics["teacher/gt_false_positive_rate"] = counts["fp"] / (counts["fp"] + counts["tn"])
    return metrics
