"""Question-level sampled validation, with no trainer or GPU dependencies.

``pass@16`` means at least one success among 16 draws; ``pass_pow4`` means
four successes among four draws. Both use the without-replacement estimators
computed from the observed independent samples, rather than powers of a
dataset-wide mean. Question IDs must identify original dataset rows, not
prompt strings: two rows with identical text are still distinct questions.
"""

from collections import defaultdict
from math import comb
from numbers import Integral

import numpy as np


def summarize_sampled_validation(data_sources, question_ids, scores, *, expected_n=32):
    """Return ``(metric_dict, dataset_rows)`` with all accuracies as fractions.

    Inputs contain one entry per generated answer. Rows may arrive in any
    order. Group identity is ``(data_source, question_id)``; each group must
    contain exactly ``expected_n`` binary outcomes. Repeated prompt text is
    intentionally irrelevant. Empty input returns an empty dict and list.

    Raises:
        ValueError: Inputs are malformed, scores are not binary, or any
            question has a different number of samples than ``expected_n``.
    """
    if isinstance(expected_n, bool) or not isinstance(expected_n, Integral) or expected_n < 16:
        raise ValueError("expected_n must be an integer greater than or equal to 16")
    expected_n = int(expected_n)
    sources = np.asarray(data_sources, dtype=object)
    identities = np.asarray(question_ids, dtype=object)
    outcomes = np.asarray(scores)
    if any(values.ndim != 1 for values in (sources, identities, outcomes)):
        raise ValueError("data_sources, question_ids, and scores must be one-dimensional")
    if not (len(sources) == len(identities) == len(outcomes)):
        raise ValueError("data_sources, question_ids, and scores must have equal lengths")
    if outcomes.dtype.kind not in "biuf" or not np.all(np.isfinite(outcomes)):
        raise ValueError("scores must contain only finite binary numbers 0 or 1")
    if not np.all((outcomes == 0) | (outcomes == 1)):
        raise ValueError("scores must contain only binary values 0 or 1")

    grouped = {}
    for source, question_id, score in zip(sources, identities, outcomes):
        if not isinstance(source, str) or not source:
            raise ValueError("each data source must be a nonempty string")
        if question_id is None:
            raise ValueError("question IDs must be nonmissing hashable scalars")
        try:
            hash(question_id)
            missing = bool(question_id != question_id)
        except (TypeError, ValueError):
            raise ValueError("question IDs must be nonmissing hashable scalars") from None
        if missing:
            raise ValueError("question IDs must be nonmissing hashable scalars")
        key = (source, question_id)
        counts = grouped.setdefault(key, [0, 0])
        counts[0] += 1
        counts[1] += int(score)

    per_dataset = defaultdict(list)
    for (source, question_id), (n, c) in grouped.items():
        if n != expected_n:
            raise ValueError(
                f"Question {question_id!r} in {source!r} has {n} samples; expected {expected_n}"
            )
        # math.comb(n, k) returns zero for k > n; all denominators are valid.
        per_dataset[source].append((
            c / n,
            1.0 - comb(n - c, 16) / comb(n, 16),
            comb(c, 4) / comb(n, 4),
        ))

    metrics = {}
    rows = []
    for source, values in per_dataset.items():
        pass_at_1, pass_at_16, pass_pow_4 = np.asarray(values, dtype=np.float64).mean(axis=0)
        row = {
            "dataset": source,
            "questions": len(values),
            "pass_at_1": float(pass_at_1),
            "pass_at_16": float(pass_at_16),
            "pass_pow_4": float(pass_pow_4),
        }
        rows.append(row)
        prefix = f"sampled-val/{source}"
        metrics.update({
            f"{prefix}/pass@1": row["pass_at_1"],
            f"{prefix}/pass@16": row["pass_at_16"],
            f"{prefix}/pass_pow4": row["pass_pow_4"],
            f"{prefix}/questions": row["questions"],
        })
    return metrics, rows
