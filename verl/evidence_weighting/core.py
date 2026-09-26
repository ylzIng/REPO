"""Ground-truth-free resampling of teacher-selected hard targets.

Bootstrap samples reuse finite evidence; they are not independent teachers and
cannot detect a teacher that is consistently wrong. The scalar weights control
supervision repeatability, not the probability that an answer is correct.
"""

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroupEvidence:
    """Immutable snapshot, deliberately excluding ground-truth metadata."""

    uid: str
    group_index: int
    task: str
    majority_answer: str | None
    majority_rate: float
    candidates: tuple  # (answer, original student frequency)
    outputs: tuple  # (candidate answer, original frequency, teacher text)
    original_label: str | None
    route: str
    vote_count: int
    verification_n: int
    prompt_digest: str
    prompt_text: str = ""
    acceptance_threshold: float = 0.5
    verdict_format: str = "repo"


def bootstrap_labels(evidence: GroupEvidence, *, samples: int, seed: int, step: int):
    """Rerun the exact hard selector, freezing route, candidates and counts.

    Parse failures are resampled as failures with the original fixed denominator.
    The native selector returns the majority answer for abstain; expose that route as
    an abstention here because its student update is zero in skip_update mode.
    """
    from verl.utils.reward_score.ttrl.teacher_judgments import select_teacher_targets

    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    material = f"{seed}:{step}:{evidence.group_index}:{evidence.prompt_digest}"
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "little"))
    buckets = {}
    for candidate, frequency, output in evidence.outputs:
        buckets.setdefault((candidate, frequency), []).append(output)
    group = {
        "prompt_group_idx": 0,
        "majority_answer": evidence.majority_answer,
        "majority_rate": evidence.majority_rate,
        "candidates": list(evidence.candidates),
    }
    frozen_route = "high" if evidence.route == "high_consensus" else "low"
    selected = []
    for _ in range(samples):
        outputs, mappings = [], []
        for (candidate, frequency), texts in buckets.items():
            # Exactly the observed number of slots is resampled. Missing slots
            # do not reduce verification_total_n or become successful evidence.
            for index in rng.integers(0, len(texts), size=len(texts)):
                outputs.append(texts[int(index)])
                mappings.append({"prompt_group_idx": 0, "candidate_answer": candidate, "frequency": frequency})
        labels, _, _, routes = select_teacher_targets(
            verification_outputs=outputs,
            verification_mapping=mappings,
            prompt_groups=[group],
            n_votes_per_prompt=evidence.vote_count,
            high_consistency_threshold=0.7,
            low_consistency_threshold=0.7,
            consistency_route_by_group={0: frozen_route},
            low_consistency_strategy="majority",
            fallback_mode="skip_update",
            verification_total_n=evidence.verification_n,
            acceptance_threshold=evidence.acceptance_threshold,
            verdict_format=evidence.verdict_format,
        )
        selected.append(None if routes[0] == "abstain" else labels[0])
    return tuple(selected)


def direction_weights(
    base,
    bootstrap,
    *,
    attenuation=0.25,
):
    """Return signed support, reliability, and the main REPO attenuation.

    Weights are ``1 - attenuation * (1 - max(0, signed_support))``.
    Zero base advantages retain a neutral coefficient of one.
    """
    base = np.asarray(base, dtype=np.float64)
    bootstrap = np.asarray(bootstrap, dtype=np.float64)
    if base.ndim != 1 or bootstrap.ndim != 2 or bootstrap.shape[1] != len(base) or len(bootstrap) < 1:
        raise ValueError("Expected base[N] and bootstrap[B,N] with B >= 1")
    if not np.isfinite(base).all() or not np.isfinite(bootstrap).all():
        raise ValueError("Non-finite native advantages")
    if not 0 <= attenuation <= 1:
        raise ValueError("attenuation must lie in [0,1]")
    support = np.sign(base[None, :] * bootstrap).mean(axis=0)
    reliability = np.maximum(0.0, support)
    multipliers = 1.0 - attenuation * (1.0 - reliability)
    # A zero original advantage stays exactly zero; the coefficient is neutral.
    multipliers[base == 0] = 1.0
    return support, reliability, multipliers
