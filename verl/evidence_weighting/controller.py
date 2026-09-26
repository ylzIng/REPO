"""Driver-side hard-label reliability attenuation; no extra rollout model."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from verl import DataProto
from verl.evidence_weighting.core import GroupEvidence, bootstrap_labels, direction_weights


def _match_candidates(task, responses, labels):
    """The exact native qwen_reward_fn semantics, caching extraction/matching.

    The inherited math and GPQA reward functions both normalize, extract, and call
    grade_answer. Extract once per response and grade once per unique answer and
    label rather than reparse every long response for every bootstrap replicate.
    """
    from verl.utils.reward_score.ttrl.latex_clean import normalize_latex
    from verl.utils.reward_score.ttrl.qwen.math_grade import grade_answer
    from verl.utils.reward_score.ttrl.qwen.qwen_math_parser import extract_answer

    if task not in {"math", "gpqa"}:
        raise ValueError(f"Unsupported native matcher task: {task}")
    predictions = [extract_answer(normalize_latex(response), task) for response in responses]
    result = {}
    for label in labels:
        cache = {}
        values = []
        for answer in predictions:
            if answer not in cache:
                cache[answer] = float(grade_answer(answer, label))
            values.append(cache[answer])
        result[label] = np.asarray(values, dtype=np.float32)
    return result


def _scalar_advantage(advantages, mask):
    lengths = mask.sum(-1).clamp_min(1)
    return (advantages.sum(-1) / lengths).detach().cpu().numpy()


class EvidenceWeightingController:
    def __init__(self, config, tokenizer):
        cfg = config.get("evidence_weighting", {})
        self.enabled = bool(cfg.get("enable", False))
        self.tokenizer = tokenizer
        self.samples = int(cfg.get("bootstrap_samples", 32))
        self.acceptance_threshold = float(config.get("teacher_acceptance_threshold", 0.5))
        self.verdict_format = str(config.get("teacher_verdict_format", "repo"))
        self.attenuation = float(cfg.get("attenuation", 0.25))
        self.seed = int(cfg.get("seed", 42))
        self.dump = bool(cfg.get("dump_evidence", True))
        self.directory = Path(cfg.get("evidence_dir") or (Path(config.trainer.default_local_dir) / "evidence"))
        self.records = {}
        self.capture_step = None
        if not self.enabled:
            return
        if not config.get("teacher_enabled", False):
            raise ValueError("evidence_weighting requires native teacher_enabled")
        if config.get("teacher_abstention_mode") != "skip_update":
            raise ValueError("Hard-label reliability requires skip_update abstention")
        high = float(config.get("teacher_high_consensus_threshold", 0.7))
        low = float(config.get("teacher_low_consensus_threshold", 0.7))
        if high != low:
            raise ValueError("Use the same high/low threshold for the two-regime method")
        if config.algorithm.adv_estimator not in {"grpo", "pass_grpo", "pass_grpo_penalized"}:
            raise ValueError("Hard-label reliability supports native outcome advantages only")
        if config.algorithm.get("use_kl_in_reward", False):
            raise ValueError("Use KL in actor loss, not token rewards, for this hard-label method")
        if self.samples < 1 or not 0 <= self.attenuation <= 1:
            raise ValueError("Invalid evidence_weighting bootstrap/attenuation")
        if not 0 <= self.acceptance_threshold <= 1:
            raise ValueError("Teacher acceptance threshold must lie in [0,1]")
        if self.verdict_format not in {"repo", "legacy"}:
            raise ValueError("Teacher verdict format must be repo or legacy")

    def capture(self, batch, groups, outputs, mapping, labels, routes, *, task, votes, verification_n, step):
        """Capture before teacher-batch filtering or student row reordering."""
        if not self.enabled:
            return
        if len(outputs) != len(mapping) or len(groups) != len(labels) or len(groups) != len(routes):
            raise ValueError("Misaligned verification evidence")
        if len(batch) != len(groups) * votes:
            raise ValueError("Expected complete native vote groups when capturing evidence")
        records = {}
        ordinals = np.zeros(len(batch), dtype=np.int64)
        group_indices = np.zeros(len(batch), dtype=np.int64)
        for group, label, route in zip(groups, labels, routes):
            group_index = int(group["prompt_group_idx"])
            start, end = group_index * votes, (group_index + 1) * votes
            uids = batch.non_tensor_batch["uid"][start:end]
            if len(uids) != votes or len(set(map(str, uids))) != 1:
                raise ValueError("Native vote-group UID alignment changed before verification")
            uid = str(uids[0])
            if uid in records or route not in {"high_consensus", "teacher_selected", "abstain"}:
                raise ValueError("Evidence must have unique UIDs and exactly two consistency regimes")
            source_outputs = tuple(
                (m["candidate_answer"], int(m["frequency"]), str(output or ""))
                for output, m in zip(outputs, mapping)
                if int(m["prompt_group_idx"]) == group_index
            )
            digest = hashlib.sha256(batch.batch["prompts"][start].detach().cpu().numpy().tobytes()).hexdigest()
            records[uid] = GroupEvidence(
                uid=uid,
                group_index=group_index,
                task=task,
                majority_answer=group.get("majority_answer"),
                majority_rate=float(group.get("majority_rate", 0.0)),
                candidates=tuple((answer, int(count)) for answer, count in group["candidates"]),
                outputs=source_outputs,
                original_label=label,
                route=route,
                vote_count=int(votes),
                verification_n=int(verification_n),
                prompt_digest=digest,
                prompt_text=self.tokenizer.decode(batch.batch["prompts"][start], skip_special_tokens=True),
                acceptance_threshold=self.acceptance_threshold,
                verdict_format=self.verdict_format,
            )
            ordinals[start:end] = np.arange(votes)
            group_indices[start:end] = group_index
        batch.non_tensor_batch["evidence_vote_ordinal"] = ordinals
        batch.non_tensor_batch["evidence_group_index"] = group_indices
        self.records, self.capture_step = records, int(step)

    def apply(self, batch, metrics, step, *, advantage_fn, advantage_kwargs):
        """Scale only final student advantages, after native skip masks.

        The native advantage is recomputed once per *distinct* resampled label
        on the fixed retained group, then indexed by bootstrap decisions. This
        is exactly the same as B native calls but avoids redundant work.
        """
        if not self.enabled:
            return
        if self.capture_step != int(step):
            raise RuntimeError("Missing current-step hard-label verification evidence")
        uids = list(map(str, batch.non_tensor_batch["uid"]))
        if set(uids) != set(self.records):
            raise ValueError("Retained student groups differ from captured verification groups")
        advantages = batch.batch["advantages"]
        mask = batch.batch["response_mask"]
        all_multipliers = np.ones(len(batch), dtype=np.float64)
        all_reliability = np.ones(len(batch), dtype=np.float64)
        json_records = []
        low_weights, flip_rates, abstention_rates = [], [], []
        for uid in dict.fromkeys(uids):
            evidence = self.records[uid]
            indices = np.flatnonzero(np.asarray(uids) == uid)
            ordinals = np.asarray(batch.non_tensor_batch["evidence_vote_ordinal"])[indices]
            if len(np.unique(ordinals)) != len(indices) or np.any(ordinals >= evidence.vote_count):
                raise ValueError("Repeated or out-of-range retained student identity")
            base_tensor = advantages[indices]
            group_mask = mask[indices]
            base = _scalar_advantage(base_tensor, group_mask)
            mass = base_tensor.abs().sum(-1).detach().cpu().numpy()
            support, reliability, multipliers = np.ones(len(indices)), np.ones(len(indices)), np.ones(len(indices))
            labels = ()
            if evidence.route == "abstain":
                if np.any(base != 0):
                    raise ValueError("abstain student advantages must be zero before reliability attenuation")
            elif evidence.route == "teacher_selected":
                labels = bootstrap_labels(evidence, samples=self.samples, seed=self.seed, step=step)
                candidate_labels = list(dict.fromkeys(label for label in labels if label is not None))
                # Validate the original hard label through the same native path.
                candidate_labels = list(dict.fromkeys([evidence.original_label] + candidate_labels))
                responses = []
                for index in indices:
                    length = int(mask[int(index)].sum().item())
                    responses.append(
                        self.tokenizer.decode(batch.batch["responses"][int(index), :length], skip_special_tokens=False)
                    )
                rewards_by_label = _match_candidates(evidence.task, responses, candidate_labels)
                adv_by_label = {None: np.zeros(len(indices), dtype=np.float64)}
                lengths = group_mask.detach().cpu().sum(-1).long()
                native_mask = group_mask.detach().cpu()
                consistency = np.asarray(batch.non_tensor_batch["consistency_rate"])[indices].copy()
                for label in candidate_labels:
                    rewards = rewards_by_label[label]
                    native_rewards = torch.zeros(native_mask.shape, dtype=base_tensor.dtype)
                    valid = lengths > 0
                    rows = torch.arange(len(indices))[valid]
                    native_rewards[rows, lengths[valid] - 1] = torch.as_tensor(rewards, dtype=base_tensor.dtype)[valid]
                    # All native PASS variants use 0 versus nonzero; preserving
                    # answer identities among negatives does not change them.
                    types = np.where(rewards > 0, 0, 1).astype(np.int64)
                    native = DataProto.from_dict(
                        tensors={"response_mask": native_mask, "token_level_rewards": native_rewards},
                        non_tensors={
                            "uid": np.asarray([uid] * len(indices), dtype=object),
                            "answer_types": types,
                            "consistency_rate": consistency,
                        },
                    )
                    native = advantage_fn(native, **advantage_kwargs)
                    adv_by_label[label] = _scalar_advantage(native.batch["advantages"], native_mask)
                reference = adv_by_label[evidence.original_label]
                skipped = (
                    np.asarray(batch.non_tensor_batch.get("zero_advantage_mask", np.zeros(len(batch))))[indices] > 0
                )
                if not np.allclose(reference[~skipped], base[~skipped], rtol=2e-5, atol=2e-6):
                    raise ValueError("Native hard-label advantage replay disagrees with original student advantages")
                bootstrap = np.stack([adv_by_label[label] for label in labels])
                bootstrap[:, skipped] = 0
                support, reliability, multipliers = direction_weights(
                    base,
                    bootstrap,
                    attenuation=self.attenuation,
                )
                low_weights.extend(reliability.tolist())
                flip_rates.append(
                    sum(label is not None and label != evidence.original_label for label in labels) / self.samples
                )
                abstention_rates.append(sum(label is None for label in labels) / self.samples)
            all_multipliers[indices] = multipliers
            all_reliability[indices] = reliability
            # Observation-only fields are assembled after deciding weights. The
            # prompt permits a later offline dataset join without exposing any
            # reference answer to the estimator; responses permit error audits.
            from verl.utils.reward_score.ttrl.latex_clean import normalize_latex
            from verl.utils.reward_score.ttrl.qwen.qwen_math_parser import extract_answer
            from verl.utils.reward_score.ttrl.teacher_judgments import parse_teacher_judgment

            student_observations = []
            for index in indices:
                length = int(mask[int(index)].sum().item())
                response = self.tokenizer.decode(
                    batch.batch["responses"][int(index), :length], skip_special_tokens=False
                )
                try:
                    answer = extract_answer(normalize_latex(response), evidence.task) or ""
                    parse_error = ""
                except Exception as error:
                    # This diagnostic must never alter the already-decided
                    # supervision weights or fail a training step.
                    answer, parse_error = "", type(error).__name__
                student_observations.append(
                    {
                        "response": response,
                        "answer": answer,
                        "answer_parse_error": parse_error,
                        "response_tokens": length,
                    }
                )
            source = batch.non_tensor_batch.get("data_source")
            source = str(source[int(indices[0])]) if source is not None else ""
            extra = batch.non_tensor_batch.get("extra_info")
            extra = extra[int(indices[0])] if extra is not None else {}
            dataset_index = str(extra.get("index", "")) if isinstance(extra, dict) else ""

            json_records.append(
                {
                    "step": int(step),
                    "uid": uid,
                    "group_index": evidence.group_index,
                    "prompt_digest": evidence.prompt_digest,
                    "problem_prompt": evidence.prompt_text,
                    "data_source": source,
                    "dataset_index": dataset_index,
                    "route": evidence.route,
                    "majority_rate": evidence.majority_rate,
                    "original_label": evidence.original_label,
                    "acceptance_threshold": evidence.acceptance_threshold,
                    "judgments_per_candidate": evidence.verification_n,
                    "verdict_format": evidence.verdict_format,
                    "candidates": [{"answer": answer, "count": count} for answer, count in evidence.candidates],
                    "verification": [
                        {
                            "answer": answer,
                            "frequency": frequency,
                            "verdict": parse_teacher_judgment(output, verdict_format=evidence.verdict_format),
                            "output": output,
                        }
                        for answer, frequency, output in evidence.outputs
                    ],
                    "bootstrap_selection": [
                        {"label": label, "count": count} for label, count in Counter(labels).items()
                    ],
                    "rows": [
                        {
                            "vote_ordinal": int(ordinal),
                            "support": float(s),
                            "reliability": float(w),
                            "multiplier": float(m),
                            "base_advantage": float(a),
                            "final_advantage": float(a * m),
                        }
                        | observation
                        for ordinal, s, w, m, a, observation in zip(
                            ordinals, support, reliability, multipliers, base, student_observations
                        )
                    ],
                    "base_absolute_token_advantage_mass": float(mass.sum()),
                    "final_absolute_token_advantage_mass": float(np.dot(mass, multipliers)),
                }
            )
        multiplier_tensor = torch.as_tensor(all_multipliers, dtype=advantages.dtype, device=advantages.device).detach()
        # Preserve exact bit patterns when all coefficients are neutral.
        if np.any(all_multipliers != 1):
            batch.batch["advantages"] = advantages * multiplier_tensor.unsqueeze(-1)
        batch.batch["evidence_multiplier"] = multiplier_tensor
        batch.batch["evidence_weighting"] = torch.as_tensor(
            all_reliability, dtype=advantages.dtype, device=advantages.device
        )
        original_mass = float(advantages.abs().sum().item())
        final_mass = float(batch.batch["advantages"].abs().sum().item())
        prefix = "train/evidence_"
        metrics.update(
            {
                prefix + "high_groups": float(sum(record.route == "high_consensus" for record in self.records.values())),
                prefix + "low_update_groups": float(sum(record.route == "teacher_selected" for record in self.records.values())),
                prefix + "low_skip_groups": float(sum(record.route == "abstain" for record in self.records.values())),
                prefix + "low_weight_mean": float(np.mean(low_weights)) if low_weights else 1.0,
                prefix + "multiplier_mean": float(all_multipliers.mean()),
                prefix + "label_flip_rate": float(np.mean(flip_rates)) if flip_rates else 0.0,
                prefix + "abstention_rate": float(np.mean(abstention_rates)) if abstention_rates else 0.0,
                prefix + "retained_advantage_mass": final_mass / original_mass if original_mass else 1.0,
            }
        )
        if self.dump:
            self.directory.mkdir(parents=True, exist_ok=True)
            destination = self.directory / f"evidence_step_{int(step):06d}.jsonl"
            temporary = destination.with_suffix(".jsonl.tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                for record in sorted(json_records, key=lambda item: item["group_index"]):
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            temporary.replace(destination)
        print(
            f"[REPO/evidence] step={step} high={int(metrics[prefix + 'high_groups'])} "
            f"low_update={int(metrics[prefix + 'low_update_groups'])} "
            f"low_skip={int(metrics[prefix + 'low_skip_groups'])} "
            f"weight={metrics[prefix + 'low_weight_mean']:.4f} label_flip={metrics[prefix + 'label_flip_rate']:.4f} "
            f"abstain={metrics[prefix + 'abstention_rate']:.4f} "
            f"retained_adv_mass={metrics[prefix + 'retained_advantage_mass']:.4f}"
        )
