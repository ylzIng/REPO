# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Student-candidate and teacher-verification utilities.

This module implements teacher judgments on candidate answers extracted from
student rollouts. The selector also serves evidence replay so that the same
acceptance rule is applied to observed and resampled judgments.

Supports two verification modes:
  - Greedy: temperature=0, n=1 per candidate. Select True + highest frequency.
  - Sampling: temperature=0.6, n=N per candidate. Select by majority True votes.

Supports two fallback strategies when all candidates verify as False:
  - "majority": fallback to the highest-frequency candidate from student.
  - "penalize": return None, signaling the trainer to apply a -1 advantage penalty.
"""

import re
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto

logger = logging.getLogger(__name__)

# ===========================================================================
# Verification Prompt Templates
# ===========================================================================

TEACHER_SYSTEM_PROMPT = "You are a careful teacher checking a candidate answer against the problem itself."


def load_teacher_template(task: str = "math") -> str:
    """Load the versioned REPO prompt shipped with the Python package."""
    if task not in {"math", "gpqa"}:
        raise ValueError(f"Unsupported teacher task: {task}")
    return (Path(__file__).resolve().parents[3] / "prompts" / f"repo_teacher_{task}.txt").read_text(encoding="utf-8")


TEACHER_USER_TEMPLATE = load_teacher_template("math")


# ===========================================================================
# Answer Extraction from student
# ===========================================================================

def collect_student_candidates(
    pass1_data: DataProto,
    tokenizer,
    n_votes_per_prompt: int,
    task: str = "math",
    max_candidates: int = 10,
) -> List[Dict]:
    """Extract unique candidate answers from student rollout data, grouped by prompt.

    For each prompt group (n_votes_per_prompt samples), extracts and deduplicates
    answers, returning them sorted by frequency (descending).

    Args:
        pass1_data: DataProto from student generation (already union'd with batch).
        tokenizer: HuggingFace tokenizer for decoding.
        n_votes_per_prompt: Number of rollout samples per prompt.
        task: Task name for answer extraction (e.g., "math", "gpqa").
        max_candidates: Maximum number of candidate answers to keep per prompt.

    Returns:
        List of dicts, one per prompt group. Each dict contains:
            - "problem_text": str, the decoded problem prompt
            - "candidates": List[Tuple[str, int]], (answer, frequency) sorted desc
            - "all_answers": List[str], all extracted answers (including duplicates)
            - "prompt_group_idx": int, the index of this prompt group
    """
    from verl.utils.reward_score.ttrl.auto_extract import auto_extract

    assert len(pass1_data) % n_votes_per_prompt == 0, (
        f"Data length {len(pass1_data)} must be divisible by n_votes_per_prompt {n_votes_per_prompt}"
    )
    num_prompts = len(pass1_data) // n_votes_per_prompt

    prompt_groups = []

    for prompt_i in range(num_prompts):
        group_responses = []
        group_extra_info = []
        problem_text = None
        ground_truth = ""

        for j in range(n_votes_per_prompt):
            idx = prompt_i * n_votes_per_prompt + j
            data_item = pass1_data[idx]

            # Decode prompt (only need once per group)
            if problem_text is None:
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum().item())
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                problem_text = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

            if not ground_truth and "reward_model" in data_item.non_tensor_batch:
                ground_truth = data_item.non_tensor_batch["reward_model"].get("ground_truth", "")

            # Decode response
            response_ids = data_item.batch["responses"]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            group_responses.append(response_str)

            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            group_extra_info.append(extra_info)

        # Extract answers using the standard extractor
        model_answers = auto_extract(task, group_responses, extra_info=group_extra_info)

        # Count frequencies
        counter = Counter(model_answers)
        # Sort by frequency descending, then limit to max_candidates if specified
        if max_candidates is not None and max_candidates > 0:
            candidates = counter.most_common(max_candidates)
        else:
            candidates = counter.most_common()

        # Extract ground truth (fallback if missing in reward_model dict)
        if not ground_truth and group_extra_info and group_extra_info[0]:
            gt_info = group_extra_info[0].get("reward_model", {})
            ground_truth = gt_info.get("ground_truth", 
                                      group_extra_info[0].get("ground_truth", 
                                                              group_extra_info[0].get("answer", 
                                                                                      group_extra_info[0].get("target", ""))))

        prompt_groups.append({
            "problem_text": problem_text,
            "candidates": candidates,  # List[(answer, count)]
            "all_answers": model_answers,
            "prompt_group_idx": prompt_i,
            "majority_rate": candidates[0][1] / n_votes_per_prompt if candidates else 0.0,
            "majority_answer": candidates[0][0] if candidates else None,
            "ground_truth": ground_truth,
        })

    return prompt_groups


# ===========================================================================
# Construct Verification DataProto
# ===========================================================================

def build_teacher_batch(
    prompt_groups: List[Dict],
    tokenizer,
    verification_mode: str = "greedy",
    verification_n: int = 1,
    max_prompt_length: int = 2048,
    max_answer_length: int = 200,
    system_prompt: str = None,
    user_template: str = None,
    task: str = "math",
) -> Tuple[Optional[DataProto], List[Dict]]:
    """Construct a DataProto for the verification pass from extracted candidate answers.

    For each prompt group and each candidate answer, creates a verification prompt
    using the template, tokenizes it, and packs it into a DataProto.

    Args:
        prompt_groups: Output of collect_student_candidates().
        tokenizer: HuggingFace tokenizer.
        verification_mode: "greedy" or "sampling".
        verification_n: Number of verification samples per candidate (for sampling mode).
        max_prompt_length: Maximum token length for the verification prompt.
        max_answer_length: Maximum character length for candidate answer strings.
        system_prompt: Custom system prompt (defaults to TEACHER_SYSTEM_PROMPT).
        user_template: Custom user template (defaults to TEACHER_USER_TEMPLATE).

    Returns:
        Tuple of:
            - DataProto for verification generation (or None if no candidates)
            - List[Dict] mapping each row in the DataProto back to:
                {"prompt_group_idx": int, "candidate_answer": str, "frequency": int}
    """
    if system_prompt is None:
        system_prompt = TEACHER_SYSTEM_PROMPT
    if user_template is None:
        user_template = load_teacher_template(task)

    all_token_ids = []
    all_attention_masks = []
    verification_mapping = []

    for group in prompt_groups:
        problem_text = group["problem_text"]
        prompt_group_idx = group["prompt_group_idx"]

        for candidate_idx, (candidate_answer, frequency) in enumerate(group["candidates"]):
            if candidate_answer is None or str(candidate_answer).strip() == "":
                continue

            # Truncate long candidate answers
            answer_str = str(candidate_answer)[:max_answer_length]

            # Build the verification prompt text
            user_content = user_template.format(
                problem=problem_text,
                candidate_answer=answer_str,
            )

            # Apply chat template if available
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback: manual ChatML formatting
                prompt_text = (
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{user_content}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )

            # Tokenize
            encoded = tokenizer(
                prompt_text,
                # The rendered template already contains any model BOS token.
                add_special_tokens=False,
                truncation=True,
                max_length=max_prompt_length,
                padding=False,
                return_tensors=None,
            )

            token_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            # In sampling mode, repeat for verification_n samples.
            # If verification_n is None or < 0, dynamically use candidate frequency.
            if verification_mode == "sampling":
                if verification_n is None or verification_n < 0:
                    repeat_count = frequency
                    n_sampling = 1
                else:
                    # Enable native n-sampling in engine, no python dup
                    repeat_count = 1
                    n_sampling = verification_n
            else:
                repeat_count = 1
                n_sampling = 1
                
            for _ in range(repeat_count):
                all_token_ids.append(token_ids)
                all_attention_masks.append(attention_mask)
                # Keep track of mapping for N samples
                for _ in range(n_sampling):
                    verification_mapping.append({
                        "prompt_group_idx": prompt_group_idx,
                        "candidate_answer": candidate_answer,
                        "candidate_idx": candidate_idx,
                        "frequency": frequency,
                        "n_sampling": n_sampling, # recorded to help matching if needed
                    })

    if not all_token_ids:
        logger.warning("[REPO/teacher] No valid candidates found; skipping teacher judgments.")
        return None, []

    # Pad all sequences to the same length (left-padded, matching verl convention)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(len(ids) for ids in all_token_ids)

    padded_input_ids = []
    padded_attention_masks = []
    padded_position_ids = []

    for ids, mask in zip(all_token_ids, all_attention_masks):
        pad_len = max_len - len(ids)
        # Left padding
        padded_ids = [pad_token_id] * pad_len + ids
        padded_mask = [0] * pad_len + mask
        # Position IDs: 0 for padding, then 0, 1, 2, ...
        pos_ids = [0] * pad_len + list(range(len(ids)))

        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)
        padded_position_ids.append(pos_ids)

    # Convert to tensors
    input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
    attention_mask_tensor = torch.tensor(padded_attention_masks, dtype=torch.long)
    position_ids_tensor = torch.tensor(padded_position_ids, dtype=torch.long)

    batch_size = input_ids_tensor.shape[0]

    # Build DataProto
    batch = TensorDict(
        {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "position_ids": position_ids_tensor,
        },
        batch_size=batch_size,
    )

    # Store raw_prompt_ids for vLLM compatibility
    non_tensor_batch = {
        "raw_prompt_ids": np.array([ids for ids in all_token_ids], dtype=object),
    }

    # Meta info: configure for verification generation
    meta_info = {
        "verification_mode": verification_mode,
        "do_sample": verification_mode != "greedy",
        "validate": False,
        "verification_n": 1 if verification_mode != "sampling" or verification_n is None or verification_n < 0 else verification_n,
    }

    verification_batch = DataProto(
        batch=batch,
        non_tensor_batch=non_tensor_batch,
        meta_info=meta_info,
    )

    logger.info(
        f"[REPO/teacher] Constructed judgment batch: {batch_size} samples "
        f"(from {len(prompt_groups)} prompt groups), max_len={max_len}"
    )

    return verification_batch, verification_mapping


# ===========================================================================
# Parse Verification Results
# ===========================================================================

def parse_teacher_judgment(text: str, verdict_format: str = "repo") -> Optional[bool]:
    """Read one unambiguous final judgment, never a keyword in the reasoning.

    REPO accepts an optional analysis block followed by exactly one judgment and
    no continuation. Explicit legacy mode is for auditing old cached records;
    it accepts only one standalone final ``Verification Result: True/False``.
    Mixing formats, duplicate judgments (even equal ones), and truncation fail.
    """
    if verdict_format not in {"repo", "legacy"}:
        raise ValueError("verdict_format must be repo or legacy")
    if not text or not isinstance(text, str):
        return None
    if verdict_format == "legacy":
        if re.search(r"</?judgment\b", text, flags=re.IGNORECASE):
            return None
        matches = list(re.finditer(r"(?im)^\s*Verification Result:\s*(True|False)\s*$", text))
        if len(matches) != 1 or text[matches[0].end():].strip():
            return None
        # A second malformed or inline conclusion is also ambiguous.
        if len(re.findall(r"verification\s+result\s*:", text, re.IGNORECASE)) != 1:
            return None
        return matches[0].group(1).lower() == "true"
    if re.search(r"verification\s+result\s*:", text, flags=re.IGNORECASE):
        return None
    if len(re.findall(r"<judgment\b", text, flags=re.IGNORECASE)) != 1:
        return None
    if len(re.findall(r"</judgment\s*>", text, flags=re.IGNORECASE)) != 1:
        return None
    match = re.fullmatch(
        r"\s*(?:<analysis>(?:(?!</?analysis\b).)*</analysis>\s*)?"
        r"<judgment>\s*(True|False)\s*</judgment>\s*",
        text, flags=re.DOTALL | re.IGNORECASE,
    )
    return None if match is None else match.group(1).lower() == "true"


def classify_consensus_bucket(
    majority_rate: float,
    high_consistency_threshold: float = 0.5,
    low_consistency_threshold: Optional[float] = None,
) -> str:
    """Classify consistency into high/low/middle buckets.

    If low_consistency_threshold is None, it falls back to the high threshold,
    which preserves the original single-threshold behavior.
    """
    if low_consistency_threshold is None:
        low_consistency_threshold = high_consistency_threshold

    if low_consistency_threshold > high_consistency_threshold:
        low_consistency_threshold = high_consistency_threshold

    if majority_rate >= high_consistency_threshold:
        return "high"
    if majority_rate <= low_consistency_threshold:
        return "low"
    return "middle"


def pad_high_consistency_candidates_to_topk(
    prompt_groups: List[Dict],
    max_candidates: int,
    high_consistency_threshold: float = 0.5,
    low_consistency_threshold: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """Pad high-consistency groups to top-k candidates using cross-prompt answers.

    For each high-consistency prompt group, if candidate count is smaller than
    ``max_candidates``, this function samples answers from other prompt groups
    in the same batch to fill the gap.

    The added candidates use frequency=1 so they can be checked by the teacher.

    Returns:
        Stats dict with keys:
            - high_consistency_group_count
            - padded_group_count
            - padded_candidate_count
    """
    stats = {
        "high_consistency_group_count": 0,
        "padded_group_count": 0,
        "padded_candidate_count": 0,
    }

    if max_candidates is None or max_candidates <= 0 or not prompt_groups:
        return stats

    rng = np.random.default_rng(seed)

    valid_answers_by_group: Dict[int, List[str]] = {}
    all_valid_answers: List[str] = []

    for group in prompt_groups:
        group_idx = group.get("prompt_group_idx", -1)
        group_answers: List[str] = []
        for ans in group.get("all_answers", []):
            if ans is None:
                continue
            ans_str = str(ans).strip()
            if not ans_str:
                continue
            group_answers.append(ans_str)
            all_valid_answers.append(ans_str)
        valid_answers_by_group[group_idx] = group_answers

    if not all_valid_answers:
        return stats

    for group in prompt_groups:
        group_idx = group.get("prompt_group_idx", -1)
        majority_rate = group.get("majority_rate", 0.0)
        bucket = classify_consensus_bucket(
            majority_rate=majority_rate,
            high_consistency_threshold=high_consistency_threshold,
            low_consistency_threshold=low_consistency_threshold,
        )

        if bucket != "high":
            continue

        stats["high_consistency_group_count"] += 1

        original_candidates = list(group.get("candidates", []))
        if len(original_candidates) >= max_candidates:
            continue

        existing_answers = {
            str(ans).strip()
            for ans, _ in original_candidates
            if ans is not None and str(ans).strip()
        }

        own_answers = set(valid_answers_by_group.get(group_idx, []))

        donor_pool: List[str] = []
        for other_group_idx, other_answers in valid_answers_by_group.items():
            if other_group_idx == group_idx:
                continue
            donor_pool.extend(other_answers)

        if not donor_pool:
            donor_pool = [ans for ans in all_valid_answers if ans not in own_answers]

        if not donor_pool:
            continue

        need = max_candidates - len(original_candidates)
        padded_answers: List[str] = []

        unique_pool: List[str] = []
        seen = set()
        for ans in donor_pool:
            if ans in seen or ans in existing_answers:
                continue
            seen.add(ans)
            unique_pool.append(ans)

        if unique_pool and need > 0:
            if len(unique_pool) <= need:
                sampled_unique = unique_pool
            else:
                sampled_unique = rng.choice(unique_pool, size=need, replace=False).tolist()
            padded_answers.extend([str(x) for x in sampled_unique])
            need -= len(sampled_unique)

        if need > 0:
            sampled_extra = rng.choice(donor_pool, size=need, replace=True).tolist()
            padded_answers.extend([str(x) for x in sampled_extra])

        if not padded_answers:
            continue

        for ans in padded_answers:
            original_candidates.append((ans, 1))

        padded_count = min(len(padded_answers), max_candidates - len(group.get("candidates", [])))
        group["candidates"] = original_candidates[:max_candidates]
        group["padded_candidate_count"] = padded_count

        if padded_count > 0:
            stats["padded_group_count"] += 1
            stats["padded_candidate_count"] += padded_count

    return stats

def select_teacher_targets(
    verification_outputs: List[str],
    verification_mapping: List[Dict],
    prompt_groups: List[Dict],
    n_votes_per_prompt: int = 8,
    high_consistency_threshold: float = 0.5,
    low_consistency_threshold: Optional[float] = None,
    consistency_route_by_group: Optional[Dict[int, str]] = None,
    low_consistency_strategy: str = "majority",
    fallback_mode: str = "skip_update",
    verification_total_n: Optional[int] = None,
    acceptance_threshold: float = 0.5,
    verdict_format: str = "repo",
) -> Tuple[List[str], List[float], List[bool], List[str]]:
    """Resolve the final pseudo-labels and decide teacher-batch participation.

    High-consistency groups directly keep the majority answer.
    Low-consistency groups use verification True/False statistics to choose a candidate.

    Args:
        verification_outputs: List of decoded verification result strings.
        verification_mapping: The mapping list from build_teacher_batch().
        prompt_groups: Output of collect_student_candidates().
        n_votes_per_prompt: Number of student voting samples per prompt.
        high_consistency_threshold: threshold for high-consistency groups.
        low_consistency_threshold: threshold for low-consistency groups.
            If None, falls back to high_consistency_threshold.
        consistency_route_by_group: optional precomputed route mapping
            (group_idx -> "high"/"low"/"middle").
        low_consistency_strategy: "true" or "majority" selection strategy for low-consistency groups.
        fallback_mode: "skip_teacher_update" or "skip_update" for low-consistency failures.
        verification_total_n: Optional fixed teacher judgment budget per candidate.
            Invalid or missing judgments remain in this denominator. When not
            supplied, every observed judgment slot (including invalids) counts.
        acceptance_threshold: Require True fraction strictly above this value.
            The historical default is 0.5; REPO's release configuration uses 0.6.

    Returns:
        Tuple of:
            - final pseudo labels for each prompt group.
            - consistency scores for each prompt group.
            - whether each prompt group should participate in teacher-batch processing.
            - route applied to each prompt group ("high_consensus", "teacher_selected", "abstain").
    """
    assert len(verification_outputs) == len(verification_mapping), (
        f"Mismatch: {len(verification_outputs)} outputs vs {len(verification_mapping)} mappings"
    )
    if not 0 <= acceptance_threshold <= 1:
        raise ValueError("acceptance_threshold must lie in [0,1]")
    if verification_total_n is not None and verification_total_n < 1:
        raise ValueError("verification_total_n must be positive when supplied")

    num_prompt_groups = len(prompt_groups)
    group_stats: Dict[int, Dict[str, Dict[str, int]]] = {}

    for output_text, mapping in zip(verification_outputs, verification_mapping):
        group_idx = mapping["prompt_group_idx"]
        ans = mapping["candidate_answer"]
        if group_idx not in group_stats:
            group_stats[group_idx] = {}
        if ans not in group_stats[group_idx]:
            group_stats[group_idx][ans] = {
                "true_count": 0, "false_count": 0, "observed_count": 0,
                "frequency": mapping["frequency"],
            }
        group_stats[group_idx][ans]["observed_count"] += 1

        parsed = parse_teacher_judgment(output_text, verdict_format=verdict_format)
        if parsed is True:
            group_stats[group_idx][ans]["true_count"] += 1
        elif parsed is False:
            group_stats[group_idx][ans]["false_count"] += 1

    pseudo_labels = [""] * num_prompt_groups
    consistencies = [0.0] * num_prompt_groups
    should_update_second = [False] * num_prompt_groups
    routes = [""] * num_prompt_groups

    for i, group in enumerate(prompt_groups):
        group_idx = group["prompt_group_idx"]
        majority_rate = group.get("majority_rate", 0.0)
        majority_answer = group.get("majority_answer")

        route_bucket = None
        if consistency_route_by_group is not None:
            route_bucket = consistency_route_by_group.get(group_idx)
        if route_bucket not in {"high", "low", "middle"}:
            route_bucket = classify_consensus_bucket(
                majority_rate=majority_rate,
                high_consistency_threshold=high_consistency_threshold,
                low_consistency_threshold=low_consistency_threshold,
            )

        if route_bucket == "high":
            pseudo_labels[i] = majority_answer
            consistencies[i] = majority_rate
            should_update_second[i] = True
            routes[i] = "high_consensus"
            continue

        if route_bucket == "middle":
            pseudo_labels[i] = majority_answer
            consistencies[i] = majority_rate
            should_update_second[i] = False
            routes[i] = "deferred"
            continue

        candidate_stats = group_stats.get(group_idx, {})
        # Explicit candidate order makes ties independent of the order in which
        # parallel teacher outputs arrive. Older callers may omit candidates.
        candidate_order = {
            answer: index for index, (answer, _) in enumerate(group.get("candidates", ()))
        }
        for answer in candidate_stats:
            if answer not in candidate_order:
                candidate_order[answer] = len(candidate_order)
        if verification_total_n is not None and any(
            info["observed_count"] > verification_total_n for info in candidate_stats.values()
        ):
            raise ValueError("Observed teacher judgments exceed verification_total_n")
        true_set_candidates = [
            (ans, info["true_count"], info["frequency"])
            for ans, info in candidate_stats.items()
            if info["true_count"] > acceptance_threshold * (
                verification_total_n if verification_total_n is not None else info["observed_count"]
            )
        ]

        if true_set_candidates:
            if low_consistency_strategy == "true":
                true_set_candidates.sort(key=lambda x: (-x[1], -x[2], candidate_order[x[0]]))
            else:
                true_set_candidates.sort(key=lambda x: (-x[2], -x[1], candidate_order[x[0]]))

            best_ans, _, best_freq = true_set_candidates[0]
            pseudo_labels[i] = best_ans
            consistencies[i] = best_freq / max(1, n_votes_per_prompt)
            should_update_second[i] = True
            routes[i] = "teacher_selected"
        else:
            pseudo_labels[i] = majority_answer
            consistencies[i] = majority_rate
            # Abstained samples do not participate in teacher-batch processing under
            # "skip_teacher_update" and "skip_update".
            should_update_second[i] = fallback_mode not in {"skip_teacher_update", "skip_update"}
            routes[i] = "abstain"

    return pseudo_labels, consistencies, should_update_second, routes

def compute_proxy_cm_reward(
    verification_outputs: List[str],
    verification_mapping: List[Dict],
    final_pseudo_labels: Dict[int, str],
    consistency_scores: Dict[int, float],
    gt_correct_scores: Optional[List[bool]] = None,
    verdict_format: str = "repo",
) -> Tuple[List[float], Dict[str, float]]:
    """Compute surrogate CM rewards for each verification sample based on standard rules.
    
    Args:
        verification_outputs: List of decoded verification response strings.
        verification_mapping: Output from build_teacher_batch().
        final_pseudo_labels: Dict mapping prompt_group_idx to the chosen pseudo label string.
        consistency_scores: Dict mapping prompt_group_idx to the consistency float.
        gt_correct_scores: Optional list of booleans indicating if the candidate answer is equal to ground truth.
        
    Returns:
        rewards: List of float rewards
        metrics: Dict with tp/tn/fp/fn, format error rates, and GT CM metrics (if gt_correct_scores is provided).
    """
    rewards = []
    
    tp_count = 0
    tn_count = 0
    fp_count = 0
    fn_count = 0
    format_error_count = 0
    total = len(verification_outputs)
    
    # GT-based confusion matrix counters
    gt_tp_count = 0
    gt_tn_count = 0
    gt_fp_count = 0
    gt_fn_count = 0

    for i, (output_text, mapping) in enumerate(zip(verification_outputs, verification_mapping)):
        group_idx = mapping["prompt_group_idx"]
        candidate = mapping["candidate_answer"]
        
        pl = final_pseudo_labels.get(group_idx)
        consistency = consistency_scores.get(group_idx, 1.0)
        
        parsed_result = parse_teacher_judgment(output_text, verdict_format=verdict_format)
        is_pl = (candidate == pl)
        
        # Calculate GT logic if gt_correct_scores is provided
        if gt_correct_scores is not None and i < len(gt_correct_scores):
            is_gt_correct = gt_correct_scores[i]
            if parsed_result is True:
                if is_gt_correct:
                    gt_tp_count += 1
                else:
                    gt_fp_count += 1
            elif parsed_result is False:
                if is_gt_correct:
                    gt_fn_count += 1
                else:
                    gt_tn_count += 1
        
        if parsed_result is None:
            # Format exists but result can't be parsed properly (e.g. "Verification Result: maybe")
            rewards.append(-1.0)
            format_error_count += 1
        elif is_pl and parsed_result is True:   # TP
            rewards.append(1.0)
            tp_count += 1
        elif not is_pl and parsed_result is False: # TN
            rewards.append(1.0)
            tn_count += 1
        elif is_pl and parsed_result is False:   # FN
            rewards.append(-0.3)
            fn_count += 1
        elif not is_pl and parsed_result is True:  # FP
            rewards.append(-0.8)
            fp_count += 1
            
    metrics = {
        "format_error_rate": format_error_count / total if total > 0 else 0.0,
        "strictness_index": (fn_count + tn_count) / (fp_count + tp_count) if (fp_count + tp_count) > 0 else 0.0,
        "reward_mean": sum(rewards) / total if total > 0 else 0.0,
    }

    if gt_correct_scores is not None:
        total_gt_valid = gt_tp_count + gt_tn_count + gt_fp_count + gt_fn_count
        metrics.update({
            "gt_tp_rate": gt_tp_count / total_gt_valid if total_gt_valid > 0 else 0.0,
            "gt_tn_rate": gt_tn_count / total_gt_valid if total_gt_valid > 0 else 0.0,
            "gt_fp_rate": gt_fp_count / total_gt_valid if total_gt_valid > 0 else 0.0,
            "gt_fn_rate": gt_fn_count / total_gt_valid if total_gt_valid > 0 else 0.0,
        })
    
    return rewards, metrics


# ===========================================================================
# Decode Verification Outputs
# ===========================================================================

def decode_teacher_outputs(
    verification_gen_output: DataProto,
    tokenizer,
) -> List[str]:
    """Decode the generated verification responses back to strings.

    Args:
        verification_gen_output: DataProto output from generate_sequences().
        tokenizer: HuggingFace tokenizer.

    Returns:
        List of decoded response strings.
    """
    decoded_texts = []
    batch_size = len(verification_gen_output)

    for i in range(batch_size):
        data_item = verification_gen_output[i]
        response_ids = data_item.batch["responses"]
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        attention_mask = data_item.batch["attention_mask"]

        # Calculate valid response length
        valid_response_length = int(attention_mask[prompt_length:].sum().item())
        valid_response_ids = response_ids[:valid_response_length]

        text = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        decoded_texts.append(text)

    return decoded_texts
