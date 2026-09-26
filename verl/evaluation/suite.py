"""Scheduled, bounded validation with distinct greedy and sampled generations.

This opt-in path never changes training rewards or the persistent rollout
configuration. Raw answers, rule-based scores, IDs, and estimator definitions
are written alongside question-level summaries for independent reconstruction.
"""

import json
import os
import time
from collections import Counter, defaultdict
from math import comb
from numbers import Integral
from pathlib import Path

import numpy as np

from verl.utils.validation_sampling import validation_request_seed


def should_save_checkpoint(step, trainer_config, *, is_last_step=False):
    """Explicit save steps override the period; optional storage guard skips saves."""
    explicit = trainer_config.get("checkpoint_steps")
    if explicit is not None:
        values = list(explicit)
        if any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0 for x in values):
            raise ValueError("trainer.checkpoint_steps must contain positive integer steps")
        if len(set(values)) != len(values):
            raise ValueError("trainer.checkpoint_steps must not contain duplicates")
        selected = step in values
    else:
        frequency = int(trainer_config.get("save_freq", -1))
        selected = frequency > 0 and (is_last_step or step % frequency == 0)
    minimum_free = int(trainer_config.get("checkpoint_min_free_bytes", 0))
    if selected and minimum_free > 0:
        import shutil
        from pathlib import Path
        directory = Path(trainer_config["default_local_dir"]).resolve()
        while not directory.exists():
            directory = directory.parent
        free = shutil.disk_usage(directory).free
        if free < minimum_free:
            print(f"[checkpoint-storage] SKIP step={step}: free_bytes={free} "
                  f"required_bytes={minimum_free}; evaluation and logs are retained", flush=True)
            return False
    return selected


def should_validate(step, trainer_config, *, is_last_step=False, before_train=False):
    """An explicit positive step list overrides periodic and pretrain validation."""
    explicit = trainer_config.get("validation_steps")
    if explicit is not None:
        values = list(explicit)
        if any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0 for x in values):
            raise ValueError("trainer.validation_steps must contain positive integer steps")
        if len(set(values)) != len(values):
            raise ValueError("trainer.validation_steps must not contain duplicates")
        return not before_train and step in values
    if before_train:
        return bool(trainer_config.get("val_before_train", True))
    frequency = int(trainer_config.get("test_freq", -1))
    return frequency > 0 and (is_last_step or step % frequency == 0)


def _binary(values):
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
        raise ValueError("Validation scores must be finite binary outcomes")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("Validation scores must be binary outcomes")
    return array.astype(int).tolist()


def question_metrics(question, *, expected_n=32, include_greedy=True):
    """Exact finite-sample metrics; ties in majority use first sampled occurrence.

    Invalid extracted answers form one failed-answer bucket and can win the
    vote. This prevents a mostly unparseable question from obtaining maj@32 by
    voting only over its few parseable outputs.
    """
    samples = question["sampled"]
    greedy = question["greedy"]
    if len(samples) != expected_n or len(greedy) != int(include_greedy) or expected_n < 16:
        raise ValueError("Each question needs the configured sampled count and optional independent greedy output")
    scores = _binary([sample["score"] for sample in samples])
    greedy_score = _binary([greedy[0]["score"]])[0] if include_greedy else None
    c, n = sum(scores), len(scores)
    predictions = [sample.get("prediction") or None for sample in samples]
    winner, count = Counter(predictions).most_common(1)[0]
    majority_score = 0 if winner is None else scores[predictions.index(winner)]
    frequencies = np.asarray(list(Counter(predictions).values()), dtype=float) / n
    result = {
        "dataset": question["dataset"],
        "question_id": question["question_id"],
        "n": n,
        "correct": c,
        "greedy_at_1": float(greedy_score) if include_greedy else None,
        "pass_at_1": c / n,
        "pass_at_16": 1.0 - comb(n - c, 16) / comb(n, 16),
        "pass_pow_4": comb(c, 4) / comb(n, 4),
        "coverage_at_32": float(c > 0),
        "majority_at_32": float(majority_score),
        "majority_ratio": count / n,
        "answer_entropy": float(-(frequencies * np.log(frequencies)).sum()),
        "parse_failure_rate": sum(prediction is None for prediction in predictions) / n,
        "sampled_response_tokens": float(np.mean([sample["response_tokens"] for sample in samples])),
        "greedy_response_tokens": float(greedy[0]["response_tokens"]) if include_greedy else None,
        "sampled_length_limit_rate": float(np.mean([sample["length_limit_hit"] for sample in samples])),
        "greedy_length_limit_rate": float(greedy[0]["length_limit_hit"]) if include_greedy else None,
    }

    return {key: value for key, value in result.items() if value is not None}


MEAN_FIELDS = (
    "greedy_at_1",
    "pass_at_1",
    "pass_at_16",
    "pass_pow_4",
    "coverage_at_32",
    "majority_at_32",
    "majority_ratio",
    "answer_entropy",
    "parse_failure_rate",
    "sampled_response_tokens",
    "greedy_response_tokens",
    "sampled_length_limit_rate",
    "greedy_length_limit_rate",
)
DISPLAY_NAMES = {
    "greedy_at_1": "greedy@1",
    "pass_at_1": "pass@1",
    "pass_at_16": "pass@16",
    "pass_pow_4": "pass_pow4",
    "coverage_at_32": "coverage@32",
    "majority_at_32": "maj@32",
}


def summarize_questions(questions, *, expected_n=32, exclude_from_macro=("dapo",), include_greedy=True):
    """Return per-dataset and explicitly distinguished aggregate metrics.

    Primary macro averages datasets equally; primary micro averages questions.
    Matching datasets (DAPO by default) remain in per-dataset reporting and are
    included only in explicitly labeled all-dataset diagnostic aggregates.
    """
    if expected_n != 32:
        raise ValueError("Full validation uses exactly 32 samples for the requested @32 metrics")
    rows = [question_metrics(question, expected_n=expected_n, include_greedy=include_greedy) for question in questions]
    mean_fields = tuple(field for field in MEAN_FIELDS if include_greedy or not field.startswith("greedy_"))
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        identity = (row["dataset"], row["question_id"])
        if identity in seen:
            raise ValueError(f"Duplicate validation question identity: {identity}")
        seen.add(identity)
        grouped[row["dataset"]].append(row)

    def aggregate(name, group, *, count=None, scope="dataset"):
        return {
            "dataset": name,
            "scope": scope,
            "questions": len(group) if count is None else count,
            "n": expected_n,
            **{field: float(np.mean([row[field] for row in group])) for field in mean_fields},
        }

    dataset_rows = [aggregate(source, group) for source, group in grouped.items()]
    primary = [
        row
        for row in dataset_rows
        if not any(str(pattern).lower() in row["dataset"].lower() for pattern in exclude_from_macro)
    ]
    aggregates = []
    for name, selected in (("PRIMARY", primary), ("ALL_DIAGNOSTIC", dataset_rows)):
        if not selected:
            continue
        selected_names = {row["dataset"] for row in selected}
        selected_questions = [row for row in rows if row["dataset"] in selected_names]
        aggregates.append(aggregate(f"{name}_MACRO", selected, count=len(selected_questions), scope="dataset_macro"))
        aggregates.append(aggregate(f"{name}_MICRO", selected_questions, scope="question_micro"))
    metrics = {}
    for row in dataset_rows + aggregates:
        prefix = f"full-val/{row['dataset']}"
        metrics[f"{prefix}/questions"] = row["questions"]
        for field in mean_fields:
            metrics[f"{prefix}/{DISPLAY_NAMES.get(field, field)}"] = row[field]
    return metrics, dataset_rows + aggregates, rows


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize validation value {type(value)}")


def _write_json(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False, default=_json_default, allow_nan=False) + "\n")


def _score_ttrl_batch(reward_manager, batch, outputs):
    """Use the existing GT teacher once, avoiding redundant legacy TTRL metrics.

    Extraction is deliberately one response at a time: the legacy auto_extract
    bulk interface drops None predictions, which would otherwise misalign rows.
    """
    from verl.utils.reward_score.ttrl.auto_extract import auto_extract
    from verl.utils.reward_score.ttrl.auto_verify import verify_many

    groups = defaultdict(list)
    reward_key = getattr(reward_manager, "reward_fn_key", "data_source")
    for index, source in enumerate(batch.non_tensor_batch[reward_key]):
        groups[reward_manager._data_source_to_task(source)].append(index)
    scores, predictions = [None] * len(batch), [None] * len(batch)
    for task, indices in groups.items():
        texts = [outputs[index] for index in indices]
        labels = [batch.non_tensor_batch["reward_model"][index]["ground_truth"] for index in indices]
        verified = verify_many(task, texts, labels)
        if len(verified) != len(indices):
            raise ValueError("Ground-truth teacher returned a different number of rows")
        for index, score in zip(indices, verified):
            scores[index] = score
            extracted = auto_extract(task, [outputs[index]], num_workers=0)
            predictions[index] = str(extracted[0]) if extracted and extracted[0] is not None else None
    return _binary(scores), predictions


def _generate_batch(trainer, batch, *, do_sample):
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

    # DataProto slices/repeats share metadata. Keep generation acknowledgments
    # and mode-specific fields local to this pass rather than the source loader.
    batch.meta_info = dict(batch.meta_info)
    tensor_keys = ["input_ids", "attention_mask", "position_ids"]
    non_tensor_keys = [
        key
        for key in ("raw_prompt_ids", "multi_modal_data", "multi_modal_inputs", "validation_seed")
        if key in batch.non_tensor_batch
    ]
    generation_batch = batch.pop(batch_keys=tensor_keys, non_tensor_batch_keys=non_tensor_keys)
    generation_batch.meta_info = {
        "eos_token_id": trainer.tokenizer.eos_token_id,
        "pad_token_id": trainer.tokenizer.pad_token_id,
        "recompute_log_prob": False,
        "do_sample": do_sample,
        "validate": True,
    }
    padded, padding = pad_dataproto_to_divisor(generation_batch, trainer.actor_rollout_wg.world_size)
    generated = trainer.actor_rollout_wg.generate_sequences(padded)
    if do_sample and not generated.meta_info.get("validation_request_seeds_applied", False):
        raise RuntimeError(
            "Rollout backend did not acknowledge per-request validation seeds; refusing duplicate sampling"
        )
    # Both validate=True and do_sample=False force native n=1 in this repository.
    # The 32 sampled rows were repeated by the caller; never expand them again.
    if len(generated) != len(padded):
        raise ValueError(f"Validation native n must be 1: received {len(generated)} outputs for {len(padded)} inputs")
    generated = unpad_dataproto(generated, pad_size=padding)
    return batch.union(generated)


def run_full_validation(trainer):
    """Evaluate all loader rows, stream evidence, then print one final line per dataset."""
    from verl import DataProto

    started = time.monotonic()
    cfg = trainer.config.trainer
    val_cfg = trainer.config.actor_rollout_ref.rollout.val_kwargs
    n = int(val_cfg.n)
    if n != 32 or not val_cfg.do_sample or float(val_cfg.temperature) <= 0:
        raise ValueError("Full validation requires val_kwargs.n=32, do_sample=true, temperature>0")
    if not hasattr(trainer.val_reward_fn, "_data_source_to_task"):
        raise ValueError("Full validation currently requires the existing TTRL rule-based evaluation manager")
    limit = int(cfg.get("validation_micro_batch_size", 64))
    if limit < n:
        raise ValueError("validation_micro_batch_size must accommodate one complete 32-sample question")
    directory = Path(cfg.get("validation_output_dir") or Path(cfg.default_local_dir) / "evaluation")
    directory.mkdir(parents=True, exist_ok=True)
    include_greedy = bool(cfg.get("validation_include_greedy", True))
    step = int(trainer.global_steps)
    base_seed = int(cfg.get("validation_seed", 42))
    response_path = directory / f"responses_step_{step:06d}.jsonl"
    temporary_path = response_path.with_suffix(".jsonl.partial")
    questions = {}
    row_offset = 0
    generation_seconds, scoring_seconds = 0.0, 0.0
    # Logging examples are bounded; all full responses live in streamed JSONL.
    log_inputs, log_outputs, log_scores = [], [], []
    max_examples = int(cfg.get("log_val_generations", 0))
    with temporary_path.open("w", encoding="utf-8") as handle:
        for test_data in trainer.val_dataloader:
            original = DataProto.from_single_dict(test_data)
            sources = original.non_tensor_batch.get("data_source", ["unknown"] * len(original))
            for local_index, source in enumerate(sources):
                qid = row_offset + local_index
                questions[qid] = {"dataset": str(source), "question_id": qid, "sampled": [], "greedy": []}
            # The greedy pass is an independent generation, never a selected sample.
            for mode, repeats in ((("greedy", 1), ("sampled", n)) if include_greedy else (("sampled", n),)):
                questions_per_batch = max(1, limit // repeats)
                for start in range(0, len(original), questions_per_batch):
                    stop = min(start + questions_per_batch, len(original))
                    batch = original[start:stop].repeat(repeat_times=repeats, interleave=True)
                    ids = np.repeat(np.arange(row_offset + start, row_offset + stop), repeats).tolist()
                    ordinals = np.tile(np.arange(repeats), stop - start).tolist()
                    seeds = (
                        [validation_request_seed(base_seed, step, qid, ordinal) for qid, ordinal in zip(ids, ordinals)]
                        if mode == "sampled"
                        else [None] * len(ids)
                    )
                    if mode == "sampled":
                        batch.non_tensor_batch["validation_seed"] = np.asarray(seeds, dtype=np.int64)
                    input_texts = [
                        trainer.tokenizer.decode(tokens, skip_special_tokens=True)
                        for tokens in batch.batch["input_ids"]
                    ]
                    tick = time.monotonic()
                    generated = _generate_batch(trainer, batch, do_sample=(mode == "sampled"))
                    generation_seconds += time.monotonic() - tick
                    response_ids = generated.batch["responses"]
                    response_lengths = generated.batch["attention_mask"][:, -response_ids.shape[-1] :].sum(-1).tolist()
                    outputs = [
                        trainer.tokenizer.decode(tokens[: int(length)], skip_special_tokens=True)
                        for tokens, length in zip(response_ids, response_lengths)
                    ]
                    tick = time.monotonic()
                    scores, predictions = _score_ttrl_batch(trainer.val_reward_fn, generated, outputs)
                    scoring_seconds += time.monotonic() - tick
                    for index, (qid, ordinal, score, prediction, output, length) in enumerate(
                        zip(ids, ordinals, scores, predictions, outputs, response_lengths)
                    ):
                        length = int(length)
                        last_token = int(response_ids[index, length - 1]) if length else None
                        limit_hit = length == response_ids.shape[-1] and last_token != trainer.tokenizer.eos_token_id
                        evidence = {
                            "score": score,
                            "prediction": prediction,
                            "response_tokens": length,
                            "length_limit_hit": limit_hit,
                        }
                        questions[qid][mode].append(evidence)
                        reward_model = generated.non_tensor_batch["reward_model"][index]
                        _write_json(
                            handle,
                            {
                                "step": step,
                                "dataset": questions[qid]["dataset"],
                                "question_id": qid,
                                "mode": mode,
                                "sample_index": ordinal,
                                "sampling_seed": seeds[index],
                                "input": input_texts[index],
                                "output": output,
                                "ground_truth": reward_model["ground_truth"],
                                **evidence,
                            },
                        )
                        if mode == "sampled" and len(log_scores) < max_examples:
                            log_inputs.append(input_texts[index])
                            log_outputs.append(output)
                            log_scores.append(score)
                    handle.flush()
                    print(
                        f"[full-val] step={step} mode={mode} rows={row_offset + stop} "
                        f"generated={len(generated)} scoring_s={scoring_seconds:.1f}",
                        flush=True,
                    )
                    del generated, batch, response_ids
            row_offset += len(original)
    metrics, summary_rows, question_rows = summarize_questions(
        list(questions.values()),
        expected_n=n,
        exclude_from_macro=cfg.get("validation_exclude_from_macro", ["dapo"]),
        include_greedy=include_greedy,
    )
    elapsed = time.monotonic() - started
    metrics.update(
        {
            "full-val/timing/total_seconds": elapsed,
            "full-val/timing/generation_seconds": generation_seconds,
            "full-val/timing/scoring_seconds": scoring_seconds,
            "full-val/generated_responses": row_offset * (n + int(include_greedy)),
        }
    )
    os.replace(temporary_path, response_path)
    question_path = directory / f"questions_step_{step:06d}.jsonl"
    with question_path.open("w", encoding="utf-8") as handle:
        for row in question_rows:
            _write_json(handle, {"step": step, **row})
    result = {
        "step": step,
        "sampling_seed_policy": {
            "base_seed": base_seed,
            "coordinates": ["base_seed", "step", "question_id", "sample_index"],
            "hash": "blake2b-64/verl-val-seeds masked to nonnegative int64",
        },
        "sampling": {
            "n": n,
            "temperature": float(val_cfg.temperature),
            "top_p": float(val_cfg.top_p),
            "top_k": int(val_cfg.top_k),
        },
        "greedy": {"n": int(include_greedy), "performed": include_greedy, "temperature": 0.0, "do_sample": False},
        "estimators": {
            "pass@1": "c/n",
            "pass@16": "1-C(n-c,16)/C(n,16)",
            "pass^4": "C(c,4)/C(n,4)",
            "coverage@32": "1[c>0]",
            "maj@32": "score of first response in most frequent extracted-answer bucket; first-seen tie",
        },
        "rows": summary_rows,
        "metrics": metrics,
        "response_file": str(response_path),
        "question_file": str(question_path),
    }
    with (directory / f"metrics_step_{step:06d}.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False)
        handle.write("\n")
    with (directory / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        _write_json(handle, result)
    for row in summary_rows:
        greedy_text = f"greedy@1={row['greedy_at_1']:.2%} " if include_greedy else ""
        print(
            f"[full-val] step={step} FINAL {row['dataset']}: "
            f"{greedy_text}pass@1={row['pass_at_1']:.2%} "
            f"pass@16={row['pass_at_16']:.2%} pass^4={row['pass_pow_4']:.2%} "
            f"maj@32={row['majority_at_32']:.2%} coverage@32={row['coverage_at_32']:.2%} "
            f"q={row['questions']} n={n} len={row['sampled_response_tokens']:.1f} "
            f"parse_fail={row['parse_failure_rate']:.2%} limit_hit={row['sampled_length_limit_rate']:.2%} "
            f"scope={row['scope']}",
            flush=True,
        )
    print(f"[full-val] step={step} COMPLETE seconds={elapsed:.1f} evidence={response_path}", flush=True)
    trainer._maybe_log_val_generations(inputs=log_inputs, outputs=log_outputs, scores=log_scores)
    return metrics
