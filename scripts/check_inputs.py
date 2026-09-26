"""Fail before GPU allocation if local model/data inputs are missing."""
from pathlib import Path
from inspect_config import read_config

cfg = read_config()
missing = [str(p) for p in (*cfg.data.train_files, *cfg.data.val_files) if not Path(p).is_file()]
if not Path(cfg.actor_rollout_ref.model.path).is_dir():
    missing.append(str(cfg.actor_rollout_ref.model.path))
if missing:
    raise SystemExit('Missing local inputs (see data/README.md):\n' + '\n'.join(missing))
if not cfg.actor_rollout_ref.teacher_ema.enabled or cfg.teacher_policy_gradient_enabled:
    raise SystemExit('REPO requires EMA teacher with no teacher optimizer/gradient training.')
if cfg.teacher_high_consensus_threshold != cfg.teacher_low_consensus_threshold:
    raise SystemExit('REPO uses one consensus threshold for its two routes.')
print('[REPO] Model/data paths and teacher configuration checked.')
