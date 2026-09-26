#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$root"
export REPO_RUN_NAME=${REPO_RUN_NAME:-repo_$(date -u +%Y%m%dT%H%M%SZ)}
export REPO_OUTPUT_DIR=${REPO_OUTPUT_DIR:-$root/outputs/$REPO_RUN_NAME}
export REPO_LOG_DIR=${REPO_LOG_DIR:-$root/logs/$REPO_RUN_NAME}
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export WANDB_MODE=${WANDB_MODE:-offline}
python_bin=${PYTHON:-python}
config=${REPO_CONFIG:-repo}
if [[ "${1:-}" == --dry-run ]]; then
  shift
  exec "$python_bin" scripts/inspect_config.py --config "$config" "$@"
fi
mkdir -p -- "$REPO_LOG_DIR" "$REPO_OUTPUT_DIR"
"$python_bin" scripts/inspect_config.py --config "$config" "$@" > "$REPO_LOG_DIR/resolved_config.yaml"
"$python_bin" scripts/check_inputs.py --config "$config" "$@"
printf 'REPO run: %s\nLogs: %s\nOutputs: %s\n' "$REPO_RUN_NAME" "$REPO_LOG_DIR" "$REPO_OUTPUT_DIR"
set +e
"$python_bin" -u -m verl.trainer.main_ppo --config-name="$config" "$@" 2>&1 | tee "$REPO_LOG_DIR/train.log"
codes=("${PIPESTATUS[@]}")
set -e
exit_code=${codes[0]}
if ((exit_code == 0 && codes[1] != 0)); then exit_code=${codes[1]}; fi
printf '{"returncode":%d}\n' "$exit_code" > "$REPO_LOG_DIR/exit_status.json"
exit "$exit_code"
