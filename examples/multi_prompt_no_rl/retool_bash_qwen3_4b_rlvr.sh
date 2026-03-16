#!/bin/bash
set -ex

export PYTHONBUFFERED=16

# Fixed config for a single node with 4x H100 and a 4B model.
# Keep these explicit to avoid environment-dependent auto-detection behavior.
NUM_GPUS=4
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-4}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.4}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-5120}"

if [ "${MAX_TOKENS_PER_GPU}" -le "${ROLLOUT_MAX_RESPONSE_LEN}" ]; then
   echo "ERROR: MAX_TOKENS_PER_GPU (${MAX_TOKENS_PER_GPU}) must be greater than ROLLOUT_MAX_RESPONSE_LEN (${ROLLOUT_MAX_RESPONSE_LEN})" >&2
   exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
   NVLINK_COUNT="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)"
else
   NVLINK_COUNT=0
fi

if [ "${NVLINK_COUNT}" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

if [ "${ACTOR_NUM_NODES}" -ne 1 ]; then
   echo "ERROR: This script is tuned for a single node; got ACTOR_NUM_NODES=${ACTOR_NUM_NODES}" >&2
   exit 1
fi

if [ "${ACTOR_NUM_GPUS_PER_NODE}" -ne 4 ]; then
   echo "ERROR: This script expects ACTOR_NUM_GPUS_PER_NODE=4; got ${ACTOR_NUM_GPUS_PER_NODE}" >&2
   exit 1
fi

if [ "${NUM_GPUS_PER_NODE}" -ne 4 ]; then
   echo "ERROR: This script expects NUM_GPUS_PER_NODE=4; got ${NUM_GPUS_PER_NODE}" >&2
   exit 1
fi

if [ "${ROLLOUT_NUM_GPUS_PER_ENGINE}" -ne 1 ]; then
   echo "ERROR: This script expects ROLLOUT_NUM_GPUS_PER_ENGINE=1; got ${ROLLOUT_NUM_GPUS_PER_ENGINE}" >&2
   exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
source "${REPO_ROOT}/scripts/models/qwen3-4B.sh"

MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-${REPO_ROOT}/../Megatron-LM}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${REPO_ROOT}/Qwen/Qwen3-4B-Instruct-2507}"
REF_LOAD="${REF_LOAD:-${REPO_ROOT}/Qwen/Qwen3-4B-Instruct-2507_torch_dist}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/outputs/qwen3-4b-bash-rlvr}"
OPEN_R1_MULTI_SUBSET="${OPEN_R1_MULTI_SUBSET:-level_4}"
OPEN_R1_MULTI_DOMAIN="${OPEN_R1_MULTI_DOMAIN:-number theory}"
PROBLEMS_PER_PROMPT="${PROBLEMS_PER_PROMPT:-100}"
export PROBLEMS_PER_PROMPT
PROMPT_DATA_DIR="${PROMPT_DATA_DIR:-${REPO_ROOT}/data/retool_bash_multi_prompt}"
PROMPT_DATA_FILE="${PROMPT_DATA_FILE:-${PROMPT_DATA_DIR}/train.parquet}"
WANDB_PROJECT="${WANDB_PROJECT:-slime-open-r1}"
WANDB_GROUP="${WANDB_GROUP:-qwen3-4B-bash-rlvr}"
SLIME_BASH_TOOL_WORKDIR="${SLIME_BASH_TOOL_WORKDIR:-/opt/NeMo/slime_bash_tool_workspace}"

mkdir -p "${PROMPT_DATA_DIR}"
pushd "${SCRIPT_DIR}" >/dev/null
python3 - "${OPEN_R1_MULTI_SUBSET}" "${PROBLEMS_PER_PROMPT}" "${PROMPT_DATA_FILE}" "${OPEN_R1_MULTI_DOMAIN}" <<'PY'
import sys
from data_utils import build_verl_parquet_openr1_bigmath_multi

subset = sys.argv[1]
problems_per_prompt = int(sys.argv[2])
local_save_file = sys.argv[3]
domain = sys.argv[4].strip() or None

train_ds = build_verl_parquet_openr1_bigmath_multi(
    subset=subset,
    problems_per_prompt=problems_per_prompt,
    domain=domain,
)

train_ds.to_parquet(local_save_file)

print('Wrote:', local_save_file)
print(f'Train examples: {len(train_ds)}')
print(f"Problems in first prompt: {train_ds[0]['prompt'].count('=== PROBLEM DELIMITER ===') + 1}")
print(train_ds[0])
PY
popd >/dev/null

PROMPT_DATA="${PROMPT_DATA_FILE}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_DIR}"
   --save-interval 20
   --rotary-base 5000000
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key solution
   --reward-key score
   --num-rollout 3000
   --rollout-batch-size 128
   --n-samples-per-prompt 8
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature 1
   --global-batch-size 1024
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 4
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project "${WANDB_PROJECT}"
   --wandb-group "${WANDB_GROUP}"
)

if [ -n "${WANDB_KEY:-}" ]; then
   WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
fi

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_bash_retool.generate
   --custom-rm-path generate_with_bash_retool.reward_func
   --custom-convert-samples-to-train-data-path custom_convert_samples_to_train_data.convert_samples_to_train_data
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
)
SGLANG_ARGS+=(--sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}")

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

pkill -9 -f ray || true
pkill -9 -f sglang || true

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${SCRIPT_DIR}:${REPO_ROOT}\",
    \"SLIME_BASH_TOOL_WORKDIR\": \"${SLIME_BASH_TOOL_WORKDIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${REPO_ROOT}/train.py" \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   --num-gpus-per-node "${NUM_GPUS_PER_NODE}" \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
