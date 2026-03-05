#!/bin/bash
set -ex

if command -v nvidia-smi >/dev/null 2>&1; then
   DETECTED_NUM_GPUS="$(nvidia-smi --list-gpus 2>/dev/null | wc -l)"
else
   DETECTED_NUM_GPUS=0
fi

NUM_GPUS="${NUM_GPUS:-${DETECTED_NUM_GPUS}}"
if [ "${NUM_GPUS}" -le 0 ]; then
   NUM_GPUS=1
fi
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-${NUM_GPUS}}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
source "${REPO_ROOT}/scripts/models/qwen3-4B.sh"

MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-${REPO_ROOT}/../Megatron-LM}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${REPO_ROOT}/checkpoints/qwen3-4b-sft}"
REF_LOAD="${REF_LOAD:-${REPO_ROOT}/checkpoints/qwen3-4b-sft_torch_dist}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/outputs/qwen3-4b-bash-rlvr}"
PROMPT_DATA="${PROMPT_DATA:-${REPO_ROOT}/data/dapo-math-17k/dapo-math-17k.jsonl}"

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
   --label-key label
   --apply-chat-template
   --reward-key score
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 256
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_bash_retool.generate
   --custom-rm-path generate_with_bash_retool.reward_func
   --custom-convert-samples-to-train-data-path custom_convert_samples_to_train_data.convert_samples_to_train_data
)

SGLANG_ARGS=()
if [ -n "${ROLLOUT_NUM_GPUS_PER_ENGINE}" ]; then
   SGLANG_ARGS+=(--rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}")
fi

ray start --head --node-ip-address 127.0.0.1 --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${SCRIPT_DIR}:${REPO_ROOT}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${REPO_ROOT}/train.py" \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${SGLANG_ARGS[@]}
