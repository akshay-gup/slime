#!/bin/bash
set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "/root/slime/scripts/models/qwen3-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/font-info/qwen3-4b-sft
   --ref-load /root/font-info/qwen3-4b-sft_torch_dist
   --save /root/font-info/qwen3-4b-sft/qwen3-4b-bash-rlvr/
   --save-interval 20
   --rotary-base 5000000
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
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

ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:/root/slime\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
