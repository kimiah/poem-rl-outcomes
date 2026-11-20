#!/bin/bash
# One P-GRPO run (step 2, App. B.2): verl GRPO with the KL penalty in the loss, a LoRA adapter of π_ref from the shared
# initialization INIT (drawn below when missing), on the P-GRPO prompts of prompts.py, saved as the PEFT adapter
# ADAPTER.
# About 45 minutes on one 80 GB GPU. Runs in the directory of prompts.py's files; verl's run goes under verl/ and is
# removed once the adapter is saved (its checkpoint and merged model take about 5 GB).
#
#   cd OUT && INIT=adapters/p_grpo_init VERL_PYTHON=/path/to/verl-env/bin/python \
#       bash /path/to/reproduce/train_p_grpo.sh ADAPTER REWARD [OVERRIDES]
#
# REWARD is reward_name=R or combined_reward_name=C, R and C names of poem.programmatic; OVERRIDES are more Hydra
# overrides (the second RL run: data.seed=2). verl runs in a second environment, whose python is VERL_PYTHON
# (DETAILS.md).
set -eo pipefail
: "${2:?usage: train_p_grpo.sh ADAPTER REWARD [OVERRIDES]}"
adapter=$1 reward=$2
shift 2
: "${INIT:?set INIT, the directory of the shared LoRA initialization}"
: "${VERL_PYTHON:?set VERL_PYTHON to the python of the verl environment (DETAILS.md)}"
run=verl/$(basename "$(dirname "$adapter")")/$(basename "$adapter")
REFERENCE=$(python -c "from huggingface_hub import snapshot_download as d; import poem
print(d(poem.REFERENCE_MODEL, revision=poem.REFERENCE_REVISION))")
SCORE=$(python -c "import poem.programmatic as m; print(m.__file__)")
python -c "import poem.programmatic as p; p.ensure_nltk()"     # before verl starts its reward workers
# a new shared LoRA initialization of π_ref (the paper's is not distributed): rank 64, α 128, the 7 projections,
# dropout 0.05, seed 0
[ -f "$INIT/adapter_model.safetensors" ] || python - "$INIT" <<'EOF'
import sys

import torch
from peft import LoraConfig, get_peft_model

from poem.policies import load_policy

torch.manual_seed(0)
modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
config = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, target_modules=modules, task_type="CAUSAL_LM")
get_peft_model(load_policy(device="cpu"), config).save_pretrained(sys.argv[1])
EOF
"$VERL_PYTHON" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files=p_grpo_train.parquet data.val_files=p_grpo_val.parquet data.train_batch_size=64 \
    data.max_prompt_length=512 data.max_response_length=512 data.filter_overlong_prompts=True \
    data.truncation=error +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$REFERENCE" actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 actor_rollout_ref.model.lora_adapter_path="$INIT" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-5 actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio=0.2 actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.load_format=safetensors actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24576 \
    reward_model.enable=False reward.num_workers=8 reward.reward_manager.name=naive \
    reward.custom_reward_function.path="$SCORE" reward.custom_reward_function.name=verl_compute_score \
    "+reward.custom_reward_function.reward_kwargs.$reward" \
    trainer.balance_batch=True 'trainer.logger=["console"]' trainer.total_epochs=1 trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=1 trainer.test_freq=-1 trainer.n_gpus_per_node=1 trainer.nnodes=1 \
    trainer.default_local_dir="$run" "$@"
"$VERL_PYTHON" -m verl.model_merger merge --backend fsdp --target_dir "$run/merged" \
    --local_dir "$run/global_step_$(cat "$run/latest_checkpointed_iteration.txt")/actor"
merged=$(dirname "$(find "$run/merged" -name adapter_config.json | head -n 1)")
# verl writes lora_alpha 0; the runs train at α = 2r
python -c "import json, sys; c = json.load(open(sys.argv[1])); c['lora_alpha'] = c['lora_alpha'] or 2 * c['r']
json.dump(c, open(sys.argv[1], 'w'), indent=2)" "$merged/adapter_config.json"
# adapter_config.json last, by rename: run.sh takes it as the sign that the adapter is complete
mkdir -p "$adapter" && cp "$merged/adapter_model.safetensors" "$adapter/"
cp "$merged/adapter_config.json" "$adapter/adapter_config.json.tmp"
mv "$adapter/adapter_config.json.tmp" "$adapter/adapter_config.json"
rm -rf "$run"
