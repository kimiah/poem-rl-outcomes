#!/bin/bash
# PoEM at weights you choose, with no calibration or fit: one decode at every composition strength γ of GAMMA, next to
# π_ref and π_tar, then their rewards, recovery and KL to π_tar. PoEM at γ is OUT/poem_g<γ>.jsonl, γ as a float
# (poem_g1.0.jsonl).
#
#   EXPERTS="experts/lora_a experts/lora_b" ALPHA="0.7 0.3" GAMMA="0.5 1 2" REWARD=my_reward.py \
#       TARGET=target/lora_t bash examples/fixed_weights.sh
#
# REWARD is a Python file that defines reward(prompt, response) -> float. Also: GAMMA (default 1), EVAL_PROMPTS
# (default eval_prompts.json), OUT (fixed_weights).
set -eo pipefail
: "${EXPERTS:?set EXPERTS}" "${ALPHA:?set ALPHA, one weight per expert}" "${REWARD:?set REWARD}" "${TARGET:?set TARGET}"
OUT=${OUT:-fixed_weights}

python -m poem sample --prompts "${EVAL_PROMPTS:-eval_prompts.json}" --experts $EXPERTS --alpha $ALPHA \
    --gamma ${GAMMA:-1} --reference --target-adapter "$TARGET" --out-dir "$OUT"
python -m poem evaluate --dir "$OUT" --reward "$REWARD" --out "$OUT/eval.json"
python -m poem kl --dir "$OUT" --out "$OUT/kl.json"
