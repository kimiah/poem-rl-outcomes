#!/bin/bash
# PoEM on your own experts: calibrate on π_ref, fit the weights to your reward, decode PoEM, π_ref and (with TARGET)
# π_tar, score them with your reward, and with TARGET compare PoEM with π_tar: recovery and KL.
#
#   EXPERTS="experts/lora_a experts/lora_b" REWARD=my_reward.py TARGET=target/lora_t bash examples/run_poem.sh
#
# REWARD is a Python file that defines reward(prompt, response) -> float. Also: GAMMA (default geom, γ_geom; 1 for
# experts trained with PPO and the KL in the reward), CALIB_PROMPTS (default calib_prompts.json), EVAL_PROMPTS
# (eval_prompts.json), OUT (poem_run). Run it from the directory that the adapter paths are relative to.
set -eo pipefail
: "${EXPERTS:?set EXPERTS, the expert adapter directories}" "${REWARD:?set REWARD, your reward file}"
OUT=${OUT:-poem_run}

python -m poem calibrate --prompts "${CALIB_PROMPTS:-calib_prompts.json}" --out "$OUT/cal.jsonl"
python -m poem fit --calibration "$OUT/cal.jsonl" --experts $EXPERTS --reward "$REWARD" --out "$OUT/weights.json"
python -m poem sample --prompts "${EVAL_PROMPTS:-eval_prompts.json}" --weights "$OUT/weights.json" --reference \
    --gamma ${GAMMA:-geom} ${TARGET:+--target-adapter "$TARGET"} --out-dir "$OUT/D"
python -m poem evaluate --dir "$OUT/D" --reward "$REWARD" --out "$OUT/D/eval.json"
[ -n "$TARGET" ] || exit 0
python -m poem kl --dir "$OUT/D" --out "$OUT/D/kl.json"
