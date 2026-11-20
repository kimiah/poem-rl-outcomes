#!/bin/bash
# Reruns the paper's programmatic and reward-model results from the experts and targets, in 12 steps:
#
#   1      the prompt sets                                                       prompts.py
#   2-6    the programmatic bases P-GRPO and P-DPO: P-GRPO training,             programmatic_bases below,
#          calibration, fits, decodes and effective ranks                        train_p_grpo.sh, ranks.py
#   7-11   the RM bases RM-RB, RM-HH, RM-Div and RM-Sky: calibration, fits,      rm_bases below, ranks.py
#          decodes and effective ranks
#   12     the paper's numbers, printed                                          summarize.py
#
# Each step's comment names the paper items it serves. DETAILS.md has the adapter layout.
#
#   SETTINGS=programmatic OUT=runs ADAPTERS=adapters VERL_PYTHON=/path/to/verl-env/bin/python bash reproduce/run.sh
#
#   OUT          the output directory; every step runs in it
#   ADAPTERS     the experts and targets; the P-GRPO adapters it lacks are trained into it
#   SETTINGS     programmatic, rm, or both (the default); rm needs the RM-basis adapters in ADAPTERS, checked first
#   VERL_PYTHON  the python of the verl environment, used only to train P-GRPO adapters (train_p_grpo.sh)
#
# Needs pip install -e ".[data]" and one 80 GB GPU.
set -eo pipefail
: "${OUT:?set OUT, the output directory}"
: "${ADAPTERS:?set ADAPTERS, the directory of the experts and targets}"
SETTINGS=${SETTINGS:-programmatic rm}
for setting in $SETTINGS; do
    case $setting in
        programmatic | rm) ;;
        *) echo "SETTINGS: unknown setting '$setting' (programmatic, rm)" >&2; exit 1 ;;
    esac
done
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$ADAPTERS" "$OUT"
ADAPTERS=$(cd "$ADAPTERS" && pwd)

# rm needs every RM-basis expert and target in ADAPTERS, and none is distributed: check them before step 1 rather than
# fail at step 7, after the programmatic steps and before step 12 prints anything
if [[ " $SETTINGS " == *" rm "* ]]; then
    missing=$(python - "$ADAPTERS" <<'EOF'
import sys
from pathlib import Path
from poem.reward_models import ROSTERS, basis_combined_rewards
need = [f"{d}/lora_{r}" for d, b in (("rm_rb", "RM-RB"), ("rm_hh", "RM-HH"), ("rm_div", "RM-Div")) for r in ROSTERS[b]]
need += [f"rm_sky/lora_{r}" for r in ROSTERS["RM-Sky"][1:]] + ["second_run/lora_skywork_second_run"]
need += [f"{d}_targets/lora_{c}" for d, b in (("rm_rb", "RM-RB"), ("rm_hh", "RM-HH")) for c in basis_combined_rewards(b)]
print(*(p for p in need if not (Path(sys.argv[1]) / p / "adapter_config.json").is_file()))
EOF
)
    if [ -n "$missing" ]; then
        echo "SETTINGS: rm needs the RM-basis adapters in ADAPTERS (DETAILS.md, \"RM bases\"), which are not" \
             "distributed; $(wc -w <<< "$missing") are missing, e.g. $ADAPTERS/${missing%% *}." \
             "SETTINGS=programmatic runs the programmatic bases alone." >&2
        exit 1
    fi
fi
cd "$OUT"

# names EXPR: the items of EXPR, a table of the poem package (e.g. programmatic.REWARD_FUNCTIONS), space-separated
names() { python -c "from poem import programmatic, reward_models, weights; print(*$1)"; }

# predict DIR TARGET_FLAG REWARD TARGET_ADAPTER [SAMPLE_FLAGS]: one target reward, end to end, into DIR:
#
#   fit       α̂ and the coverage (and γ_geom on the programmatic bases)     DIR/weights.json
#   sample    π_ref, π_tar and PoEM, decoded in one engine on EVAL_PROMPTS    DIR/<policy>.jsonl
#   evaluate  their rewards and recovery                                      DIR/eval.json
#   kl        their KL to π_tar                                               DIR/kl.json
#
# kl.json is written last: a DIR that holds it is kept (delete DIR to redo it), so a rerun resumes at the first
# unfinished target.
#
# TARGET_FLAG is --combined-reward or --heldout-reward. The basis is in variables the caller sets:
#
#   EXPERTS        (array) the expert adapters
#   CALIBRATION    the calibration set
#   PHI            the experts' log-ratios Φ on it: the first fit computes them, the next ones read them
#   FIT_FLAGS      (array) more fit flags
#   RM_FLAGS       (array) --rm-basis B on the RM bases, empty on the programmatic ones
#   EVAL_PROMPTS   the evaluation prompts
#   DECODE_FLAGS   (array) the decode flags
predict() {
    local dir=$1 flag=$2 reward=$3 target=$4 phi
    shift 4
    if [ -f "$dir/kl.json" ]; then echo "$dir: done, kept"; return 0; fi
    if [ -f "$PHI" ]; then phi=(--logratios "$PHI"); else phi=(--save-logratios "$PHI"); fi
    python -m poem fit --calibration "$CALIBRATION" "${phi[@]}" --experts "${EXPERTS[@]}" "${RM_FLAGS[@]}" \
        "$flag" "$reward" "${FIT_FLAGS[@]}" --out "$dir/weights.json"
    python -m poem sample --prompts "$EVAL_PROMPTS" --weights "$dir/weights.json" "${DECODE_FLAGS[@]}" --reference \
        --target-adapter "$target" --out-dir "$dir" "$@"
    python -m poem evaluate --dir "$dir" "${RM_FLAGS[@]}" "$flag" "$reward" --out "$dir/eval.json"
    python -m poem kl --dir "$dir" --out "$dir/kl.json"
}


# ============================================================================= steps 2-6: the programmatic bases

# train ADAPTER REWARD [OVERRIDES]: train_p_grpo.sh, unless ADAPTERS already holds ADAPTER
train() {
    [ -f "$1/adapter_config.json" ] || INIT=$ADAPTERS/p_grpo_init bash "$SCRIPTS/train_p_grpo.sh" "$@"
}

programmatic_bases() {
    local REWARDS COMBINED SECOND_RUN BASES b c r second_run
    local EXPERTS CALIBRATION PHI FIT_FLAGS RM_FLAGS=() EVAL_PROMPTS=eval_prompts.json DECODE_FLAGS
    REWARDS=($(names programmatic.REWARD_FUNCTIONS))        # the 20 rewards, one expert each
    COMBINED=($(names programmatic.COMBINED_REWARDS))       # the 32 combined rewards, one π_tar each
    DECODE_FLAGS=(--gamma geom $(names weights.GAMMA_GRID))

    # ---- 2. P-GRPO training (App. B.2): one LoRA adapter per reward and per combined reward, all from one shared
    #         initialization. The paper's initialization is not distributed: a new one is drawn when
    #         ADAPTERS/p_grpo_init is missing
    for r in "${REWARDS[@]}"; do train "$ADAPTERS/p_grpo/lora_$r" "reward_name=$r"; done
    for c in "${COMBINED[@]}"; do train "$ADAPTERS/p_grpo_targets/lora_$c" "combined_reward_name=$c"; done
    # the second RL run: the same recipe with the training prompts reordered, on 9 of the combined rewards
    SECOND_RUN=(clean_k10_conc_medium clean_k16_conc_medium clean_k16_r1 clean_k16_r2
                clean_k2_r0_code_content_concreteness clean_k4_conc_medium clean_k4_r1_code_content_formality_readabi
                clean_k8_r0_code_content_concreteness_conj clean_k8_r1_code_content_concreteness_conj)
    for c in "${SECOND_RUN[@]}"; do
        train "$ADAPTERS/p_grpo_second_run/lora_$c" "combined_reward_name=$c" data.seed=2
    done

    # ---- 3. Calibration sets (Table 4): π_ref's 4 responses to each pool prompt (a prompt of 511 or more tokens
    #         keeps none: 452 prompts remain); for the held-out rewards, 4 responses per remaining prompt from the
    #         experts, round-robin. A calibration set that exists is kept, so that it stays the one its saved Φ was
    #         computed on
    [ -f calibration.jsonl ] || python -m poem calibrate --prompts calibration_prompts.json --out calibration.jsonl
    BASES=()                                                # P-DPO only when ADAPTERS holds its experts
    for b in p_grpo p_dpo; do [ -d "$ADAPTERS/$b" ] && BASES+=("$b"); done
    for b in "${BASES[@]}"; do
        [ -f "$b/calibration.jsonl" ] || python -m poem calibrate --prompts calibration.jsonl \
            --experts "${REWARDS[@]/#/$ADAPTERS/$b/lora_}" --out "$b/calibration.jsonl"
    done

    # ---- 4. Combined rewards (Table 1, Sec. 4.1, Fig. 9): ridge fits on π_ref's responses; PoEM(α̂) and
    #         PoEM(α*) at γ_geom and on the γ grid, the top expert and the second RL run (greedy, 96 tokens)
    for b in "${BASES[@]}"; do
        EXPERTS=("${REWARDS[@]/#/$ADAPTERS/$b/lora_}") CALIBRATION=calibration.jsonl PHI=$b/phi_reference.npz
        FIT_FLAGS=(--exclude-prompts first_50_prompts.json)
        for c in "${COMBINED[@]}"; do
            second_run=()
            [ -d "$ADAPTERS/${b}_second_run/lora_$c" ] && second_run=(--adapter "$ADAPTERS/${b}_second_run/lora_$c")
            predict "$b/combined/$c" --combined-reward "$c" "$ADAPTERS/${b}_targets/lora_$c" \
                --arms alpha_hat alpha_star top_expert "${second_run[@]}"
        done
    done

    # ---- 5. Held-out rewards (Fig. 5, Table 7, App. C.3.2): each expert held out in turn, NNLS on the other
    #         experts' responses, γ_geom from the Gram matrix of π_ref's responses
    for b in "${BASES[@]}"; do
        EXPERTS=("${REWARDS[@]/#/$ADAPTERS/$b/lora_}") CALIBRATION=$b/calibration.jsonl PHI=$b/phi_experts.npz
        FIT_FLAGS=(--exclude-prompts first_50_prompts.json --gram-calibration "$b/phi_reference.npz")
        for r in "${REWARDS[@]}"; do predict "$b/heldout/$r" --heldout-reward "$r" "$ADAPTERS/$b/lora_$r"; done
    done

    # ---- 6. Effective ranks (Fig. 3, Fig. 12's P-DPO panel): R̃ and Φ̃ on π_ref's 2,400 responses to the 600
    #         prompts of App. B.4, sampled as in the RM bases' training, and ΔΘ of the experts. The RM bases draw
    #         the same set again and score it (step 11), so that this setting needs no reward model. The ranks are
    #         also saved, and step 12 prints them again
    [ -f geometry/reference.jsonl ] || python -m poem calibrate --sampler rollout --prompts calibration_600.json \
        --out geometry/reference.jsonl
    python "$SCRIPTS/ranks.py" programmatic "$ADAPTERS" "${BASES[@]}" | tee geometry/ranks_programmatic.txt
}


# ============================================================================= steps 7-11: the RM bases

# rollout_calibration PROMPTS EXPERT ...: the calibration set CALIBRATION, unless it exists: the experts' rollouts on
# PROMPTS, scored by the reward models of the basis in RM_FLAGS
rollout_calibration() {
    [ -f "$CALIBRATION" ] || python -m poem calibrate --sampler rollout "${RM_FLAGS[@]}" --prompts "$1" \
        --experts "${@:2}" --out "$CALIBRATION"
}

# predict_combined BASIS DIR GROUP: every combined reward of BASIS, into DIR/combined, its π_tar in
# ADAPTERS/GROUP_targets
predict_combined() {
    local combined c
    combined=($(names "reward_models.basis_combined_rewards('$1')"))      # an assignment, so that set -e applies
    for c in "${combined[@]}"; do
        predict "$2/combined/$c" --combined-reward "$c" "$ADAPTERS/${3}_targets/lora_$c" \
            --arms alpha_hat alpha_star top_expert
    done
}

# Each basis sets the variables predict reads; every decode is at γ = 1, 512 new tokens
rm_bases() {
    local RB HH DIV SKY SKY_EXPERTS others k r
    local EXPERTS CALIBRATION PHI FIT_FLAGS RM_FLAGS EVAL_PROMPTS DECODE_FLAGS=(--gamma 1 --max-new-tokens 512)

    # ---- 7. RM-RB (Table 1, Fig. 9, App. C.3.3): fits on the 150 training prompts of the calibration set
    RB=($(names "reward_models.ROSTERS['RM-RB']"))
    RM_FLAGS=(--rm-basis RM-RB) CALIBRATION=rb/calibration.jsonl PHI=rb/phi.npz
    EXPERTS=("${RB[@]/#/$ADAPTERS/rm_rb/lora_}") EVAL_PROMPTS=rb_eval.json
    FIT_FLAGS=(--exclude-prompts not_train_benign.json)
    rollout_calibration calibration_600.json "${EXPERTS[@]}"
    predict_combined RM-RB rb rm_rb
    for r in "${RB[@]}"; do predict "rb/heldout/$r" --heldout-reward "$r" "$ADAPTERS/rm_rb/lora_$r"; done

    # ---- 8. RM-HH (Table 1, Fig. 9): calibration responses on the 300 training prompts
    HH=($(names "reward_models.ROSTERS['RM-HH']"))
    RM_FLAGS=(--rm-basis RM-HH) CALIBRATION=hh/calibration.jsonl PHI=hh/phi.npz
    EXPERTS=("${HH[@]/#/$ADAPTERS/rm_hh/lora_}") EVAL_PROMPTS=hh_eval.json
    FIT_FLAGS=()
    rollout_calibration calibration_300.json "${EXPERTS[@]}"
    predict_combined RM-HH hh rm_hh

    # ---- 9. RM-Div (the recovery and relative KL of Fig. 5 and Sec. 4.3, App. C.3.3, the two recoveries of
    #         App. C.2.2, Fig. 4): each expert held out in turn: PoEM over the other nine, and each of the nine
    #         decoded alone
    DIV=($(names "reward_models.ROSTERS['RM-Div']"))
    RM_FLAGS=(--rm-basis RM-Div) CALIBRATION=div/calibration.jsonl PHI=div/phi.npz
    EXPERTS=("${DIV[@]/#/$ADAPTERS/rm_div/lora_}") EVAL_PROMPTS=hh_eval.json
    FIT_FLAGS=(--exclude-prompts heldout_splits.json)
    rollout_calibration calibration_600.json "${EXPERTS[@]}"
    for r in "${DIV[@]}"; do
        others=()
        for k in "${DIV[@]}"; do [ "$k" = "$r" ] || others+=("$ADAPTERS/rm_div/lora_$k"); done
        predict "div/heldout/$r" --heldout-reward "$r" "$ADAPTERS/rm_div/lora_$r" --adapter "${others[@]}"
    done

    # ---- 10. RM-Sky (Fig. 4, App. C.3.1): the experts of the five smaller Skywork reward models; π_tar is
    #          RM-Div's Skywork-Reward-V2-Llama-3.1-8B expert, held out, next to its second RL run
    SKY=($(names "reward_models.ROSTERS['RM-Sky'][1:]"))      # the roster without its first, the 8B reward model
    SKY_EXPERTS=("${SKY[@]/#/$ADAPTERS/rm_sky/lora_}")
    RM_FLAGS=(--rm-basis RM-Sky) CALIBRATION=sky/calibration.jsonl PHI=sky/phi.npz
    EXPERTS=("$ADAPTERS/rm_div/lora_skywork" "${SKY_EXPERTS[@]}") EVAL_PROMPTS=hh_eval.json
    FIT_FLAGS=(--exclude-prompts heldout_splits.json)
    rollout_calibration calibration_600.json "${SKY_EXPERTS[@]}"
    predict sky/heldout/skywork --heldout-reward skywork "$ADAPTERS/rm_div/lora_skywork" \
        --adapter "${SKY_EXPERTS[@]}" "$ADAPTERS/second_run/lora_skywork_second_run"

    # ---- 11. Effective ranks (App. C.2.1, set up as in App. B.4): π_ref's 2,400 responses to the 600 prompts,
    #          scored by the RM-Div and RM-RB reward models; also saved, and step 12 prints them again
    [ -f geometry/reference_rm.jsonl ] || python -m poem calibrate --sampler rollout --rm-basis RM-Div RM-RB \
        --prompts calibration_600.json --out geometry/reference_rm.jsonl
    python "$SCRIPTS/ranks.py" rm "$ADAPTERS" | tee geometry/ranks_rm.txt
}


python "$SCRIPTS/prompts.py"                                                     # 1
for setting in $SETTINGS; do "${setting}_bases"; done                            # 2-6, 7-11
python "$SCRIPTS/summarize.py" $SETTINGS                                         # 12
