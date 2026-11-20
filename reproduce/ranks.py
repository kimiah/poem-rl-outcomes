"""The effective ranks of App. B.4 (steps 6 and 11 of reproduce/run.sh), printed. run.sh runs it in OUT:

    python ranks.py programmatic ADAPTERS BASIS ...   Fig. 3 and Fig. 12's P-DPO panel
    python ranks.py rm ADAPTERS                       App. C.2.1
"""
import sys

# poem before numpy: its BLAS kernel setting (poem/__init__.py)
from poem.utils import load_logratios, read_json, read_jsonl
from poem.policies import lora_update_gram, row_log_ratios
from poem.programmatic import REWARD_FUNCTIONS
from poem.reward_models import ROSTERS, zscores
from poem.weights import calibration_mask, effective_rank, gram_effective_rank

import numpy as np


def programmatic(A, bases):
    """R̃ on π_ref's responses to the 600 prompts, then Φ̃ on the same responses and ΔΘ of each basis's experts."""
    rows = read_jsonl("geometry/reference.jsonl")
    pids = np.array([r["prompt_id"] for r in rows])
    R = np.array([[f(r["response"]) for f in REWARD_FUNCTIONS.values()] for r in rows])
    print(f"R̃ {effective_rank(R, pids):.2f}")
    for b in bases:
        experts = [f"{A}/{b}/lora_{r}" for r in REWARD_FUNCTIONS]
        Phi = row_log_ratios(experts, rows)
        print(f"{b}: Φ̃ {effective_rank(Phi, pids):.2f}, ΔΘ {gram_effective_rank(lora_update_gram(experts)):.2f}")


def ranks(rows, Phi, basis, keep, groups):
    """Effective ranks of Φ̃ and R̃ (z-scored rewards), on the rows `keep` with no reward beyond 20 standard
    deviations, centered within `groups`."""
    z = zscores(rows, basis)
    m = calibration_mask(z, [r.get("expert", "") for r in rows]) & keep
    return effective_rank(Phi[m], groups[m]), effective_rank(z[m], groups[m])


def rm(A):
    """RM-Div and RM-RB: Φ̃ and R̃ on π_ref's responses to the 600 prompts and on the experts' own responses, then
    RM-Div's ΔΘ."""
    reference, train_benign = read_jsonl("geometry/reference_rm.jsonl"), set(read_json("calibration_300.json")[:150])
    for basis, d, group in (("RM-Div", "div", "rm_div"), ("RM-RB", "rb", "rm_rb")):
        experts = [f"{A}/{group}/lora_{r}" for r in ROSTERS[basis]]
        Phi = row_log_ratios(experts, reference)
        every, pids = np.ones(len(reference), dtype=bool), np.array([r["prompt_id"] for r in reference])
        print("%s, π_ref's responses: Φ̃ %.2f, R̃ %.2f" % (basis, *ranks(reference, Phi, basis, every, pids)))
        rows = read_jsonl(f"{d}/calibration.jsonl")     # the experts' own responses (RM-RB: the 150 training prompts)
        keep = np.array([basis == "RM-Div" or r["prompt"] in train_benign for r in rows])
        groups = np.array([f"{r['prompt_id']}|{r['expert']}" for r in rows])
        Phi = load_logratios(f"{d}/phi.npz", list(ROSTERS[basis]))[0]
        print("%s, the experts' responses: Φ̃ %.2f, R̃ %.2f" % (basis, *ranks(rows, Phi, basis, keep, groups)))
    experts = [f"{A}/rm_div/lora_{r}" for r in ROSTERS["RM-Div"]]
    print("RM-Div, ΔΘ: %.2f" % gram_effective_rank(lora_update_gram(experts)))


def main():
    setting, adapters, *bases = sys.argv[1:]
    if setting == "programmatic":
        programmatic(adapters, bases)
    elif setting == "rm":
        rm(adapters)
    else:
        sys.exit(f"ranks.py: unknown setting {setting!r} (programmatic, rm)")


if __name__ == "__main__":
    main()
