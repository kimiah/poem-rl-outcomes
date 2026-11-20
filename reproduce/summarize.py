"""Step 12 of reproduce/run.sh: the numbers of the settings run, printed from the target directories run.sh writes
in OUT. Medians over targets; reward error |1 − rec|; closer: relative KL below 1; covered: coverage above 0.3.

    python summarize.py programmatic rm      (the settings run)
"""
import sys
from pathlib import Path

# poem before numpy: its BLAS kernel setting (poem/__init__.py)
from poem.utils import read_json
from poem.metrics import boot_ci
from poem.programmatic import COMBINED_REWARDS, REWARD_FUNCTIONS
from poem.reward_models import ROSTERS, basis_combined_rewards

import numpy as np
from scipy.stats import spearmanr


def target(d):
    """(weights JSON, {policy: (recovery, relative KL, per-prompt KL)}) of one target directory; where the fit stored
    γ_geom (the programmatic bases), the γ_geom arms also as alpha_hat_geom and alpha_star_geom."""
    w, e, k = (read_json(Path(d, f)) for f in ("weights.json", "eval.json", "kl.json"))
    t = {n: (e["policies"][n]["recovery"], v["relative_kl"], v["per_prompt"]) for n, v in k["policies"].items()}
    if w["gamma_geom"] is not None:
        t["alpha_hat_geom"] = t[f"alpha_hat_g{round(w['gamma_geom'], 6)}"]
        if "alpha_star" in w:
            t["alpha_star_geom"] = t[f"alpha_star_g{round(w['alpha_star']['gamma_geom'], 6)}"]
    return w, t


def top(w):
    """The plan name of the top expert of a held-out reward: the largest α̂, the first on ties."""
    return "adapter_" + Path(w["experts"][int(np.argmax(w["alpha_hat"]))]).name


def med(x):
    return float(np.median(x))


def ci(x):
    """The 95 % bootstrap interval of the median of x."""
    return "[%.2f, %.2f]" % boot_ci(x)


def err(ts, arm):
    """The median reward error |1 − recovery| of one arm over the targets ts."""
    return med([abs(1 - t[arm][0]) for t in ts])


def programmatic_combined(b):
    """Table 1, Fig. 9 and Sec. 4.1 on the programmatic basis b: its 32 combined rewards."""
    ts = [target(f"{b}/combined/{c}")[1] for c in COMBINED_REWARDS]
    second = [{"run": v} for t in ts for n, v in t.items() if n.startswith("adapter_")]   # the second RL runs
    rel_star = [t["alpha_star_geom"][1] for t in ts]
    print(f"Table 1, {b}: reward error PoEM(α̂) {err(ts, 'alpha_hat_geom'):.2f}, "
          f"PoEM(α*) {err(ts, 'alpha_star_geom'):.2f}, top expert {err(ts, 'top_expert'):.2f}, "
          f"second RL run {err(second, 'run'):.2f} ({len(second)} targets); PoEM(α*) relative KL {med(rel_star):.2f}, "
          f"closer {sum(r < 1 for r in rel_star)}/{len(rel_star)}")
    for arm, name in (("alpha_hat_geom", "PoEM(α̂)"), ("alpha_star_geom", "PoEM(α*)")):
        rec, rel = ([t[arm][i] for t in ts] for i in (0, 1))
        print(f"Fig. 9, {b}, {name}: recovery {med(rec):.2f} {ci(rec)}, "
              f"relative KL {med(rel):.2f} {ci(rel)} (medians, 95% bootstrap intervals over combined rewards)")
    print(f"Sec. 4.1, {b}: median recovery of PoEM(α*) at γ = 1 {med([t['alpha_star_g1.0'][0] for t in ts]):.2f}, "
          f"at γ_geom {med([t['alpha_star_geom'][0] for t in ts]):.2f}")


def programmatic_heldout(b):
    """Fig. 5 and Table 7 on the programmatic basis b: its 20 held-out rewards."""
    held = [target(f"{b}/heldout/{r}") for r in REWARD_FUNCTIONS]
    cov = np.array([w["coverage"] for w, _ in held])
    rec, rel = (np.array([t["alpha_hat_geom"][i] for _, t in held]) for i in (0, 1))
    print(f"Fig. 5, {b}: ρ(coverage, reward error) {spearmanr(cov, abs(1 - rec))[0]:.2f}, "
          f"ρ(coverage, relative KL) {spearmanr(cov, rel)[0]:.2f}")
    for name, m in (("covered", cov > 0.3), ("not covered", cov <= 0.3)):
        print(f"Table 7, {b}, {name}: recovery {med(rec[m]):.2f}, relative KL {med(rel[m]):.2f}, "
              f"closer {int((rel[m] < 1).sum())}/{int(m.sum())}")


def rm_combined(basis, d):
    """Table 1 and Fig. 9 on the RM basis, from its directory d: its combined rewards."""
    names = basis_combined_rewards(basis)
    ts = [target(f"{d}/combined/{c}")[1] for c in names]
    rel_star = [t["alpha_star_g1.0"][1] for t in ts]
    print(f"Table 1, {basis}: reward error PoEM(α̂) {err(ts, 'alpha_hat_g1.0'):.2f}, "
          f"PoEM(α*) {err(ts, 'alpha_star_g1.0'):.2f}, top expert {err(ts, 'top_expert'):.2f}; "
          f"PoEM(α*) relative KL {med(rel_star):.2f}, closer {sum(r < 1 for r in rel_star)}/{len(rel_star)}; "
          f"median recovery of PoEM(α*) {med([t['alpha_star_g1.0'][0] for t in ts]):.2f}")
    for c, t in zip(names, ts):
        print("  Fig. 9, %s: recovery and relative KL, PoEM(α̂) %.2f %.2f, PoEM(α*) %.2f %.2f"
              % (c, *t["alpha_hat_g1.0"][:2], *t["alpha_star_g1.0"][:2]))


def rm_rb_heldout():
    """App. C.3.3: RM-RB's experts held out in turn."""
    for r in ROSTERS["RM-RB"]:
        w, t = target(f"rb/heldout/{r}")
        print(f"RM-RB held out, {r}: coverage {w['coverage']:.3f}, recovery {t['alpha_hat_g1.0'][0]:.2f}")


def rm_div():
    """Fig. 5 and Sec. 4.3, App. C.3.3, App. C.2.2 and Fig. 4: RM-Div's experts held out
    in turn, PoEM against the top expert. The coverage printed is the fit's, on the experts' rollouts, not the
    within-expert coverage the paper reports for RM-Div (all below 0.3)."""
    div = {r: target(f"div/heldout/{r}") for r in ROSTERS["RM-Div"]}
    for r, (w, t) in div.items():
        print(f"RM-Div held out, {r}: coverage {w['coverage']:.3f}, recovery {t['alpha_hat_g1.0'][0]:.3f}, relative KL "
              f"{t['alpha_hat_g1.0'][1]:.3f}; top expert {top(w)[len('adapter_lora_'):]}: recovery {t[top(w)][0]:.3f}, "
              f"relative KL {t[top(w)][1]:.3f}")
    rec_poem = [t["alpha_hat_g1.0"][0] for w, t in div.values()]
    rec_top = [t[top(w)][0] for w, t in div.values()]
    closer = sum(t["alpha_hat_g1.0"][1] < t[top(w)][1] for w, t in div.values())
    more = sum(p > b for p, b in zip(rec_poem, rec_top))
    print(f"RM-Div, PoEM against the top expert: closer {closer}/{len(div)}, more reward {more}/{len(div)}, "
          f"median recovery {med(rec_poem):.2f} against {med(rec_top):.2f}; ρ(coverage, recovery) "
          f"{spearmanr([w['coverage'] for w, _ in div.values()], rec_poem)[0]:.2f}")
    w, t = div["skywork"]
    singles = [t[n][1] for n in t if n.startswith("adapter_")]
    print(f"Fig. 4, RM-Div without Skywork: recovery {t['alpha_hat_g1.0'][0]:.3f}, relative KL "
          f"{t['alpha_hat_g1.0'][1]:.3f}; single experts' relative KL {min(singles):.2f} to {max(singles):.2f}")


def rm_sky():
    """Fig. 4 and App. C.3.1: RM-Sky, against the single experts and the second RL run."""
    w, t = target("sky/heldout/skywork")
    p, s = t["alpha_hat_g1.0"], t.pop("adapter_lora_skywork_second_run")
    singles = ", ".join(f"{n[len('adapter_lora_'):]} {v[1]:.3f}" for n, v in t.items() if n.startswith("adapter_"))
    print(f"Fig. 4, RM-Sky: α̂ {dict(zip((Path(e).name for e in w['poem_experts']), w['poem_weights']))}; "
          f"recovery {p[0]:.3f}, relative KL {p[1]:.3f}; single experts' relative KL {singles}; "
          f"second RL run: recovery {s[0]:.3f}, relative KL {s[1]:.3f}, PoEM closer on "
          f"{sum(a < b for a, b in zip(p[2], s[2]) if a is not None and b is not None)}/{len(p[2])} prompts")


def main():
    settings = sys.argv[1:]
    if "programmatic" in settings:
        for b in ("p_grpo", "p_dpo"):
            if Path(b).is_dir():
                programmatic_combined(b)
                programmatic_heldout(b)
    if "rm" in settings:
        rm_combined("RM-RB", "rb")
        rm_combined("RM-HH", "hh")
        rm_rb_heldout()
        rm_div()
        rm_sky()
    for s in settings:                                  # the effective ranks steps 6 and 11 saved
        if Path(f"geometry/ranks_{s}.txt").is_file():
            print(f"Effective ranks, {s}:\n" + Path(f"geometry/ranks_{s}.txt").read_text(), end="")


if __name__ == "__main__":
    main()
