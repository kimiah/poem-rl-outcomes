"""The PoEM command line: the steps of the framework (Sec. 2) on files.

    python -m poem calibrate --prompts P --out CAL.jsonl [--experts DIR ...]
    python -m poem fit --calibration CAL.jsonl --experts DIR ... --out W.json
                       [--reward FILE | --combined-reward C | --heldout-reward R]
    python -m poem sample --prompts P --out-dir D --weights W.json [--arms alpha_hat alpha_star top_expert]
                          [--gamma geom 1 ...] [--reference] [--target-adapter DIR] [--adapter DIR ...]
    python -m poem evaluate --dir D --out R.json [--reward FILE | --combined-reward C | --heldout-reward R] [--drift]
    python -m poem kl --dir D --out K.json

calibrate   step 1: rows {"prompt_id", "prompt", "response"[, "expert"]} sampled from π_ref, or from the experts
            (--sampler rollout: the RM bases' sampler, rows also hold "truncated"; --rm-basis: and "scores")
fit         steps 2-3: the experts' policy log-ratios Φ on the calibration rows (one GPU, or --logratios), the
            target reward of every row, α̂, the coverage and γ_geom -> the weights JSON
sample      step 4: π_ref, π_tar, PoEM and other policies decoded in one vLLM engine -> D/<policy>.jsonl, D/plan.json
evaluate    reward and recovery of every policy of D
kl          KL(π_tar ‖ π) and the relative KL of every policy of D

A prompts file is a .json list, a .jsonl with a "prompt" per line ("prompt_id" or "prompt_idx" names the prompt),
or text with one prompt per line. The target reward of fit and evaluate is --reward, your Python file that defines
reward(prompt, response) -> float; or --combined-reward or --heldout-reward, a target of the programmatic setting
(poem.programmatic), or with --rm-basis B of the RM basis B (poem.reward_models); without any of them each
calibration or response row carries its target reward as "reward".
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from . import REFERENCE_MODEL, programmatic, reward_models
from .calibration import expert_calibration, reference_calibration, rollout_calibration
from .metrics import kl_to_target, mean_kl, mean_logprob, recovery, relative_kl
from .policies import load_policy, load_tokenizer, policy_log_ratios, row_log_ratios
from .programmatic import combined_reward, heldout_reward
from .reward_models import EVAL_BATCH, ROSTERS, calibration_order, decode_order, reward_scorer, score_rows
from .sampling import sample
from .utils import (expert_name, load_logratios, load_reward, prompt_id, read_json, read_jsonl, read_prompts,
                    revision_for, save_logratios, write_json, write_jsonl)
from .weights import calibration_mask, centered_gram, composition_weights, coverage, gamma_geom, poem_members


# -------------------------------------------------------------------------------------------------------------- helpers

def reference_model(a, recorded):
    """(π_ref, its revision): --reference-model and --revision, else those `recorded` names, else REFERENCE_MODEL at
    its pinned revision. A weights JSON names π_ref's model only; plan.json also names its revision."""
    return a.reference_model or recorded.get("reference_model", REFERENCE_MODEL), a.revision or recorded.get("revision")


def setting(a):
    """The setting of the target rewards: the RM bases with --rm-basis, else the programmatic one."""
    return reward_models if a.rm_basis else programmatic


# ------------------------------------------------------------------------------------------------------------ calibrate

def cmd_calibrate(a):
    """Step 1: n responses per prompt from π_ref, or with --experts from the experts sorted by name (response j of
    prompt i from expert (i·n + j) mod K). --sampler rollout: n responses per prompt from π_ref, or from each expert
    in the given order. Each sampler has its own defaults for the flags left out. --rm-basis then scores every row
    with the reward models of those bases (reward_models.score_rows, one source at a time, longest response first)."""
    model, revision = reference_model(a, {})
    prompts = read_prompts(a.prompts)
    given = {k: v for k, v in (("max_new_tokens", a.max_new_tokens), ("temperature", a.temperature)) if v is not None}
    kw = dict(n=a.n, seed=a.seed, model=model, revision=revision, device=a.device, **given)
    batch = {} if a.batch_size is None else {"batch_size": a.batch_size}
    if a.sampler == "rollout":
        rows = rollout_calibration(a.experts or [], prompts, **batch, **kw)
    elif a.experts:
        rows = expert_calibration(sorted(a.experts, key=expert_name), prompts, **batch, **kw)
    else:
        rows = reference_calibration(prompts, **kw)
    if a.rm_basis:
        rewards = list(dict.fromkeys(r for b in a.rm_basis for r in ROSTERS[b]))
        score_rows(rewards, rows, calibration_order(rows), device=a.device)
    write_jsonl(a.out, rows)
    print(f"{len(rows)} calibration rows on {len(prompts)} prompts -> {a.out}")


# ------------------------------------------------------------------------------------------------------------------ fit

def logratios_for(a, path, names):
    """(Φ (N, K), prompt ids, row index) of a calibration set: loaded from a log-ratio npz, whose row n is line
    pair_idx[n] of its calibration JSONL, or computed on every row of a calibration JSONL (one GPU), in the RM bases'
    convention with --rm-basis (policies.row_log_ratios)."""
    if str(path).endswith(".npz"):
        Phi, _, idx, pids = load_logratios(path, names)
        return Phi, pids, idx
    rows = read_jsonl(path)
    model, revision = reference_model(a, {})
    if a.rm_basis:
        Phi = row_log_ratios(a.experts, rows, model=model, revision=revision, device=a.device)
    else:
        Phi = policy_log_ratios(a.experts, [r["prompt"] for r in rows], [r["response"] for r in rows], model=model,
                                revision=revision, device=a.device)
    return Phi, np.array([prompt_id(r) for r in rows]), np.arange(len(rows))


def fit_rows(a, rows, names):
    """(mask, target, cols, estimator) of the fit: the rows it uses, the target reward of every row, the expert
    columns it regresses on and the estimator. --heldout-reward R is the held-out protocol: R's own column and the
    rows R's expert wrote are left out, and NNLS fits the rest. With --rm-basis B the rewards are the rows' scores
    z-scored with B's z-stats (reward_models.zscores, fit_target) and NNLS fits every target. Under both, rows with a
    basis reward beyond 20 standard deviations are left out (weights.calibration_mask). Otherwise the target is the
    --combined-reward scorer's, --reward's or each row's "reward", fitted by ridge. --exclude-prompts drops the rows
    of those prompts."""
    cols, sources = list(range(len(names))), [r.get("expert", "") for r in rows]
    if a.rm_basis:
        z = reward_models.zscores(rows, a.rm_basis)
        target = reward_models.fit_target(z, a.rm_basis, a.combined_reward, a.heldout_reward)
        mask, estimator = calibration_mask(z, sources, a.heldout_reward), "nnls"
    elif a.heldout_reward:
        z = programmatic.zscores([r["response"] for r in rows], names)
        target = z[:, names.index(a.heldout_reward)]
        mask, estimator = calibration_mask(z, sources, a.heldout_reward), "nnls"
    else:
        if a.combined_reward:
            score = combined_reward(a.combined_reward)
            target = np.array([score(r["response"]) for r in rows])
        elif a.reward:
            score = load_reward(a.reward)
            target = np.array([float(score(r["prompt"], r["response"])) for r in rows])
        else:
            target = np.array([float(r["reward"]) for r in rows])
        mask, estimator = np.ones(len(rows), dtype=bool), "ridge"
    if a.heldout_reward:
        cols.remove(names.index(a.heldout_reward))
    if a.exclude_prompts:
        excluded = {p for _, p in read_prompts(a.exclude_prompts)}
        mask &= np.array([r["prompt"] not in excluded for r in rows])
    return mask, target, cols, a.estimator or estimator


def cmd_fit(a):
    """Steps 2-3. The weights JSON holds reference_model, target, estimator, experts (the columns fitted), alpha_hat,
    then alpha_hat_raw (before Σ_k |α̂_k| = 1), lambda, lambda_neg, r2_cv, r2_heldout for ridge or topk for NNLS, then
    coverage, poem_experts / poem_weights (the experts PoEM decodes with, and their α̂), gamma_geom of those weights
    (6 dp) and, for a combined reward of the experts' own rewards, alpha_star {experts, weights, gamma_geom of |α*|
    (4 dp)}. The Gram matrix of γ_geom comes from π_ref's responses: this calibration set, or --gram-calibration.
    With --rm-basis the JSON also holds rm_basis, NNLS drops zero weights from the support, and both gamma_geom are
    null: the RM bases decode at γ = 1."""
    rows = read_jsonl(a.calibration)
    if not (a.rm_basis or a.gram_calibration) and any("expert" in r for r in rows):
        sys.exit("fit: the calibration responses come from the experts, and γ_geom needs the Gram matrix of π_ref's "
                 "responses: give --gram-calibration")
    names = [expert_name(e) for e in a.experts]
    Phi, pids, idx = logratios_for(a, a.logratios or a.calibration, names)
    if a.save_logratios:
        save_logratios(a.save_logratios, Phi, names, idx, pids)
    mask, target, cols, estimator = fit_rows(a, [rows[int(i)] for i in idx], names)
    X, t, fit_pids = Phi[mask][:, cols], target[mask], pids[mask]
    if not any(np.ptp(t[fit_pids == p]) > 0 for p in np.unique(fit_pids)):
        sys.exit("fit: the target reward varies within no prompt of the fit's rows (one response per prompt, or "
                 "identical responses), and the fit centers within each prompt")
    out = {"reference_model": reference_model(a, {})[0], "target": a.combined_reward or a.heldout_reward or "reward",
           "estimator": estimator, "experts": [a.experts[k] for k in cols]}
    if a.rm_basis:
        out["rm_basis"] = a.rm_basis
    alpha, keep, w, report = composition_weights(X, t, fit_pids, estimator, topk=a.topk, drop_zeros=bool(a.rm_basis))
    out["alpha_hat"] = [float(x) for x in alpha]
    out.update(report)
    out["coverage"] = coverage(X, t, fit_pids)
    if a.rm_basis:
        G = None
    elif a.gram_calibration:
        Phi_ref, pids_ref, _ = logratios_for(a, a.gram_calibration, names)
        G = centered_gram(Phi_ref, pids_ref)
    else:
        G = centered_gram(Phi, pids)
    decode, w = [cols[k] for k in keep], np.asarray(w, dtype=np.float64)
    out.update(poem_experts=[a.experts[k] for k in decode], poem_weights=[float(x) for x in w],
               gamma_geom=None if G is None else round(gamma_geom(G, w, decode), 6))
    rewards, alpha_star = setting(a).COMBINED_REWARDS.get(a.combined_reward, ((), ()))
    if rewards and set(rewards) <= set(names):
        j = [names.index(r) for r in rewards]
        out["alpha_star"] = {"experts": [a.experts[k] for k in j], "weights": [float(x) for x in alpha_star],
                             "gamma_geom": None if G is None else
                             round(gamma_geom(G, np.abs(np.asarray(alpha_star, dtype=float)), j), 4)}
    write_json(a.out, out)
    print(f"{out['target']}: {estimator} on {int(mask.sum())} rows, coverage {out['coverage']:.3f}; PoEM decodes "
          f"{len(decode)} of {len(cols)} experts at γ_geom {out['gamma_geom']} -> {a.out}")


# --------------------------------------------------------------------------------------------------------------- sample

def plan_entry(experts, weights, gamma, kind):
    """A plan.json entry. Every policy decodes as weights.poem_members(experts, weights, gamma): PoEM (kind "poem"),
    π_ref with no experts ("reference"), or one adapter at weight 1, its own policy ("target", "adapter")."""
    return {"experts": [str(e) for e in experts], "weights": [float(w) for w in weights], "gamma": gamma, "kind": kind}


def plan_policies(a):
    """({name: plan.json entry} of the policies `sample` decodes, in decode order; the weights JSON or {}).
    alpha_hat is PoEM over the weights JSON's poem_experts / poem_weights and alpha_star over its alpha_star entry,
    each at every --gamma ("geom" is the stored γ_geom; γ is rounded to 6 dp). top_expert is the expert with the
    largest α* weight (else the largest α̂; the first on ties) alone at γ = 1. --experts / --alpha is PoEM over the
    given weights ("geom" is 1)."""
    w = read_json(a.weights) if a.weights else {}
    policies = {}
    if a.reference:
        policies["reference"] = plan_entry([], [], 1.0, "reference")
    if a.target_adapter:
        policies["target"] = plan_entry([a.target_adapter], [1.0], 1.0, "target")
    for d in a.adapter:
        policies[f"adapter_{Path(d).name}"] = plan_entry([d], [1.0], 1.0, "adapter")
    if a.weights:
        arms = {"alpha_hat": (w["poem_experts"], w["poem_weights"], w["gamma_geom"])}
        if "alpha_star" in w:
            arms["alpha_star"] = tuple(w["alpha_star"][k] for k in ("experts", "weights", "gamma_geom"))
    else:
        arms = {"poem": (a.experts, a.alpha, 1.0)} if a.experts else {}
    for arm in (a.arms if a.weights else arms):
        if arm == "top_expert":
            experts, weights, _ = arms.get("alpha_star", arms["alpha_hat"])
            policies[arm] = plan_entry([experts[int(np.argmax(weights))]], [1.0], 1.0, "poem")
            continue
        if arm not in arms:
            print(f"sample: the weights JSON has no {arm}; skipped", file=sys.stderr)
            continue
        experts, weights, geom = arms[arm]
        if "geom" in a.gamma and geom is None:
            sys.exit("sample: the weights JSON has no γ_geom; give --gamma values")
        for g in a.gamma:
            g = round(float(geom if g == "geom" else g), 6)
            policies[f"{arm}_g{g}"] = plan_entry(experts, weights, g, "poem")
    return policies, w


def cmd_sample(a):
    """Step 4: every policy of plan_policies on every prompt, in one vLLM engine, so that π_ref, π_tar and PoEM are
    decoded together. Writes D/<policy>.jsonl, one {"prompt", "response"} line per prompt, and D/plan.json."""
    policies, w = plan_policies(a)
    model, revision = reference_model(a, w)
    prompts = [p for _, p in read_prompts(a.prompts)]
    print(f"{len(policies)} policies x {len(prompts)} prompts: {', '.join(policies)}")
    responses = sample({name: poem_members(p["experts"], p["weights"], p["gamma"]) for name, p in policies.items()},
                       prompts, model=model, revision=revision, max_new_tokens=a.max_new_tokens,
                       temperature=a.temperature, seed=a.seed, gpu_mem=a.gpu_memory_utilization,
                       repeatable=not a.vllm_default_kernels)
    for name, texts in responses.items():
        write_jsonl(Path(a.out_dir) / f"{name}.jsonl", [{"prompt": p, "response": r} for p, r in zip(prompts, texts)])
    write_json(Path(a.out_dir) / "plan.json", {"reference_model": model, "revision": revision_for(model, revision),
                                               "target_adapter": a.target_adapter, "policies": policies})


# ------------------------------------------------------------------------------------------------------ evaluate and kl

def rewards_of(a, files):
    """{policy: per-prompt rewards}: by your --reward, the --combined-reward scorer (programmatic: α* in float32, as
    the paper scored decodes) or the --heldout-reward scorer, else each row's "reward". With --rm-basis B the target's
    reward models first score the rows of every file together (reward_models.score_rows, shortest first, at B's
    EVAL_BATCH)."""
    if a.reward:
        score = load_reward(a.reward)
        return {name: [float(score(r["prompt"], r["response"])) for r in rows] for name, rows in files.items()}
    if not (a.combined_reward or a.heldout_reward):
        return {name: [float(r["reward"]) for r in rows] for name, rows in files.items()}
    if a.rm_basis:
        rows = [r for rs in files.values() for r in rs]
        rewards = reward_models.COMBINED_REWARDS[a.combined_reward][0] if a.combined_reward else [a.heldout_reward]
        score_rows(rewards, rows, decode_order(rows), batch_size=EVAL_BATCH.get(a.rm_basis), device=a.device)
        score = reward_scorer(a.rm_basis, a.combined_reward, a.heldout_reward)
        return {name: [score(r) for r in rows] for name, rows in files.items()}
    score = combined_reward(a.combined_reward, np.float32) if a.combined_reward else heldout_reward(a.heldout_reward)
    return {name: [score(r["response"]) for r in rows] for name, rows in files.items()}


def cmd_evaluate(a):
    """Per policy file of D: the reward (mean over prompts), the recovery against D/reference.jsonl and
    D/target.jsonl (with --rm-basis on the per-prompt differences, as the RM bases' numbers; null without
    target.jsonl), and the per-prompt rewards. --drift adds logp_ref, the mean per-token log-likelihood of the
    policy's responses under π_ref, and drift = logp_ref(reference) − logp_ref(policy), D(π) of Eq. 21."""
    files = {f.stem: read_jsonl(f) for f in sorted(Path(a.dir).glob("*.jsonl"))}
    if "reference" not in files:
        sys.exit(f"evaluate: {a.dir} has no reference.jsonl: sample with --reference")
    prompts = [r["prompt"] for r in files["reference"]]
    assert all([r["prompt"] for r in rows] == prompts for rows in files.values()), "the files differ in prompts"
    rewards = rewards_of(a, files)
    report = {name: {"reward": float(np.mean(r)),
                     "recovery": recovery(r, rewards["reference"], rewards["target"], paired=bool(a.rm_basis))
                     if "target" in rewards else None,
                     "per_prompt": r} for name, r in rewards.items()}
    if a.drift:
        model, revision = reference_model(a, read_json(Path(a.dir) / "plan.json"))
        tok, ref = load_tokenizer(model, revision), load_policy(model=model, revision=revision, device=a.device)
        for name, rows in files.items():
            report[name]["logp_ref"] = mean_logprob(ref, tok, prompts, [r["response"] for r in rows], a.device)
        for v in report.values():
            v["drift"] = report["reference"]["logp_ref"] - v["logp_ref"]
    write_json(a.out, {"target": a.combined_reward or a.heldout_reward or "reward", "policies": report})


def cmd_kl(a):
    """KL(π_tar ‖ π) per prompt along π_tar's responses (D/target.jsonl), its mean over prompts and the relative KL,
    for π_ref and every other policy of D/plan.json. A policy of kind "poem" is scored with the logits of
    Eq. 6 at the weights γα_k (metrics.poem_logits), top_expert too, as a one-expert PoEM, as the paper
    scored it; only kind "adapter" is scored with its own logits. per_prompt holds None where π_tar's response is
    empty."""
    plan, target = read_json(Path(a.dir) / "plan.json"), read_jsonl(Path(a.dir) / "target.jsonl")
    policies = {"reference": plan_entry([], [], 1.0, "reference"), **plan["policies"]}
    scored = {name: (p["experts"], None if p["kind"] == "adapter" else [p["gamma"] * w for w in p["weights"]])
              for name, p in policies.items() if p["kind"] != "target"}
    model, revision = reference_model(a, plan)
    kl = kl_to_target(plan["target_adapter"], scored, [r["prompt"] for r in target], [r["response"] for r in target],
                      model=model, revision=revision, device=a.device)
    write_json(a.out, {"target_adapter": plan["target_adapter"], "policies": {
        name: {"gamma": policies[name]["gamma"], "kl": mean_kl(v), "relative_kl": relative_kl(v, kl["reference"]),
               "per_prompt": v} for name, v in kl.items()}})


# ----------------------------------------------------------------------------------------------------------------- main

def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--reference-model", help=f"π_ref (default {REFERENCE_MODEL}; sample, evaluate --drift and kl "
                                                  "default to the one the weights JSON or plan.json names)")
    common.add_argument("--revision", help="π_ref's revision (default: the paper's, for the default π_ref)")
    device = argparse.ArgumentParser(add_help=False)
    device.add_argument("--device", default="cuda:0",
                        help="the GPU of calibrate, fit, evaluate (--drift, --rm-basis) and kl")
    target = argparse.ArgumentParser(add_help=False)
    g = target.add_mutually_exclusive_group()
    g.add_argument("--reward", metavar="FILE",
                   help="target reward: your own, a Python file that defines reward(prompt, response) -> float")
    g.add_argument("--combined-reward", metavar="NAME",
                   help="target reward: a combined reward of the programmatic setting, or of the --rm-basis basis")
    g.add_argument("--heldout-reward", metavar="NAME",
                   help="target reward: a held-out reward of the programmatic setting, or of the --rm-basis basis")
    target.add_argument("--rm-basis", choices=ROSTERS,
                        help="the targets are this RM basis's (poem.reward_models): fit reads the rows' scores "
                             "(calibrate --rm-basis); evaluate scores the responses with its reward models (one GPU)")

    ap = argparse.ArgumentParser(prog="python -m poem", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("calibrate", parents=[common, device], help="step 1: sample a calibration set")
    p.add_argument("--prompts", required=True, help="the calibration prompts")
    p.add_argument("--out", required=True, help="the calibration JSONL to write")
    p.add_argument("--experts", nargs="+", help="sample from these expert adapters instead of π_ref")
    p.add_argument("--sampler", choices=("rollout",),
                   help="rollout: the RM bases' PPO sampler (T 0.7, top-p 0.8, top-k 20), --n responses per prompt "
                        "from π_ref or from each of --experts, in the given order")
    p.add_argument("--n", type=int, default=4, help="responses per prompt (default %(default)s)")
    p.add_argument("--max-new-tokens", type=int, help="default 128 (rollout: 512)")
    p.add_argument("--temperature", type=float, help="default 1.0; 0 is greedy (--experts only)")
    p.add_argument("--seed", type=int, default=0, help="default %(default)s")
    p.add_argument("--batch-size", type=int, help="with --experts (default 16) or rollout (default 32)")
    p.add_argument("--rm-basis", nargs="+", choices=ROSTERS,
                   help="with --sampler rollout: also score every row with the reward models of these RM bases "
                        "(one GPU)")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("fit", parents=[common, device, target], help="steps 2-3: α̂, coverage and γ_geom")
    p.add_argument("--calibration", required=True, help="the calibration JSONL")
    p.add_argument("--experts", nargs="+", required=True, help="the expert adapters, in the columns' order")
    p.add_argument("--out", required=True, help="the weights JSON to write")
    p.add_argument("--exclude-prompts", help="leave out the calibration rows of these prompts")
    p.add_argument("--gram-calibration", help="π_ref's calibration JSONL, or its log-ratio npz: γ_geom's Gram matrix")
    p.add_argument("--logratios", help="load Φ from this npz instead of computing it")
    p.add_argument("--save-logratios", help="also save Φ to this npz")
    p.add_argument("--estimator", choices=("ridge", "nnls"),
                   help="default: nnls with --heldout-reward or --rm-basis, else ridge with a penalty on negative "
                        "weights")
    p.add_argument("--topk", type=int, default=8, help="nnls: the largest weights kept (default %(default)s)")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("sample", parents=[common], help="step 4: decode PoEM, π_ref and π_tar")
    p.add_argument("--prompts", required=True, help="the evaluation prompts")
    p.add_argument("--out-dir", required=True, help="the directory D of the responses and plan.json")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--weights", help="the weights JSON of fit")
    src.add_argument("--experts", nargs="+", help="PoEM over these expert adapters, with --alpha")
    p.add_argument("--alpha", nargs="+", type=float, help="the weights of --experts")
    p.add_argument("--arms", nargs="+", default=["alpha_hat"], choices=("alpha_hat", "alpha_star", "top_expert"),
                   help="with --weights (default %(default)s)")
    p.add_argument("--gamma", nargs="+", default=["geom"], type=lambda g: g if g == "geom" else float(g),
                   help="composition strengths γ: numbers, or geom for the stored γ_geom (1 with --experts) "
                        "(default %(default)s)")
    p.add_argument("--reference", action="store_true", help="also decode π_ref")
    p.add_argument("--target-adapter", help="also decode π_tar, this adapter")
    p.add_argument("--adapter", nargs="+", default=[], help="also decode these adapters, e.g. a second RL run")
    p.add_argument("--max-new-tokens", type=int, default=96, help="default %(default)s")
    p.add_argument("--temperature", type=float, default=0.0, help="0 is greedy (default)")
    p.add_argument("--seed", type=int, default=0, help="prompt i of every policy samples with seed + i")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6,
                   help="the share of GPU memory vLLM takes (default %(default)s); raise it when groups lose lockstep")
    p.add_argument("--vllm-default-kernels", action="store_true",
                   help="vLLM's default LoRA kernels, whose repeated decodes may differ, instead of split-K 1")
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("evaluate", parents=[common, device, target], help="reward and recovery of the policies of D")
    p.add_argument("--dir", required=True, help="a sample output directory D")
    p.add_argument("--out", required=True, help="the report JSON to write")
    p.add_argument("--drift", action="store_true", help="also the drift D(π) under π_ref (one GPU)")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("kl", parents=[common, device], help="KL(π_tar ‖ π) of the policies of D (one GPU)")
    p.add_argument("--dir", required=True, help="a sample output directory D with target.jsonl")
    p.add_argument("--out", required=True, help="the report JSON to write")
    p.set_defaults(func=cmd_kl)
    return ap


def check_target(ap, a):
    """The target of fit or evaluate must be one of the setting's: with --rm-basis B a combined reward of B's rewards
    or one of B's rewards, else a programmatic one. --rm-basis needs a target."""
    rewards = ROSTERS[a.rm_basis] if a.rm_basis else programmatic.REWARD_FUNCTIONS
    combined = reward_models.basis_combined_rewards(a.rm_basis) if a.rm_basis else list(programmatic.COMBINED_REWARDS)
    if a.combined_reward not in (None, *combined) or a.heldout_reward not in (None, *rewards):
        valid = combined if a.combined_reward else list(rewards)
        ap.error(f"{a.command}: {a.combined_reward or a.heldout_reward} is not a target of the "
                 f"{a.rm_basis or 'programmatic'} setting; one of: {', '.join(valid)}")
    if a.reward and a.rm_basis:
        ap.error(f"{a.command}: --rm-basis is for the RM bases' targets, not --reward")
    if a.rm_basis and not (a.combined_reward or a.heldout_reward):
        ap.error(f"{a.command} --rm-basis: give --combined-reward or --heldout-reward")


def check_args(ap, a):
    """The flag combinations argparse does not check, per command."""
    if a.command == "calibrate":
        if a.sampler == "rollout" and a.temperature is not None:
            ap.error("calibrate: the rollout sampler has its own temperature")
        if a.temperature == 0 and not (a.experts or a.sampler):
            ap.error("calibrate: π_ref's greedy responses to a prompt are identical, and fit centers within each "
                     "prompt: give a temperature above 0")
        if a.rm_basis and a.sampler != "rollout":
            ap.error("calibrate: --rm-basis needs --sampler rollout, the RM bases' sampler")
    elif a.command == "fit":
        if a.save_logratios and not a.save_logratios.endswith(".npz"):
            ap.error("fit: --save-logratios names an .npz file")
        if a.rm_basis and a.gram_calibration:
            ap.error("fit: --gram-calibration is for γ_geom, which --rm-basis does not compute")
        if a.heldout_reward and a.heldout_reward not in [expert_name(e) for e in a.experts]:
            ap.error(f"fit --heldout-reward {a.heldout_reward}: its expert (lora_{a.heldout_reward}) must be among "
                     "--experts")
        check_target(ap, a)
    elif a.command == "sample":
        if a.experts and len(a.alpha or []) != len(a.experts):
            ap.error("sample: give one --alpha per expert")
        if a.alpha and not a.experts:
            ap.error("sample: --alpha goes with --experts")
        if a.max_new_tokens > 1024:
            ap.error("sample: at most 1024 new tokens, as the engine holds 2048 tokens and prompts are cut to 1024")
    elif a.command == "evaluate":
        check_target(ap, a)


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    check_args(ap, a)
    a.func(a)


if __name__ == "__main__":
    main()
