"""Evaluation (Sec. 3.2, "Metrics"): recovery of π_tar's reward gain, KL(π_tar ‖ π) along π_tar's own responses, the
relative KL, the drift D(π) of Eq. 21 and the bootstrap interval over targets.

    recovery        rec(π) = (r(π) − r(π_ref)) / (r(π_tar) − r(π_ref)) on rewards averaged over prompts (or on the
                    per-prompt differences, paired)
    boot_ci         percentile-bootstrap 95% interval of a statistic over targets
    token_logprobs  one policy's teacher-forced next-token log-probs on one response
    poem_logits     the PoEM logits ℓ_ref + Σ_k w_k (ℓ_k − ℓ_ref), w_k = γ α_k (Eq. 6)
    kl_per_token    KL(π_tar ‖ π) at each response position
    kl_to_target    per-prompt KL(π_tar ‖ π) of several policies on π_tar's responses
    mean_kl         KL(π_tar ‖ π) averaged over prompts; relative_kl divides it by KL(π_tar ‖ π_ref)
    mean_logprob    mean per-token log-likelihood of responses under a model, the terms of D(π)

The KL uses the form ℓ_ref + Σ_k w_k (ℓ_k − ℓ_ref), and the decoder (logits_processor) the form Σ_m w_m ℓ_m. Each is
the form its reported numbers come from. token_logprobs scores one response per forward pass, while
policies.sequence_logprobs batches the calibration rows; a batched bf16 forward is not bit-equal to an unbatched one.
"""
from __future__ import annotations

import numpy as np
import torch

from . import REFERENCE_MODEL
from .policies import format_prompt, load_policy, load_tokenizer, response_ids


def recovery(policy, reference, target, eps=1e-9, paired=False):
    """rec(π) from the per-prompt rewards of π, π_ref and π_tar, each averaged over prompts first, or with paired (the
    RM bases' arithmetic) mean(π − π_ref) / mean(π_tar − π_ref) over the per-prompt differences; NaN when the
    denominator is at most eps in magnitude."""
    if paired:
        ref = np.asarray(reference)
        gain, gap = float(np.mean(np.asarray(policy) - ref)), float(np.mean(np.asarray(target) - ref))
    else:
        ref = float(np.mean(reference))
        gain, gap = float(np.mean(policy)) - ref, float(np.mean(target)) - ref
    return gain / gap if abs(gap) > eps else float("nan")


def boot_ci(x, stat=np.median, B=10000, seed=0):
    """Percentile-bootstrap 95% interval of `stat` over the items of x, e.g. one value per target. Non-finite items are
    dropped; B resamples of the items are drawn with default_rng(seed). Returns the 2.5th and 97.5th percentiles as a
    tuple of floats, or (nan, nan) for fewer than two items."""
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    s = stat(x[rng.integers(0, len(x), (B, len(x)))], axis=1)
    return tuple(float(v) for v in np.percentile(s, [2.5, 97.5]))


def _response_logits(model, full, positions):
    """The fp32 logits of `model` at the response positions of the (1, L) prompt and response ids."""
    return model(input_ids=full).logits[0, positions].float()


@torch.no_grad()
def token_logprobs(model, tok, prompt, response, device):
    """log π(· | x, y_<t) of `model` at each response position t, fp32 (T, V), on the prompt as format_prompt gives it;
    the end-of-turn token is not scored. Returns (log-probs, the (1, L) prompt and response ids, the slice of the
    response positions, the response ids), or None for an empty response."""
    ids = response_ids(tok, response)
    if not ids:
        return None
    prompt_ids = format_prompt(tok, prompt).to(device)
    resp = torch.tensor(ids, device=device)
    full = torch.cat([prompt_ids, resp[None]], dim=1)
    positions = slice(prompt_ids.shape[1] - 1, prompt_ids.shape[1] - 1 + len(ids))
    return torch.log_softmax(_response_logits(model, full, positions), dim=-1), full, positions, resp


def poem_logits(l_ref, expert_logits, weights):
    """ℓ_ref + Σ_k w_k (ℓ_k − ℓ_ref) (Eq. 6), added expert by expert in order with float(w_k), where
    w_k = γ α_k is the caller's product; a copy of ℓ_ref when there are no experts."""
    out = l_ref.clone()
    for lk, w in zip(expert_logits, weights):
        out = out + float(w) * (lk - l_ref)
    return out


def kl_per_token(lp_target, lp_policy):
    """KL(π_tar ‖ π) = Σ_v p_tar(v) (log p_tar(v) − log p(v)) at each position, from (T, V) log-probs; returns (T,)."""
    return (lp_target.exp() * (lp_target - lp_policy)).sum(-1)


@torch.no_grad()
def kl_to_target(target, policies, prompts, target_texts, *, model=REFERENCE_MODEL, revision=None, device="cuda:0"):
    """KL(π_tar ‖ π) of each policy, teacher-forced on π_tar's response target_texts[i] to prompts[i] and averaged
    over its tokens. `target` is π_tar's adapter. policies = {name: (adapters, weights)}:

        (experts, [γ α_k])   PoEM, log_softmax(poem_logits(ℓ_ref, [ℓ_k], [γ α_k]))
        ([], [])             π_ref
        ([adapter], None)    that adapter's own policy, log_softmax(ℓ), e.g. a second RL run

    π_ref (bf16), π_tar and every adapter are loaded as separate models, each adapter on its own copy of π_ref, so that
    an adapter keeps the dtype it was saved in (one PEFT model holding several adapters rounds the later fp32 adapters
    to bf16 under PEFT 0.18); this takes one π_ref copy per adapter. Returns {name: [KL per prompt]}, with None where
    π_tar's response is empty."""
    tok = load_tokenizer(model, revision)
    adapters = list(dict.fromkeys(a for members, _ in policies.values() for a in members))
    reference = load_policy(model=model, revision=revision, device=device)
    tar = load_policy(target, model=model, revision=revision, device=device)
    copies = {a: load_policy(a, model=model, revision=revision, device=device) for a in adapters}
    out = {name: [] for name in policies}
    for prompt, text in zip(prompts, target_texts):
        r = token_logprobs(tar, tok, prompt, text, device)
        if r is None:
            for kl in out.values():
                kl.append(None)
            continue
        lp_target, full, positions, _ = r
        l_ref = _response_logits(reference, full, positions)
        logits = {a: _response_logits(copies[a], full, positions) for a in adapters}
        for name, (members, weights) in policies.items():
            l = logits[members[0]] if weights is None else poem_logits(l_ref, [logits[a] for a in members], weights)
            out[name].append(float(kl_per_token(lp_target, torch.log_softmax(l, dim=-1)).mean()))
    return out


def mean_kl(kl):
    """KL(π_tar ‖ π) over prompts: the mean of kl_to_target's per-prompt values, skipping None."""
    return float(np.mean([v for v in kl if v is not None]))


def relative_kl(kl_policy, kl_reference):
    """KL(π_tar ‖ π) / KL(π_tar ‖ π_ref) from kl_to_target's per-prompt values of π and of π_ref, each through mean_kl.
    Below 1, π is closer to π_tar than π_ref is."""
    return mean_kl(kl_policy) / mean_kl(kl_reference)


def mean_logprob(model, tok, prompts, texts, device):
    """Mean over responses (empty ones skipped) of the mean over response tokens of log π(y_t | x, y_<t) under `model`.
    Under π_ref, the drift of Eq. 21 is D(π) = mean_logprob(π_ref's responses) − mean_logprob(π's responses)."""
    per = []
    for prompt, text in zip(prompts, texts):
        r = token_logprobs(model, tok, prompt, text, device)
        per.append(float("nan") if r is None else float(r[0].gather(1, r[3][:, None]).squeeze(1).mean()))
    return float(np.nanmean(per))
