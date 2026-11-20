"""Step 1 of PoEM: the calibration set (Sec. 2.4, "The calibration set").

Responses are sampled for each calibration prompt from the reference model π_ref (reference_calibration) or from the
experts (expert_calibration), or for the RM bases with PPO's rollout sampler from π_ref or from each expert
(rollout_calibration). `prompts` is a list of (prompt_id, prompt) pairs, one per prompt; each sampler returns one row
{"prompt_id", "prompt", "response"} per kept response, in prompt order ("expert" names the expert that wrote an
expert-sampled response).
"""
from __future__ import annotations

import torch

from . import REFERENCE_MODEL
from .policies import chat_text, each_adapter, load_policy, load_tokenizer
from .utils import expert_name


def _kept(tok, text, response):
    """Whether a π_ref response is kept: it is not blank, and the formatted prompt text plus the response, cut to 512
    tokens, holds at least two tokens past the prompt (also cut to 512)."""
    if not response.strip():
        return False
    total = len(tok(text + response, truncation=True, max_length=512)["input_ids"])
    return total > len(tok(text, truncation=True, max_length=512)["input_ids"]) + 1


def reference_calibration(prompts, *, n=4, max_new_tokens=128, temperature=1.0, top_p=1.0, seed=0,
                          model=REFERENCE_MODEL, revision=None, device="cuda:0", log=print):
    """n responses per prompt from π_ref, one generate call each, on the prompt cut to 256 tokens. The seed is set
    before π_ref loads; top-k comes from π_ref's generation config (20 for Qwen3-0.6B). At temperature 0 the responses
    are greedy, so the n responses of a prompt repeat and fit cannot use them. A response is dropped by _kept, so a
    prompt of 511 or more formatted tokens keeps none."""
    torch.manual_seed(seed)
    tok = load_tokenizer(model, revision)
    policy = load_policy(model=model, revision=revision, device=device)
    gkw = dict(do_sample=True, temperature=temperature, top_p=top_p) if temperature > 0 else dict(do_sample=False)
    rows = []
    for i, (prompt_id, prompt) in enumerate(prompts):
        text = chat_text(tok, prompt)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        plen = int(enc["input_ids"].shape[1])
        responses = []
        for _ in range(n):
            out = policy.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id, **gkw)
            responses.append(tok.decode(out[0][plen:], skip_special_tokens=True))
        rows += [{"prompt_id": prompt_id, "prompt": prompt, "response": r} for r in responses if _kept(tok, text, r)]
        if (i + 1) % 10 == 0:
            log(f"[calibration] {i + 1}/{len(prompts)} prompts, {len(rows)} rows")
    return rows


def expert_calibration(experts, prompts, *, n=4, max_new_tokens=128, temperature=1.0, batch_size=16, seed=0,
                       model=REFERENCE_MODEL, revision=None, device="cuda:0", log=print):
    """n responses per prompt from the experts (adapter directories): response j of prompt i comes from expert
    (i·n + j) mod K, in the given order. Each expert samples its rows after torch.manual_seed(seed·1000 + k), in
    left-padded batches with the prompt cut to 1024 tokens, at top-p 1 and top-k 0 (greedy at temperature 0)."""
    slots = [[] for _ in experts]                     # expert k -> [(row, prompt_id, prompt)]
    for i, (prompt_id, prompt) in enumerate(prompts):
        for j in range(n):
            slots[(i * n + j) % len(experts)].append((i * n + j, prompt_id, prompt))
    tok = load_tokenizer(model, revision)
    tok.padding_side = "left"
    rows = [None] * (len(prompts) * n)
    for k, policy in each_adapter(load_policy(model=model, revision=revision, device=device), experts):
        torch.manual_seed(seed * 1000 + k)
        for b in range(0, len(slots[k]), batch_size):
            chunk = slots[k][b:b + batch_size]
            enc = tok([chat_text(tok, p) for _, _, p in chunk], return_tensors="pt", padding=True, truncation=True,
                      max_length=1024).to(device)
            gkw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
            if temperature > 0:
                gkw.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
            else:
                gkw.update(do_sample=False)
            new = policy.generate(**enc, **gkw)[:, enc["input_ids"].shape[1]:]
            for ids, (row, prompt_id, prompt) in zip(new, chunk):
                rows[row] = {"prompt_id": prompt_id, "prompt": prompt,
                             "response": tok.decode(ids, skip_special_tokens=True), "expert": expert_name(experts[k])}
        log(f"  [{k + 1}/{len(experts)}] {expert_name(experts[k])}: {len(slots[k])} rows")
    return rows


def rollout_calibration(experts, prompts, *, n=4, max_new_tokens=512, batch_size=32, seed=0, model=REFERENCE_MODEL,
                        revision=None, device="cuda:0", log=print):
    """n responses per prompt from π_ref (no experts) or from each expert in the given order, with the sampler of the
    RM bases' PPO training (Table 4, "sampled as in training"): temperature 0.7, top-p 0.8, top-k 20. Each policy is
    loaded fresh, then torch.manual_seed(seed); it samples batches of batch_size prompts (left padding,
    num_return_sequences=n). A response is the decode of its tokens without pad and special tokens; it is "truncated"
    when its tokens, end-of-turn included, reach max_new_tokens. Rows are source-major, then prompt, then sample."""
    tok = load_tokenizer(model, revision)
    tok.padding_side = "left"
    rows = []
    for adapter in experts or [None]:
        policy = load_policy(adapter, model=model, revision=revision, device=device)
        torch.manual_seed(seed)
        for b in range(0, len(prompts), batch_size):
            chunk = prompts[b:b + batch_size]
            enc = tok([chat_text(tok, p) for _, p in chunk], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(device)
            out = policy.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.8,
                                  top_k=20, num_return_sequences=n, pad_token_id=tok.pad_token_id)
            for j, ids in enumerate(out[:, enc["input_ids"].shape[1]:]):
                ids = ids[ids != tok.pad_token_id].tolist()
                prompt_id, prompt = chunk[j // n]
                row = {"prompt_id": prompt_id, "prompt": prompt, "response": tok.decode(ids, skip_special_tokens=True),
                       "truncated": len(ids) >= max_new_tokens}
                rows.append({**row, "expert": expert_name(adapter)} if adapter else row)
        log(f"  [rollout] {expert_name(adapter) if adapter else 'reference'}: {len(prompts) * n} rows")
        del policy
        torch.cuda.empty_cache()
    return rows
