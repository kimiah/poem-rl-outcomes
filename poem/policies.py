"""The reference model π_ref, the experts and their policy log-ratios (Eq. 5).

An expert is a PEFT LoRA adapter trained from π_ref. A prompt is one user turn in π_ref's chat template, with the
generation prompt and thinking off. φ_k(x, y) = log π_k(y | x) − log π_ref(y | x) is a teacher-forced sum over the
response tokens: without the end-of-turn token on the programmatic bases (policy_log_ratios), with it on the RM bases
and in App. B.4 (chat_log_ratios). The weight updates ΔΘ_k = B_k A_k of the experts give the Gram matrix of
App. B.4.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import REFERENCE_MODEL
from .utils import revision_for


def chat_text(tok, prompt):
    """The prompt as one user turn with the generation prompt, thinking off."""
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def format_prompt(tok, prompt, max_len=1024):
    """(1, L) token ids of chat_text, cut to the first max_len tokens."""
    return tok(chat_text(tok, prompt), return_tensors="pt", truncation=True, max_length=max_len)["input_ids"]


def response_ids(tok, text):
    """The token ids of a response, without special tokens."""
    return tok(text, add_special_tokens=False)["input_ids"]


def eos_ids(tok):
    """Stop-token ids: the eos and eot ids plus whichever of <|im_end|>, <|endoftext|>, <|eot_id|> exist."""
    out = set()
    if tok.eos_token_id is not None:
        out.add(int(tok.eos_token_id))
    if getattr(tok, "eot_token_id", None) is not None:
        out.add(int(tok.eot_token_id))
    for s in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        i = tok.convert_tokens_to_ids(s)
        if i is not None and i != tok.unk_token_id:
            out.add(int(i))
    return out


def load_tokenizer(model=REFERENCE_MODEL, revision=None):
    """π_ref's tokenizer; its pad token is the eos token when it has none."""
    tok = AutoTokenizer.from_pretrained(model, revision=revision_for(model, revision))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_policy(adapter=None, *, model=REFERENCE_MODEL, revision=None, device="cuda:0"):
    """π_ref in bf16 and eval mode; with `adapter`, that LoRA adapter on this copy of π_ref."""
    policy = AutoModelForCausalLM.from_pretrained(model, revision=revision_for(model, revision), dtype=torch.bfloat16,
                                                  device_map=device).eval()
    return policy if adapter is None else PeftModel.from_pretrained(policy, adapter).eval()


def add_adapter(model, path, name):
    """Load the adapter at `path` as `name`. The first adapter wraps π_ref in a PeftModel, which is returned."""
    if isinstance(model, PeftModel):
        model.load_adapter(path, adapter_name=name)
        return model
    return PeftModel.from_pretrained(model, path, adapter_name=name).eval()


def each_adapter(model, adapters):
    """Yield (k, the model with adapter k active) for each adapter in turn; adapter k is deleted after its turn."""
    for k, path in enumerate(adapters):
        model = add_adapter(model, path, f"e{k}")
        model.set_adapter(f"e{k}")
        yield k, model
        model.delete_adapter(f"e{k}")


@torch.no_grad()
def sequence_logprobs(model, tok, prompt_ids, response_ids, *, batch_size=32, device="cuda:0"):
    """log π(y | x) of each row (N,) float64: teacher forcing in right-padded batches, with an fp32 log-softmax over
    the response positions. A row with an empty response gets 0."""
    n = len(prompt_ids)
    out = np.zeros(n, dtype=np.float64)
    pad = tok.pad_token_id
    for i in range(0, n, batch_size):
        js = list(range(i, min(i + batch_size, n)))
        seqs = [prompt_ids[j] + response_ids[j] for j in js]
        plens = [len(prompt_ids[j]) for j in js]
        rlens = [len(response_ids[j]) for j in js]
        maxlen = max(len(s) for s in seqs)
        ids = torch.full((len(js), maxlen), pad, dtype=torch.long)
        attn = torch.zeros((len(js), maxlen), dtype=torch.long)
        for b, s in enumerate(seqs):
            ids[b, :len(s)] = torch.tensor(s, dtype=torch.long)
            attn[b, :len(s)] = 1
        ids = ids.to(device)
        attn = attn.to(device)
        logits = model(input_ids=ids, attention_mask=attn).logits
        for b, j in enumerate(js):
            pl, rl = plens[b], rlens[b]
            if rl == 0:
                continue
            # positions pl-1 ... pl+rl-2 predict the response tokens; the log-softmax is row-wise, so it is taken on
            # those rows only
            pred = torch.log_softmax(logits[b, pl - 1: pl + rl - 1, :].float(), dim=-1)
            tgt = ids[b, pl: pl + rl]
            out[j] = float(pred.gather(1, tgt.unsqueeze(1)).squeeze(1).sum().item())
    return out


def policy_log_ratios(experts, prompts, responses, *, model=REFERENCE_MODEL, revision=None, batch_size=32,
                      device="cuda:0", log=print):
    """Φ (N, K) float64 of the programmatic bases: Φ[n, k] = φ_k(x, y) (Eq. 5) on the response
    tokens without the end-of-turn, on calibration row n, x = prompts[n] and y = responses[n]; Φ̃ is Φ centered
    within each prompt. π_ref is scored first, then each expert adapter in turn on the same copy of π_ref."""
    tok = load_tokenizer(model, revision)
    p_ids = [format_prompt(tok, p)[0].tolist() for p in prompts]
    r_ids = [response_ids(tok, r) for r in responses]
    reference = load_policy(model=model, revision=revision, device=device)
    reference_lp = sequence_logprobs(reference, tok, p_ids, r_ids, batch_size=batch_size, device=device)
    Phi = np.zeros((len(prompts), len(experts)))
    for k, policy in each_adapter(reference, experts):
        Phi[:, k] = sequence_logprobs(policy, tok, p_ids, r_ids, batch_size=batch_size, device=device) - reference_lp
        log(f"  phi {k + 1}/{len(experts)} {experts[k]}: mean |phi| {np.abs(Phi[:, k]).mean():.3f}")
    return Phi


def chat_ids(tok, prompt, response):
    """(token ids of the whole chat, number of prompt tokens): chat_text's prompt followed by the response as the
    assistant turn of the chat template, without added special tokens. The tokens after the prompt are the response
    and the template's end-of-turn ("<|im_end|>" and a newline for Qwen3)."""
    prompt_text = chat_text(tok, prompt)
    full = tok.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
                                   tokenize=False, add_generation_prompt=False)
    assert full.startswith(prompt_text), "the chat template does not extend the prompt"
    full_ids, prompt_ids = (tok(x, add_special_tokens=False)["input_ids"] for x in (full, prompt_text))
    return full_ids, len(prompt_ids)


@torch.no_grad()
def _chat_logprobs(model, tok, enc, groups, *, batch_size, max_len, device):
    """(log π(y | x), tail) per row, float64, from enc[n] = chat_ids of row n. Each group of rows (a list of row
    indices) runs on its own, its rows by decreasing token length (numpy's default argsort, over the group in order),
    in right-padded batches of batch_size cut to max_len tokens. One fp32 log-softmax over the whole batch gives the
    log-probs; log π sums them over the tokens after the prompt, and tail sums the last two of these, the end-of-turn
    (0 when the row was cut). A row cut within its prompt gets 0 for both."""
    out, tail = np.zeros(len(enc)), np.zeros(len(enc))
    pad = tok.pad_token_id
    for group in groups:
        order = np.asarray(group)[np.argsort([-len(enc[i][0]) for i in group])]
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            seqs = [enc[i][0][:max_len] for i in idx]
            T = max(len(x) for x in seqs)
            ids = torch.full((len(seqs), T), pad, dtype=torch.long)
            for b, x in enumerate(seqs):
                ids[b, :len(x)] = torch.tensor(x)
            ids = ids.to(device)
            lengths = torch.tensor([len(x) for x in seqs], device=device)
            attn = torch.arange(T, device=device)[None, :] < lengths[:, None]
            logits = model(input_ids=ids, attention_mask=attn).logits.float()
            lp = torch.log_softmax(logits[:, :-1], dim=-1).gather(2, ids[:, 1:, None]).squeeze(-1)
            for b, i in enumerate(idx):
                p_len, n = enc[i][1], len(seqs[b])
                if p_len < n:
                    span = lp[b, p_len - 1:n - 1]
                    out[i] = float(span.sum().item())
                    tail[i] = float(span[-2:].sum().item()) if n == len(enc[i][0]) else 0.0
    return out, tail


def chat_log_ratios(experts, prompts, responses, truncated, groups, *, batch_size=16, max_len=1536,
                    model=REFERENCE_MODEL, revision=None, device="cuda:0", log=print):
    """Φ (N, K) float64, Φ[n, k] = φ_k(x, y) on row n in the RM bases' convention: log π(y | x) scores the response
    and the end-of-turn of chat_ids, except on a truncated row (a response cut at the sampler's token limit, whose
    end-of-turn no policy sampled), where the end-of-turn is left out. π_ref and each expert are loaded fresh (one copy
    each) and run by _chat_logprobs on `groups`, the row indices of each source. Unlike policy_log_ratios (the
    programmatic bases), the batches are sorted by length and the log-softmax runs over the whole batch: each Φ is
    computed the way its reported numbers were."""
    tok = load_tokenizer(model, revision)
    enc = [chat_ids(tok, p, r) for p, r in zip(prompts, responses)]
    cut = np.asarray(truncated, dtype=float)

    def logprob(adapter):
        policy = load_policy(adapter, model=model, revision=revision, device=device)
        lp, tail = _chat_logprobs(policy, tok, enc, groups, batch_size=batch_size, max_len=max_len, device=device)
        del policy
        torch.cuda.empty_cache()
        return lp - cut * tail

    reference_lp = logprob(None)
    Phi = np.zeros((len(prompts), len(experts)))
    for k, expert in enumerate(experts):
        Phi[:, k] = logprob(expert) - reference_lp
        log(f"  phi {k + 1}/{len(experts)} {expert}: mean |phi| {np.abs(Phi[:, k]).mean():.3f}")
    return Phi


def source_groups(rows):
    """The row indices of each source, the expert that wrote the row ("expert"; π_ref for rows without it), in order of
    first appearance."""
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row.get("expert"), []).append(i)
    return list(groups.values())


def row_log_ratios(experts, rows, **kw):
    """chat_log_ratios on calibration rows {"prompt", "response", "truncated"[, "expert"]}, grouped by source_groups;
    `kw` goes to chat_log_ratios."""
    return chat_log_ratios(experts, [r["prompt"] for r in rows], [r["response"] for r in rows],
                           [r["truncated"] for r in rows], source_groups(rows), **kw)


def lora_update_gram(adapters):
    """(K, K) float64 uncentered Gram matrix of the experts' weight updates ΔΘ_k, the flattened LoRA products B_k A_k
    over all LoRA modules (App. B.4). ⟨B_i A_i, B_j A_j⟩_F = Σ (B_iᵀ B_j) ∘ (A_i A_jᵀ), so no product B A is
    formed. It compares experts only when they share one LoRA initialization."""
    files = [safe_open(str(Path(a) / "adapter_model.safetensors"), framework="pt") for a in adapters]
    modules = sorted({key.rsplit(".", 2)[0] for key in files[0].keys() if key.endswith("lora_A.weight")})
    G = torch.zeros(len(files), len(files), dtype=torch.float64)
    for m in modules:
        A = torch.stack([h.get_tensor(f"{m}.lora_A.weight").double() for h in files])   # K, r, in
        B = torch.stack([h.get_tensor(f"{m}.lora_B.weight").double() for h in files])   # K, out, r
        G += (torch.einsum("kor,lot->klrt", B, B) * torch.einsum("kri,lti->klrt", A, A)).sum((2, 3))
    return G.numpy()
