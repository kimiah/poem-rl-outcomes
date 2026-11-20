"""Step 4 of PoEM: sampling PoEM with vLLM (Eq. 6).

A policy is a list of members [(adapter or None, w)] whose weights sum to 1; None is the reference model π_ref.
`weights.poem_members` gives PoEM(α) at composition strength γ, [(expert_k, γα_k)] + [(None, 1 − γΣ_k α_k)]; π_ref is
[(None, 1.0)], and one adapter (π_tar, an expert, a second RL run) is [(adapter, 1.0)].

`sample` decodes every (policy, prompt) pair as one group in one engine: one request per member, all on the prompt's
token ids, whose logits `PoEMLogitsProcessor` replaces by Σ_m w_m ℓ_m at every step, so the rows pick the same token.
The engine runs in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0), so every request of a wave is added before the wave's
first step, and a wave is sized to be scheduled whole in that step by its rows, prompt tokens and adapters (not by
the KV cache: a GPU too small for a wave makes its groups lose lockstep). A group whose rows return different tokens is
not a PoEM decode. Prompts are formatted by `policies.format_prompt` (cut to 1024 tokens) and passed as token ids; a response
is the decode of its tokens without the stop tokens. Decoding is greedy; at temperature > 0 the rows of a group share
one seed.

Repeatable kernels: vLLM's default LoRA shrink kernel splits its reduction (split-K 64 or 8) and adds the parts
atomically, so two decodes of the same groups can differ in the last bits and then in the tokens. `build_engine`
installs SPLITK1_SHRINK, the same kernel with split-K 1, for the current GPU unless repeatable=False or
VLLM_TUNED_CONFIG_FOLDER is already set; repeated decodes on one GPU model are then token-identical. The paper's
decodes used vLLM's default kernels.
"""
from __future__ import annotations

import json
import os
import tempfile
import time

import torch

from . import REFERENCE_MODEL
from .policies import eos_ids, format_prompt, load_tokenizer
from .utils import lora_rank, revision_for

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MAX_LORAS = 32                  # adapters on the GPU at once
MAX_NUM_SEQS = 512              # rows per step; within vLLM's CUDA-graph sizes, above which every step runs eager
MAX_BATCHED_TOKENS = 131072     # prompt tokens per step

# vLLM's LoRA shrink config, keyed max_loras -> num_slices -> batch -> hidden size -> rank. Every key but num_slices
# is looked up by the nearest value, so this covers every dense model.
_SPLITK1 = {"block_m": 32, "block_n": 16, "block_k": 256, "split_k": 1, "num_warps": 4, "num_ctas": 1,
            "group_size_m": 8, "num_stages": 2, "max_nreg": None}
SPLITK1_SHRINK = {"32": {s: {"1": {"1024": {"64": _SPLITK1}}} for s in "123"}}


def _use_splitk1_kernels():
    """Write SPLITK1_SHRINK as this GPU's tuned shrink config and point VLLM_TUNED_CONFIG_FOLDER at it."""
    folder = tempfile.mkdtemp()
    gpu = torch.cuda.get_device_name().replace(" ", "_").replace("-", "_")
    with open(os.path.join(folder, f"{gpu}_SHRINK.json"), "w") as f:
        json.dump(SPLITK1_SHRINK, f)
    os.environ["VLLM_TUNED_CONFIG_FOLDER"] = folder


def build_engine(model=REFERENCE_MODEL, revision=None, *, max_lora_rank=64, gpu_mem=0.6, repeatable=True):
    """The in-process vLLM engine: π_ref in bf16 with LoRA, the PoEM logits processor, no prefix caching, seed 0.
    repeatable=True installs the split-K-1 LoRA kernels first, unless VLLM_TUNED_CONFIG_FOLDER is set.
    VLLM_TUNED_CONFIG_FOLDER stays set for the process, so later engines in the same process use the same kernels."""
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0", "atomic waves need the in-process engine"
    if repeatable and "VLLM_TUNED_CONFIG_FOLDER" not in os.environ:
        _use_splitk1_kernels()
    from vllm import LLM

    from .logits_processor import PoEMLogitsProcessor
    revision = revision_for(model, revision)
    return LLM(model=model, revision=revision, tokenizer_revision=revision, dtype="bfloat16", enable_lora=True,
               max_lora_rank=max_lora_rank, max_loras=MAX_LORAS, max_cpu_loras=128, gpu_memory_utilization=gpu_mem,
               max_model_len=2048, max_num_seqs=MAX_NUM_SEQS, max_num_batched_tokens=MAX_BATCHED_TOKENS,
               enable_prefix_caching=False, logits_processors=[PoEMLogitsProcessor], seed=0)


def _waves(groups, prompt_ids):
    """Split the groups, in order, into waves the engine schedules whole in one step: at most MAX_NUM_SEQS rows,
    MAX_BATCHED_TOKENS prompt tokens and MAX_LORAS distinct adapters each."""
    waves, cur, rows, toks, ls = [], [], 0, 0, set()
    for g in groups:
        r, t = len(g["members"]), len(g["members"]) * len(prompt_ids[g["prompt"]])
        g_loras = {m[0] for m in g["members"] if m[0] is not None}
        if cur and (rows + r > MAX_NUM_SEQS or toks + t > MAX_BATCHED_TOKENS or len(ls | g_loras) > MAX_LORAS):
            waves.append(cur)
            cur, rows, toks, ls = [], 0, 0, set()
        cur.append(g)
        rows += r
        toks += t
        ls |= g_loras
    if cur:
        waves.append(cur)
    return waves


def decode_groups(llm, groups, tok, *, max_new_tokens=96, temperature=0.0, log=print):
    """Decode groups [{"id", "prompt", "members": [(adapter or None, w)], "seed"}] with an engine of build_engine
    -> ({id: response}, stats). "seed" is read only at temperature > 0. `tok` is π_ref's HF tokenizer, which cuts a
    long prompt on the right (vLLM's would cut it on the left). Raises RuntimeError after the first wave with a group
    whose rows lost lockstep."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    from .logits_processor import GROUP_KEY, SEEDED_KEY, SIZE_KEY, WEIGHT_KEY
    eos = eos_ids(tok)
    # one adapter registry per engine: vLLM caches adapters by id, so ids restarting at 1 in a second call on the same
    # engine would decode with the first call's adapters
    loras = llm.__dict__.setdefault("_poem_loras", {})

    def lora(path):
        if path is None:
            return None
        if path not in loras:
            loras[path] = LoRARequest(f"a{len(loras) + 1}", len(loras) + 1, str(path))
        return loras[path]

    prompt_ids = {}
    for g in groups:
        if g["prompt"] not in prompt_ids:
            prompt_ids[g["prompt"]] = format_prompt(tok, g["prompt"], 1024)[0].tolist()
        s = sum(w for _, w in g["members"])
        assert abs(s - 1.0) < 1e-4, f"group {g['id']} weights sum to {s}"
    waves = _waves(groups, prompt_ids)

    engine = llm.llm_engine
    out, t0 = {}, time.time()
    for wi, wave in enumerate(waves):
        for gi, g in enumerate(wave):
            for mi, (path, w) in enumerate(g["members"]):
                extra = {GROUP_KEY: f"w{wi}g{gi}", WEIGHT_KEY: float(w), SIZE_KEY: len(g["members"])}
                if temperature > 0:
                    extra[SEEDED_KEY] = True
                    sp = SamplingParams(temperature=temperature, top_p=1.0, top_k=-1, seed=int(g["seed"]),
                                        max_tokens=max_new_tokens, stop_token_ids=sorted(eos), extra_args=extra)
                else:
                    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, stop_token_ids=sorted(eos),
                                        extra_args=extra)
                engine.add_request(f"w{wi}g{gi}m{mi}", TokensPrompt(prompt_token_ids=prompt_ids[g["prompt"]]), sp,
                                   lora_request=lora(path))
        done = {}
        while engine.has_unfinished_requests():
            for o in engine.step():
                if o.finished:
                    done[o.request_id] = list(o.outputs[0].token_ids)
        split = 0
        for gi, g in enumerate(wave):
            seqs = [done[f"w{wi}g{gi}m{mi}"] for mi in range(len(g["members"]))]
            if any(s != seqs[0] for s in seqs):
                split += 1
            out[g["id"]] = tok.decode([t for t in seqs[0] if t not in eos], skip_special_tokens=True)
        if split:
            raise RuntimeError(f"wave {wi + 1}/{len(waves)}: {split} groups lost lockstep, so this is not a PoEM "
                               "decode. vLLM did not schedule the wave whole, most likely for lack of KV cache: give "
                               "more GPU memory (gpu_mem, --gpu-memory-utilization), or fewer policies or prompts")
        log(f"  wave {wi + 1}/{len(waves)}: {len(wave)} groups, {sum(len(g['members']) for g in wave)} rows, "
            f"{time.time() - t0:.0f}s elapsed")
    stats = dict(groups=len(groups), waves=len(waves), decode_seconds=time.time() - t0,
                 distinct_loras=len(loras))
    return out, stats


def sample(policies, prompts, *, model=REFERENCE_MODEL, revision=None, max_new_tokens=96, temperature=0.0, seed=0,
           gpu_mem=0.6, repeatable=True, log=print):
    """Decode every policy {name: members} on every prompt in one engine -> {name: [response per prompt]}.
    Groups are policy-major in the dict's order, and group (name, i) has seed `seed + i`. Raises ValueError, before
    building the engine, when a policy has more than MAX_LORAS adapters, and RuntimeError when a group lost lockstep
    (decode_groups). gpu_mem is the share of GPU memory vLLM takes."""
    for name, members in policies.items():
        if len({a for a, _ in members if a is not None}) > MAX_LORAS:
            raise ValueError(f"policy {name}: a PoEM decode holds at most {MAX_LORAS} experts")
    groups = [dict(id=(name, i), prompt=p, members=members, seed=seed + i)
              for name, members in policies.items() for i, p in enumerate(prompts)]
    adapters = {a for members in policies.values() for a, _ in members if a is not None}
    llm = build_engine(model, revision, max_lora_rank=lora_rank(adapters), gpu_mem=gpu_mem, repeatable=repeatable)
    responses, stats = decode_groups(llm, groups, load_tokenizer(model, revision), max_new_tokens=max_new_tokens,
                                     temperature=temperature, log=log)
    return {name: [responses[(name, i)] for i in range(len(prompts))] for name in policies}
