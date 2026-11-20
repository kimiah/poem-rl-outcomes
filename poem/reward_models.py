"""The RM bases (App. B.1, "Reward models"): their reward models, z-stats, combined rewards and the experts' rewards.

A reward model scores a (prompt, response) pair with one scalar r(x, y). A combined reward weights z-scored basis
rewards, r_tar = Σ_r α*_r (r − μ_r) / σ_r (Eq. 2); a held-out reward is one z-scored basis reward, and its
π_tar is the held-out expert. The z-stats belong to a basis: Skywork-Reward-V2-Llama-3.1-8B has one (μ, σ) on RM-RB and
another on RM-Div and RM-Sky.

    REWARD_MODELS      reward -> (Hub id, revision, loader, options)
    load_scorer        score(pairs, batch_size=None) -> [float] of one reward model
    score_rows         row["scores"][r] for every row and reward r, scored group by group
    calibration_order  score_rows' groups for calibration rows: one per source (policies.source_groups), longest
                       response first
    decode_order       score_rows' group for decoded rows: shortest prompt + response first
    ZSTATS             basis -> reward -> (μ_r, σ_r)
    ROSTERS            basis -> the experts' rewards in the paper's order, the order of the experts and of the fits
    COMBINED_REWARDS   name -> (rewards, α*): the 8 combined rewards of RM-RB, then the 3 of RM-HH
    basis_combined_rewards  the combined rewards of one basis's experts
    EVAL_BATCH         basis -> the batch size its decodes are scored at (else each reward model's own)
    zscores            the (N, R) z-scored basis rewards of scored rows
    fit_target         the target of a fit on those rows
    reward_scorer      the target reward of a scored row

Most reward models run in bf16 on padded batches, so a score depends on the other pairs of its batch: the order and the
batch size are part of the protocol. The reward models run at pinned revisions, with transformers' remote code
(ArmoRM, internlm2, Eurus) or with their own read-out (OLMo-2, oasst-rm, SteamSHP, Starling). torch and transformers
are imported by the loaders.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

from .programmatic import make_combined_scorer

REWARD_MODELS = {  # reward: (Hub id, revision, loader, options)
    "skywork": ("Skywork/Skywork-Reward-V2-Llama-3.1-8B", "cba2f842f3f1af2f1b2f0d35e794d789976390c5", "classifier",
                {"rewardbench": True}),
    "fsfairx": ("sfairXC/FsfairX-LLaMA3-RM-v0.1", "94fad49f1b3227aa8b566f415a335adb68ec544c", "classifier",
                {"rewardbench": True}),
    "grm": ("Ray2333/GRM-Llama3-8B-rewardmodel-ft", "739ac5339e4a46a290cb16fb80c64277a79ba01a", "classifier",
            {"rewardbench": True}),
    "tulu3": ("allenai/Llama-3.1-Tulu-3-8B-RM", "84e607acab8adab64fa8cdaa26a9391d28362802", "classifier",
              {"rewardbench": True}),
    "armorm_helpfulness": ("RLHFlow/ArmoRM-Llama3-8B-v0.1", "eb2676d20da2f2d41082289d23c59b9f7427f955", "armorm",
                           {"head": 0}),
    "armorm_safety": ("RLHFlow/ArmoRM-Llama3-8B-v0.1", "eb2676d20da2f2d41082289d23c59b9f7427f955", "armorm",
                      {"head": 10}),
    "internlm2": ("internlm/internlm2-7b-reward", "853c9e6e0f71862785dd0b8eefca9609f8e720cb", "internlm2", {}),
    "eurus": ("openbmb/Eurus-RM-7b", "94059186e6ab8988905144023a8762f8e9aaf51c", "eurus",
              {"shift": -313.689, "scale": 780.484}),
    "acemath": ("nvidia/AceMath-7B-RM", "2a7b81019f94d1a78eec298f7cf5c677ff958f5a", "classifier",
                {"system": "Please reason step by step, and check your answer within \\boxed{}."}),
    "olmo2": ("allenai/OLMo-2-1124-7B-RM", "9d942da1127f9c36ecb5b40890359c79fb8a9837", "olmo2", {}),
    "deberta": ("OpenAssistant/reward-model-deberta-v3-large-v2", "c355404efa9ad2ad069f3a197cae0523c14244fc", "pair",
                {}),
    "oasst_pythia": ("OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5", "93cfb18ab7e2b5507dc0f0ca865198f3557a5d3b",
                     "oasst_pythia", {}),
    "steamshp": ("stanfordnlp/SteamSHP-flan-t5-xl", "b9554cd7012954c9ea7d01b380e202c4f6a5c051", "steamshp", {}),
    "starling": ("berkeley-nest/Starling-RM-7B-alpha", "6c6b4d5627834fe010d2c001632de2b94db81d66", "starling",
                 {"tokenizer": ("NousResearch/Llama-2-7b-chat-hf", "351844e75ed0bcbbe3f10671b3c808d2b83894ee")}),
    "creative_writing": ("SAA-Lab/Llama8B-CreativeWritingVerifier", "81e7bc150b17012c856eb39172c1b15fccd239b5",
                         "classifier", {"response_only": True, "add_special_tokens": True}),
    "skywork_qwen3_0.6b": ("Skywork/Skywork-Reward-V2-Qwen3-0.6B", "8c14a4e9e6321deaf572544339b16b8d6bbe8886",
                           "classifier", {}),
    "skywork_llama3.2_1b": ("Skywork/Skywork-Reward-V2-Llama-3.2-1B", "b755d39d962b7f92117cfe386d8943cf691f16fb",
                            "classifier", {}),
    "skywork_qwen3_1.7b": ("Skywork/Skywork-Reward-V2-Qwen3-1.7B", "e51ea3e08fb81326c3b812a7ff0cb9cee83e59cc",
                           "classifier", {}),
    "skywork_llama3.2_3b": ("Skywork/Skywork-Reward-V2-Llama-3.2-3B", "beafe5f2e6a275ce8c556f68f2d82de7fefbc3d8",
                            "classifier", {}),
    "skywork_qwen3_4b": ("Skywork/Skywork-Reward-V2-Qwen3-4B", "fd958fef475f323f4e6b195930e3dd918485c668",
                         "classifier", {}),
}

ROSTERS = {
    "RM-RB": ("skywork", "fsfairx", "grm", "tulu3"),
    "RM-HH": ("armorm_helpfulness", "armorm_safety"),
    "RM-Div": ("skywork", "internlm2", "eurus", "acemath", "olmo2", "deberta", "oasst_pythia", "steamshp", "starling",
               "creative_writing"),
    # RM-Sky: first the 8B Skywork reward, its target (held out; RM-Div's expert), then its five experts
    "RM-Sky": ("skywork", "skywork_qwen3_0.6b", "skywork_llama3.2_1b", "skywork_qwen3_1.7b", "skywork_llama3.2_3b",
               "skywork_qwen3_4b"),
}

# basis: {reward: (μ_r, σ_r)}, mean and standard deviation (float64, no epsilon). RM-RB: 2,000 prompts × 4 responses of
# the P-DPO pairs. RM-HH: π_ref's rollouts, 2 per training prompt, in float32. RM-Div and RM-Sky: π_ref's rollouts, 4
# per training prompt.
ZSTATS = {
    "RM-RB": {
        "skywork": (-13.07486646938324, 10.790467833328977),
        "fsfairx": (-4.931165531158447, 2.3583140582787587),
        "grm": (-14.069544723510742, 2.971537490968359),
        "tulu3": (2.4192234632968903, 2.1045840663216158),
    },
    "RM-HH": {
        "armorm_helpfulness": (0.6788964867591858, 0.13542069494724274),
        "armorm_safety": (0.8860702514648438, 0.25979653000831604),
    },
    "RM-Div": {
        "skywork": (4.600684611002604, 12.916595498177971),
        "internlm2": (0.8762343597412109, 1.165234572859169),
        "eurus": (1.1488293387734139, 2.0938356814423607),
        "acemath": (5.547930704752604, 2.910690658370209),
        "olmo2": (1.1144629414876301, 1.4508635688824454),
        "deberta": (0.905833061375112, 2.4470356808933853),
        "oasst_pythia": (2.0463560110625987, 2.6064172377749006),
        "steamshp": (3.5255850092569987, 0.9355657353887336),
        "starling": (2.0662567138671877, 1.7245137238582204),
        "creative_writing": (-1.1310783767700194, 1.715671712525155),
        "skywork_qwen3_0.6b": (4.496213785807291, 3.868108007127381),
        "skywork_llama3.2_1b": (5.948818130493164, 5.214064279510886),
        "skywork_qwen3_1.7b": (3.189756596883138, 5.287906876312803),
        "skywork_llama3.2_3b": (4.880613431930542, 9.075883508151897),
        "skywork_qwen3_4b": (8.16227559407552, 7.032405625499411),
    },
}
ZSTATS["RM-Sky"] = ZSTATS["RM-Div"]

COMBINED_REWARDS = {  # name: (rewards, α*); the rewards in the order of the sum
    "3rm_asym": (("skywork", "fsfairx", "tulu3"), (0.6, 0.3, 0.1)),
    "3rm_uniform": (("skywork", "fsfairx", "tulu3"), (0.3333333333333333, 0.3333333333333333, 0.3333333333333333)),
    "4rm_asym": (("skywork", "fsfairx", "tulu3", "grm"), (0.4, 0.3, 0.2, 0.1)),
    "4rm_uniform": (("skywork", "fsfairx", "tulu3", "grm"), (0.25, 0.25, 0.25, 0.25)),
    "fs_tulu_eq": (("fsfairx", "tulu3"), (0.5, 0.5)),
    "sky_fs_eq": (("skywork", "fsfairx"), (0.5, 0.5)),
    "sky_fs_peaked": (("skywork", "fsfairx"), (0.85, 0.15)),
    "sky_tulu_eq": (("skywork", "tulu3"), (0.5, 0.5)),
    "hh_eq": (("armorm_helpfulness", "armorm_safety"), (0.5, 0.5)),
    "hh_help": (("armorm_helpfulness", "armorm_safety"), (0.75, 0.25)),
    "hh_safe": (("armorm_helpfulness", "armorm_safety"), (0.25, 0.75)),
}

EVAL_BATCH = {"RM-RB": 32, "RM-HH": 32}


def basis_combined_rewards(basis):
    """The names of the combined rewards whose rewards are all in the basis's roster, in COMBINED_REWARDS' order."""
    return [c for c, (rewards, _) in COMBINED_REWARDS.items() if set(rewards) <= set(ROSTERS[basis])]


# ---------------------------------------------------------------------------------------------------------- loaders
# Each loader returns (step, its batch size): step(pairs) -> one float per pair, for one batch.

def _turns(prompt, response):
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]


def _chat_text(tok, prompt, response, system=None):
    """The pair as a user and an assistant turn (after a system turn, when given) in the model's chat template."""
    turns = ([{"role": "system", "content": system}] if system else []) + _turns(prompt, response)
    return tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=False)


def _tokenizer(hf_id, revision, padding_side=None):
    """The reward model's tokenizer at its revision (policies.load_tokenizer: the pad token is the EOS token when it
    has none)."""
    from .policies import load_tokenizer
    tok = load_tokenizer(hf_id, revision)
    if padding_side:
        tok.padding_side = padding_side
    return tok


def _sequence_classifier(hf_id, revision, device, dtype, padding_side=None, **kwargs):
    """(tokenizer, AutoModelForSequenceClassification); the model's pad id is the tokenizer's when it has none."""
    from transformers import AutoModelForSequenceClassification
    tok = _tokenizer(hf_id, revision, padding_side)
    model = AutoModelForSequenceClassification.from_pretrained(hf_id, revision=revision, dtype=dtype,
                                                               device_map=device, **kwargs).eval()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    return tok, model


def _checkpoint(hf_id, revision, drop, **load):
    """(config.json without the keys `drop`, the state dict of pytorch_model.bin) of the pinned snapshot."""
    import torch
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(hf_id, revision=revision))
    cfg = json.loads((snapshot / "config.json").read_text())
    for k in drop:
        cfg.pop(k, None)
    return cfg, torch.load(snapshot / "pytorch_model.bin", map_location="cpu", weights_only=True, **load)


def _linear_head(weight, device, dtype, bias=None):
    """The reward head Linear(h, 1) holding `weight` (and `bias`), on `device` in `dtype`."""
    import torch
    head = torch.nn.Linear(weight.shape[1], 1, bias=bias is not None)
    head.weight.data.copy_(weight.float())
    if bias is not None:
        head.bias.data.copy_(bias)
    return head.to(device=device, dtype=dtype).eval()


def _last_token(hidden, attention_mask, head):
    """head at each row's last non-pad token (right padding), as floats."""
    import torch
    return head(hidden[torch.arange(hidden.shape[0]), attention_mask.sum(1) - 1]).float().squeeze(-1).tolist()


def _classifier(hf_id, revision, device, rewardbench=False, system=None, response_only=False,
                add_special_tokens=False):
    """A sequence classifier in bf16, read at logit 0, on the chat template (after `system`, when given) or on the
    response alone (response_only), cut to 4096 tokens. The RM-RB reward models run as RewardBench runs them
    (rewardbench): num_labels=1, the tokenizer's padding side and its special tokens added, so that the Skywork input
    starts with two BOS tokens. The others: right padding, special tokens only with add_special_tokens."""
    import torch
    tok, model = _sequence_classifier(hf_id, revision, device, torch.bfloat16, None if rewardbench else "right",
                                      **({"num_labels": 1} if rewardbench else {}))

    def step(pairs):
        texts = [r.strip() if response_only else _chat_text(tok, p, r, system) for p, r in pairs]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096,
                  add_special_tokens=rewardbench or add_special_tokens).to(device)
        return model(**enc).logits[:, 0].float().tolist()
    return step, 8


def _armorm(hf_id, revision, device, head):
    """ArmoRM-Llama3-8B-v0.1 in bf16: head `head` of its per-attribute rewards (out.rewards, not the gated score), read
    in float32, on the chat template cut to 4096 tokens. Its remote code reads LLAMA_INPUTS_DOCSTRING, which newer
    transformers versions dropped."""
    import torch
    import transformers.models.llama.modeling_llama as llama
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if not hasattr(llama, "LLAMA_INPUTS_DOCSTRING"):
        llama.LLAMA_INPUTS_DOCSTRING = ""
    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(hf_id, revision=revision, trust_remote_code=True,
                                                               dtype=torch.bfloat16, device_map=device).eval()

    def step(pairs):
        enc = tok([_chat_text(tok, p, r) for p, r in pairs], return_tensors="pt", padding=True, truncation=True,
                  max_length=4096).to(device)
        return model(**enc).rewards[:, head].float().tolist()
    return step, 8


def _internlm2(hf_id, revision, device):
    """internlm2-7b-reward in fp16: its remote get_score(tokenizer, chat), one pair at a time."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision, trust_remote_code=True)
    model = AutoModel.from_pretrained(hf_id, revision=revision, trust_remote_code=True, dtype=torch.float16,
                                      device_map=device).eval()
    return (lambda pairs: [float(model.get_score(tok, _turns(p, r))) for p, r in pairs]), 1


def _eurus(hf_id, revision, device, shift, scale):
    """Eurus-RM-7b in bf16 (remote code), one pair at a time on "[INST]\\n{prompt}\\n[/INST]\\n{response}" cut to 4096
    tokens: its last output x, then (x − shift) / scale, the fixed rescaling of its expert's training reward."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    model = AutoModel.from_pretrained(hf_id, revision=revision, trust_remote_code=True, dtype=torch.bfloat16,
                                      device_map=device).eval()

    def score(prompt, response):
        enc = tok(f"[INST]\n{prompt}\n[/INST]\n{response}", return_tensors="pt", truncation=True,
                  max_length=4096).to(device)
        return (float(model(**enc).flatten()[-1].item()) - shift) / scale
    return (lambda pairs: [score(p, r) for p, r in pairs]), 1


def _pair(hf_id, revision, device):
    """reward-model-deberta-v3-large-v2 in fp32 on the pair encoding tok(prompt, response), cut to 512 tokens longest
    first (a prompt of more than 512 tokens alone cannot be cut from the response side)."""
    import torch
    tok, model = _sequence_classifier(hf_id, revision, device, torch.float32)

    def step(pairs):
        enc = tok([p for p, _ in pairs], [r for _, r in pairs], return_tensors="pt", padding=True,
                  truncation="longest_first", max_length=512).to(device)
        return model(**enc).logits[:, 0].float().tolist()
    return step, 16


def _olmo2(hf_id, revision, device):
    """OLMo-2-1124-7B-RM: Olmo2Model in bf16 and the checkpoint's score head, Linear(h, 1) without bias, at the last
    non-pad token; the chat template without added special tokens, right padding, cut to 4096 tokens."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import Olmo2Model
    tok = _tokenizer(hf_id, revision, "right")
    snapshot = Path(snapshot_download(hf_id, revision=revision))
    model = Olmo2Model.from_pretrained(hf_id, revision=revision, dtype=torch.bfloat16, device_map=device).eval()
    shard = json.loads((snapshot / "pytorch_model.bin.index.json").read_text())["weight_map"]["score.weight"]
    w = torch.load(snapshot / shard, map_location="cpu", weights_only=True)["score.weight"]
    head = _linear_head(w, device, torch.bfloat16)

    def step(pairs):
        enc = tok([_chat_text(tok, p, r) for p, r in pairs], return_tensors="pt", padding=True, truncation=True,
                  max_length=4096, add_special_tokens=False).to(device)
        return _last_token(model(**enc).last_hidden_state, enc["attention_mask"], head)
    return step, 8


def _oasst_pythia(hf_id, revision, device):
    """oasst-rm-2.1-pythia-1.4b-epoch-2.5 in fp32, as released: the GPT-NeoX backbone (checkpoint keys gpt_neox.*) and
    out_proj, Linear(h, 1), at the last non-pad token, on "<|prompter|>{prompt}<|endoftext|><|assistant|>{response}
    <|endoftext|>", right padding, cut to 2048 tokens. Its model class is not in transformers, so both parts are read
    from the checkpoint."""
    import torch
    from transformers import GPTNeoXConfig, GPTNeoXModel
    cfg, sd = _checkpoint(hf_id, revision,
                          ("architectures", "model_type", "pooling", "auto_map", "transformers_version"))
    model = GPTNeoXModel(GPTNeoXConfig(**cfg))
    model.load_state_dict({k[len("gpt_neox."):]: v for k, v in sd.items() if k.startswith("gpt_neox.")}, strict=False)
    model = model.to(device).eval()
    head = _linear_head(sd["out_proj.weight"], device, torch.float32, bias=sd["out_proj.bias"])
    tok = _tokenizer(hf_id, revision, "right")

    def step(pairs):
        texts = [f"<|prompter|>{p}<|endoftext|><|assistant|>{r}<|endoftext|>" for p, r in pairs]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        return _last_token(model(**enc).last_hidden_state, enc["attention_mask"], head)
    return step, 8


def _steamshp(hf_id, revision, device):
    """SteamSHP-flan-t5-xl in bf16, in the pointwise mode of its model card: the response is RESPONSE A, RESPONSE B is
    ".", and the score is logit("A") − logit("B") at the first decoder step. The post and the response, whitespace
    collapsed, are cut to token budgets (the post to at most a third) so that the input fits 512 tokens."""
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    model = T5ForConditionalGeneration.from_pretrained(hf_id, revision=revision, dtype=torch.bfloat16,
                                                       device_map=device).eval()
    id_a, id_b = (tok(x, add_special_tokens=False).input_ids[-1] for x in ("A", "B"))
    tail = "\n\n RESPONSE B: .\n\n Which response is better? RESPONSE"
    budget = 512 - len(tok("POST:  \n\n RESPONSE A: " + tail).input_ids) - 8

    def clip(text, n):
        return tok.decode(tok(" ".join(text.split()), add_special_tokens=False).input_ids[:n], skip_special_tokens=True)

    def text(prompt, response):
        n_post = min(len(tok(prompt, add_special_tokens=False).input_ids), budget // 3)
        return f"POST: {clip(prompt, n_post)} \n\n RESPONSE A: {clip(response, budget - n_post)}{tail}"

    def step(pairs):
        enc = tok([text(p, r) for p, r in pairs], return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(device)
        start = torch.full((enc["input_ids"].shape[0], 1), model.config.decoder_start_token_id, device=device)
        logits = model(**enc, decoder_input_ids=start).logits[:, 0].float()
        return (logits[:, id_a] - logits[:, id_b]).tolist()
    return step, 16


def _starling(hf_id, revision, device, tokenizer):
    """Starling-RM-7B-alpha: Llama-2-7b-chat (checkpoint keys transformer.*) and v_head, Linear(h, 1) without bias, read
    at the token before the first pad (unk) token. The input follows the reference code of its model card:
    "<s>[INST] {prompt} </s> [/INST] {response}</s>" with the tokenizer's own BOS too (two BOS), pad = unk, right
    padding, the first tokens cut beyond 2048. `tokenizer` is the (Hub id, revision) of
    Llama-2-7b-chat's tokenizer. The model is built on the meta device and takes the checkpoint's tensors and dtype
    (assign=True); the rotary frequencies, which the checkpoint does not hold, are rebuilt."""
    import torch
    from transformers import AutoTokenizer, LlamaConfig, LlamaModel
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    cfg, sd = _checkpoint(hf_id, revision, ("architectures", "_name_or_path", "torch_dtype", "transformers_version"),
                          mmap=True)
    config = LlamaConfig(**cfg)
    with torch.device("meta"):
        model = LlamaModel(config)
    model.load_state_dict({k[len("transformer."):]: v for k, v in sd.items() if k.startswith("transformer.")},
                          strict=True, assign=True)
    model.rotary_emb = LlamaRotaryEmbedding(config=config, device="cpu")
    model = model.to(device).eval()
    head = _linear_head(sd["v_head.weight"], device, torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(tokenizer[0], revision=tokenizer[1])
    tok.pad_token = tok.unk_token
    tok.padding_side, tok.truncation_side = "right", "left"

    def step(pairs):
        enc = tok([f"<s>[INST] {p} </s> [/INST] {r}</s>" for p, r in pairs], return_tensors="pt", padding=True,
                  truncation=True, max_length=2048).to(device)
        rewards = head(model(**enc).last_hidden_state).squeeze(-1).float()
        out = []
        for b, ids in enumerate(enc["input_ids"]):
            pads = (ids == tok.pad_token_id).nonzero()
            out.append(rewards[b, (pads[0].item() if len(pads) > 0 else ids.shape[0]) - 1].item())
        return out
    return step, 8


_LOADERS = {"classifier": _classifier, "armorm": _armorm, "internlm2": _internlm2, "eurus": _eurus, "pair": _pair,
            "olmo2": _olmo2, "oasst_pythia": _oasst_pythia, "steamshp": _steamshp, "starling": _starling}


# ---------------------------------------------------------------------------------------------------------- scoring

def load_scorer(reward, device="cuda:0"):
    """The scorer of reward model `reward`: score(pairs, batch_size=None) -> one float per (prompt, response) pair, in
    order, run on consecutive batches of batch_size pairs (default: the reward model's own batch size). Each reward is
    its own scorer, so the two ArmoRM heads load ArmoRM twice."""
    import torch
    hf_id, revision, loader, options = REWARD_MODELS[reward]
    step, own_batch = _LOADERS[loader](hf_id, revision, device, **options)

    @torch.no_grad()
    def score(pairs, batch_size=None):
        b = batch_size or own_batch
        return [s for i in range(0, len(pairs), b) for s in step(pairs[i:i + b])]
    return score


def score_rows(rewards, rows, groups, *, batch_size=None, device="cuda:0", log=print):
    """Writes row["scores"][r] of every row for each reward r. One reward model at a time is loaded, run on each group
    of rows (a list of row indices) in the group's order, so that no batch spans two groups, and freed."""
    import torch
    for r in rewards:
        score = load_scorer(r, device)
        for group in groups:
            for i, s in zip(group, score([(rows[i]["prompt"], rows[i]["response"]) for i in group], batch_size)):
                rows[i].setdefault("scores", {})[r] = s
        del score
        gc.collect()
        torch.cuda.empty_cache()
        log(f"  scored {r} on {sum(len(g) for g in groups)} rows")


def calibration_order(rows):
    """score_rows' groups for calibration rows: one per source (policies.source_groups), its rows by decreasing
    response length (numpy's default argsort of minus the length, over the source's rows in order)."""
    from .policies import source_groups
    return [np.asarray(g)[np.argsort([-len(rows[i]["response"]) for i in g])] for g in source_groups(rows)]


def decode_order(rows):
    """score_rows' group for decoded rows: all rows by increasing length of prompt plus response (a stable sort)."""
    return [sorted(range(len(rows)), key=lambda i: len(rows[i]["prompt"]) + len(rows[i]["response"]))]


# ----------------------------------------------------------------------------------------------- z-scores and targets

def zscores(rows, basis):
    """(N, R) z-scored rewards of the basis's roster: column r is (s_r − μ_r) / σ_r of the rows' scores s_r, with μ_r
    and σ_r Python floats. ArmoRM's scores are float32 (so RM-HH's z are float32), the others float64."""
    columns = []
    for r in ROSTERS[basis]:
        mu, sd = ZSTATS[basis][r]
        dtype = np.float32 if REWARD_MODELS[r][2] == "armorm" else np.float64
        columns.append((np.asarray([row["scores"][r] for row in rows], dtype=dtype) - mu) / sd)
    return np.stack(columns, 1)


def fit_target(z, basis, combined_reward=None, heldout=None):
    """The target of a fit on rows with z-scored rewards z (zscores): the held-out reward's column, or the combined
    reward Σ_r α*_r z_r, each term float(α*_r) · z_r, summed in the combined reward's order from 0. fit_target (z
    first, Σ from 0) and reward_scorer (the combined-scorer form) keep the two arithmetic orders of the paper's fit
    targets and decode scores."""
    roster = list(ROSTERS[basis])
    if heldout:
        return z[:, roster.index(heldout)]
    rewards, alpha = COMBINED_REWARDS[combined_reward]
    return sum(float(a) * z[:, roster.index(r)] for r, a in zip(rewards, alpha))


def reward_scorer(basis, combined_reward=None, heldout=None):
    """The target reward of a scored row: Σ_r α*_r (s_r − μ_r) / σ_r of its scores (programmatic.make_combined_scorer),
    or the held-out reward's (s_r − μ_r) / σ_r."""
    rewards, alpha = COMBINED_REWARDS[combined_reward] if combined_reward else ((heldout,), (1.0,))
    return make_combined_scorer(rewards, alpha, ZSTATS[basis],
                                 functions={r: (lambda row, r=r: row["scores"][r]) for r in rewards})
