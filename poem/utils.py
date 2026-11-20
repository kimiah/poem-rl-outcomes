"""Helpers of the package and the recipe: the files of the command line (DETAILS.md, "File formats"), that is JSON and
JSONL, prompts files, the experts' log-ratios as npz and your reward as a Python file; and the adapter directories,
that is an expert's name, the LoRA rank vLLM needs and the revision of π_ref to load."""
from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np

from . import REFERENCE_MODEL, REFERENCE_REVISION


# -------------------------------------------------------------------------------------------------------------- files

def read_json(path):
    with open(path) as f:
        return json.load(f)


def read_jsonl(path):
    """The rows of a JSONL file, skipping blank lines."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)


def write_json(path, obj, indent=1):
    _write(path, json.dumps(obj, indent=indent))


def write_jsonl(path, rows):
    _write(path, "".join(json.dumps(r) + "\n" for r in rows))


def prompt_id(row, default=None):
    """The prompt id of a row: "prompt_id", or "prompt_idx"."""
    return row.get("prompt_id", row.get("prompt_idx", default))


def read_prompts(path):
    """[(prompt_id, prompt)] of a prompts file: a .json list, a .jsonl with a "prompt" per line, or text with one
    prompt per line. Ids are positions, or a .jsonl row's prompt id, whose first row is kept."""
    if str(path).endswith(".json"):
        return list(enumerate(read_json(path)))
    if str(path).endswith(".jsonl"):
        prompts = {}
        for i, row in enumerate(read_jsonl(path)):
            prompts.setdefault(prompt_id(row, i), row["prompt"])
        return list(prompts.items())
    with open(path) as f:
        return list(enumerate(line.rstrip("\n") for line in f if line.strip()))


def save_logratios(path, Phi, names, pair_idx, prompt_ids):
    """Write Φ as an npz {logratios, names, pair_idx, prompt_ids}; row n is line pair_idx[n] of the calibration set."""
    np.savez(path, logratios=Phi, names=np.array(names), pair_idx=pair_idx, prompt_ids=prompt_ids)


def load_logratios(path, names=None):
    """(Phi, names, pair_idx, prompt_ids) from save_logratios' npz. `names`, when given, must be its columns in
    order."""
    d = np.load(path)
    stored = [str(x) for x in d["names"]]
    if names is not None and list(names) != stored:
        raise ValueError(f"the experts must be the columns of {path} in order: {stored}")
    return d["logratios"], stored, d["pair_idx"], d["prompt_ids"]


def load_reward(path):
    """reward(prompt, response) -> float, from the Python file at path (the command line's --reward)."""
    return runpy.run_path(path)["reward"]


# ------------------------------------------------------------------------------------------------------------- models

def revision_for(model, revision=None):
    """The revision to load: the pinned one for π_ref when none is given, else `revision`."""
    return REFERENCE_REVISION if model == REFERENCE_MODEL and revision is None else revision


def expert_name(path):
    """An expert's name: its adapter directory name without the "lora_" prefix."""
    return Path(path).name.removeprefix("lora_")


def lora_rank(adapters):
    """The smallest of the LoRA ranks 8 ... 512 that holds every adapter's rank r (8 for no adapters)."""
    rank = max((json.loads((Path(a) / "adapter_config.json").read_text())["r"] for a in adapters), default=8)
    return min(r for r in (8, 16, 32, 64, 128, 256, 320, 512) if r >= rank)
