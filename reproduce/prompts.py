"""Step 1 of reproduce/run.sh, both settings: every prompt set, from UltraChat and PKU-SafeRLHF-prompts at the
revisions the paper's sets come from, written to the working directory (run.sh runs it in OUT)."""
import json
import random

import numpy as np
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download

from poem.utils import write_json
from poem.policies import format_prompt, load_tokenizer

ULTRACHAT = ("HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a")    # (Hub id, revision)
PKU = ("PKU-Alignment/PKU-SafeRLHF-prompts", "66465198fb8b9cbc6a7bdff124afaf35158d807c")


def load_ultrachat(n=5000):
    """The first user turn of each of the first n train_sft rows whose first message is from the user. Prompts 0-499
    are held out (the programmatic bases' calibration and evaluation pool, the RM bases' evaluation pool), 500-4999
    are training prompts."""
    prompts = []
    for row in load_dataset(ULTRACHAT[0], split="train_sft", revision=ULTRACHAT[1]):
        if row["messages"] and row["messages"][0]["role"] == "user":
            prompts.append(row["messages"][0]["content"])
        if len(prompts) == n:
            break
    return prompts


def load_pku():
    """Every PKU-SafeRLHF prompt, in file order."""
    with open(hf_hub_download(PKU[0], "data/PKU-SafeRLHF-prompts.jsonl", repo_type="dataset", revision=PKU[1])) as f:
        return [json.loads(line)["prompt"] for line in f]


def programmatic_sets(evaluation):
    """The programmatic bases' pool: the first 30 are evaluated, and the weight and coverage fits skip the first 50."""
    write_json("calibration_prompts.json", evaluation)
    write_json("eval_prompts.json", evaluation[:30])
    write_json("first_50_prompts.json", evaluation[:50])


def rm_training_sets(training, pku):
    """RM-RB trains on training prompts 0-1899 (1900-1999 validate). RM-HH, RM-Div and RM-Sky train on 1,000 of those
    and 1,000 PKU-SafeRLHF prompts, shuffled; the second RL run of RM-Div's Skywork expert on the same prompts in
    another order. Returns (rb_train, rb_val, mix_train, mix_val), mix_train as (prompt, "benign" or "harm") pairs."""
    rb_train, rb_val = training[:1900], training[1900:2000]
    rng = random.Random(42)
    harm, benign = sorted(set(pku)), list(rb_train)
    rng.shuffle(harm)
    rng.shuffle(benign)
    mix_train = [(p, "benign") for p in benign[:1000]] + [(p, "harm") for p in harm[:1000]]
    mix_val = benign[1000:1050] + harm[1000:1050]
    rng.shuffle(mix_train)
    write_json("rm_rb_train.json", rb_train)
    write_json("rm_rb_val.json", rb_val)
    write_json("rm_mix_train.json", [p for p, _ in mix_train])
    write_json("rm_mix_val.json", mix_val)
    write_json("rm_mix_train_second_run.json",
          [mix_train[i][0] for i in np.random.RandomState(2).permutation(len(mix_train))])
    return rb_train, rb_val, mix_train, mix_val


def calibration_sets(evaluation, pku, rb_train, mix_train, mix_val):
    """The 600 calibration prompts (App. B.4): 150 UltraChat and 150 PKU-SafeRLHF training prompts of RM-HH and
    RM-Div, and 150 of each held out from every training set. Returns the four splits of 150."""
    seen = {p for p, _ in mix_train} | set(mix_val) | set(rb_train)
    splits = {"train_benign": [p for p, s in mix_train if s == "benign"][:150],
              "train_harm": [p for p, s in mix_train if s == "harm"][:150],
              "heldout_benign": [p for p in evaluation if p not in seen][:150],
              "heldout_harm": sorted(set(pku) - seen)}
    random.Random(0).shuffle(splits["heldout_harm"])
    splits["heldout_harm"] = splits["heldout_harm"][:150]
    assert all(len(s) == 150 for s in splits.values())
    write_json("calibration_600.json", sum(splits.values(), []))
    write_json("calibration_300.json", splits["train_benign"] + splits["train_harm"])
    write_json("not_train_benign.json", splits["train_harm"] + splits["heldout_benign"] + splits["heldout_harm"])
    write_json("heldout_splits.json", splits["heldout_benign"] + splits["heldout_harm"])
    return splits


def eval_sets(evaluation, rb_train, rb_val, mix_train, mix_val, splits):
    """The RM bases' evaluation prompts: chat-formatted in at most 1,024 tokens and in no training set (texts compared
    after strip()). RM-RB: the first 300 held-out UltraChat prompts. RM-HH, RM-Div, RM-Sky: benign = the first 50
    RM-RB validation prompts, then the held-out UltraChat calibration prompts; red-team = the held-out PKU-SafeRLHF
    calibration prompts, also not in a validation set."""
    tok = load_tokenizer()

    def fits(p):
        return format_prompt(tok, p, 10 ** 9).shape[1] <= 1024

    trained = {p.strip() for p in rb_train + [p for p, _ in mix_train]}
    used = trained | {p.strip() for p in rb_val + mix_val}
    write_json("rb_eval.json", [p for p in evaluation if fits(p) and p.strip() not in used][:300])
    hh_eval, kept = [], set()
    for candidates, excluded in ((rb_val[:50] + splits["heldout_benign"], trained), (splits["heldout_harm"], used)):
        for p in candidates:
            if fits(p) and p.strip() not in kept and p.strip() not in excluded:
                hh_eval.append(p)
                kept.add(p.strip())
    write_json("hh_eval.json", hh_eval)


def p_grpo_sets(training):
    """P-GRPO's training prompts, as verl's parquet files: 2,000 of the 4,500 training prompts (64 more fill verl's
    validation file, which verl decodes once before training and not again, as test_freq is -1). The paper's 2,000
    were the prompts of the first 2,000 rows of the P-DPO length-pair set after a seed-42 shuffle, with repeats (Table
    4 counts the 1,792 rows its 28 steps trained on: 1,518 distinct prompts); that set needs the P-DPO responses, which
    are not included, so a seed-42 draw of distinct prompts stands in for it."""
    train = random.Random(42).sample(training, 2064)
    for split, prompts in (("train", train[:2000]), ("val", train[2000:])):
        Dataset.from_list([{"data_source": "ultrachat", "prompt": [{"role": "user", "content": p}], "ability": "style",
                            "reward_model": {"style": "rule", "ground_truth": ""}, "extra_info": {"index": i}}
                           for i, p in enumerate(prompts)]).to_parquet(f"p_grpo_{split}.parquet")


def main():
    ultrachat, pku = load_ultrachat(), load_pku()
    evaluation, training = ultrachat[:500], ultrachat[500:]
    write_json("ultrachat.json", ultrachat)
    programmatic_sets(evaluation)
    rb_train, rb_val, mix_train, mix_val = rm_training_sets(training, pku)
    splits = calibration_sets(evaluation, pku, rb_train, mix_train, mix_val)
    eval_sets(evaluation, rb_train, rb_val, mix_train, mix_val, splits)
    p_grpo_sets(training)


if __name__ == "__main__":
    main()
