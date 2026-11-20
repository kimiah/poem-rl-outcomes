<div align="center">
<h1>PoEM: Predicting RL Outcomes from Existing Policies</h1>

#### [Paper](https://arxiv.org/abs/2609.30226) | [BibTeX](#citation)

[**Kimia Hamidieh**](https://kimiah.github.io),
[**Giannis Daras**](https://giannisdaras.github.io),
[**Antonio Torralba**](https://groups.csail.mit.edu/vision/torralbalab/)

MIT CSAIL, NeurIPS 2026

</div>

**TL;DR** Given a new reward, PoEM predicts the policy that RL on that reward would produce, without running RL. It
composes LoRA experts that were already trained with RL on other rewards from the same reference model.

## Overview

Given a target reward $r_{\text{tar}}$, PoEM predicts the policy $\pi_{\text{tar}}$ that RL on $r_{\text{tar}}$ would
produce, without running RL. It reuses *experts*: LoRA policies $\pi_k$ already trained with RL on basis rewards $r_k$,
all from one reference model $\pi_{\text{ref}}$. When $r_{\text{tar}} = \sum_k \alpha_k r_k$ and every expert is the
KL-regularized optimum of its reward, the optimum for $r_{\text{tar}}$ is a product of experts:

$$
\pi^{\star}_{\text{tar}} \propto \pi_{\text{ref}}^{1 - \sum_k \alpha_k} \prod_k \pi_k^{\alpha_k}.
$$

PoEM estimates the weights from a few scored samples and decodes this product:

1. **Calibrate.** Sample a few responses from $\pi_{\text{ref}}$ and score them with $r_{\text{tar}}$.
2. **Fit.** Regress $r_{\text{tar}}$ on the experts' log-ratios
   $\phi_k(x, y) = \log \pi_k(y \mid x) - \log \pi_{\text{ref}}(y \mid x)$ to get weights $\hat{\alpha}$, and score how
   well the experts cover $r_{\text{tar}}$.
3. **Decode.** Sample token by token from the logits
   $\gamma \sum_k \hat\alpha_k \ell_k + \bigl(1 - \gamma \sum_k \hat\alpha_k\bigr) \ell_{\text{ref}}$, where
   $\ell_k$ and $\ell_{\text{ref}}$ are the experts' and the reference model's logits and $\gamma$ is the composition
   strength.

<details>
<summary><b>Abstract</b></summary>

Foundation models are post-trained with reinforcement learning (RL) to maximize specific rewards, such as helpfulness,
correctness, or instruction following. This post-training process is computationally intensive, sometimes unstable,
and has to be run from scratch every time the reward model changes or when we want to combine multiple rewards. We
hence ask: given a new reward function, *is it possible to predict the policy RL would produce without actually
training on that reward*? We answer this in the affirmative by introducing PoEM, a framework for predicting RL
outcomes using models previously trained on other rewards. First, we build on the fact that if the new reward function
can be written as a linear combination of existing ones, then the optimal policy can be expressed as a combination of
the existing policies in log-probability space. Surprisingly, even in cases where the rewards are *not* linearly
connected, we observe that policies learned through RL often span an approximately low-rank subspace across rewards.
To our benefit, the weights for this combination can be estimated using only the reward or basis policy outputs on the
samples. We turn these observations into an algorithm that takes post-trained models and a target reward function, and
approximates the target RL policy without any additional RL training. We experimentally evaluate our approach across
programmatic and learned rewards for open-ended generation, with additional experiments on math reasoning, tool
calling, and image generation.

</details>

## Setup

Requirements:

- Python 3.11 on x86-64 Linux
- One NVIDIA GPU with a driver for CUDA 12.8 or newer

```bash
git clone https://github.com/kimiah/poem-rl-outcomes.git && cd poem-rl-outcomes
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .                  # torch, vLLM and the rest: about 11 GB, 10 minutes
```

$\pi_{\text{ref}}$ defaults to `Qwen/Qwen3-0.6B` (`--reference-model` sets another); it downloads from the Hugging Face
Hub on first use.

## Run

What you bring:

- Expert LoRA adapters of $\pi_{\text{ref}}$, one directory each.
- Your target reward, a Python file that defines `reward(prompt, response) -> float`.
- Prompts, as JSON lists of strings: `calib_prompts.json` and `eval_prompts.json` (or set `CALIB_PROMPTS` and
  `EVAL_PROMPTS`).
- To check the prediction: $\pi_{\text{tar}}$, an adapter trained with RL on $r_{\text{tar}}$.

The main entry point is:

```bash
EXPERTS="experts/lora_a experts/lora_b experts/lora_c" REWARD=my_reward.py TARGET=target/lora_t \
    bash examples/run_poem.sh
```

It runs these steps (`OUT`, the output directory, defaults to `poem_run`):

```bash
python -m poem calibrate --prompts calib_prompts.json --out $OUT/cal.jsonl
python -m poem fit --calibration $OUT/cal.jsonl --experts $EXPERTS --reward $REWARD --out $OUT/weights.json
python -m poem sample --prompts eval_prompts.json --weights $OUT/weights.json --reference --target-adapter $TARGET \
    --out-dir $OUT/D
python -m poem evaluate --dir $OUT/D --reward $REWARD --out $OUT/D/eval.json
python -m poem kl --dir $OUT/D --out $OUT/D/kl.json
```

Without `TARGET` it skips $\pi_{\text{tar}}$, the recovery and `kl.json`. Every
`python -m poem` step but `evaluate` runs on the GPU. Run it from the directory the adapter paths are relative to, as
`weights.json` keeps them as given.

## How the pipeline works

**Calibration** (`poem/calibration.py`). `calibrate` samples 4 responses per prompt from $\pi_{\text{ref}}$ at
temperature 1 (with the top-k of 20 of its generation config), up to 128 new tokens. Keep calibration prompts short:
the prompt in $\pi_{\text{ref}}$'s chat template is cut to its first 256 tokens, so a longer one is cut before
$\pi_{\text{ref}}$'s turn begins, and one of 511 or more tokens gets no responses.

**Weights** (`poem/policies.py`, `poem/weights.py`). `fit` regresses $r_{\text{tar}}$ on the log-ratios $\phi_k$ by
ridge regression with a penalty on negative weights, rescales $\hat{\alpha}$ to
$\sum_k \lvert \hat\alpha_k \rvert = 1$, and leaves experts with $\lvert \hat\alpha_k \rvert < 0.01$ out of the
decode. It also prints the *coverage*, how well the experts span $r_{\text{tar}}$: the held-out $R^2$ of a non-negative
fit, averaged over 20 splits of the prompts. The higher the coverage, the closer PoEM tends to get to RL. The paper
counts a reward as covered above 0.3, with the coverage computed on the experts' responses
([DETAILS.md](DETAILS.md#command-line-notes), "A new reward").

**Decoding** (`poem/sampling.py`, `poem/logits_processor.py`). `sample` decodes PoEM greedily in vLLM, up to 96 new
tokens, composing the logits at every token. $\gamma$ defaults to $\gamma_{\text{geom}}$, which needs no
$\pi_{\text{tar}}$; experts trained with PPO and the KL in the reward decode at $\gamma = 1$ (`GAMMA=1`).

**Evaluation** (`poem/metrics.py`). `evaluate` gives each policy's mean reward and its *recovery*:

$$
\text{recovery} = \frac{r(\pi_{\text{PoEM}}) - r(\pi_{\text{ref}})}{r(\pi_{\text{tar}}) - r(\pi_{\text{ref}})}.
$$

A recovery of 1 means PoEM gains as much reward as RL did. On a few prompts one outlier can swing it. `kl` gives
$\mathrm{KL}(\pi_{\text{tar}} \parallel \pi)$ per token along $\pi_{\text{tar}}$'s responses, and the *relative KL*,
PoEM's over $\pi_{\text{ref}}$'s: below 1, PoEM is closer to $\pi_{\text{tar}}$ than $\pi_{\text{ref}}$ is.

Every step is also a Python function ([DETAILS.md](DETAILS.md#the-python-flow)).

## Options

The main flags; `python -m poem <command> --help` lists all of them.

| command | option | default | |
|---|---|---|---|
| `calibrate` | `--n` | `4` | Responses per prompt |
| `fit` | `--estimator` | `ridge` | `ridge`, or `nnls` (non-negative, the `--topk 8` largest weights) |
| `sample` | `--gamma` | `geom` | Composition strengths: numbers, or `geom` for $\gamma_{\text{geom}}$ (1 with `--experts`); all in one decode |
| `sample` | `--experts`, `--alpha` | none | PoEM over these experts at these weights, in place of `--weights` |
| `sample` | `--reference`, `--target-adapter` | off | Also decode $\pi_{\text{ref}}$, and $\pi_{\text{tar}}$ from this adapter |
| `sample` | `--max-new-tokens`, `--temperature` | `96`, `0` | Decoding; temperature 0 is greedy |
| `sample` | `--vllm-default-kernels` | off | vLLM's default LoRA kernels, whose repeated decodes may differ |
| `fit`, `evaluate` | `--reward` | none | Your reward: a Python file that defines `reward(prompt, response)` |
| `fit`, `evaluate` | `--combined-reward`, `--heldout-reward` | none | A reward of the paper, in place of `--reward` |
| all | `--reference-model`, `--revision` | `Qwen/Qwen3-0.6B` | $\pi_{\text{ref}}$, at the paper's revision by default |

`fit --heldout-reward` fits with `nnls`.

## Examples

```
examples/run_poem.sh        the Run above
examples/fixed_weights.sh   PoEM at weights you choose, at several γ in one decode, with no calibration or fit
```

For example, PoEM at $\alpha = (0.7, 0.3)$ and $\gamma \in \lbrace 0.5, 1, 2 \rbrace$ (`fixed_weights.sh` needs
`TARGET`):

```bash
EXPERTS="experts/lora_a experts/lora_b" ALPHA="0.7 0.3" GAMMA="0.5 1 2" REWARD=my_reward.py TARGET=target/lora_t \
    bash examples/fixed_weights.sh
```

## Outputs

`examples/run_poem.sh` writes, under `OUT`:

```
cal.jsonl            calibration responses
weights.json         alpha_hat, the experts PoEM decodes and their weights (poem_experts, poem_weights), coverage,
                     gamma_geom
D/<policy>.jsonl     responses: reference (π_ref), alpha_hat_g<γ> (PoEM at γ), target (π_tar)
D/plan.json          every decoded policy: its experts, weights and γ
D/eval.json          each policy's reward (and with TARGET its recovery), and its reward per prompt
D/kl.json            with TARGET: each policy's KL(π_tar ‖ π) and relative KL, and its KL per prompt
```

`examples/fixed_weights.sh` writes the files of `D/` directly under its `OUT` (default `fixed_weights`), with PoEM at
$\gamma$ as `poem_g<γ>.jsonl`, $\gamma$ as a float (`poem_g1.0.jsonl`). [DETAILS.md](DETAILS.md#file-formats) lists
every field.

## Reproducing the paper

```bash
pip install -e ".[data]"
SETTINGS=programmatic OUT=runs ADAPTERS=adapters VERL_PYTHON=/path/to/verl-env/bin/python bash reproduce/run.sh
```

This builds the prompt sets from their public sources, trains the 61 P-GRPO adapters (20 experts, 32 $\pi_{\text{tar}}$
and 9 second RL runs) with verl (in its own environment, `VERL_PYTHON`), reruns the programmatic results on one 80 GB
GPU and prints the numbers behind the paper's tables and figures. Training alone takes about 46 hours, 45 minutes per
adapter. The retrained adapters are not the paper's, so expect its trends rather than its exact values. P-DPO runs too
when `ADAPTERS` holds its adapters. Without `SETTINGS`, the script also runs the reward-model bases, whose adapters are
not distributed (no data or model weights are); it stops before the first step when they are missing.

```
reproduce/run.sh            the 12 steps in order: calibration, fits, decodes and effective ranks on the programmatic
                            bases (P-GRPO, P-DPO) and the RM bases (RM-RB, RM-HH, RM-Div, RM-Sky)
reproduce/prompts.py        step 1: the prompt sets, from UltraChat and PKU-SafeRLHF-prompts
reproduce/train_p_grpo.sh   one P-GRPO run with verl, from the LoRA initialization the runs share
reproduce/ranks.py          the effective ranks
reproduce/summarize.py      step 12: prints the paper's numbers
```

[DETAILS.md](DETAILS.md) has the adapter layout, the training settings and the pinned environment.

## Citation

If you find our work useful, please cite our paper:

```bibtex
@inproceedings{hamidieh2026poem,
  title     = {PoEM: Predicting RL Outcomes from Existing Policies},
  author    = {Hamidieh, Kimia and Daras, Giannis and Torralba, Antonio},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {39},
  year      = {2026}
}
```

## License

MIT (`LICENSE`). Models and datasets keep their own licences.
