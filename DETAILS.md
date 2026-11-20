# Details

What the [README](README.md) leaves out: how to reproduce the paper, how the release's numbers relate to the paper's
runs, the rest of the command line, the Python API, the file formats and the pinned environment.

## Reproducing the paper

The scripts in `reproduce/` rerun the paper's programmatic and reward-model results from the experts and targets.
`reproduce/run.sh` runs 12 steps in order; the scripts' comments name the paper items each step serves, and the last
step prints the numbers.

```bash
pip install -e ".[data]"
SETTINGS=programmatic OUT=runs ADAPTERS=adapters VERL_PYTHON=/path/to/verl-env/bin/python bash reproduce/run.sh
```

| variable        |                                                                                                                                               |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `OUT`         | The output directory; every step runs in it                                                                                                   |
| `ADAPTERS`    | The experts and targets, in the layouts below. The P-GRPO adapters it lacks are trained into it                                               |
| `SETTINGS`    | `programmatic`, `rm`, or both (the default). `rm` needs the RM-basis adapters in `ADAPTERS`; `run.sh` checks for them before step 1 |
| `VERL_PYTHON` | The python of the verl environment ([Training settings](#training-settings)); only P-GRPO training uses it                                     |

No data or model weights are distributed, the paper's experts and targets included. The recipe builds every prompt set
from its public source and trains the P-GRPO experts and targets; the P-DPO and RM bases need adapters trained with the
[Training settings](#training-settings). The retrained P-GRPO adapters start from a new LoRA initialization on stand-in
training prompts, so they are not the paper's: expect the paper's trends, not its exact values. `run.sh` keeps what
earlier runs finished (adapters, calibration sets and each target's results), so a rerun with the same `OUT` resumes.

### Prompt sets (step 1)

`prompts.py` writes the prompt sets of both settings. The UltraChat prompts are the first user turns of the first 5,000
`HuggingFaceH4/ultrachat_200k` `train_sft` rows whose first message is from the user (revision
`8049631c405ae6576f93f445c6b8166f76f5505a`). Prompts 0-499 are held out: on the programmatic bases they are the
calibration and evaluation pool (the first 30 are evaluated, and the weight and coverage fits skip the first 50, which
hold them); prompts 500-4999 are training prompts. The RM bases add `PKU-Alignment/PKU-SafeRLHF-prompts` (revision
`66465198fb8b9cbc6a7bdff124afaf35158d807c`). A comment states the rule of each set the script writes: the programmatic
pool, the RM-RB training prompts, the training mix of RM-HH, RM-Div and RM-Sky and its second-run order, the 600
calibration prompts of App. B.4, the evaluation prompts of each basis and the P-GRPO training prompts, which stand in
for the paper's set.

### Programmatic bases (`SETTINGS=programmatic`)

`ADAPTERS` holds:

- `p_grpo/lora_<r>` and `p_dpo/lora_<r>`: the 20 experts of each basis, `<r>` a key of
  `poem.programmatic.REWARD_FUNCTIONS`;
- `p_grpo_targets/lora_<c>` and `p_dpo_targets/lora_<c>`: $\pi_{\text{tar}}$ of each combined reward `<c>` of
  `COMBINED_REWARDS`;
- `p_grpo_second_run/lora_<c>` and `p_dpo_second_run/lora_<c>`, where a second RL run exists;
- `p_grpo_init`: the LoRA initialization the P-GRPO runs share (`train_p_grpo.sh` draws a new one when it is
  missing).

The steps (`programmatic_bases` in `run.sh`):

2. P-GRPO training with verl (`train_p_grpo.sh`, about 45 minutes per adapter on one 80 GB GPU, so about 46 hours for
   all 61: 20 experts, 32 targets and 9 second runs), for every P-GRPO adapter `ADAPTERS` does not hold. Each run
   needs about 5 GB under `OUT/verl` for its checkpoint, removed once its adapter is saved.
3. $\pi_{\text{ref}}$'s calibration set: 4 responses to each of the 500 pool prompts (a prompt of 511 or more tokens
   keeps none, so 452 prompts remain). Per basis, the experts' calibration set on the same prompts.
4. The 32 combined rewards of each basis: ridge fits on $\pi_{\text{ref}}$'s responses, then PoEM($\hat{\alpha}$) and
   PoEM($\alpha^\star$) at $\gamma_{\text{geom}}$ and at $\gamma \in \lbrace 0.5, 1, 1.5, 2, 3 \rbrace$, the top
   expert and the second RL run, decoded with $\pi_{\text{ref}}$ and $\pi_{\text{tar}}$ (greedy, 96 tokens); rewards,
   recovery and KL to $\pi_{\text{tar}}$.
5. The 20 held-out rewards of each basis: NNLS on the other experts' responses, $\gamma_{\text{geom}}$ from the Gram
   matrix of $\pi_{\text{ref}}$'s responses, PoEM($\hat{\alpha}$) at $\gamma_{\text{geom}}$ and on the $\gamma$ grid.
6. The effective ranks of Fig. 3 and of the P-DPO panel of Fig. 12, also saved in `geometry/`.

Step 12 prints Table 1's programmatic rows, Fig. 9's medians with 95 % bootstrap intervals, the
$\gamma = 1$ and $\gamma_{\text{geom}}$ recoveries of Sec. 4.1, the $\rho$ of Fig. 5 (a Spearman correlation
over the held-out rewards) and Table 7. The $\gamma$-grid medians of App. C.3.2,
Fig. 10 and the second run's KL against PoEM($\alpha^\star$) are in each target's `weights.json`,
`eval.json` and `kl.json`, with no printed line.

### RM bases (`SETTINGS=rm`)

`ADAPTERS` holds, with `<r>` a reward key of the table below and `<c>` a combined reward of
`poem.reward_models.COMBINED_REWARDS`:

- `rm_rb/lora_<r>` and `rm_rb_targets/lora_<c>`;
- `rm_hh/lora_<r>` and `rm_hh_targets/lora_<c>` (the safety expert is its step-100 checkpoint);
- `rm_div/lora_<r>` (SteamSHP: its step-100 checkpoint);
- `rm_sky/lora_<r>` and `second_run/lora_skywork_second_run`.

The steps (`rm_bases` in `run.sh`) each sample rollout calibration responses, score them with the basis's reward
models, fit with NNLS and decode PoEM($\hat{\alpha}$) (and PoEM($\alpha^\star$) and the top expert) at $\gamma = 1$
with 512 new tokens:

7. RM-RB: Table 1, Fig. 9, App. C.3.3.
8. RM-HH: Table 1, Fig. 9.
9. RM-Div: the recovery and relative KL of Fig. 5 and Sec. 4.3 (not their RM-Div coverage, which is
   within-expert coverage), App. C.3.3, the two recoveries of App. C.2.2, and Fig. 4.
10. RM-Sky: Fig. 4 and App. C.3.1.
11. The effective ranks of App. C.2.1, also saved in `geometry/`.

Step 12 prints the numbers. The RM bases take about 30 GPU-hours on one 80 GB GPU, most of them scoring the
calibration responses with the 7-8B reward models.

| key                     | model                                                | bases                 |
| ----------------------- | ---------------------------------------------------- | --------------------- |
| `skywork`             | `Skywork/Skywork-Reward-V2-Llama-3.1-8B`           | RM-RB, RM-Div, RM-Sky |
| `fsfairx`             | `sfairXC/FsfairX-LLaMA3-RM-v0.1`                   | RM-RB                 |
| `grm`                 | `Ray2333/GRM-Llama3-8B-rewardmodel-ft`             | RM-RB                 |
| `tulu3`               | `allenai/Llama-3.1-Tulu-3-8B-RM`                   | RM-RB                 |
| `armorm_helpfulness`  | `RLHFlow/ArmoRM-Llama3-8B-v0.1`, head 0            | RM-HH                 |
| `armorm_safety`       | `RLHFlow/ArmoRM-Llama3-8B-v0.1`, head 10           | RM-HH                 |
| `internlm2`           | `internlm/internlm2-7b-reward`                     | RM-Div                |
| `eurus`               | `openbmb/Eurus-RM-7b`                              | RM-Div                |
| `acemath`             | `nvidia/AceMath-7B-RM`                             | RM-Div                |
| `olmo2`               | `allenai/OLMo-2-1124-7B-RM`                        | RM-Div                |
| `deberta`             | `OpenAssistant/reward-model-deberta-v3-large-v2`   | RM-Div                |
| `oasst_pythia`        | `OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5` | RM-Div                |
| `steamshp`            | `stanfordnlp/SteamSHP-flan-t5-xl`                  | RM-Div                |
| `starling`            | `berkeley-nest/Starling-RM-7B-alpha`               | RM-Div                |
| `creative_writing`    | `SAA-Lab/Llama8B-CreativeWritingVerifier`          | RM-Div                |
| `skywork_qwen3_0.6b`  | `Skywork/Skywork-Reward-V2-Qwen3-0.6B`             | RM-Sky                |
| `skywork_llama3.2_1b` | `Skywork/Skywork-Reward-V2-Llama-3.2-1B`           | RM-Sky                |
| `skywork_qwen3_1.7b`  | `Skywork/Skywork-Reward-V2-Qwen3-1.7B`             | RM-Sky                |
| `skywork_llama3.2_3b` | `Skywork/Skywork-Reward-V2-Llama-3.2-3B`           | RM-Sky                |
| `skywork_qwen3_4b`    | `Skywork/Skywork-Reward-V2-Qwen3-4B`               | RM-Sky                |

Each reward model runs at the revision `poem.reward_models.REWARD_MODELS` pins. Starling's tokenizer is
`NousResearch/Llama-2-7b-chat-hf` at `351844e75ed0bcbbe3f10671b3c808d2b83894ee`. ArmoRM, internlm2-7b-reward and
Eurus-RM-7b run transformers remote code, at the pinned revisions. Starling's input format follows the reference code
on its model card, and its reward is read at the token before the first pad (unk) token.

The z-stats in `poem.reward_models.ZSTATS` are part of each target reward's definition:

- RM-RB: the mean and standard deviation of its four reward models on 2,000 prompts $\times$ 4 responses of the
  P-DPO pairs (the RM-RB $\pi_{\text{tar}}$ were trained with these values);
- RM-HH: over $\pi_{\text{ref}}$'s rollouts, 2 per training calibration prompt (300 prompts), in float32;
- RM-Div and RM-Sky: over $\pi_{\text{ref}}$'s rollouts, 4 per training calibration prompt.

### Training settings

The recipe trains P-GRPO (`train_p_grpo.sh`); for the other bases, the settings below are the paper's recipe
(App. B.2, Table 4).

|                | P-GRPO                                                                              | P-DPO                                                                                                                                                                      | RM bases                                                                                                                                                                               |
| -------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| algorithm      | GRPO, verl at commit `a4351480` (0.8.0.dev0), KL 0.04 in the loss (k3)            | DPO, $\beta = 0.1$                                                                                                                                                       | PPO, verl at `a4351480`, KL in the reward (k3), adaptive from $10^{-3}$ to target 0.1; GAE without discounting; a separate full-parameter critic (lr $10^{-5}$, 5 warm-up steps) |
| lr, batch      | $10^{-5}$; 64 prompts $\times$ 8 rollouts, 2 updates per step                   | $5 \times 10^{-5}$; 16 pairs                                                                                                                                             | $3 \times 10^{-6}$; 32 prompts $\times$ 8 rollouts, 4 updates per step                                                                                                             |
| length         | 1 epoch (28 steps in the paper's run)                                               | 6 epochs                                                                                                                                                                   | 232 steps                                                                                                                                                                              |
| prompts        | 2,000 training prompts, a stand-in for the paper's set (`prompts.py`); 512 + 512 tokens | the 4,500 training prompts, 4 pairs each; 512 tokens                                                                                                                       | 1,861 (RM-RB) or 1,979 (the mix); 1,024 + 512 tokens                                                                                                                                   |
| rollouts       | temperature 1.0, top-p 1.0                                                          | pairs of 8 responses per prompt (temperature 1.0, top-p 0.95, 256 tokens), ranked by the reward: best against worst, second against second worst, and so on              | temperature 0.7, top-p 0.8, top-k 20                                                                                                                                                   |
| expert reward  | `programmatic.verl_compute_score`                                                 | the reward                                                                                                                                                                 | raw on RM-RB and RM-HH; z-scored with `ZSTATS` on RM-Div and RM-Sky (Eurus after its fixed affine rescaling); $\pi_{\text{tar}}$: $\sum_r \alpha^\star_r z_r$                    |
| initialization | one shared                                                                          | one shared; the targets one per $k$ group (2-8, 10, 16)                                                                                                                  | one per run on RM-RB and RM-HH, one shared on RM-Div and RM-Sky                                                                                                                        |
| second RL run  | the same recipe with the prompts reordered (`data.seed=2`)                        | another seed                                                                                                                                                               | the RM-Div training prompts reordered (`rm_mix_train_second_run.json`)                                                                                                               |

All runs train LoRA adapters of rank 64 and `lora_alpha` 128 on the 7 projection matrices of `Qwen/Qwen3-0.6B`,
thinking off, with dropout 0.05 on the programmatic bases and none on the RM bases, and gradients clipped at 1.0 in GRPO
and PPO. The RM-Div experts were screened as in Table 4. The paper's shared initializations are not distributed;
`train_p_grpo.sh` draws a new one.

verl is not in `pyproject.toml`: verl requires numpy < 2.0 and the fits need numpy 2.2.6. Install verl at commit
`a4351480` with its vLLM extra (`pip install -e ".[vllm]"`), with flash-attn 2.8.3 and nltk 3.9.4, in a second
environment, and set `VERL_PYTHON` to that environment's python. The paper's verl environment had Python 3.10, torch
2.9.0, vLLM 0.12.0, transformers 4.57.6, PEFT 0.18.0, numpy 1.26.4, tensordict 0.10.0 and ray 2.55.1.

## Command line notes

What the README's options table leaves out; `python -m poem <command> --help` lists every flag.

- **A new reward, as in the paper.** On the programmatic bases, the paper fits a combination of the experts' rewards
  with ridge on $\pi_{\text{ref}}$'s responses (the default), and a new reward with `--estimator nnls` on the
  experts' responses: `calibrate --experts …`, then `fit --gram-calibration` with $\pi_{\text{ref}}$'s calibration
  file or its saved log-ratios, from which $\gamma_{\text{geom}}$ is computed. Its threshold of 0.3 is for the
  coverage on the experts' responses. The RM bases fit every target with NNLS on the experts' rollouts
  (`calibrate --sampler rollout --experts …`).
- **More policies in one decode.** `sample --arms alpha_hat alpha_star top_expert` adds PoEM over the true weights
  $\alpha^\star$ (when the weights JSON has them) and the top expert: the expert with the largest $\alpha^\star$, else
  the largest $\hat{\alpha}$. `--adapter DIR …` decodes other adapters alone (a second RL run, an expert); they are
  named after their directory, so give them distinct directory names.
- **At most 32 experts** per PoEM decode (`poem.sampling.MAX_LORAS`, the adapters vLLM schedules in one step).
- **Drift.** `evaluate --drift` adds the drift $D(\pi)$ of Eq. 21.
- **Another revision of $\pi_{\text{ref}}$.** With `--reference-model … --revision R`, pass both to `calibrate` and
  `fit` (a calibration file does not record $\pi_{\text{ref}}$), and `--revision R` to `sample` too: a weights JSON
  names $\pi_{\text{ref}}$'s model only.
- **One comparison per directory.** $\pi_{\text{ref}}$, $\pi_{\text{tar}}$ and PoEM must come from one `sample` call,
  as a bf16 forward pass depends on its batch. `evaluate` covers every `*.jsonl` of a directory, `kl` only the
  policies of its `plan.json`, and `sample` does not clear its `--out-dir`. Decode each comparison
  into a new directory (rerunning `run_poem.sh` with another reward, experts or $\gamma$ needs a new `OUT`).

## The Python flow

Every step is a function of the `poem` package, one module per step:

| module                                       | holds                                                                                                                                                |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `poem.calibration`                         | Calibrate: the $\pi_{\text{ref}}$ sampler, the round-robin expert sampler, the RM bases' rollout sampler                                           |
| `poem.policies`                            | $\pi_{\text{ref}}$, the experts (PEFT LoRA adapters), the log-ratios $\tilde{\Phi}$, the Gram matrix of the weight updates $\Delta\Theta$      |
| `poem.weights`                             | Fit: $\hat{\alpha}$ (ridge with a penalty on negative weights, or NNLS), the coverage, $\gamma_{\text{geom}}$, $\gamma^\star$, effective ranks |
| `poem.sampling`, `poem.logits_processor` | Decode: the vLLM decoder of Eq. 6                                                                                                                    |
| `poem.metrics`                             | Recovery, $\mathrm{KL}(\pi_{\text{tar}} \parallel \pi)$, relative KL, drift, bootstrap intervals                                                   |
| `poem.programmatic`                        | The programmatic setting: 20 rewards, their z-stats, 32 combined rewards, the verl reward function                                                   |
| `poem.reward_models`                       | The RM bases: 20 rewards from 19 reward models at pinned revisions, their z-stats, 11 combined rewards                                               |
| `poem.utils`                               | Helpers: the files below (JSON and JSONL, prompts, the log-ratio npz, your reward file), expert names, LoRA ranks     |

The log-ratios are centered within each prompt. Ridge rescales $\hat{\alpha}$ to
$\sum_k \lvert \hat\alpha_k \rvert = 1$ and NNLS to $\sum_k \hat\alpha_k = 1$, so $\gamma$ sets the overall
strength. Each of the coverage's 20 splits holds out 20 % of the prompts and fits NNLS, with every expert and no
rescaling, on the rest. $\gamma_{\text{geom}}$ scales the composed
log-ratio so that its norm on $\pi_{\text{ref}}$'s responses matches the root mean square of the experts' norms,
weighted by $\lvert \alpha_k \rvert$.

```python
import poem                      # first, before numpy (its BLAS kernel setting: poem/__init__.py)
import numpy as np
from poem.calibration import reference_calibration
from poem.metrics import kl_to_target, recovery, relative_kl
from poem.policies import policy_log_ratios
from poem.sampling import sample
from poem.weights import centered_gram, composition_weights, coverage, gamma_geom, poem_members

rows = reference_calibration(list(enumerate(calibration_prompts)))           # calibrate
prompts, responses = [r["prompt"] for r in rows], [r["response"] for r in rows]
pids = np.array([r["prompt_id"] for r in rows])
Phi = policy_log_ratios(experts, prompts, responses)                         # Phi, Eq. 5
t = np.array([r_tar(x, y) for x, y in zip(prompts, responses)])
alpha_hat, keep, w, _ = composition_weights(Phi, t, pids)                    # fit; decodes |alpha_hat_k| >= 0.01
cov = coverage(Phi, t, pids)                                                 # Eq. 11
gamma = gamma_geom(centered_gram(Phi, pids), w, keep)                        # Eq. 20
poem_policy = poem_members([experts[k] for k in keep], w, gamma)             # Eq. 6 at strength gamma
texts = sample({"reference": [(None, 1.0)], "target": [(target, 1.0)], "poem": poem_policy}, eval_prompts)   # decode

rewards = {name: [r_tar(x, y) for x, y in zip(eval_prompts, ys)] for name, ys in texts.items()}
rec = recovery(rewards["poem"], rewards["reference"], rewards["target"])
kl = kl_to_target(target, {"reference": ([], []), "poem": ([experts[k] for k in keep], [gamma * float(x) for x in w])},
                  eval_prompts, texts["target"])
rel = relative_kl(kl["poem"], kl["reference"])
```

`composition_weights(…, "nnls")` is the held-out estimator; `drift_target` and `gamma_star` give the drift-matched
$\gamma^\star$ of Eq. 21; `programmatic.reward_zstats` gives the z-stats of the 20 rewards on your own
responses; and `effective_rank`, `gram_effective_rank` and `policies.lora_update_gram` give the effective ranks of
App. B.4. Each function's docstring states its equation and arithmetic.

## File formats

| file                | format                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| prompts             | a `.json` list of strings, a `.jsonl` with `"prompt"` (and `"prompt_id"` or `"prompt_idx"`) per line, or text with one single-line prompt per line                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| calibration set     | JSONL, one response per line: `prompt_id`, `prompt`, `response`; `expert` on expert-sampled rows; `truncated` and `scores` ({reward: score}) on the RM bases' rows; `reward` when `fit` reads $r_{\text{tar}}$ from the file                                                                                                                                                                                                                                                                                                                                                                                  |
| experts             | PEFT LoRA directories of $\pi_{\text{ref}}$ named `lora_<name>`; `<name>` is the expert's reward in the paper's settings                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| log-ratios          | npz `{logratios (N, K), names, pair_idx, prompt_ids}`; row $n$ is line `pair_idx[n]` of its calibration set                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| weights JSON        | `reference_model`, `target`, `estimator`, `experts` (the columns fitted), `alpha_hat`; ridge: `alpha_hat_raw`, `lambda`, `lambda_neg`, `r2_cv` (the cross-validated $R^2$ of the chosen $\lambda$), `r2_heldout` (the ridge fit's $R^2$ on its held-out 20 % of prompts); NNLS: `topk`; `coverage`; `poem_experts`, `poem_weights` (the experts PoEM decodes with, and their $\hat{\alpha}$); `gamma_geom` (6 dp; `null` on the RM bases); `alpha_star` {`experts`, `weights`, `gamma_geom` (4 dp)} for a combined reward of the experts' rewards; `rm_basis` on the RM bases |
| sample directory    | `<policy>.jsonl` with one `{prompt, response}` line per prompt, named `reference`, `target`, `alpha_hat_g<gamma>`, `alpha_star_g<gamma>`, `top_expert`, `adapter_<directory name>`, `poem_g<gamma>` (`--experts`); `plan.json` {`reference_model`, `revision`, `target_adapter`, `policies`: {name: {`experts`, `weights`, `gamma`, `kind`}}}                                                                                                                                                                                                                                             |
| `evaluate` report | {`target`, `policies`: {name: {`reward`, `recovery` (`null` without `target.jsonl`), `per_prompt`, and with `--drift` `logp_ref`, `drift`}}}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `kl` report       | {`target_adapter`, `policies`: {name: {`gamma`, `kl`, `relative_kl`, `per_prompt`}}}; `per_prompt` is `null` where $\pi_{\text{tar}}$'s response is empty                                                                                                                                                                                                                                                                                                                                                                                                                                                     |

## Environment

`pyproject.toml` is the one environment file. It joins the paper's two environments: the decoding stack (torch 2.9.0,
vLLM 0.12.0, transformers 4.57.6) and the environment of the fits (numpy 2.2.6, scipy 1.17.1, PEFT 0.19.1), with nltk
3.9.4. The paper's KL ran under torch 2.6, transformers 5.6.2 and PEFT 0.19.1. The package needs Linux on x86-64 and
an NVIDIA driver for CUDA 12.8; every step that runs a model takes one GPU. Models and datasets download from the
Hugging Face Hub on first use. $\pi_{\text{ref}}$ is `Qwen/Qwen3-0.6B` at the revision the paper used
(`poem.REFERENCE_REVISION`), which the code passes whenever it loads $\pi_{\text{ref}}$.
`ensure_nltk` downloads the NLTK packages the POS-tag, stopword and sentence rewards read, and warns when one differs
from the paper's (md5).

The environment the release was checked in (`pip install -e .` on Python 3.11, then `pip freeze --exclude-editable`):

<details>
<summary>pip freeze</summary>

```text
accelerate==1.13.0
agent-detector==2.0.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.5
aiosignal==1.4.0
annotated-doc==0.0.6
annotated-types==0.8.0
anthropic==0.71.0
anyio==4.16.0
apache-tvm-ffi==0.1.16
astor==0.8.1
attrs==26.1.0
blake3==1.0.11
cachetools==8.0.0
cbor2==6.3.0
certifi==2026.10.23
charset-normalizer==3.5.2
click==8.5.1
cloudpickle==3.1.2
compressed-tensors==0.12.2
cuda-bindings==13.5.0
cuda-pathfinder==1.9.0
cupy-cuda12x==14.2.0
depyf==0.20.0
detect-installer==0.2.1
dill==0.4.1
diskcache==5.6.3
distro==1.9.0
dnspython==2.9.2
docstring_parser==0.18.0
einops==0.8.2
email-validator==2.3.0
fastapi==0.163.0
fastapi-cli==0.0.34
fastapi-cloud-cli==0.28.0
fastar==0.12.0
filelock==4.2.10
flashinfer-python==0.5.3
frozenlist==1.8.0
fsspec==2026.9.0
gguf==0.19.0
googleapis-common-protos==1.75.0
h11==0.16.0
hf-xet==1.8.0
httpcore==1.0.9
httpcore2==2.16.0
httptools==0.9.0
httpx==0.28.1
httpx2==2.16.0
huggingface_hub==0.36.2
idna==3.20
interegular==0.3.3
Jinja2==3.1.6
jiter==0.19.0
jmespath==1.1.0
joblib==1.6.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
lark==1.2.2
llguidance==1.3.0
llvmlite==0.44.0
lm-format-enforcer==0.11.3
loguru==0.8.0
markdown-it-py==4.3.0
MarkupSafe==3.0.4
mdurl==0.1.2
mistral_common==1.13.1
model-hosting-container-standards==0.1.16
mpmath==1.3.0
msgpack==1.2.4
msgspec==0.22.0
multidict==7.2.1
networkx==3.6.1
ninja==1.13.2
nltk==3.9.4
numba==0.61.2
numpy==2.2.6
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cudnn-frontend==1.32.3
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-cutlass-dsl==4.8.1
nvidia-cutlass-dsl-libs-base==4.8.1
nvidia-cutlass-dsl-libs-core==4.8.1
nvidia-cutlass-dsl-libs-cu12==4.8.1
nvidia-cutlass-dsl-libs-cu13==4.8.1
nvidia-ml-py==13.615.71
nvidia-nccl-cu12==2.27.5
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvshmem-cu12==3.3.20
nvidia-nvtx-cu12==12.8.90
openai==3.51.0
openai-harmony==0.0.8
opencv-python-headless==5.0.0.93
opentelemetry-api==1.45.2
opentelemetry-exporter-http-transport==0.66b2
opentelemetry-exporter-otlp-common==0.66b2
opentelemetry-exporter-otlp-proto-common==1.45.2
opentelemetry-exporter-otlp-proto-http==1.45.2
opentelemetry-proto==1.45.2
opentelemetry-sdk==1.45.2
opentelemetry-semantic-conventions==0.66b2
outlines_core==0.2.11
packaging==26.3
partial-json-parser==0.2.1.1.post7
peft==0.19.1
pillow==13.0.0
prometheus-fastapi-instrumentator==8.1.0
prometheus_client==0.26.0
propcache==0.5.4
protobuf==5.29.6
psutil==7.2.2
py-cpuinfo==9.0.0
pybase64==1.5.2
pycountry==26.2.16
pydantic==2.14.3
pydantic-extra-types==2.11.1
pydantic-settings==2.16.3
pydantic_core==2.50.4
Pygments==2.21.0
python-dotenv==1.2.4
python-json-logger==4.2.0
python-multipart==0.0.33
PyYAML==6.0.3
pyzmq==27.3.0
ray==2.60.0
referencing==0.37.0
regex==2026.11.9
requests==2.34.2
rich==15.0.0
rich-toolkit==0.20.6
rignore==0.8.1
rpds-py==2026.9.1
safetensors==0.7.0
scipy==1.17.1
sentencepiece==0.2.1
sentry-sdk==2.75.0
setproctitle==1.3.8
shellingham==1.5.4
sniffio==1.3.1
starlette==1.10.0
supervisor==4.3.0
sympy==1.14.0
tabulate==0.10.0
tiktoken==0.14.0
tokenizers==0.22.2
torch==2.9.0
torchaudio==2.9.0
torchvision==0.24.0
tqdm==4.70.1
transformers==4.57.6
triton==3.5.0
truststore==0.10.5
typer==0.29.1
typing-inspection==0.4.5
typing_extensions==4.16.0
urllib3==2.8.0
uvicorn==0.54.0
uvloop==0.23.0
vllm==0.12.0
watchfiles==1.3.0
websockets==17.4
xgrammar==0.1.27
yarl==1.25.1
```

</details>

`pip install -e ".[data]"`, which the recipe needs, adds datasets 4.8.5 and its dependencies (pandas, pyarrow,
multiprocess and xxhash) and moves fsspec down to 2026.2.0, the newest that datasets 4.8.5 allows. Only
`reproduce/prompts.py` uses them.
