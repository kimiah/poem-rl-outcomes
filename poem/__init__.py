"""PoEM (Product of Experts Mixing): predict the policy that RL on a new target reward r_tar would produce by composing
policies already trained with RL from one reference model π_ref on other rewards (the basis; here LoRA experts).

The framework (Sec. 2) and the module of each step:
    calibration       1. a calibration set of responses, sampled from π_ref or from the experts
    policies          π_ref, the experts and their policy log-ratios φ_k (Eq. 5) on that set
    weights           2. the composition weights α̂ (Eq. 10), 3. the coverage
                      (Eq. 11), and the composition strength γ (Eqs. 20, 21)
    sampling          4. sampling PoEM(α) at γ with vLLM (Eq. 6); logits_processor composes the logits
    metrics           recovery, KL(π_tar ‖ π), relative KL and drift
    programmatic      the programmatic setting: the 20 rewards, the 32 combined rewards and the held-out rewards
    reward_models     the RM bases: their reward models, z-stats, combined rewards and the experts' rewards
`python -m poem calibrate | fit | sample | evaluate | kl` runs the steps on files (poem/__main__.py).

Importing poem before numpy sets OPENBLAS_CORETYPE=Haswell on CPUs with AVX2 (unless it is already set): numpy's
bundled OpenBLAS picks its kernels by CPU, and with the Haswell kernels the fits of α̂ give the same bits on AVX2 and
AVX-512 CPUs.
"""
import os
import sys

REFERENCE_MODEL = "Qwen/Qwen3-0.6B"                               # π_ref of the programmatic and RM bases
REFERENCE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"   # the version of π_ref the paper used

if "numpy" not in sys.modules:
    try:
        with open("/proc/cpuinfo") as f:
            if "avx2" in f.read():
                os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
    except OSError:
        pass
