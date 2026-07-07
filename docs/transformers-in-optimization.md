# Where transformers can be applied in optimization

Notes for the thesis *Accelerating evolutionary algorithms with transformers*
(CMA-ES + transformer surrogate models, benchmarked with COCO).

## 1. Transformers as surrogate models — the thesis topic

In surrogate-assisted black-box optimization, the expensive objective function is
mostly replaced by a cheap regression model trained on already-evaluated points.
The established baseline for CMA-ES is Gaussian-process surrogates:

- **DTS-CMA-ES** — L. Bajer, Z. Pitra, J. Repický, M. Holeňa.
  *Gaussian process surrogate models for the CMA evolution strategy.*
  Evolutionary Computation 27:665–697, 2019.
  In each generation the GP ranks the sampled offspring; only the most promising
  or most uncertain points get real evaluations (doubly-trained scheme).

The thesis replaces the GP with a transformer. Relevant prior transformer work:

- **PFNs4BO** — S. Müller, M. Feurer, N. Hollmann, F. Hutter.
  *PFNs4BO: In-Context Learning for Bayesian Optimization.* ICML 2023.
  Prior-Fitted Networks: a transformer pre-trained offline on millions of
  synthetic functions sampled from a prior. At run time the evaluated points are
  fed in as the input sequence and the model predicts the posterior for new
  points **in-context, with no retraining** — a natural fit for CMA-ES, where
  each generation produces a fresh small training set.
  Code: https://github.com/automl/PFNs4BO

- **RTDK-BO** — A. Shmakov et al.
  *RTDK-BO: High Dimensional Bayesian Optimization with Reinforced Transformer
  Deep Kernels.* IEEE CASE 2023.
  Keeps the GP but learns its kernel with a transformer (deep kernel learning),
  targeting high-dimensional problems.

**Integration point in CMA-ES:** in every generation, CMA-ES samples λ offspring
from N(m, σ²C). The transformer surrogate predicts/ranks their fitness; only a
subset is evaluated on the true function (selection follows the DTS or lq-CMA-ES
evaluation-control schemes). This directly reduces the number of expensive
function evaluations — the quantity COCO measures.

## 2. Dynamic algorithm configuration (DAC)

A learned policy (potentially a transformer over the run history) adapts the
algorithm's parameters online — step-size σ, population size λ, restart
strategy of CMA-ES:

- S. Adriaensen, A. Biedenkapp, G. Shala, N. Awad, T. Eimer, et al.
  *Automated Dynamic Algorithm Configuration.* JAIR 75:1633–1699, 2022.

Here the transformer does not model the objective function; it controls the
optimizer. Complementary to surrogate modelling.

## 3. Learned optimizers / optimization as sequence modeling

The whole optimization trajectory (x, f(x) pairs) is treated as a sequence and a
transformer is trained to *continue* it, i.e. to propose the next query point —
e.g. **OptFormer** (Chen et al., NeurIPS 2022) trained on large hyperparameter-
tuning corpora. The transformer acts as the optimizer itself rather than as a
surrogate; useful context and a source of architecture/tokenization ideas.

## Toolchain used in this repo

| Component | Package |
|---|---|
| Benchmarking (BBOB testbed, INRIA Saclay) | `coco-experiment` (cocoex), `cocopp` |
| Reference CMA-ES | `cma` (pycma, N. Hansen) |
| Modular CMA-ES (assignment point 5) | `modcma` |
| Transformer surrogate | `torch` |
| Baseline surrogates for comparison | `scikit-learn` (GP, RF) |
| Analysis & plots | `pandas`, `matplotlib`, `seaborn` |
