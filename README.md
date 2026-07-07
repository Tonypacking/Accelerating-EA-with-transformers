# Accelerating evolutionary algorithms with transformers

Master's thesis project: a version of **CMA-ES** that uses **transformers as
surrogate models**, benchmarked with the
[COCO platform](https://coco-platform.org/) ([numbbo/coco](https://github.com/numbbo/coco), INRIA Saclay).

Background notes: [docs/transformers-in-optimization.md](docs/transformers-in-optimization.md)

## Setup (local, WSL2/Linux)

Requires Python ≥ 3.8 (developed with 3.12).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

The extra index installs the small **CPU-only PyTorch** build locally.
Heavy experiments run on **MetaCentrum** — there install the CUDA build instead:

```bash
pip install -r requirements.txt   # default PyPI torch = CUDA build
```

Keep model code device-agnostic (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`).

## COCO quick check

Download the official example experiments (template for our own):

```bash
python -m cocoex.download_example   # run inside experiments/
```

Minimal smoke test of the whole toolchain:

```bash
python experiments/smoke_test.py    # tiny random-search run on bbob + cocopp post-processing
```

Experiment output goes to `exdata/`, post-processing to `ppdata/` (both gitignored).

## Repository layout

```
src/           # CMA-ES + transformer surrogate implementation (to be done)
experiments/   # COCO experiment scripts
docs/          # notes and literature
results/       # experiment outputs (gitignored)
```
