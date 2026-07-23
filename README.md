# PyTorch CUDA Environment Check

[![CI](https://github.com/chshdh/pytorch-cuda-check/actions/workflows/ci.yml/badge.svg)](https://github.com/chshdh/pytorch-cuda-check/actions/workflows/ci.yml)

A reproducible PyTorch and CUDA development environment for WSL2, VS Code Remote SSH, and NVIDIA GPUs.

## Verified Environment

- Ubuntu 26.04 LTS on WSL2
- Python 3.12
- PyTorch 2.13 with CUDA 12.6
- NVIDIA GeForce RTX 2060
- `uv` for Python and dependency management
- Ruff for linting and formatting
- pytest for automated tests
- Jupyter for interactive GPU experiments

## Features

- CUDA availability verification
- GPU matrix multiplication test
- Reproducible dependencies with `uv.lock`
- VS Code Python and pytest integration
- Ruff formatting and linting
- Pre-commit quality checks
- Jupyter notebook GPU example
- Lightweight GitHub Actions CI

## Setup

Clone the repository:

```bash
git clone git@github.com:chshdh/pytorch-cuda-check.git
cd pytorch-cuda-check
```

Install the locked environment:

```bash
uv sync --locked
```

## Verify CUDA

Run the standalone GPU check:

```bash
uv run python gpu_check.py
```

Expected output includes:

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 2060
Result device: cuda:0
```

## Run Tests

```bash
uv run pytest -v
```

The CUDA test performs matrix multiplication directly on the GPU. On systems without CUDA, that test is skipped instead of failing.

## Code Quality

```bash
uv run ruff check .
uv run ruff format --check .
uv run pre-commit run --all-files
```

## Jupyter Notebook

Open `gpu_notebook.ipynb` in VS Code and select:

```text
.venv/bin/python
```

The notebook verifies that PyTorch tensors and matrix operations run on `cuda:0`.

## CI Strategy

GitHub-hosted runners do not provide an NVIDIA GPU. The cloud CI workflow therefore checks:

- dependency lockfile consistency
- Ruff linting
- Ruff formatting

CUDA execution is tested locally on the WSL2 development machine with the RTX 2060. A self-hosted GPU runner can be added later for automated CUDA CI.
