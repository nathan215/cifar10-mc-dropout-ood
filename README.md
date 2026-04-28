# Deep Learning Project: CIFAR-10 MC Dropout

This repository trains and evaluates a CIFAR-10 `ResNet-18` with multiple MC Dropout configurations, then analyzes OOD behavior on CIFAR-10-C.

## Repository structure

- `train.py` - train one dropout mode and save best checkpoint
- `evaluate_ood.py` - run OOD evaluation on CIFAR-10-C from a checkpoint
- `analyze_paper.py` - aggregate metrics and generate report artifacts
- `models.py` - CIFAR ResNet-18 model and dropout mode definitions
- `dataset.py` - CIFAR-10 / CIFAR-10-C data loaders and helpers
- `metrics.py` - uncertainty and calibration metrics (MI, Brier, ACE)
- `runtime_utils.py` - shared runtime helpers (device, JSON/CSV writing)
- `Project.tex` - paper/report source
- `figures/`, `tables/`, `checkpoints/` - generated outputs (kept mostly untracked)

## Setup

```bash
pip install -r requirements.txt
```

## Quick usage

Train:

```bash
python train.py --mode all --epochs 100 --device cuda
```

Evaluate OOD:

```bash
python evaluate_ood.py --checkpoint checkpoints/resnet18_all_best.pt --device cuda
```

Generate analysis outputs:

```bash
python analyze_paper.py
```

## Notes

- Supported modes are defined in `models.py` (`none`, `all`, `first_block`, `last_layer_strict`, `last_block_plus_fc`).
- `evaluate_ood.py` expects `data/CIFAR-10-C.tar`.
