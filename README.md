# calibration-repro

Reproduction of **Guo, Pleiss, Sun & Weinberger (2017), "On Calibration of Modern Neural Networks"**, ICML 2017
([arXiv:1706.04599](https://arxiv.org/abs/1706.04599)) on CIFAR-100.

## Goal

1. Measure how miscalibrated a ResNet-110 trained on CIFAR-100 is (ECE, MCE, NLL, reliability diagrams) and how well
   **temperature scaling** fixes it (paper, Section 4.2 and Table 1).
2. Compare temperature scaling with **histogram binning** and **isotonic regression** (one-vs-all).
3. One Section 3 ablation: effect of **weight decay** (and, cheaply, **depth**) on calibration (paper, Figure 2).

## Status

| Part | Status |
|---|---|
| Metrics, temperature scaling, binning baselines, reliability diagrams | done, 21 tests pass (`uv run pytest -q`) |
| Training script + CPU smoke test (512 images, 2 epochs) | done, passes |
| Kaggle/Colab notebook | done (`notebooks/run_kaggle.ipynb`); job runner tested locally on the smoke config only |
| GPU training (17 runs) | **pending**: run `notebooks/run_kaggle.ipynb` |
| Results on CIFAR-100 | **pending**: run the notebook, then `uv run python src/analyze.py` |

## Repository layout

```
src/calibration_metrics.py   ECE (15 equal-width bins), MCE, NLL, stable softmax
src/temperature_scaling.py   fit T on validation logits: L-BFGS on log T, minimizing NLL
src/baselines.py             histogram binning (15 equal-width bins) and isotonic regression, one-vs-all
src/reliability_diagram.py   reliability diagrams in the paper's style
src/resnet_cifar.py          CIFAR ResNet (6n+2 layers, option-A shortcuts, He et al. 2016)
src/data.py                  CIFAR-100 loading, stratified 45k/5k split
src/train.py                 training + val/test logits -> results/logits/<model>_wd<wd>_seed<s>.npz
src/analyze.py               all metrics before/after calibration, mean ± std over seeds, tables + figures
src/demo_synthetic.py        sanity check on synthetic logits
configs/                     base recipe, experiment grids, smoke test, val_split_seed0.json (5,000 indices)
notebooks/run_kaggle.ipynb   one-click GPU training on Kaggle/Colab
tests/                       unit tests (metrics, calibrators, split, model, augmentation, analysis, smoke run)
```

## How to reproduce

Local setup (Windows/PowerShell or any OS with [uv](https://docs.astral.sh/uv/)):

```powershell
uv sync
uv run pytest -q
uv run python src/train.py --config configs/smoke.yaml --device cpu
uv run python src/demo_synthetic.py
```

GPU training (Kaggle: GPU T4 x2 or P100, Internet on, "Save & Run All"; or Colab): open
`notebooks/run_kaggle.ipynb`. It clones this repo, installs from `requirements.txt` only the training packages the
platform lacks, runs one part of the jobs (`PART` = 1-4, one job per GPU in parallel) and zips `results/logits/` to
`logits_part<PART>.zip`. Single runs can also be
launched directly:

```bash
python src/train.py --config configs/resnet110_main.yaml              # ResNet-110, wd 1e-4, seeds 0,1,2
python src/train.py --config configs/wd_ablation.yaml --wd 1e-5 --seed 0
python src/train.py --config configs/depth_ablation.yaml --depth 20 --seed 0
```

Then, locally, unzip the logits into `results/logits/` and run:

```powershell
uv run python src/analyze.py
```

This writes `results/tables/{per_run,main_resnet110,wd_ablation,depth_ablation,train_times}.csv`,
`results/tables/main_resnet110.md` and `results/figures/{reliability_resnet110,ece_vs_weight_decay,ece_vs_depth}.png`.

### Planned runs and expected GPU time (estimates, not measurements)

| Config | Runs | Estimated time per run (1x T4, AMP) |
|---|---|---|
| `resnet110_main.yaml`: ResNet-110, wd 1e-4, seeds 0-2 | 3 | ~3 h |
| `wd_ablation.yaml`: ResNet-56, wd in {1e-5, 1e-4, 5e-4, 1e-3, 5e-3}, seeds 0-1 | 10 | ~1.5 h |
| `depth_ablation.yaml`: ResNet-20/32 (56 and 110 reused), wd 1e-4, seeds 0-1 | 4 | ~0.6-0.9 h |

Total: about 27 GPU-hours (estimate), roughly 14 h of wall time on Kaggle's T4 x2, split into four parts
(one Kaggle session each, ~3-4.5 h estimated, so each stays under the 12-hour limit even if the estimates are 2x too low). Real times are stored in each `.npz` and written to `results/tables/train_times.csv` by `analyze.py`.

## Results

**Pending: run `notebooks/run_kaggle.ipynb`, then `uv run python src/analyze.py`.** No CIFAR-100 numbers are
reported yet.

### Comparison with the paper (CIFAR-100, ResNet-110)

Paper numbers come from Table 1 (ECE), S1 (MCE), S2 (error) and S3 (NLL) of arXiv:1706.04599v2.

| Method | Paper error (%) | Paper NLL | Paper ECE (%) | Paper MCE (%) | Ours |
|---|---|---|---|---|---|
| Uncalibrated | 27.83 | 1.4978 | 16.53 | 35.5 | pending |
| Temperature scaling | 27.83 | 1.0442 | 1.26 | 4.74 | pending |
| Histogram binning | 34.78 | 1.4379 | 2.66 | 7.03 | pending |
| Isotonic regression | 28.41 | 1.207 | 4.99 | 10.36 | pending |

Once the logits exist, `results/tables/main_resnet110.md` lists our mean ± std over 3 seeds next to the paper's ECE.
Differences will be reported as they are, without tuning anything to match.

### Pipeline sanity check on synthetic logits (not CIFAR-100)

`uv run python src/demo_synthetic.py`: K = 100 classes, 5,000 val / 10,000 test, labels drawn from softmax(z), the
"model" outputs z x 2.5 (overconfident), 3 seeds, mean ± std (`results/tables/synthetic_check.csv`):

| Method | Error (%) | ECE (%) | NLL |
|---|---|---|---|
| Uncalibrated | 74.22 | 38.83 ± 0.29 | 4.318 |
| Temperature scaling | 74.22 | 1.12 ± 0.27 | 2.999 |
| Histogram binning | 77.63 | 6.55 ± 0.60 | 5.837 |
| Isotonic regression | 74.66 | 6.60 ± 0.57 | 3.452 |

Fitted T = 2.497 ± 0.035 (true value 2.5). This only shows that the code behaves as intended: the synthetic
miscalibration is exactly a temperature, so temperature scaling is the correct model here by construction.

## Assumptions and deviations

The paper only says that each model follows its original training recipe. Everything below is our choice.

- **Training recipe** (He et al. 2016 CIFAR-10 recipe, applied to CIFAR-100): SGD, momentum 0.9, lr 0.1, batch 128,
  weight decay 1e-4, 64k iterations (about 182 epochs), lr / 10 at 32k and 48k, pad-4 random crop + horizontal
  flip, per-channel mean/std normalization (statistics of the 45k training split; He et al. used per-pixel mean
  subtraction). ResNet-110 uses lr 0.01 for the first 400 iterations (He et al. warm up until training error is
  below 80%, "about 400 iterations"; we use a fixed 400). Option-A (zero-padding) shortcuts, Kaiming-normal init.
- **Validation split**: 5,000 images from the CIFAR-100 training set, stratified (50 per class), `RandomState(0)`;
  indices in `configs/val_split_seed0.json`. The same split is used for every run and seed; the other 45,000 images
  are used for training. Calibrators are fit on validation logits only; the 10,000 test images are only evaluated.
- **Checkpoint**: final iteration, no early stopping, no model selection.
- **Weight-decay ablation uses ResNet-56**, not ResNet-110 as in Figure 2, to save GPU time. The paper's x-axis
  spans 1e-5 to 1e-2; ours is {1e-5, 1e-4, 5e-4, 1e-3, 5e-3}.
- **Depth ablation**: depths {20, 32, 56, 110} at wd 1e-4; the paper does not list its exact depths.
- **Histogram binning**: 15 equal-width bins per class (one-vs-all); empty bins predict the bin centre; outputs are
  renormalized across classes. **Isotonic regression**: scikit-learn, one-vs-all, clipped to [0, 1], renormalized.
  The NLL of the binning methods clips probabilities at 1e-12 (they can give the true class probability 0). Ties
  after binning are broken by `argmax` (first class), which can change the error.
- **Seeds**: 3 for ResNet-110, 2 per ablation point; mean ± sample std (ddof = 1).
- **Mixed precision** (fp16 autocast + GradScaler) for GPU training; logits are computed in fp32. cuDNN benchmark
  mode is on, so runs are not bit-exact, and a resumed run is not bit-identical to an uninterrupted one.
- **Data augmentation runs on the GPU** (dataset kept on the device; zero padding applied after normalization); the
  last partial batch of each epoch is dropped (351 steps per epoch).
- **Kaggle install**: training needs only torch, torchvision, numpy and pyyaml, which Kaggle and Colab preinstall,
  so the notebook installs from `requirements.txt` only the ones that are missing (installing every pin could replace
  the platform's numpy under its CUDA build of torch). The analysis runs locally with the pinned versions (`uv sync`).
  The lock file pins a CPU build of torch for local Windows use.
- BBQ, Platt scaling, vector and matrix scaling from Table 1 are not implemented.

## Limitations

- No CIFAR-100 results yet (GPU runs pending); the GPU times above are estimates.
- 2-3 seeds give only a rough idea of variance, and one validation split is used for all runs.
- The He et al. recipe was tuned on CIFAR-10. The paper's ResNet-110 reached 27.83% test error on CIFAR-100; ours
  may differ, which also changes the ECE.
- Only CIFAR-100 and ResNets; no other datasets or architectures from the paper.
