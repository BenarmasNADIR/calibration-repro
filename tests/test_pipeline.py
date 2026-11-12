import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

import analyze
from calibration_metrics import softmax
from data import make_stratified_split
from resnet_cifar import CifarResNet
from train import GpuLoader, lr_at, run_name


def test_stratified_split():
    labels = np.repeat(np.arange(100), 500)
    val = make_stratified_split(labels, seed=0)
    assert len(val) == 5000 == len(set(val))
    assert np.all(np.bincount(labels[val], minlength=100) == 50)
    assert val == make_stratified_split(labels, seed=0)


@pytest.mark.parametrize("depth,params", [(20, 0.278e6), (110, 1.737e6)])
def test_resnet_param_count(depth, params):
    m = CifarResNet(depth, num_classes=100)
    n = sum(p.numel() for p in m.parameters())
    assert abs(n - params) / params < 0.02  # He et al.: 0.27M / 1.7M (CIFAR-10 head)
    assert m(torch.zeros(2, 3, 32, 32)).shape == (2, 100)


def test_lr_schedule_with_warmup():
    cfg = {"train": {"lr": 0.1, "milestones": [32000, 48000], "warmup_iters": 400,
                     "warmup_lr": 0.01, "warmup_min_depth": 110}}
    assert lr_at(0, cfg, 110) == 0.01 and lr_at(399, cfg, 110) == 0.01
    assert lr_at(400, cfg, 110) == 0.1 and lr_at(0, cfg, 56) == 0.1
    assert lr_at(32000, cfg, 56) == pytest.approx(0.01)
    assert lr_at(63999, cfg, 56) == pytest.approx(0.001)


def test_gpu_loader_crop_flip_matches_naive():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, size=(8, 32, 32, 3), dtype=np.uint8)
    loader = GpuLoader(x, np.zeros(8, dtype=np.int64), [0, 0, 0], [1, 1, 1],
                       torch.device("cpu"), 8, seed=0)
    state = loader.gen.get_state()
    out = loader.augment(loader.x)
    loader.gen.set_state(state)
    oy = torch.randint(0, 9, (8,), generator=loader.gen)
    ox = torch.randint(0, 9, (8,), generator=loader.gen)
    flip = torch.rand(8, generator=loader.gen) < 0.5
    xp = torch.nn.functional.pad(loader.x.float() / 255.0, (4, 4, 4, 4))
    for i in range(8):
        ref = xp[i, :, oy[i]:oy[i] + 32, ox[i]:ox[i] + 32]
        if flip[i]:
            ref = ref.flip(-1)
        torch.testing.assert_close(out[i], ref)


def test_run_name():
    assert run_name(110, 1e-4, 2) == "resnet110_wd1e-04_seed2"


def _fake_run(path, depth, wd, seed, temp, n_val=1000, n_test=2000, k=100):
    rng = np.random.default_rng(seed + depth)
    z = rng.normal(0, 2.0, size=(n_val + n_test, k))
    y = (softmax(z).cumsum(1) > rng.random((n_val + n_test, 1))).argmax(1)
    z = z * temp  # overconfident by a known factor
    np.savez_compressed(path, val_logits=z[:n_val], val_labels=y[:n_val], test_logits=z[n_val:],
                        test_labels=y[n_val:], meta=json.dumps({"train_time_s": 60.0, "gpu": "fake"}))


def test_analyze_on_synthetic_logits(tmp_path, monkeypatch):
    logits = tmp_path / "logits"
    logits.mkdir()
    for s in [0, 1]:
        _fake_run(logits / f"{run_name(110, 1e-4, s)}.npz", 110, 1e-4, s, 2.5)
        _fake_run(logits / f"{run_name(20, 1e-4, s)}.npz", 20, 1e-4, s, 1.5)
        for wd in [1e-5, 1e-4, 1e-3]:
            _fake_run(logits / f"{run_name(56, wd, s)}.npz", 56, wd, s, 2.0)
    monkeypatch.setattr(sys, "argv", ["analyze.py", "--logits-dir", str(logits), "--out-dir", str(tmp_path)])
    analyze.main()
    main = pd.read_csv(tmp_path / "tables" / "main_resnet110.csv").set_index("method")
    assert (main.n_seeds == 2).all()
    assert main.loc["temp_scaling", "ece_mean"] < 0.3 * main.loc["uncalibrated", "ece_mean"]
    assert main.loc["temp_scaling", "temperature_mean"] == pytest.approx(2.5, rel=0.1)
    assert main.loc["temp_scaling", "error_mean"] == pytest.approx(main.loc["uncalibrated", "error_mean"])
    for f in ["per_run.csv", "wd_ablation.csv", "depth_ablation.csv", "main_resnet110.md"]:
        assert (tmp_path / "tables" / f).exists()
    for f in ["reliability_resnet110.png", "ece_vs_weight_decay.png", "ece_vs_depth.png"]:
        assert (tmp_path / "figures" / f).stat().st_size > 1000
