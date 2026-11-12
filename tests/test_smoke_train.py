import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "data" / "cifar-100-python").exists(), reason="CIFAR-100 not downloaded")
def test_cpu_smoke_training(tmp_path):
    """512 training images, 2 epochs on CPU (configs/smoke.yaml); checks the saved logits file."""
    subprocess.run([sys.executable, str(ROOT / "src" / "train.py"), "--config", str(ROOT / "configs" / "smoke.yaml"),
                    "--device", "cpu", "--overwrite"], check=True, cwd=ROOT, capture_output=True)
    d = np.load(ROOT / "results" / "smoke" / "logits" / "resnet20_wd1e-04_seed0.npz")
    assert d["val_logits"].shape == (256, 100) and d["test_logits"].shape == (256, 100)
    assert np.isfinite(d["val_logits"]).all()
