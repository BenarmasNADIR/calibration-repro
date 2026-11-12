"""CIFAR-100 loading and the fixed stratified 45k/5k train/validation split."""
import json
from pathlib import Path

import numpy as np

VAL_PER_CLASS = 50  # 100 classes x 50 = 5,000 validation images


def load_cifar100(data_dir="data"):
    """Return (x_train, y_train, x_test, y_test) as uint8 NHWC arrays and int64 labels."""
    from torchvision.datasets import CIFAR100

    tr = CIFAR100(data_dir, train=True, download=True)
    te = CIFAR100(data_dir, train=False, download=True)
    return (tr.data, np.asarray(tr.targets, dtype=np.int64),
            te.data, np.asarray(te.targets, dtype=np.int64))


def make_stratified_split(labels, per_class=VAL_PER_CLASS, seed=0):
    """Pick `per_class` validation indices per class with a fixed seed; return sorted indices."""
    rng = np.random.RandomState(seed)
    val = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        val.extend(rng.choice(idx, per_class, replace=False).tolist())
    return sorted(int(i) for i in val)


def load_or_create_split(path, train_labels, seed=0):
    path = Path(path)
    if path.exists():
        val = json.loads(path.read_text())["val_indices"]
    else:
        val = make_stratified_split(train_labels, seed=seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"seed": seed, "per_class": VAL_PER_CLASS, "n_val": len(val),
                                    "source": "CIFAR-100 training set (50,000 images)",
                                    "val_indices": val}))
    val = np.asarray(val, dtype=np.int64)
    mask = np.ones(len(train_labels), dtype=bool)
    mask[val] = False
    return np.flatnonzero(mask), val
