"""Train CIFAR-100 ResNets with the He et al. (2016) CIFAR recipe and save val/test logits.

Usage:
    uv run python src/train.py --config configs/resnet110_main.yaml
    uv run python src/train.py --config configs/smoke.yaml          # CPU smoke test

The config defines a grid (depths x weight_decays x seeds); every run writes
results/logits/resnet<depth>_wd<wd>_seed<seed>.npz with val/test logits and labels.
Finished runs are skipped, interrupted runs resume from results/checkpoints/.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from data import load_cifar100, load_or_create_split
from resnet_cifar import CifarResNet

ROOT = Path(__file__).resolve().parents[1]


def deep_update(base, new):
    out = copy.deepcopy(base)
    for k, v in new.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if "defaults" in cfg:
        base = load_config(path.parent / cfg.pop("defaults"))
        cfg = deep_update(base, cfg)
    return cfg


def run_name(depth, wd, seed):
    return f"resnet{depth}_wd{wd:.0e}_seed{seed}"


class GpuLoader:
    """Whole dataset on the device; pad-4 random crop + horizontal flip done per batch."""

    def __init__(self, x_uint8, y, mean, std, device, batch_size, seed):
        self.x = torch.from_numpy(x_uint8).permute(0, 3, 1, 2).contiguous().to(device)
        self.y = torch.from_numpy(y).to(device)
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.device, self.bs = device, batch_size
        self.gen = torch.Generator().manual_seed(seed)
        self.steps_per_epoch = len(self.y) // batch_size  # drop the last partial batch

    def epoch(self):
        perm = torch.randperm(len(self.y), generator=self.gen)
        for i in range(self.steps_per_epoch):
            idx = perm[i * self.bs:(i + 1) * self.bs].to(self.device)
            yield self.augment(self.x[idx]), self.y[idx]

    def augment(self, xb):
        b = xb.shape[0]
        xb = (xb.float() / 255.0 - self.mean) / self.std
        xp = F.pad(xb, (4, 4, 4, 4))  # zero padding after normalization
        oy = torch.randint(0, 9, (b,), generator=self.gen).to(self.device)
        ox = torch.randint(0, 9, (b,), generator=self.gen).to(self.device)
        flip = (torch.rand(b, generator=self.gen) < 0.5).to(self.device)
        ar = torch.arange(32, device=self.device)
        rows = (oy[:, None] + ar)[:, None, :, None]          # b,1,32,1
        cols = (ox[:, None] + ar)[:, None, None, :]          # b,1,1,32
        cols = torch.where(flip[:, None, None, None], ox[:, None, None, None] + 31 - ar, cols)
        bi = torch.arange(b, device=self.device)[:, None, None, None]
        ci = torch.arange(3, device=self.device)[None, :, None, None]
        return xp[bi, ci, rows, cols]


def lr_at(it, cfg, depth):
    t = cfg["train"]
    if it < t.get("warmup_iters", 0) and depth >= t.get("warmup_min_depth", 10**9):
        return t["warmup_lr"]
    return t["lr"] * 0.1 ** sum(it >= m for m in t["milestones"])


@torch.no_grad()
def predict_logits(model, x_uint8, mean, std, device, bs=1000):
    model.eval()
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(x_uint8), bs):
        xb = torch.from_numpy(x_uint8[i:i + bs]).permute(0, 3, 1, 2).to(device).float() / 255.0
        out.append(model((xb - mean_t) / std_t).float().cpu())
    return torch.cat(out).numpy()


def train_one(cfg, depth, wd, seed, arrays, device, out_dirs, overwrite=False):
    name = run_name(depth, wd, seed)
    logits_path = out_dirs["logits"] / f"{name}.npz"
    if logits_path.exists() and not overwrite:
        print(f"[skip] {name} (logits exist)")
        return
    x_tr, y_tr, x_val, y_val, x_te, y_te = arrays
    t = cfg["train"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean = (x_tr / 255.0).mean(axis=(0, 1, 2)).tolist()
    std = (x_tr / 255.0).std(axis=(0, 1, 2)).tolist()

    model = CifarResNet(depth, num_classes=100).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=t["momentum"], weight_decay=wd)
    use_amp = bool(t.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loader = GpuLoader(x_tr, y_tr, mean, std, device, t["batch_size"], seed)

    total = t["iterations"]
    ckpt_path = out_dirs["ckpt"] / f"{name}_last.pt"
    it, elapsed_prev = 0, 0.0
    if ckpt_path.exists() and not overwrite:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        loader.gen.set_state(ck["gen"])
        it, elapsed_prev = ck["it"], ck["elapsed"]
        print(f"[resume] {name} from iteration {it}")

    print(f"[train] {name}: {total} iterations, {loader.steps_per_epoch} steps/epoch, "
          f"~{total / loader.steps_per_epoch:.0f} epochs, device={device}, amp={use_amp}")
    t0 = time.time()
    run_loss, run_correct, run_n = 0.0, 0, 0
    model.train()
    while it < total:
        for xb, yb in loader.epoch():
            if it >= total:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(it, cfg, depth)
            xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(xb)
                loss = F.cross_entropy(out, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            it += 1
            run_loss += loss.item() * len(yb)
            run_correct += (out.argmax(1) == yb).sum().item()
            run_n += len(yb)
            if it % t.get("log_every", 1000) == 0 or it == total:
                el = elapsed_prev + time.time() - t0
                print(f"  it {it:6d}/{total} lr {opt.param_groups[0]['lr']:.4f} "
                      f"loss {run_loss / run_n:.4f} train_err {1 - run_correct / run_n:.4f} "
                      f"elapsed {el / 60:.1f} min", flush=True)
                run_loss, run_correct, run_n = 0.0, 0, 0
            if t.get("ckpt_every") and it % t["ckpt_every"] == 0 and it < total:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "scaler": scaler.state_dict(), "gen": loader.gen.get_state(),
                            "it": it, "elapsed": elapsed_prev + time.time() - t0}, ckpt_path)
    train_time = elapsed_prev + time.time() - t0

    # Final-iteration model (no early stopping, no checkpoint selection).
    val_logits = predict_logits(model, x_val, mean, std, device)
    test_logits = predict_logits(model, x_te, mean, std, device)
    meta = {"depth": depth, "weight_decay": wd, "seed": seed, "iterations": total,
            "train_time_s": round(train_time, 1), "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "amp": use_amp, "torch": torch.__version__, "n_train": len(y_tr),
            "n_val": len(y_val), "n_test": len(y_te), "config": cfg.get("name", "")}
    np.savez_compressed(logits_path, val_logits=val_logits, val_labels=y_val,
                        test_logits=test_logits, test_labels=y_te, meta=json.dumps(meta))
    torch.save(model.state_dict(), out_dirs["ckpt"] / f"{name}_final.pt")
    if ckpt_path.exists():
        ckpt_path.unlink()
    test_err = float((test_logits.argmax(1) != y_te).mean())
    print(f"[done] {name}: test error {test_err:.4f}, train time {train_time / 60:.1f} min "
          f"-> {logits_path.relative_to(ROOT) if logits_path.is_relative_to(ROOT) else logits_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--depth", type=int, help="override: run only this depth")
    ap.add_argument("--wd", type=float, help="override: run only this weight decay")
    ap.add_argument("--seed", type=int, help="override: run only this seed")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    torch.backends.cudnn.benchmark = True

    d = cfg["data"]
    x_train_all, y_train_all, x_te, y_te = load_cifar100(ROOT / d.get("data_dir", "data"))
    tr_idx, val_idx = load_or_create_split(ROOT / d["val_split"], y_train_all, seed=d["split_seed"])
    x_tr, y_tr = x_train_all[tr_idx], y_train_all[tr_idx]
    x_val, y_val = x_train_all[val_idx], y_train_all[val_idx]
    if d.get("subset_train"):  # smoke test only
        rng = np.random.RandomState(0)
        sel = rng.choice(len(y_tr), d["subset_train"], replace=False)
        x_tr, y_tr = x_tr[sel], y_tr[sel]
    if d.get("subset_eval"):
        n = d["subset_eval"]
        x_val, y_val, x_te, y_te = x_val[:n], y_val[:n], x_te[:n], y_te[:n]

    t = cfg["train"]
    if "epochs" in t:  # convert an epoch budget to iterations (used by the smoke test)
        steps = len(y_tr) // t["batch_size"]
        t["iterations"] = t["epochs"] * steps
        t["milestones"] = [math.ceil(f * t["iterations"]) for f in t.get("milestone_fracs", [0.5, 0.75])]

    out = cfg.get("output", {})
    out_dirs = {"logits": ROOT / out.get("logits_dir", "results/logits"),
                "ckpt": ROOT / out.get("checkpoint_dir", "results/checkpoints")}
    for p in out_dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    depths = [args.depth] if args.depth else cfg["depths"]
    wds = [args.wd] if args.wd is not None else cfg["weight_decays"]
    seeds = [args.seed] if args.seed is not None else cfg["seeds"]
    arrays = (x_tr, y_tr, x_val, y_val, x_te, y_te)
    print(f"config={cfg.get('name')} train={len(y_tr)} val={len(y_val)} test={len(y_te)} "
          f"runs={len(depths) * len(wds) * len(seeds)}")
    for depth in depths:
        for wd in wds:
            for seed in seeds:
                train_one(cfg, depth, float(wd), seed, arrays, device, out_dirs, args.overwrite)


if __name__ == "__main__":
    main()
