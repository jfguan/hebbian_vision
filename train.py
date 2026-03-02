#!/usr/bin/env python3
"""Train HebbianVision on CIFAR-100 with DeiT augmentation recipe."""

import argparse, json, math, os, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from model import Config, HebbianVision

# ---- config ----

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch-size",   type=int,   default=2048)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--n-layers",     type=int,   default=6)
    p.add_argument("--d-model",      type=int,   default=128)
    p.add_argument("--patch-size",   type=int,   default=4)
    p.add_argument("--memory-alpha", type=float, default=1.0)
    p.add_argument("--eval-every",   type=int,   default=10)
    p.add_argument("--tag",          type=str,   default="cifar100")
    p.add_argument("--compile",      action="store_true")
    p.add_argument("--resume",       type=str,   default=None)
    # [DeiT] augmentation flags
    p.add_argument("--no-deit",      action="store_true", help="disable DeiT augmentations")
    p.add_argument("--drop-path",    type=float, default=0.1)
    p.add_argument("--label-smooth", type=float, default=0.1)
    p.add_argument("--mixup-alpha",  type=float, default=0.8)
    p.add_argument("--cutmix-alpha", type=float, default=1.0)
    return p.parse_args()

# ---- data ----

CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD  = (0.2675, 0.2565, 0.2761)

def make_loaders(bs, deit=True):
    train_tf = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    if deit:
        train_tf.append(T.RandAugment(num_ops=2, magnitude=9))  # [DeiT] random augmentation
    train_tf += [T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)]
    if deit:
        train_tf.append(T.RandomErasing(p=0.25))                # [DeiT] random erasing
    train_tf = T.Compose(train_tf)
    test_tf  = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    kw = dict(root="./data", download=True)
    train = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100(train=True,  transform=train_tf, **kw),
        batch_size=bs, shuffle=True, pin_memory=True, drop_last=True)
    test = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100(train=False, transform=test_tf,  **kw),
        batch_size=bs, pin_memory=True)
    return train, test

# ---- [DeiT] mixup / cutmix ----

def rand_bbox(W, H, lam):
    """Random bbox for cutmix."""
    cut_rat = math.sqrt(1 - lam)
    cw, ch = int(W * cut_rat), int(H * cut_rat)
    cx = torch.randint(0, W, (1,)).item()
    cy = torch.randint(0, H, (1,)).item()
    x1, y1 = max(cx - cw // 2, 0), max(cy - ch // 2, 0)
    x2, y2 = min(cx + cw // 2, W), min(cy + ch // 2, H)
    return x1, y1, x2, y2

def mix_data(imgs, labels, mixup_alpha, cutmix_alpha):
    """[DeiT] Randomly apply mixup or cutmix each batch."""
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    use_cutmix = torch.rand(1).item() < 0.5

    if use_cutmix:
        lam = torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample().item()
        x1, y1, x2, y2 = rand_bbox(imgs.size(3), imgs.size(2), lam)
        imgs = imgs.clone()
        imgs[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
        lam = 1 - (x2 - x1) * (y2 - y1) / (imgs.size(2) * imgs.size(3))  # adjust for actual area
    else:
        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
        imgs = lam * imgs + (1 - lam) * imgs[idx]

    return imgs, labels, labels[idx], lam

# ---- training ----

def cosine_lr(step, warmup, total, lr_max, lr_min):
    if step < warmup:
        return lr_max * (step + 1) / warmup
    t = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(imgs, labels)
        loss_sum += loss.item() * imgs.size(0)
        correct  += (logits.argmax(-1) == labels).sum().item()
        n += imgs.size(0)
    model.train()
    return loss_sum / n, correct / n

def main():
    args = parse_args()
    deit = not args.no_deit

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_loader, test_loader = make_loaders(args.batch_size, deit=deit)

    cfg = Config(n_classes=100, img_size=32, patch_size=args.patch_size,
                 d_model=args.d_model, n_layers=args.n_layers,
                 memory_alpha=args.memory_alpha,
                 drop_path=args.drop_path if deit else 0.0)
    model = HebbianVision(cfg).to(device)
    if args.compile:
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    decay    = [p for p in model.parameters() if p.dim() >= 2]
    nodecay  = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay,   "weight_decay": 0.05},
        {"params": nodecay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95))

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)

    total_steps  = args.epochs * len(train_loader)
    warmup_steps = len(train_loader) * 5
    lr_min       = args.lr * 0.01

    aug_str = "DeiT(mixup+cutmix+randaug+erase+label_smooth+drop_path)" if deit else "basic(crop+flip)"
    print(f"{n_params/1e6:.2f}M params | {device} | B={args.batch_size} lr={args.lr}")
    print(f"aug: {aug_str}")
    os.makedirs("checkpoints", exist_ok=True)
    log_path = f"checkpoints/history_{args.tag}.jsonl"
    log_file = open(log_path, "a" if args.resume else "w")
    step = start_epoch * len(train_loader)
    best_acc = 0.0

    # ---- main loop ----

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        epoch_loss, nb = 0.0, 0

        for imgs, labels in train_loader:
            lr = cosine_lr(step, warmup_steps, total_steps, args.lr, lr_min)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            imgs, labels = imgs.to(device), labels.to(device)

            # [DeiT] mixup / cutmix
            if deit:
                imgs, labels_a, labels_b, lam = mix_data(imgs, labels, args.mixup_alpha, args.cutmix_alpha)
            else:
                labels_a, labels_b, lam = labels, labels, 1.0

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(imgs)
                # [DeiT] label smoothing via cross_entropy
                ls = args.label_smooth if deit else 0.0
                loss = lam * F.cross_entropy(logits, labels_a, label_smoothing=ls) \
                     + (1 - lam) * F.cross_entropy(logits, labels_b, label_smoothing=ls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            nb += 1
            step += 1

        epoch_loss /= nb
        dt = time.time() - t0
        entry = {"step": step, "epoch": epoch, "train_loss": epoch_loss}
        print(f"ep {epoch:3d} | loss {epoch_loss:.4f} | lr {lr:.2e} | {dt:.1f}s", flush=True)

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            vl, va = evaluate(model, test_loader, device)
            entry["val_loss"], entry["val_acc"] = vl, va
            print(f"         val {vl:.4f} | acc {va*100:.1f}%", flush=True)
            if va > best_acc:
                best_acc = va
                raw = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save({"model": raw.state_dict(), "config": cfg,
                            "epoch": epoch+1, "val_acc": va},
                           f"checkpoints/best_{args.tag}.pt")

        log_file.write(json.dumps(entry) + "\n"); log_file.flush()

    log_file.close()

    # ---- save + plot ----

    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({"model": raw.state_dict(), "optimizer": optimizer.state_dict(),
                "config": cfg, "epoch": args.epochs}, f"checkpoints/model_{args.tag}.pt")

    hist = [json.loads(l) for l in open(log_path)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist], alpha=0.7)
    vpts = [(h["epoch"], h["val_loss"]) for h in hist if "val_loss" in h]
    if vpts: a1.plot(*zip(*vpts), "o-", ms=3)
    a1.set(xlabel="epoch", ylabel="loss"); a1.grid(True, alpha=0.3); a1.legend(["train","val"])
    apts = [(h["epoch"], h["val_acc"]) for h in hist if "val_acc" in h]
    if apts: a2.plot(*zip(*apts), "o-", color="green", ms=3)
    a2.set(xlabel="epoch", ylabel="acc"); a2.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f"checkpoints/loss_{args.tag}.png", dpi=150); plt.close()

    print(f"best acc {best_acc*100:.1f}% | saved checkpoints/model_{args.tag}.pt")

if __name__ == "__main__":
    main()
