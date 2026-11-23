import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.mvtec import CATEGORIES, build_loaders
from src.evaluation.anomaly_map import upsample_and_blur
from src.evaluation.metrics import image_auroc, pixel_auroc
from src.models.backbone import FeatureExtractor
from src.models.padim import PaDiM


def evaluate_category(root, cat, rd, seed, device):
    train_l, _, test_l = build_loaders(root, cat, batch_size=8)
    model = PaDiM(FeatureExtractor(), rd=rd, seed=seed, device=device).fit(train_l)
    img_scores, labels, amaps, masks = [], [], [], []
    for image, mask, label, path in test_l:
        smaps = model.score_maps(image)                                  # (B, 28, 28)
        img_scores.append(smaps.flatten(1).max(dim=1).values.cpu())    # điểm ảnh
        amaps.append(upsample_and_blur(smaps).cpu())                     # (B, 224, 224)
        masks.append(mask.squeeze(1))                                  # (B, 224, 224)
        labels.append(label)
    labels, img_scores = torch.cat(labels), torch.cat(img_scores)
    masks, amaps = torch.cat(masks), torch.cat(amaps)
    return {
        "n_train": len(train_l.dataset),
        "n_test": len(labels),
        "image_auroc": image_auroc(labels.numpy(), img_scores.numpy()),
        "pixel_auroc": pixel_auroc(masks.numpy(), amaps.numpy()),
    }


def main():
    p = argparse.ArgumentParser(description="PaDiM mini: Image/Pixel AUROC trên MVTec AD")
    p.add_argument("--data-root", default=str(REPO_ROOT / "data" / "datasets" / "mvtec_ad"))
    p.add_argument("--categories", nargs="+", default=CATEGORIES)
    p.add_argument("--rd", type=int, default=550)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for cat in args.categories:
        results[cat] = evaluate_category(args.data_root, cat, args.rd, args.seed, device)
        print(f"{cat:<10} image {results[cat]['image_auroc']:.4f}  pixel {results[cat]['pixel_auroc']:.4f}")

    n = len(results)
    mean = {k: sum(r[k] for r in results.values()) / n for k in ("image_auroc", "pixel_auroc")}
    print(f"{'mean':<10} image {mean['image_auroc']:.4f}  pixel {mean['pixel_auroc']:.4f}")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    record = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "device": device,
        "torch": torch.__version__,
        "config": {"backbone": "wide_resnet50_2", "layers": ["layer2", "layer3"], "rd": args.rd,
                   "eps": 0.01, "seed": args.seed, "resize": 256, "crop": 224, "blur_sigma": 4.0},
        "results": results,
        "mean": mean,
    }
    out = Path(args.out) if args.out else REPO_ROOT / "experiments" / f"padim_rd{args.rd}_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("saved:", out)


if __name__ == "__main__":
    main()
