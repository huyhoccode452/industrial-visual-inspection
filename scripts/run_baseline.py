import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.mvtec import CATEGORIES, build_loaders
from src.evaluation.metrics import image_auroc
from src.models.backbone import FeatureExtractor
from src.models.knn_baseline import extract_image_features, knn_scores


def main():
    parser = argparse.ArgumentParser(description="kNN baseline: Image AUROC trên MVTec AD")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data" / "datasets" / "mvtec_ad"))
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--k", type=int, default=1)
    args = parser.parse_args()

    extractor = FeatureExtractor()
    for cat in args.categories:
        train_l, _, test_l = build_loaders(args.data_root, cat, batch_size=8)
        train_f, _ = extract_image_features(train_l, extractor)
        test_f, test_y = extract_image_features(test_l, extractor)
        scores = knn_scores(train_f, test_f, k=args.k)
        auroc = image_auroc(test_y.numpy(), scores.numpy())
        print(f"{cat:<10} image AUROC = {auroc:.4f}")


if __name__ == "__main__":
    main()