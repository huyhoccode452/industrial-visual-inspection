import numpy as np
from sklearn.metrics import roc_auc_score


def image_auroc(labels, scores):
    """labels: (N,) 0/1 cho mỗi mẫu; scores: (N,) điểm bất thường của mỗi mẫu."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.shape != scores.shape:
        raise ValueError(f"labels {labels.shape} và scores {scores.shape} khác kích thước")
    if len(np.unique(labels)) < 2:
        raise ValueError("Cần có cả mẫu tốt và mẫu lỗi để tính AUROC")
    return float(roc_auc_score(labels, scores))


def pixel_auroc(masks, anomaly_maps):
    """masks: (N, H, W) 0/1; anomaly_maps: (N, H, W) điểm bất thường theo pixel."""
    return image_auroc(np.asarray(masks).ravel(), np.asarray(anomaly_maps).ravel())