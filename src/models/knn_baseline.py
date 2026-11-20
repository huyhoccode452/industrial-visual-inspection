import torch


@torch.no_grad()
def extract_image_features(loader, extractor):
    """Mỗi ảnh -> 1 vector (1024,). Trả về (features (N, 1024), labels (N,))."""
    feats, labels = [], []
    for image, mask, label, path in loader:
        f = extractor(image)["layer3"]        # (B, 1024, 14, 14)
        f = f.mean(dim=(2, 3))            # (B, 1024)
        feats.append(f)
        labels.append(label)
    return torch.cat(feats), torch.cat(labels)


def knn_scores(train_feats, test_feats, k=1):
    """Điểm bất thường = khoảng cách trung bình tới k ảnh train gần nhất."""
    d = torch.cdist(test_feats, train_feats)  # (N_test, N_train)
    return d.topk(k, dim=1, largest=False).values.mean(dim=1)