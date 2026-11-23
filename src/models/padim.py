import torch
import torch.nn.functional as F


def make_channel_index(c_total=1536, rd=100, seed=42):
    """Chọn ngẫu nhiên rd kênh trong c_total kênh (cố định theo seed)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(c_total, generator=g)[:rd]


def build_embedding(feats, idx):
    """feats: {"layer2": (B,512,28,28), "layer3": (B,1024,14,14)} -> (B, Rd, 28, 28)."""
    l2 = feats["layer2"]
    l3 = F.interpolate(feats["layer3"], size=l2.shape[-2:], mode="nearest")  # (B, 1024, 28, 28)
    emb = torch.cat([l2, l3], dim=1)      # (B, 1536, 28, 28)
    return emb[:, idx]                      # giữ lại rd kênh

def fit_gaussians(emb, eps=0.01):
    """emb: (N, C, H, W) -> mean (C, H*W), inv_cov (H*W, C, C)."""
    n, c, h, w = emb.shape
    x = emb.reshape(n, c, h * w)                       # (N, C, HW)
    mean = x.mean(dim=0)                             # (C, HW): trung bình qua N ảnh
    centered = x - mean
    cov = torch.einsum("nci,ndi->icd", centered, centered) / (n - 1)   # (HW, C, C)
    cov = cov + eps * torch.eye(c, device=emb.device)
    inv_cov = torch.linalg.inv(cov)
    return mean, inv_cov


def mahalanobis_map(emb, mean, inv_cov):
    """emb: (B, C, H, W) -> bản đồ điểm (B, H, W)."""
    b, c, h, w = emb.shape
    delta = emb.reshape(b, c, h * w) - mean            # (B, C, HW)
    d2 = torch.einsum("bci,icd,bdi->bi", delta, inv_cov, delta)
    return d2.clamp(min=0).sqrt().reshape(b, h, w)

class PaDiM:
    def __init__(self, extractor, rd=100, eps=0.01, seed=42, device = "cpu"):
        self.extractor = extractor.to(device)
        self.idx = make_channel_index(1536, rd, seed).to(device)
        self.eps = eps
        self.device = device

    @torch.no_grad()
    def _embed(self, images):
        return build_embedding(self.extractor(images.to(self.device)), self.idx)

    def fit(self, loader):
        embs = [self._embed(image) for image, mask, label, path in loader]
        self.mean, self.inv_cov = fit_gaussians(torch.cat(embs, dim=0), self.eps)
        return self

    @torch.no_grad()
    def score_maps(self, images):
        return mahalanobis_map(self._embed(images), self.mean, self.inv_cov)    # (B, 28, 28)

    def score_images(self, images):
        return self.score_maps(images).flatten(1).max(dim=1).values              # (B,)