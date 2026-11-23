import torch.nn.functional as F
import torchvision.transforms.functional as TF


def upsample_and_blur(score_maps, size=224, sigma=4.0):
    """score_maps: (B, 28, 28) -> (B, size, size)."""
    x = score_maps.unsqueeze(1)                                       # (B, 1, 28, 28)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    k = int(4 * sigma) * 2 + 1                                        # cỡ kernel lẻ, = 33
    x = TF.gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])
    return x.squeeze(1)


def normalize(maps, lo, hi):
    """Chuẩn hóa về [0, 1] bằng lo, hi cho trước."""
    return ((maps - lo) / (hi - lo)).clamp(0, 1)