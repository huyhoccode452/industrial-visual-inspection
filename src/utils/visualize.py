import matplotlib.pyplot as plt
import torch

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denorm(img):
    """Đảo phép Normalize để vẽ lại ảnh."""
    return (img * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


def show_overlay(image, mask, amap, title=""):
    """image: (3,H,W) đã chuẩn hóa; mask: (H,W) 0/1; amap: (H,W) trong [0,1]."""
    img = denorm(image)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title("ảnh")
    axes[1].imshow(img)
    axes[1].imshow(mask.numpy(), cmap="Reds", alpha=0.4)
    axes[1].set_title("mask thật")
    axes[2].imshow(img)
    axes[2].imshow(amap.numpy(), cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title("bản đồ dự đoán")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    plt.show()