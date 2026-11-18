from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader, Subset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CATEGORIES = ["bottle", "cable", "capsule", "metal_nut", "screw"]


class MVTecDataset:
    def __init__(self, root, category, split):
        # root: thư mục chứa mvtec_ad, category: "bottle"..., split: "train" hoặc "test"
        self.root = Path(root)
        self.category = category
        self.split = split
        self.samples = self._collect_samples()

        self.image_tf = T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.mask_tf = T.Compose([
            T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
            T.CenterCrop(224),
            T.ToTensor(),
        ])

    def _collect_samples(self):
        samples = []
        if self.split == "train":
            train_dir = self.root / self.category / "train" / "good"
            for img_path in sorted(train_dir.glob("*.png")):
                samples.append((img_path, "good"))

        elif self.split == "test":
            test_dir = self.root / self.category / "test"
            for defect_dir in sorted(test_dir.iterdir()):
                if defect_dir.is_dir():
                    defect_type = defect_dir.name
                    for img_path in sorted(defect_dir.glob("*.png")):
                        samples.append((img_path, defect_type))
        else:
            raise ValueError(f"split phải là 'train' hoặc 'test', nhận được: {self.split!r}")
        return samples

    def __len__(self):
        return len(self.samples)

    def _mask_path(self, img_path, defect_type):
        if defect_type == "good":
            return None

        mask_dir = self.root / self.category / "ground_truth" / defect_type
        mask_path = mask_dir / (img_path.stem + "_mask.png")
        return mask_path

    def __getitem__(self, idx):
        img_path, defect_type = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        mask_path = self._mask_path(img_path, defect_type)
        if mask_path is None:
            mask = np.zeros((image.height, image.width), dtype=np.uint8)  # (cao, rộng)
        else:
            mask = np.array(Image.open(mask_path).convert("L"))
            mask = (mask > 0).astype(np.uint8)  # 255 -> 1
        image = self.image_tf(image)
        mask = self.mask_tf(Image.fromarray(mask * 255))
        label = 0 if defect_type == "good" else 1
        return image, mask, label, str(img_path)
    
def split_train_calib(n, calib_ratio=0.1, seed=42):
    """Trả về (train_idx, calib_idx): hai danh sách chỉ số không giao nhau."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_calib = max(1, round(n * calib_ratio))
    calib_idx = perm[:n_calib]
    train_idx = perm[n_calib:]
    return sorted(train_idx), sorted(calib_idx)

def build_loaders(root, category, batch_size=32, calib_ratio=0.1, seed=42, num_workers=0):
    full_train = MVTecDataset(root, category, "train")
    train_idx, calib_idx = split_train_calib(len(full_train), calib_ratio, seed)

    train_loader = DataLoader(Subset(full_train, train_idx), batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    calib_loader = DataLoader(Subset(full_train, calib_idx), batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(MVTecDataset(root, category, "test"), batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, calib_loader, test_loader