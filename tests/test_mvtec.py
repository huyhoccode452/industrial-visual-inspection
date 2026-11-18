import numpy as np
import pytest
from PIL import Image

from src.data.mvtec import MVTecDataset, split_train_calib


@pytest.fixture
def fake_root(tmp_path):
    """Dataset MVTec giả: train/good 3 ảnh; test gồm 1 good + 1 scratch có mask."""
    cat = tmp_path / "bottle"
    for d in ["train/good", "test/good", "test/scratch", "ground_truth/scratch"]:
        (cat / d).mkdir(parents=True)

    img = np.full((300, 300, 3), 128, dtype=np.uint8)
    for i in range(3):
        Image.fromarray(img).save(cat / "train" / "good" / f"{i:03d}.png")
    Image.fromarray(img).save(cat / "test" / "good" / "000.png")
    Image.fromarray(img).save(cat / "test" / "scratch" / "000.png")

    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[100:200, 100:200] = 255
    Image.fromarray(mask).save(cat / "ground_truth" / "scratch" / "000_mask.png")
    return tmp_path


def test_split_is_disjoint_and_complete():
    train_idx, calib_idx = split_train_calib(100)
    assert set(train_idx).isdisjoint(calib_idx)
    assert sorted(train_idx + calib_idx) == list(range(100))

def test_split_is_deterministic():
    train_idx1, calib_idx1 = split_train_calib(100, seed=42)
    train_idx2, calib_idx2 = split_train_calib(100, seed=42)
    assert train_idx1 == train_idx2
    assert calib_idx1 == calib_idx2

def test_good_sample_has_zero_mask(fake_root):
    ds = MVTecDataset(fake_root, "bottle", "test")
    image, mask, label, path = ds[0]
    assert image.shape == (3, 224, 224)
    assert mask.sum() == 0
    assert label == 0

def test_defect_sample_has_binary_mask(fake_root):
    ds = MVTecDataset(fake_root, "bottle", "test")
    image, mask, label, path = ds[1]
    assert image.shape == (3, 224, 224)
    assert mask.sum() > 0
    assert label == 1
    assert mask.shape == (1, 224, 224)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}

def test_invalid_split_raises_error(fake_root):
    with pytest.raises(ValueError):
        MVTecDataset(fake_root, "bottle", "val")