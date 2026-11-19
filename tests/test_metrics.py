import numpy as np
import pytest

from src.evaluation.metrics import image_auroc, pixel_auroc

LABELS = [0, 0, 1, 1]


@pytest.mark.parametrize("scores, expected", [
    ([0.1, 0.2, 0.8, 0.9], 1.0),   # A: xếp hạng hoàn hảo
    ([0.8, 0.9, 0.1, 0.2], 0.0),   # B: xếp ngược
    ([0.5, 0.5, 0.5, 0.5], 0.5),   # C: hòa hết
])
def test_image_auroc_matches_hand_computation(scores, expected):
    assert image_auroc(LABELS, scores) == pytest.approx(expected)

def test_image_auroc_rejects_single_class():
    with pytest.raises(ValueError):
        image_auroc([0, 0, 0], [0.1, 0.2, 0.3])

def test_pixel_auroc_perfect_map():
    masks = np.array([[[0,0],[1,1]]])
    maps = np.array([[[0.1, 0.2],[0.8, 0.9]]])
    assert pixel_auroc(masks, maps) == pytest.approx(1.0)

def test_image_auroc_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        image_auroc([0, 0, 1, 1], [0.1, 0.2, 0.3])