
import torch
from src.evaluation.anomaly_map import upsample_and_blur, normalize

def test_output_shape():
    score_maps = torch.rand(2, 28, 28)
    output = upsample_and_blur(score_maps, size=224, sigma=4.0)
    assert output.shape == (2, 224, 224)

def test_normalize_range():
    maps = torch.tensor([-1.0, 0.5, 3.0])
    lo = 0.0
    hi = 1.0
    normalized = normalize(maps, lo, hi)
    assert torch.all(normalized >= 0.0) and torch.all(normalized <= 1.0)

def test_blur_keeps_peak_location():
    score_maps = torch.zeros(1, 28, 28)
    score_maps[0, 14, 14] = 10.0  # Set a peak in the center
    blurred = upsample_and_blur(score_maps, size=224, sigma=4.0)
    peak_location = blurred[0].flatten().argmax()
    r,c = peak_location // 224, peak_location % 224
    assert abs(r - 112) <= 8  # The peak should be within 8 pixels of the center after blurring
    assert abs(c - 112) <= 8