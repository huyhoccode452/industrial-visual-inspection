import torch

from src.models.padim import build_embedding, make_channel_index
from src.models.padim import build_embedding, fit_gaussians, mahalanobis_map, make_channel_index
from src.models.padim import PaDiM

def make_feats(b=2):
    return {"layer2": torch.randn(b, 512, 28, 28), "layer3": torch.randn(b, 1024, 14, 14)}


def test_embedding_shape():
    idx = make_channel_index(1536, 100)
    emb = build_embedding(make_feats(), idx)
    assert emb.shape == (2, 100, 28, 28)

def test_layer3_is_repeated_2x2() :
    idx = torch.arange(1536)
    emb = build_embedding(make_feats(), idx)
    assert torch.equal(emb[0, 512:, 0, 0], emb[0, 512:, 1, 1])

def test_channel_index_is_deterministic_and_unique():
    idx1 = make_channel_index(1536, 100, seed=42)
    idx2 = make_channel_index(1536, 100, seed=42)
    assert torch.equal(idx1, idx2)
    assert len(torch.unique(idx1)) == 100

def test_hand_example_matches_notes():
    train = torch.tensor([[1., 2.], [3., 2.], [2., 4.], [2., 0.]]).reshape(4, 2, 1, 1)
    mean, inv_cov = fit_gaussians(train, eps=0.0)
    test = torch.tensor([[4., 2.], [2., 4.]]).reshape(2, 2, 1, 1)
    scores = mahalanobis_map(test, mean, inv_cov).reshape(2)
    assert torch.allclose(scores, torch.tensor([6.0, 1.5]).sqrt(), atol=1e-5)

class FakeExtractor:
    def __call__(self, x):
        b = x.shape[0]
        return {"layer2": torch.randn(b, 512, 28, 28), "layer3": torch.randn(b, 1024, 14, 14)}


def test_padim_end_to_end_shapes():
    train_loader = [(torch.randn(4, 3, 224, 224), None, None, None) for _ in range(3)]  # 12 ảnh
    model = PaDiM(FakeExtractor(), rd=20).fit(train_loader)
    x = torch.randn(2, 3, 224, 224)
    maps = model.score_maps(x)
    scores = model.score_images(x)
    assert maps.shape == (2, 28, 28)
    assert scores.shape == (2,)
    assert torch.isfinite(maps).all()
    assert (maps >= 0).all()