import torch
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights


class FeatureExtractor(torch.nn.Module):
    """Backbone WideResNet50 đã học sẵn, đóng băng; trả về feature map layer2, layer3."""

    def __init__(self):
        super().__init__()
        self.model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        m = self.model
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        x1 = m.layer1(x)
        x2 = m.layer2(x1)
        x3 = m.layer3(x2)
        return {"layer2": x2, "layer3": x3}