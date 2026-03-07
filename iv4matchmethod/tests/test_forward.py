import torch

from iv4matchmethod.config import ModelConfig
from iv4matchmethod.geometry import decode_pose
from iv4matchmethod.models.network import TemplateConditionedLocator


def test_locator_forward_shapes():
    config = ModelConfig(template_size=64, search_size=256, backbone_variant="legacy", pretrained_backbone=False)
    model = TemplateConditionedLocator(config)
    template = torch.randn(2, 3, 64, 64)
    search = torch.randn(2, 3, 256, 256)

    outputs = model(template, search)

    assert outputs["heatmap"].shape == (2, 1, config.response_size, config.response_size)
    assert outputs["offset"].shape == (2, 2, config.response_size, config.response_size)
    poses = decode_pose(outputs)
    assert len(poses) == 2
    assert poses[0].sx > 0.0
    assert poses[0].sy > 0.0


def test_locator_forward_shapes_torchvision_backbone():
    config = ModelConfig(
        template_size=64,
        search_size=256,
        backbone_variant="torchvision",
        pretrained_backbone=False,
    )
    model = TemplateConditionedLocator(config)
    template = torch.randn(1, 3, 64, 64)
    search = torch.randn(1, 3, 256, 256)

    outputs = model(template, search)

    assert outputs["heatmap"].shape == (1, 1, config.response_size, config.response_size)
