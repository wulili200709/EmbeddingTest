from dataclasses import dataclass


@dataclass(slots=True)
class ModelConfig:
    backbone_variant: str = "legacy"
    pretrained_backbone: bool = False
    template_size: int = 128
    search_size: int = 384
    feature_stride: int = 8
    fuse_channels: int = 96
    head_channels: int = 64
    input_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    input_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def validate(self) -> None:
        if self.backbone_variant not in {"legacy", "torchvision"}:
            raise ValueError("backbone_variant must be one of: legacy, torchvision")
        if self.pretrained_backbone and self.backbone_variant != "torchvision":
            raise ValueError("pretrained_backbone is only supported with backbone_variant='torchvision'")
        if self.template_size % self.feature_stride != 0:
            raise ValueError("template_size must be divisible by feature_stride")
        if self.search_size % self.feature_stride != 0:
            raise ValueError("search_size must be divisible by feature_stride")
        if self.search_size <= self.template_size:
            raise ValueError("search_size must be larger than template_size")
        if len(self.input_mean) != 3 or len(self.input_std) != 3:
            raise ValueError("input_mean and input_std must contain 3 channels")

    @property
    def template_kernel(self) -> int:
        return self.template_size // self.feature_stride

    @property
    def search_feature_size(self) -> int:
        return self.search_size // self.feature_stride

    @property
    def response_size(self) -> int:
        return self.search_feature_size - self.template_kernel + 1
