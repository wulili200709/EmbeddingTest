"""Template-conditioned lightweight locator."""

from iv4matchmethod.config import ModelConfig
from iv4matchmethod.geometry import PosePrediction
from iv4matchmethod.models.network import TemplateConditionedLocator

__all__ = ["ModelConfig", "PosePrediction", "TemplateConditionedLocator"]
__version__ = "0.1.0"

