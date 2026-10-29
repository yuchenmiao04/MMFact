"""Image-only supervised visual backbone with a classification head (C04)."""

from mmfact.models_classification._neural_baselines import _NeuralBaseline
from mmfact.registry import register_model


@register_model("ImageOnly")
class ImageOnly(_NeuralBaseline):
    modalities = ("image",)
