"""Raw PLM and supervised image features concatenated before classification (C05)."""

from mmfact.models_classification._neural_baselines import _NeuralBaseline
from mmfact.registry import register_model


@register_model("ConcatMLP")
class ConcatMLP(_NeuralBaseline):
    pass
