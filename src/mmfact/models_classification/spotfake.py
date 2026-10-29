"""SpotFake: frozen BERT/image features with supervised multimodal fusion."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder
from mmfact.models_classification._encoders import BertEncoder
from mmfact.models_classification._supervised_vision import supervised_encoder
from mmfact.registry import register_model


def _projection(input_dim: int, widths: Sequence[int], dropout: float) -> nn.Sequential:
    """The final representation receives dropout after the two branches join."""
    if not widths or any(not isinstance(width, int) or width <= 0 for width in widths):
        raise ValueError("SpotFake projection widths must be positive integers.")
    layers: list[nn.Module] = []
    for index, width in enumerate(widths):
        layers.extend((nn.Linear(input_dim, width), nn.ReLU()))
        if index < len(widths) - 1:
            layers.append(nn.Dropout(dropout))
        input_dim = width
    return nn.Sequential(*layers)


@register_model("SpotFake")
class SpotFake(AbstractDetector):
    """Paper-led architecture; the preset records source/code discrepancies."""

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        parameters = config["model"]["parameters"]
        self.text_encoder = material_encoder(
            config, "text_encoder", BertEncoder, **parameters["text"]
        )
        self.image_encoder = material_encoder(
            config, "image_encoder", supervised_encoder, **parameters["image"]
        )
        dropout = parameters["dropout"]
        text_hidden = parameters["text_hidden"]
        image_hidden = parameters["image_hidden"]
        self.text_projection = _projection(self.text_encoder.output_dim, text_hidden, dropout)
        self.image_projection = _projection(self.image_encoder.output_dim, image_hidden, dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(text_hidden[-1] + image_hidden[-1], parameters["fusion_hidden"]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(parameters["fusion_hidden"], 1),
        )

    def forward(self, batch: Mapping[str, Any]) -> Tensor:
        text = self.text_projection(self.text_encoder(batch["text"]))
        image = self.image_projection(self.image_encoder(batch["image_path"]))
        return self.classifier(torch.cat((text, image), dim=-1)).squeeze(-1)

    def calculate_loss(self, batch: Mapping[str, Any]) -> Tensor:
        return F.binary_cross_entropy_with_logits(self(batch), batch["labels"].float())

    def predict(self, batch: Mapping[str, Any]) -> Tensor:
        logits = self(batch)
        # softmax([0, z]) == [1 - sigmoid(z), sigmoid(z)].
        return torch.stack((torch.zeros_like(logits), logits), dim=-1)
