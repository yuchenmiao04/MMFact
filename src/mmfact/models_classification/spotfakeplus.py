"""SpotFake+: sentence-level BERT and image features with regularized fusion."""

import torch
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder
from mmfact.models_classification._encoders import SentenceEncoder
from mmfact.models_classification._supervised_vision import supervised_encoder
from mmfact.models_classification._layers import SingletonBatchNorm
from mmfact.registry import register_model


def _projection(input_dim, widths, dropout, normalize=False):
    if not widths or any(not isinstance(width, int) or width < 1 for width in widths):
        raise ValueError("SpotFakePlus projection widths must be positive integers.")
    layers = []
    for width in widths:
        layer = nn.Linear(input_dim, width)
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        nn.init.zeros_(layer.bias)
        layers.extend((layer, nn.ReLU()))
        input_dim = width
    if normalize:
        # Keras momentum 0.99 gives the same running-mean update as PyTorch 0.01.
        layers.append(SingletonBatchNorm(input_dim, eps=1e-3, momentum=0.01))
    layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


@register_model("SpotFakePlus")
class SpotFakePlus(AbstractDetector):
    """Keep ordered sentence slots; train both modality heads and fusion jointly."""

    def __init__(self, config):
        super().__init__(config)
        parameters = config["model"]["parameters"]
        dropout, self.kernel_l2 = parameters["dropout"], parameters["kernel_l2"]
        if self.kernel_l2 < 0:
            raise ValueError("SpotFakePlus kernel_l2 must be nonnegative.")
        self.text_encoder = material_encoder(
            config, "text_encoder", SentenceEncoder, **parameters["text"]
        )
        self.image_encoder = material_encoder(
            config, "image_encoder", supervised_encoder, **parameters["image"]
        )
        text_hidden, image_hidden = parameters["text_hidden"], parameters["image_hidden"]
        self.text_projection = _projection(
            self.text_encoder.max_sentences * self.text_encoder.output_dim,
            text_hidden,
            dropout,
            normalize=True,
        )
        self.image_projection = _projection(
            self.image_encoder.output_dim,
            image_hidden,
            dropout,
            normalize=True,
        )
        self.fusion = _projection(
            text_hidden[-1] + image_hidden[-1],
            parameters["fusion_hidden"],
            dropout,
        )
        self.classifier = nn.Linear(parameters["fusion_hidden"][-1], 2)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, batch):
        sentences = self.text_encoder(batch["text"])
        text = self.text_projection(sentences.flatten(start_dim=1))
        image = self.image_projection(self.image_encoder(batch["image_path"]))
        return self.classifier(self.fusion(torch.cat((text, image), dim=-1)))

    def calculate_loss(self, batch):
        regularization = sum(
            layer.weight.square().sum()
            for branch in (self.text_projection, self.image_projection, self.fusion)
            for layer in branch
            if isinstance(layer, nn.Linear)
        )
        return {
            "classification": F.cross_entropy(self(batch), batch["labels"]),
            "regularization": self.kernel_l2 * regularization,
        }

    def predict(self, batch):
        return self(batch)
