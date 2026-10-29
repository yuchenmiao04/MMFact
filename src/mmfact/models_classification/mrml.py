"""MRML: alternating intra-modal metrics and cross-modal pairwise learning."""

import torch
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder
from mmfact.models_classification._encoders import BertEncoder
from mmfact.models_classification._supervised_vision import supervised_encoder
from mmfact.models_classification._layers import SingletonBatchNorm
from mmfact.registry import register_model


def _distances(values):
    return (values[:, None] - values[None, :]).square().sum(-1)


def _triplet_loss(target, complementary, labels, metric, margin):
    """Equations 3--5, with hardest valid in-batch triplets and W = A.T @ A."""
    distances = _distances(metric(target))
    same = labels[:, None].eq(labels[None, :])
    positive = same & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    negative = ~same
    valid = positive.any(1) & negative.any(1)
    if not valid.any():
        return distances.sum() * 0
    with torch.no_grad():
        positives = distances.masked_fill(~positive, -torch.inf).argmax(1)[valid]
        negatives = distances.masked_fill(~negative, torch.inf).argmin(1)[valid]
        anchors = valid.nonzero(as_tuple=True)[0]
        other = complementary.detach()
        delta = (
            (other[anchors] - other[positives]).square().sum(-1)
            - (other[anchors] - other[negatives]).square().sum(-1)
        ).sign()
    difference = distances[anchors, positives] - distances[anchors, negatives]
    return F.relu(margin - delta * difference).sum() / len(labels)


def _pairwise_loss(scores, labels, margin):
    """Equation 6, author release's hardest opposite-label pair in each direction."""
    valid = labels[:, None].ne(labels[None, :])
    diagonal = scores.diag()
    image_anchor = F.relu(margin + scores - diagonal[:, None]).masked_fill(~valid, 0)
    text_anchor = F.relu(margin + scores - diagonal[None, :]).masked_fill(~valid, 0)
    return (image_anchor.amax(1).sum() + text_anchor.amax(0).sum()) / len(labels)


@register_model("MRML")
class MRML(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        parameters = config["model"]["parameters"]
        dimension = parameters["hidden_dim"]
        fusion_dim = parameters["fusion_dim"]
        if min(dimension, fusion_dim, parameters["classifier_dim"], parameters["rank"]) < 1:
            raise ValueError("MRML dimensions and fusion rank must be positive.")
        if parameters["iteration_epochs"] < 1:
            raise ValueError("MRML iteration_epochs must be positive.")
        if not 0 <= parameters["dropout"] < 1:
            raise ValueError("MRML dropout must be in [0, 1).")
        if min(parameters[key] for key in ("triplet_margin", "pair_margin")) < 0:
            raise ValueError("MRML margins must be nonnegative.")
        if min(parameters[key] for key in ("triplet_weight", "pair_weight")) <= 0:
            raise ValueError("MRML auxiliary loss weights must be positive.")
        self.text_encoder = material_encoder(
            config, "text_encoder", BertEncoder, **parameters["text"]
        )
        self.image_encoder = material_encoder(
            config, "image_encoder", supervised_encoder, **parameters["image"]
        )
        self.image_input_relu = parameters["image"].get("backbone") != "vit"

        def projection(input_dim):
            return nn.Sequential(
                nn.Linear(input_dim, dimension), SingletonBatchNorm(dimension), nn.ReLU()
            )

        self.text_projection = projection(self.text_encoder.output_dim)
        self.image_projection = projection(self.image_encoder.output_dim)
        self.text_metric = nn.Linear(dimension, dimension, bias=False)
        self.image_metric = nn.Linear(dimension, dimension, bias=False)
        self.text_factors = nn.ModuleList(
            [nn.Linear(dimension, fusion_dim) for _ in range(parameters["rank"])]
        )
        self.image_factors = nn.ModuleList(
            [nn.Linear(dimension, fusion_dim) for _ in range(parameters["rank"])]
        )
        self.dropout = nn.Dropout(parameters["dropout"])
        self.similarity = nn.Linear(fusion_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, parameters["classifier_dim"]),
            SingletonBatchNorm(parameters["classifier_dim"]),
            nn.ReLU(),
            nn.Linear(parameters["classifier_dim"], 2),
            SingletonBatchNorm(2),
        )
        self.iteration_epochs = parameters["iteration_epochs"]
        self.triplet_margin, self.triplet_weight = (
            parameters["triplet_margin"],
            parameters["triplet_weight"],
        )
        self.pair_margin, self.pair_weight = parameters["pair_margin"], parameters["pair_weight"]
        self.metric_modality = "image"

    def on_epoch_start(self, epoch, total_epochs):
        self.metric_modality = "image" if (epoch - 1) // self.iteration_epochs % 2 == 0 else "text"

    def _project(self, batch):
        text = self.text_projection(self.text_encoder(batch["text"]))
        image = self.image_encoder(batch["image_path"])
        # ResNet avgpool is nonnegative; ViT's final-normalized CLS is signed.
        if self.image_input_relu:
            image = image.relu()
        image = self.image_projection(image)
        return text, image

    def _fuse(self, text, image, pairwise=False):
        features = []
        for text_factor, image_factor in zip(self.text_factors, self.image_factors):
            textual, visual = text_factor(self.dropout(text)), image_factor(self.dropout(image))
            features.append(visual[:, None] * textual[None, :] if pairwise else visual * textual)
        return torch.stack(features).sum(0)

    def forward(self, batch):
        return self.classifier(self._fuse(*self._project(batch)))

    def calculate_loss(self, batch):
        text, image = self._project(batch)
        pairs = self._fuse(text, image, pairwise=True)
        indices = torch.arange(len(text), device=text.device)
        logits = self.classifier(pairs[indices, indices])
        target, complementary, metric = (
            (image, text, self.image_metric)
            if self.metric_modality == "image"
            else (text, image, self.text_metric)
        )
        return {
            "classification": F.cross_entropy(logits, batch["labels"]),
            "triplet": self.triplet_weight
            * _triplet_loss(target, complementary, batch["labels"], metric, self.triplet_margin),
            "pairwise": self.pair_weight
            * _pairwise_loss(
                self.similarity(pairs).squeeze(-1).sigmoid(), batch["labels"], self.pair_margin
            ),
        }

    def predict(self, batch):
        was_training = self.training
        self.eval()
        try:
            return self(batch)
        finally:
            self.train(was_training)
