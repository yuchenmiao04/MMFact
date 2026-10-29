"""FND-CLIP: CLIP similarity guidance and modality-wise feature attention."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder
from mmfact.models_classification._encoders import BertEncoder, CLIPImageEncoder, CLIPTextEncoder
from mmfact.models_classification._supervised_vision import supervised_encoder
from mmfact.models_classification._layers import SingletonBatchNorm
from mmfact.registry import register_model


def _projection(input_dim, hidden_dim, output_dim, dropout):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        SingletonBatchNorm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
        SingletonBatchNorm(output_dim),
        nn.ReLU(),
    )


class _SimilarityNormalizer(nn.Module):
    """Cumulative population moments of training cosine scores (parallel Welford)."""

    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("m2", torch.zeros(()))

    @torch.no_grad()
    def update(self, similarity):
        values = similarity.detach().to(self.mean)
        batch_mean = values.mean()
        batch_m2 = (values - batch_mean).square().sum()
        total = self.count + len(values)
        delta = batch_mean - self.mean
        self.m2.add_(batch_m2 + delta.square() * self.count * len(values) / total)
        self.mean.add_(delta * len(values) / total)
        self.count.copy_(total)

    def forward(self, similarity):
        variance = self.m2 / self.count.clamp_min(1)
        # Before training, use an identity standardization; no evaluation fitting.
        std = torch.where(self.count > 0, variance.sqrt(), torch.ones_like(variance))
        return ((similarity - self.mean) / std.clamp_min(self.epsilon)).sigmoid()


@register_model("FND-CLIP")
class FNDCLIP(AbstractDetector):
    """Paper-led detection architecture with explicit content/language adaptation."""

    def __init__(self, config):
        super().__init__(config)
        parameters = config["model"]["parameters"]
        dimensions = [
            parameters[key] for key in ("projection_hidden", "feature_dim", "classifier_hidden")
        ]
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in dimensions
        ):
            raise ValueError("FND-CLIP hidden dimensions must be positive integers.")
        dropout, epsilon = parameters["dropout"], parameters["normalization_epsilon"]
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("FND-CLIP dropout must be finite and in [0, 1).")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("FND-CLIP normalization_epsilon must be finite and positive.")
        if parameters["text"].get("output") != "cls_layers":
            raise ValueError("FND-CLIP requires BERT CLS representations from every layer.")
        if any(
            not parameters[key].get("freeze", True) for key in ("text", "clip_text", "clip_image")
        ):
            raise ValueError("FND-CLIP freezes BERT and both CLIP encoders.")
        if parameters["clip_text"]["model_name"] != parameters["clip_image"]["model_name"]:
            raise ValueError("FND-CLIP requires CLIP text/image from the same checkpoint.")
        if parameters["clip_text"].get("revision", "main") != parameters["clip_image"].get(
            "revision", "main"
        ):
            raise ValueError("FND-CLIP CLIP text/image revisions must match.")
        if any(parameters[key].get("normalize", False) for key in ("clip_text", "clip_image")):
            raise ValueError("FND-CLIP projections consume unnormalized CLIP features.")
        self.text_encoder = material_encoder(
            config, "text_encoder", BertEncoder, **parameters["text"]
        )
        self.image_encoder = material_encoder(
            config, "image_encoder", supervised_encoder, **parameters["image"]
        )
        self.clip_text_encoder = material_encoder(
            config, "clip_text_encoder", CLIPTextEncoder, **parameters["clip_text"]
        )
        self.clip_image_encoder = material_encoder(
            config, "clip_image_encoder", CLIPImageEncoder, **parameters["clip_image"]
        )
        if self.clip_text_encoder.output_dim != self.clip_image_encoder.output_dim:
            raise ValueError("FND-CLIP paired CLIP feature dimensions must match.")
        self.layer_weights = nn.Parameter(torch.rand(self.text_encoder.sequence_length))
        hidden_dim, feature_dim, classifier_dim = dimensions
        clip_dim = self.clip_text_encoder.output_dim
        self.text_projection = _projection(
            self.text_encoder.output_dim + clip_dim, hidden_dim, feature_dim, dropout
        )
        self.image_projection = _projection(
            self.image_encoder.output_dim + clip_dim, hidden_dim, feature_dim, dropout
        )
        self.fusion_projection = _projection(2 * clip_dim, hidden_dim, feature_dim, dropout)
        self.modality_attention = nn.Sequential(nn.Linear(3, 3), nn.GELU(), nn.Linear(3, 3))
        self.normalizer = _SimilarityNormalizer(epsilon)
        # The selected implementation calibrates the standardized similarity with
        # a learned affine map; this is not the paper's bounded sigmoid gate.
        self.similarity_scale = nn.Parameter(torch.ones(()))
        self.similarity_bias = nn.Parameter(torch.zeros(()))
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, classifier_dim),
            SingletonBatchNorm(classifier_dim),
            nn.ReLU(),
            nn.Linear(classifier_dim, 2),
        )

    def _text_features(self, texts):
        layers = self.text_encoder(texts)
        # The reference sums over the batch; use independent per-news layer scores.
        scores = layers.sum(-1) * self.layer_weights
        return (layers * scores.softmax(-1).unsqueeze(-1)).sum(1)

    def forward(self, batch, update_statistics=False):
        text = self._text_features(batch["text"])
        image = self.image_encoder(batch["image_path"])
        clip_text = self.clip_text_encoder(batch["text"])
        clip_image = self.clip_image_encoder(batch["image_path"])
        similarity = F.cosine_similarity(clip_text, clip_image, dim=-1)
        if update_statistics and self.training:
            self.normalizer.update(similarity)
        text = self.text_projection(torch.cat((text, clip_text), dim=-1))
        image = self.image_projection(torch.cat((image, clip_image), dim=-1))
        fusion = self.fusion_projection(torch.cat((clip_text, clip_image), dim=-1))
        gate = self.similarity_scale * self.normalizer(similarity) + self.similarity_bias
        fusion = fusion * gate.unsqueeze(-1)
        features = torch.stack((text, image, fusion), dim=1)
        # Shared SE MLP for average/max descriptors, following the reference.
        attention = (
            self.modality_attention(features.mean(-1)) + self.modality_attention(features.amax(-1))
        ).sigmoid()
        aggregated = (features * attention.unsqueeze(-1)).sum(1)
        return self.classifier(aggregated)

    def calculate_loss(self, batch):
        logits = self(batch, update_statistics=self.training)
        return {"classification": F.cross_entropy(logits, batch["labels"])}

    def predict(self, batch):
        was_training = self.training
        self.eval()
        try:
            return self(batch)
        finally:
            self.train(was_training)
