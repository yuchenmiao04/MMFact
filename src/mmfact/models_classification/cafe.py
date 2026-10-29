"""CAFE: semantic alignment and ambiguity-weighted cross-modal correlations."""

import math

import torch
from torch import nn
from torch.distributions import kl_divergence
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder
from mmfact.models_classification._distributions import (
    GaussianEncoder as _GaussianEncoder,
)
from mmfact.models_classification._cafe_normalization import marginal_kl_pair
from mmfact.models_classification._encoders import BertEncoder
from mmfact.models_classification._supervised_vision import supervised_encoder
from mmfact.registry import register_model


def _mlp(input_dim, hidden_dims):
    layers = []
    for dimension in hidden_dims:
        layers.extend((nn.Linear(input_dim, dimension), nn.ReLU()))
        input_dim = dimension
    return nn.Sequential(*layers)


def _correlation(text, image):
    """Equations 8--12: two attention maps and their full outer-product feature."""
    scores = text.unsqueeze(-1) * image.unsqueeze(-2) / math.sqrt(text.shape[-1])
    text_updated = torch.bmm(scores.softmax(-1), text.unsqueeze(-1)).squeeze(-1)
    image_updated = torch.bmm(scores.transpose(-1, -2).softmax(-1), image.unsqueeze(-1)).squeeze(-1)
    return (text_updated.unsqueeze(-1) * image_updated.unsqueeze(-2)).flatten(1)


@register_model("CAFE")
class CAFE(AbstractDetector):
    """Paper architecture; auxiliary alignment precedes detection in each epoch."""

    def __init__(self, config):
        super().__init__(config)
        parameters = config["model"]["parameters"]
        self.text_encoder = material_encoder(
            config, "text_encoder", BertEncoder, **parameters["text"]
        )
        self.image_encoder = material_encoder(
            config, "image_encoder", supervised_encoder, **parameters["image"]
        )
        dimension = parameters["alignment_dim"]
        self.text_projection = _mlp(self.text_encoder.output_dim, [parameters["text_dim"]])
        self.image_projection = _mlp(self.image_encoder.output_dim, [parameters["image_dim"]])
        self.text_alignment = _mlp(parameters["text_dim"], [dimension] * 3)
        self.image_alignment = _mlp(parameters["image_dim"], [dimension] * 3)
        self.shared_alignment = _mlp(dimension, [dimension])
        self.text_distribution = _GaussianEncoder(dimension, parameters["latent_dim"])
        self.image_distribution = _GaussianEncoder(dimension, parameters["latent_dim"])
        self.classifier = nn.Sequential(
            _mlp(dimension**2 + 2 * dimension, [parameters["classifier_dim"]] * 2),
            nn.Linear(parameters["classifier_dim"], 2),
        )
        self.margin, self.beta = parameters["margin"], parameters["beta"]
        self.normalization = parameters["normalization"]
        if not -1 <= self.margin <= 1 or not 0 < self.beta <= 1:
            raise ValueError("CAFE requires margin in [-1, 1] and beta in (0, 1].")
        if self.normalization["reference_size"] < 1:
            raise ValueError("CAFE mixture reference size must be positive.")
        if self.normalization["epsilon"] <= 0:
            raise ValueError("CAFE mixture epsilon must be positive.")
        if self.normalization.get("method") != "gauss_hermite":
            raise ValueError("CAFE requires the explicit converged marginal KL integration recipe.")
        self.register_buffer("marginal_kl", torch.ones(2))
        self.register_buffer("marginal_kl_estimate", torch.ones(2))
        self.register_buffer("marginal_kl_error", torch.zeros(2, dtype=torch.float64))
        self.register_buffer("marginal_kl_order", torch.zeros((), dtype=torch.long))
        self.register_buffer("normalizer_ready", torch.tensor(False))
        self.phase = "detection"
        self._real_records = []
        self._reference_records = []

    def prepare(self, train_dataset):
        self._real_records = [row for row in train_dataset.records if row["label"] == 0]
        if len(self._real_records) < 2:
            raise ValueError("CAFE alignment requires at least two real training news items.")
        generator = torch.Generator().manual_seed(self.normalization["seed"])
        indices = torch.randperm(len(train_dataset.records), generator=generator)
        self._reference_records = [
            {key: train_dataset.records[index][key] for key in ("text", "image_path")}
            for index in indices[: self.normalization["reference_size"]].tolist()
        ]

    def training_phases(self):
        return ("alignment", "detection")

    def loss_weight(self, batch):
        if self.phase == "alignment":
            return 2 * int(batch["labels"].eq(0).sum())
        return super().loss_weight(batch)

    def set_training_phase(self, phase):
        if phase not in self.training_phases():
            raise ValueError(f"Unknown CAFE training phase: {phase}")
        self.phase = phase
        if phase == "detection":
            self._refresh_normalizer()

    @torch.no_grad()
    def _refresh_normalizer(self):
        if not self._reference_records:
            raise ValueError("Prepare CAFE on training data before fitting its KL normalizer.")
        was_training = self.training
        self.eval()
        try:
            text_parameters, image_parameters = [], []
            for start in range(0, len(self._reference_records), 64):
                records = self._reference_records[start : start + 64]
                text, image = self._aligned(
                    [row["text"] for row in records], [row["image_path"] for row in records]
                )
                text_parameters.append(self.text_distribution(text).base_dist)
                image_parameters.append(self.image_distribution(image).base_dist)
            text = tuple(
                torch.cat([getattr(value, key) for value in text_parameters])
                for key in ("loc", "scale")
            )
            image = tuple(
                torch.cat([getattr(value, key) for value in image_parameters])
                for key in ("loc", "scale")
            )
            estimates, error, order = marginal_kl_pair(
                text,
                image,
                orders=self.normalization["orders"],
                rtol=self.normalization["rtol"],
                atol=self.normalization["atol"],
            )
            estimates = estimates.to(dtype=self.marginal_kl.dtype).to(self.marginal_kl.device)
            if not torch.isfinite(estimates).all():
                raise ValueError("CAFE estimated a non-finite training-mixture KL divergence.")
            self.marginal_kl_estimate.copy_(estimates)
            self.marginal_kl.copy_(estimates.clamp_min(self.normalization["epsilon"]))
            self.marginal_kl_error.copy_(error.to(self.marginal_kl_error.device))
            self.marginal_kl_order.fill_(order)
            self.normalizer_ready.fill_(True)
        finally:
            self.train(was_training)

    def on_epoch_end(self, epoch):
        # Refresh once more after Gaussian parameters changed, before validation.
        self._refresh_normalizer()

    def _aligned(self, text, images):
        text_features = self.text_projection(self.text_encoder(text))
        image_features = self.image_projection(self.image_encoder(images))
        return (
            self.shared_alignment(self.text_alignment(text_features)),
            self.shared_alignment(self.image_alignment(image_features)),
        )

    def _alignment_pairs(self, batch):
        indices = batch["labels"].eq(0).nonzero(as_tuple=True)[0].tolist()
        if not indices:
            return None
        texts = [batch["text"][index] for index in indices]
        matched = [batch["image_path"][index] for index in indices]
        if len(indices) > 1:
            shift = torch.randint(1, len(indices), ()).item()
            unmatched = matched[shift:] + matched[:shift]
        else:
            sample_id = batch["sample_id"][indices[0]]
            candidates = [row for row in self._real_records if row["sample_id"] != sample_id]
            if not candidates:
                raise ValueError("Prepare CAFE before sampling singleton alignment batches.")
            unmatched = [candidates[torch.randint(len(candidates), ()).item()]["image_path"]]
        return texts, matched, unmatched

    def calculate_loss(self, batch):
        if self.phase == "alignment":
            pairs = self._alignment_pairs(batch)
            if pairs is None:
                return None
            texts, matched, unmatched = pairs
            text, image = self._aligned(texts * 2, matched + unmatched)
            labels = torch.ones(len(text), device=text.device)
            labels[len(texts) :] = -1
            return {
                "alignment": self.beta
                * F.cosine_embedding_loss(text, image, labels, margin=self.margin)
            }
        return {"classification": F.cross_entropy(self(batch), batch["labels"])}

    def forward(self, batch):
        if not self.normalizer_ready.item():
            raise ValueError("Train CAFE or load a checkpoint with its training KL normalizer.")
        # The two tasks update disjoint parameters, as in the released optimizers.
        with torch.no_grad():
            text, image = self._aligned(batch["text"], batch["image_path"])
        text_distribution = self.text_distribution(text)
        image_distribution = self.image_distribution(image)
        ambiguity = (
            (
                0.5
                * (
                    kl_divergence(text_distribution, image_distribution) / self.marginal_kl[0]
                    + kl_divergence(image_distribution, text_distribution) / self.marginal_kl[1]
                )
            )
            .sigmoid()
            .unsqueeze(-1)
        )
        features = torch.cat(
            (
                ambiguity * _correlation(text, image),
                (1 - ambiguity) * text,
                (1 - ambiguity) * image,
            ),
            dim=-1,
        )
        return self.classifier(features)

    def predict(self, batch):
        return self(batch)
