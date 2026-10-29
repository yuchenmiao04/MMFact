"""COOLANT: consistency-guided contrastive learning and ambiguity attention."""

import math

import torch
from torch import nn
from torch.distributions import kl_divergence
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder
from mmfact.models_classification._distributions import GaussianEncoder, mixture_kl
from mmfact.models_classification._encoders import BertEncoder
from mmfact.models_classification._supervised_vision import supervised_encoder
from mmfact.models_classification._layers import SingletonBatchNorm
from mmfact.registry import register_model


def _mlp(dimensions, dropout=0.0):
    layers = []
    for index, (source, target) in enumerate(zip(dimensions, dimensions[1:])):
        layers.extend((nn.Linear(source, target), SingletonBatchNorm(target), nn.ReLU()))
        if index == 0 and dropout:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class _TextProjection(nn.Module):
    """Author FastCNN aggregates BERT tokens within each news item."""

    def __init__(self, input_dim, parameters):
        super().__init__()
        channels = parameters["cnn_channels"]
        self.convolutions = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(input_dim, channels, kernel),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(parameters["dropout"]),
                nn.AdaptiveMaxPool1d(1),
            )
            for kernel in parameters["cnn_kernels"]
        )
        self.projection = _mlp(
            [
                channels * len(parameters["cnn_kernels"]),
                parameters["projection_dim"],
                parameters["shared_dim"],
            ],
            parameters["dropout"],
        )

    def forward(self, tokens):
        tokens = tokens.transpose(1, 2)
        features = torch.cat([layer(tokens).squeeze(-1) for layer in self.convolutions], dim=-1)
        return self.projection(features)


class _Consistency(nn.Module):
    def __init__(self, text_dim, image_dim, parameters):
        super().__init__()
        shared, aligned = parameters["shared_dim"], parameters["alignment_dim"]
        self.text_projection = _TextProjection(text_dim, parameters)
        self.image_projection = nn.Linear(image_dim, image_dim)
        self.image_features = _mlp(
            [image_dim, parameters["image_hidden_dim"], shared], parameters["dropout"]
        )
        self.text_alignment = _mlp([shared, shared, aligned])
        self.image_alignment = _mlp([shared, shared, aligned])

    def forward(self, tokens, image):
        text = self.text_projection(tokens)
        image = self.image_features(F.normalize(self.image_projection(image), dim=-1))
        return self.text_alignment(text), self.image_alignment(image)


def _correlation(text, image):
    """Equations 8--10: both coordinate-attention directions and full outer product."""
    scores = image.unsqueeze(-1) * text.unsqueeze(-2) / math.sqrt(text.shape[-1])
    image_updated = torch.bmm(scores.softmax(-1), image.unsqueeze(-1)).squeeze(-1)
    text_updated = torch.bmm(scores.transpose(-1, -2).softmax(-1), text.unsqueeze(-1)).squeeze(-1)
    return (image_updated.unsqueeze(-1) * text_updated.unsqueeze(-2)).flatten(1)


def _attention_kl(attention, guidance):
    """Equation 14, after making both positive gate vectors categorical probabilities."""
    epsilon = torch.finfo(attention.dtype).eps
    attention = attention.clamp_min(epsilon)
    guidance = guidance.clamp_min(epsilon)
    attention = attention / attention.sum(-1, keepdim=True)
    guidance = guidance / guidance.sum(-1, keepdim=True)
    return (attention * (attention.log() - guidance.log())).sum(-1).mean()


def _validate(parameters):
    dimensions = (
        "cnn_channels",
        "shared_dim",
        "projection_dim",
        "image_hidden_dim",
        "alignment_dim",
        "unimodal_dim",
        "classifier_dim",
        "attention_hidden_dim",
        "latent_dim",
    )
    if any(type(parameters[key]) is not int or parameters[key] < 1 for key in dimensions):
        raise ValueError("COOLANT dimensions must be positive integers.")
    kernels = parameters["cnn_kernels"]
    if (
        not isinstance(kernels, (list, tuple))
        or not kernels
        or any(type(kernel) is not int or kernel < 1 for kernel in kernels)
    ):
        raise ValueError("COOLANT requires positive integer CNN kernels.")
    for key in ("dropout", "margin", "temperature", "semantic_weight", "guidance_weight"):
        if type(parameters[key]) not in (int, float) or not math.isfinite(parameters[key]):
            raise ValueError(f"COOLANT {key} must be a finite number.")
    if not 0 <= parameters["dropout"] < 1 or not -1 <= parameters["margin"] <= 1:
        raise ValueError("COOLANT dropout or matching margin is out of range.")
    if parameters["temperature"] <= 0:
        raise ValueError("COOLANT requires a positive temperature.")
    if min(parameters["semantic_weight"], parameters["guidance_weight"]) <= 0:
        raise ValueError("COOLANT auxiliary loss weights must be positive.")
    text = parameters["text"]
    if text["output"] != "tokens" or text["freeze"] is not True:
        raise ValueError("COOLANT requires frozen BERT token features.")
    if type(text["max_length"]) is not int or text["max_length"] <= max(kernels):
        raise ValueError(
            "COOLANT token length must be an integer exceeding its largest CNN kernel."
        )
    if type(text["mask_padding"]) is not bool:
        raise ValueError("COOLANT mask_padding must be boolean.")
    for key, frozen in (("image", True), ("contrastive_image", False)):
        image = parameters[key]
        if image["freeze"] is not frozen or image["output"] != "avgpool":
            raise ValueError("COOLANT requires frozen ITM and trainable contrastive global image features.")
        if image.get("backbone") != "vit" and image["architecture"] not in {
            "resnet18", "resnet34", "resnet50", "resnet101"
        }:
            raise ValueError("COOLANT requires a supported supervised visual architecture.")
        if image.get("backbone") != "vit" and type(image["stem_maxpool"]) is not bool:
            raise ValueError("COOLANT stem_maxpool must be boolean.")
    normalization = parameters["normalization"]
    if any(
        type(normalization[key]) is not int or normalization[key] < 1
        for key in ("reference_size", "samples")
    ):
        raise ValueError("COOLANT mixture reference size and samples must be positive integers.")
    if type(normalization["seed"]) is not int or not -(2**63) <= normalization["seed"] < 2**64:
        raise ValueError("COOLANT mixture seed must be a valid integer torch seed.")
    epsilon = normalization["epsilon"]
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("COOLANT mixture epsilon must be a positive finite number.")


class _Aggregation(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        dimension, hidden = parameters["alignment_dim"], parameters["classifier_dim"]
        self.fusion = _mlp([dimension**2, dimension])
        self.text_projection = _mlp([dimension, dimension, parameters["unimodal_dim"]])
        self.image_projection = _mlp([dimension, dimension, parameters["unimodal_dim"]])
        self.attention = nn.Sequential(
            nn.Linear(3, parameters["attention_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(parameters["attention_hidden_dim"], 3),
            nn.Sigmoid(),
        )
        self.text_distribution = GaussianEncoder(dimension, parameters["latent_dim"])
        self.image_distribution = GaussianEncoder(dimension, parameters["latent_dim"])
        self.classifier = nn.Sequential(
            _mlp([2 * parameters["unimodal_dim"] + dimension, hidden, hidden]),
            nn.Linear(hidden, 2),
        )

    def forward(self, text, image):
        fusion = self.fusion(_correlation(text, image))
        # Squeeze over feature coordinates, then excite the three modalities (§3.4.1).
        attention = self.attention(torch.stack((text, image, fusion), dim=-1).mean(dim=1))
        features = torch.cat(
            (
                attention[:, :1] * self.text_projection(text),
                attention[:, 1:2] * self.image_projection(image),
                attention[:, 2:] * fusion,
            ),
            dim=-1,
        )
        return self.classifier(features), attention

    def guidance(self, text, image, marginal_kl):
        text = self.text_distribution(text)
        image = self.image_distribution(image)
        ambiguity = (
            (
                0.5
                * (
                    kl_divergence(text, image) / marginal_kl[0]
                    + kl_divergence(image, text) / marginal_kl[1]
                )
            )
            .sigmoid()
            .unsqueeze(-1)
        )
        return torch.cat((1 - ambiguity, 1 - ambiguity, ambiguity), dim=-1)


@register_model("COOLANT")
class COOLANT(AbstractDetector):
    """Paper objectives and fusion, with author CNN/MLP sequence aggregation."""

    def __init__(self, config):
        super().__init__(config)
        parameters = config["model"]["parameters"]
        _validate(parameters)
        self.text_encoder = material_encoder(
            config, "text_encoder", BertEncoder, **parameters["text"]
        )
        self.image_encoder = material_encoder(
            config, "image_encoder", supervised_encoder, **parameters["image"]
        )
        self.contrastive_image_encoder = material_encoder(
            config, "contrastive_image_encoder", supervised_encoder, **parameters["contrastive_image"]
        )
        self.consistency = _Consistency(
            self.text_encoder.output_dim, self.image_encoder.output_dim, parameters
        )
        self.contrastive_text = nn.Sequential(
            _TextProjection(self.text_encoder.output_dim, parameters),
            nn.Linear(parameters["shared_dim"], parameters["alignment_dim"]),
        )
        self.contrastive_image = nn.Linear(
            self.contrastive_image_encoder.output_dim, parameters["alignment_dim"]
        )
        self.log_temperature = nn.Parameter(torch.tensor(math.log(parameters["temperature"])))
        self.aggregation = _Aggregation(parameters)
        self.margin = parameters["margin"]
        self.semantic_weight, self.guidance_weight = (
            parameters["semantic_weight"],
            parameters["guidance_weight"],
        )
        self.normalization = parameters["normalization"]
        self.register_buffer("marginal_kl", torch.ones(2))
        self.register_buffer("marginal_kl_estimate", torch.ones(2))
        self.register_buffer("normalizer_ready", torch.tensor(False))
        self._real_records, self._reference_records = [], []
        self.phase = "detection"
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "phase"):
            self.consistency.train(mode and self.phase == "alignment")
            for module in (
                self.contrastive_text,
                self.contrastive_image,
                self.contrastive_image_encoder,
            ):
                module.train(mode and self.phase == "contrastive")
            self.aggregation.train(mode and self.phase == "detection")
        return self

    def prepare(self, train_dataset):
        self._real_records = [row for row in train_dataset.records if row["label"] == 0]
        if len(self._real_records) < 2:
            raise ValueError("COOLANT matching requires at least two real training news items.")
        generator = torch.Generator().manual_seed(self.normalization["seed"])
        indices = torch.randperm(len(train_dataset.records), generator=generator)
        self._reference_records = [
            {key: train_dataset.records[index][key] for key in ("text", "image_path")}
            for index in indices[: self.normalization["reference_size"]].tolist()
        ]

    def training_phases(self):
        return ("alignment", "contrastive", "detection")

    def training_phase_interval(self):
        return "batch"

    def set_training_phase(self, phase):
        if phase not in self.training_phases():
            raise ValueError(f"Unknown COOLANT training phase: {phase}")
        self.phase = phase
        self.train(self.training)

    def loss_weight(self, batch):
        if self.phase == "alignment":
            return 2 * int(batch["labels"].eq(0).sum())
        return super().loss_weight(batch)

    def _matching_pairs(self, batch):
        indices = batch["labels"].eq(0).nonzero(as_tuple=True)[0].tolist()
        if not indices:
            return None
        text = [batch["text"][index] for index in indices]
        images = [batch["image_path"][index] for index in indices]
        if len(indices) > 1:
            shift = torch.randint(1, len(indices), ()).item()
            negatives = images[shift:] + images[:shift]
        else:
            identifier = batch["sample_id"][indices[0]]
            candidates = [row for row in self._real_records if row["sample_id"] != identifier]
            if not candidates:
                raise ValueError("Prepare COOLANT before sampling singleton matching pairs.")
            negatives = [candidates[torch.randint(len(candidates), ()).item()]["image_path"]]
        return text, images, negatives

    def _consistent(self, text, images):
        return self.consistency(self.text_encoder(text), self.image_encoder(images))

    def _aligned(self, text, images):
        return (
            F.normalize(self.contrastive_text(self.text_encoder(text)), dim=-1),
            F.normalize(self.contrastive_image(self.contrastive_image_encoder(images)), dim=-1),
        )

    def calculate_loss(self, batch):
        if self.phase == "alignment":
            pairs = self._matching_pairs(batch)
            if pairs is None:
                return None
            text, matched, mismatched = pairs
            text, image = self._consistent(text * 2, matched + mismatched)
            targets = text.new_ones(len(text))
            targets[len(text) // 2 :] = -1
            return {"matching": F.cosine_embedding_loss(text, image, targets, margin=self.margin)}
        if self.phase == "contrastive":
            text, image = self._aligned(batch["text"], batch["image_path"])
            temperature = self.log_temperature.exp().clamp_min(torch.finfo(text.dtype).eps)
            logits = image @ text.T / temperature
            with torch.no_grad():
                teacher_text, teacher_image = self._consistent(batch["text"], batch["image_path"])
                teacher = teacher_image @ teacher_text.T / temperature.detach()
                image_targets, text_targets = teacher.softmax(-1), teacher.T.softmax(-1)
            targets = torch.arange(len(text), device=text.device)
            return {
                "contrastive": 0.5
                * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)),
                "semantic": -0.5
                * self.semantic_weight
                * (
                    (image_targets * logits.log_softmax(-1)).sum(-1).mean()
                    + (text_targets * logits.T.log_softmax(-1)).sum(-1).mean()
                ),
            }
        if not self.normalizer_ready.item():
            raise ValueError("Initialize COOLANT's training mixture normalizer before CA training.")
        with torch.no_grad():
            text, image = self._aligned(batch["text"], batch["image_path"])
        logits, attention = self.aggregation(text, image)
        guidance = self.aggregation.guidance(text, image, self.marginal_kl)
        return {
            "classification": F.cross_entropy(logits, batch["labels"]),
            "guidance": self.guidance_weight * _attention_kl(attention, guidance),
        }

    @torch.no_grad()
    def _refresh_normalizer(self):
        if not self._reference_records:
            raise ValueError("Prepare COOLANT on training inputs before fitting its KL normalizer.")
        was_training = self.training
        self.eval()
        try:
            text_parameters, image_parameters = [], []
            for start in range(0, len(self._reference_records), 64):
                records = self._reference_records[start : start + 64]
                text, image = self._aligned(
                    [row["text"] for row in records], [row["image_path"] for row in records]
                )
                text_parameters.append(self.aggregation.text_distribution(text).base_dist)
                image_parameters.append(self.aggregation.image_distribution(image).base_dist)
            text, image = (
                tuple(
                    torch.cat([getattr(value, key) for value in distributions])
                    for key in ("loc", "scale")
                )
                for distributions in (text_parameters, image_parameters)
            )
            estimates = torch.stack(
                [
                    mixture_kl(
                        first, second, self.normalization["samples"], self.normalization["seed"]
                    )
                    for first, second in ((text, image), (image, text))
                ]
            ).to(self.marginal_kl)
            if not torch.isfinite(estimates).all():
                raise ValueError("COOLANT estimated a non-finite training mixture KL divergence.")
            self.marginal_kl_estimate.copy_(estimates)
            self.marginal_kl.copy_(estimates.clamp_min(self.normalization["epsilon"]))
            self.normalizer_ready.fill_(True)
        finally:
            self.train(was_training)

    def on_epoch_start(self, epoch, total_epochs):
        self._refresh_normalizer()

    def on_epoch_end(self, epoch):
        self._refresh_normalizer()

    def forward(self, batch):
        text, image = self._aligned(batch["text"], batch["image_path"])
        return self.aggregation(text, image)[0]

    def predict(self, batch):
        was_training = self.training
        self.eval()
        try:
            return self(batch)
        finally:
            self.train(was_training)
