"""ADOSE-FS/Corrected: three experts, domain alignment and active selection."""

import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder
from mmfact.models_classification._adose_data import _ADOSETraining
from mmfact.models_classification._adose_select import _variance_grid
from mmfact.models_classification._encoders import BertEncoder
from mmfact.models_classification._supervised_vision import supervised_encoder
from mmfact.models_classification.eann import GradientReversal
from mmfact.registry import register_model


def _poe(logits):
    return F.log_softmax(sum(F.log_softmax(values, dim=-1) for values in logits), dim=-1)


def _negotiation(logits, labels):
    """Equation 16: standard per-news JS, averaged over genuine news only."""
    real = labels.eq(0)
    if not real.any():
        return sum(values.sum() for values in logits) * 0
    text, image, cross = [F.log_softmax(values[real], dim=-1) for values in logits]

    def js(first, second):
        log_mixture = torch.logaddexp(first, second) - math.log(2)
        return (
            (first.exp() * (first - log_mixture)).sum(-1)
            + (second.exp() * (second - log_mixture)).sum(-1)
        ) / 2

    return ((js(text, cross) + js(image, cross)) / 2).mean()


def _contrastive(text, image, visual, labels, temperature, threshold):
    """Equations 5--6: real anchors and fixed visual negative-sampling weights."""
    if not labels.eq(0).any():
        return (text.sum() + image.sum()) * 0
    visual = F.normalize(visual.detach(), dim=-1)
    weights = (threshold - (visual @ visual.T + 1) / 2).clamp_min(0)
    weights.fill_diagonal_(0)
    scores = text @ image.T / temperature
    positive = scores.diagonal()
    negatives = scores + weights.masked_fill(weights.eq(0), 1).log()
    negatives = negatives.masked_fill(weights.eq(0), -torch.inf)
    denominator = torch.logsumexp(torch.cat((positive[:, None], negatives), dim=-1), dim=-1)
    return (denominator - positive)[labels.eq(0)].mean()


class _Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, values):
        hidden = self.hidden(values)
        return hidden, self.output(hidden)


@register_model("ADOSE")
class ADOSE(_ADOSETraining, AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        self._validate(p)
        self.parameters_spec = deepcopy(p)
        self._init_training(p)
        self.text_encoder = material_encoder(config, "text_encoder", BertEncoder, **p["text"])
        self.image_encoder = material_encoder(config, "image_encoder", supervised_encoder, **p["image"])
        dimension, dropout = p["hidden_dim"], p["dropout"]
        self.convolutions = nn.ModuleList(
            nn.Conv1d(self.text_encoder.output_dim, p["filter_num"], kernel)
            for kernel in p["kernel_sizes"]
        )
        self.text_projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(p["filter_num"] * len(p["kernel_sizes"]), dimension),
            nn.LayerNorm(dimension),
        )
        self.image_projection = nn.Sequential(
            nn.Linear(self.image_encoder.output_dim, p["image_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(p["image_hidden_dim"], dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dimension),
        )

        def alignment():
            return nn.Sequential(
                nn.Linear(dimension, dimension), nn.ReLU(), nn.Linear(dimension, dimension)
            )

        self.text_alignment, self.image_alignment = alignment(), alignment()
        self.experts = nn.ModuleList(
            _Expert(input_dim, p["classifier_dim"], dropout)
            for input_dim in (dimension, dimension, 2 * dimension)
        )
        self.domain_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(dimension, dimension),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dimension, 4),
            )
            for _ in range(2)
        )

    @staticmethod
    def _validate(p):
        for key in ("hidden_dim", "image_hidden_dim", "classifier_dim", "filter_num"):
            if type(p[key]) is not int or p[key] < 1:
                raise ValueError(f"ADOSE {key} must be a positive integer.")
        kernels = p["kernel_sizes"]
        if (
            not isinstance(kernels, (list, tuple))
            or not kernels
            or any(type(k) is not int or k < 1 for k in kernels)
        ):
            raise ValueError("ADOSE kernel_sizes must be nonempty positive integers.")
        text, image = p["text"], p["image"]
        if (
            text.get("output") != "embeddings"
            or text.get("freeze") is not True
            or text.get("mask_padding") is not False
            or type(text.get("max_length")) is not int
            or text["max_length"] < max(kernels)
            or image.get("output") != "avgpool"
            or image.get("freeze") is not True
            or (
                image.get("backbone") != "vit"
                and image.get("architecture") != "resnet50"
            )
            or type(image.get("center_crop")) is not bool
            or image.get("stem_maxpool", True) is not True
        ):
            raise ValueError(
                "ADOSE requires frozen fixed-length input embeddings and supervised global image features."
            )
        for key in (
            "dropout",
            "temperature",
            "threshold",
            "classifier_weight",
            "adversarial_weight",
            "contrastive_weight",
            "negotiation_weight",
            "reverse_lambda",
        ):
            value = p[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"ADOSE {key} must be finite.")
            if (key == "dropout" and not 0 <= value < 1) or (key != "dropout" and value <= 0):
                raise ValueError(f"ADOSE {key} is outside its valid range.")
        if p["threshold"] > 1:
            raise ValueError("ADOSE threshold must be in (0,1].")
        active = p["active"]
        _variance_grid(active["variances"])
        for key in ("multiplier", "samples_per_variance", "chunk_size"):
            if type(active[key]) is not int or active[key] < 1:
                raise ValueError(f"ADOSE active.{key} must be a positive integer.")

    def features(self, batch):
        embeddings = self.text_encoder(batch["text"]).transpose(1, 2)
        text = self.text_projection(
            torch.cat([conv(embeddings).relu().amax(-1) for conv in self.convolutions], dim=-1)
        )
        image = self.image_projection(self.image_encoder(batch["image_path"]))
        aligned_text = F.normalize(self.text_alignment(text), dim=-1)
        aligned_image = F.normalize(self.image_alignment(image), dim=-1)
        hidden, logits = zip(
            *(
                expert(values)
                for expert, values in zip(
                    self.experts,
                    (text, image, torch.cat((aligned_text, aligned_image), dim=-1)),
                    strict=True,
                )
            ),
            strict=True,
        )
        return {
            "text": text,
            "image": image,
            "aligned_text": aligned_text,
            "aligned_image": aligned_image,
            "expert_hidden": hidden,
            "expert_logits": logits,
            "logits": _poe(logits),
        }

    def selection_inputs(self, batch):
        return self.features(batch)["expert_hidden"]

    def selection_heads(self):
        return tuple(expert.output for expert in self.experts)

    def forward(self, batch):
        return self.features(batch)["logits"]

    def calculate_loss(self, batch):
        result = self.features(batch)
        p = self.parameters_spec
        count = len(result["logits"])
        masks = [batch[key] for key in ("supervised_mask", "adversarial_mask")]
        if any(
            not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (count,)
            for mask in masks
        ):
            raise ValueError(
                "ADOSE supervision and adversarial masks must be boolean batch vectors."
            )
        supervised, adversarial = masks
        labels = batch["labels"][supervised]
        domains = batch["domain_labels"]
        if labels.dtype != torch.long or ((labels < 0) | (labels > 1)).any():
            raise ValueError("ADOSE supervised news must have valid binary labels.")
        if (
            domains.dtype != torch.long
            or domains.shape != (count,)
            or ((domains < 0) | (domains > 3)).any()
        ):
            raise ValueError("ADOSE domain labels must be integers in [0,3].")
        logits = tuple(values[supervised] for values in result["expert_logits"])
        zero = result["logits"].sum() * 0
        classification = F.nll_loss(result["logits"][supervised], labels) if len(labels) else zero
        individual = (
            sum(F.cross_entropy(values, labels) for values in logits) if len(labels) else zero
        )
        domain_loss = (
            sum(
                F.cross_entropy(
                    head(GradientReversal.apply(result[key][adversarial], p["reverse_lambda"])),
                    domains[adversarial],
                )
                for key, head in zip(("text", "image"), self.domain_heads, strict=True)
            )
            if adversarial.any()
            else zero
        )
        return {
            "classification": classification,
            "experts": p["classifier_weight"] * individual,
            "adversarial": p["adversarial_weight"] * domain_loss,
            "contrastive": p["contrastive_weight"]
            * _contrastive(
                result["aligned_text"][supervised],
                result["aligned_image"][supervised],
                result["image"][supervised],
                labels,
                p["temperature"],
                p["threshold"],
            ),
            "negotiation": p["negotiation_weight"] * _negotiation(logits, labels),
        }

    def predict(self, batch):
        was_training = self.training
        self.eval()
        try:
            return self(batch)
        finally:
            self.train(was_training)

    def get_extra_state(self):
        return {"version": 1, "parameters": deepcopy(self.parameters_spec)}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state != self.get_extra_state():
            raise ValueError("ADOSE checkpoint parameters differ from the configured model.")
