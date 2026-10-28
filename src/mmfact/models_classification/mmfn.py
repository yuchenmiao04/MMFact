"""MMFN multi-grained fusion with fixed OCR/translation/summary resources."""

import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder
from mmfact.materials.resources import material_resources
from mmfact.models_classification._encoders import (
    BertEncoder,
    CLIPImageEncoder,
    CLIPTextEncoder,
    MAETokenEncoder,
)
from mmfact.models_classification._generated import GeneratedTextStore, combine_news
from mmfact.models_classification._layers import SingletonBatchNorm
from mmfact.registry import register_model


class _ResourceCLIPText(CLIPTextEncoder):
    """The cache sees the same generated English input as the online text tower."""

    def __init__(self, resources, **options):
        super().__init__(**options)
        self._lookup = resources.lookup
        self.encoding_spec.update(
            input_recipe="mmfn_generated_clip_text_v1", generator=deepcopy(resources.protocol)
        )

    def inputs_from_rows(self, rows):
        return [self._lookup(row["text"], row["image_path"])["clip_text"] for row in rows]


class _CrossTransformer(nn.Module):
    """Shared bidirectional CT; Eq.1 divides logits by d_h, Figure2 uses post-norm."""

    def __init__(self, dimension, heads, ffn_dim, dropout, norm_eps):
        super().__init__()
        self.heads, self.head_dim = heads, dimension // heads
        self.query, self.key, self.value = (nn.Linear(dimension, dimension) for _ in range(3))
        self.output = nn.Linear(dimension, dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, dimension)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(dimension, eps=norm_eps) for _ in range(2))
        self.attention_dropout, self.residual_dropout = nn.Dropout(dropout), nn.Dropout(dropout)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def _attend(self, query, key, key_mask):
        shape = (len(query), -1, self.heads, self.head_dim)
        q, k, v = (
            layer(values).reshape(shape).transpose(1, 2)
            for layer, values in ((self.query, query), (self.key, key), (self.value, key))
        )
        scores = (q @ k.transpose(-2, -1)) / self.head_dim
        weights = scores.masked_fill(~key_mask[:, None, None, :], -torch.inf).softmax(-1)
        values = (self.attention_dropout(weights) @ v).transpose(1, 2).reshape_as(query)
        return self.output(values)

    def _direction(self, query, key, query_mask, key_mask):
        values = self.norms[0](query + self.residual_dropout(self._attend(query, key, key_mask)))
        values = self.norms[1](values + self.residual_dropout(self.ffn(values)))
        return values * query_mask.unsqueeze(-1).to(values.dtype)

    def forward(self, text, image, text_mask):
        image_mask = torch.ones(image.shape[:2], dtype=torch.bool, device=image.device)
        return (
            self._direction(text, image, text_mask, image_mask),
            self._direction(image, text, image_mask, text_mask),
        )


def _mean(values, mask):
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(1) / weights.sum(1).clamp_min(1)


def _branch(input_dim, hidden_dim, dropout):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        SingletonBatchNorm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


def _projection(input_dim, hidden_dim, output_dim, dropout):
    return nn.Sequential(
        *_branch(input_dim, hidden_dim, dropout),
        nn.Linear(hidden_dim, output_dim),
        SingletonBatchNorm(output_dim),
        nn.ReLU(),
    )


def _validate(p):
    sizes = [
        p[key]
        for key in (
            "attention_dim",
            "num_heads",
            "ffn_dim",
            "hidden_dim",
            "projection_dim",
            "classifier_dim",
        )
    ]
    if any(type(value) is not int or value < 1 for value in sizes):
        raise ValueError("MMFN dimensions and head count must be positive integers.")
    if p["attention_dim"] % p["num_heads"]:
        raise ValueError("MMFN attention_dim must be divisible by num_heads.")
    for key in ("attention_dropout", "branch_dropout", "projection_dropout", "layer_norm_eps"):
        value = p[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("MMFN dropout and LayerNorm epsilon must be finite numbers.")
        if (key == "layer_norm_eps" and value <= 0) or (
            key != "layer_norm_eps" and not 0 <= value < 1
        ):
            raise ValueError("MMFN dropout or LayerNorm epsilon is outside its valid range.")
    text, image, clip_text, clip_image = (
        p[key] for key in ("text", "image", "clip_text", "clip_image")
    )
    if (
        text.get("freeze") is not False
        or image.get("freeze") is not False
        or text.get("output") != "tokens"
        or text.get("mask_padding") is not True
        or image.get("output") != "patches"
        or type(text.get("max_length")) is not int
        or text["max_length"] < 2
    ):
        raise ValueError(
            "MMFN requires trainable BERT tokens with padding masks and MAE patch tokens."
        )
    if (
        clip_text.get("model_name") != clip_image.get("model_name")
        or clip_text.get("revision", "main") != clip_image.get("revision", "main")
        or any(options.get("freeze") is not True for options in (clip_text, clip_image))
        or any(options.get("normalize", False) is not False for options in (clip_text, clip_image))
        or any(options.get("output", "pooled") != "pooled" for options in (clip_text, clip_image))
        or type(clip_text.get("max_length")) is not int
        or not 2 <= clip_text["max_length"] <= 77
    ):
        raise ValueError(
            "MMFN requires paired frozen raw pooled CLIP towers and native text length."
        )
    for options in (text, image, clip_text, clip_image):
        if not isinstance(options.get("model_name"), str) or not options["model_name"].strip():
            raise ValueError("MMFN backbone names must be nonempty strings.")
        if (
            not isinstance(options.get("revision", "main"), str)
            or not options.get("revision", "main").strip()
        ):
            raise ValueError("MMFN backbone revisions must be nonempty strings.")
    resources = p["resources"]
    if (
        not isinstance(resources.get("language"), str)
        or resources["language"] not in {"en", "zh"}
        or not isinstance(resources.get("manifest"), str)
        or not resources["manifest"].strip()
    ):
        raise ValueError("MMFN requires an en/zh fixed resource manifest.")


@register_model("MMFN")
class MMFN(AbstractDetector):
    """Paper CT and multi-grained cosine fusion; explicitly adapted generated resources."""

    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        _validate(p)
        self.parameters_spec = deepcopy(p)
        self.resources = material_resources(
            config, "mmfn", GeneratedTextStore, task="mmfn", **p["resources"]
        )
        self.text_encoder = material_encoder(config, "text_encoder", BertEncoder, **p["text"])
        self.image_encoder = material_encoder(
            config, "image_encoder", MAETokenEncoder, **p["image"]
        )
        self.clip_text_encoder = material_encoder(
            config, "clip_text_encoder", _ResourceCLIPText, self.resources, **p["clip_text"]
        )
        self.clip_image_encoder = material_encoder(
            config, "clip_image_encoder", CLIPImageEncoder, **p["clip_image"]
        )
        if self.clip_text_encoder.output_dim != self.clip_image_encoder.output_dim:
            raise ValueError("MMFN CLIP projected dimensions must match.")
        text_dim, image_dim, clip_dim = (
            self.text_encoder.output_dim,
            self.image_encoder.output_dim,
            self.clip_text_encoder.output_dim,
        )
        dimension, hidden, output = p["attention_dim"], p["hidden_dim"], p["projection_dim"]
        self.text_mapping = nn.Linear(text_dim, dimension)
        self.image_mapping = nn.Linear(image_dim, dimension)
        self.cross_transformer = _CrossTransformer(
            dimension, p["num_heads"], p["ffn_dim"], p["attention_dropout"], p["layer_norm_eps"]
        )
        self.fine = _branch(2 * dimension, hidden, p["branch_dropout"])
        self.coarse = _branch(2 * clip_dim, hidden, p["branch_dropout"])
        self.multimodal_projection = _projection(
            2 * hidden, hidden, output, p["projection_dropout"]
        )
        self.text_projection = _projection(
            text_dim + clip_dim, hidden, output, p["projection_dropout"]
        )
        self.image_projection = _projection(
            image_dim + clip_dim, hidden, output, p["projection_dropout"]
        )
        self.classifier = nn.Sequential(
            nn.Linear(3 * output, p["classifier_dim"]),
            SingletonBatchNorm(p["classifier_dim"]),
            nn.ReLU(),
            nn.Linear(p["classifier_dim"], 2),
        )

    def get_extra_state(self):
        return {"format_version": 1, "parameters": deepcopy(self.parameters_spec)}

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or state.get("format_version") != 1
            or state.get("parameters") != self.parameters_spec
        ):
            raise ValueError("MMFN checkpoint parameters differ from this model.")

    def optimizer_parameter_groups(self):
        backbone = [
            parameter
            for encoder in (self.text_encoder, self.image_encoder)
            for parameter in encoder.parameters()
            if parameter.requires_grad
        ]
        backbone_ids = {id(parameter) for parameter in backbone}
        return {
            "default": [
                parameter
                for parameter in self.parameters()
                if parameter.requires_grad and id(parameter) not in backbone_ids
            ],
            "backbone": backbone,
        }

    def _inputs(self, batch):
        records = [
            self.resources.lookup(text, image)
            for text, image in zip(batch["text"], batch["image_path"])
        ]
        texts = [combine_news(text, record["ocr"]) for text, record in zip(batch["text"], records)]
        mask = (
            self.text_encoder.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.text_encoder.max_length,
                return_tensors="pt",
            )["attention_mask"]
            .to(self.text_mapping.weight.device)
            .bool()
        )
        return texts, [record["clip_text"] for record in records], mask

    def forward(self, batch):
        texts, clip_texts, mask = self._inputs(batch)
        text, image = self.text_encoder(texts), self.image_encoder(batch["image_path"])
        clip_text, clip_image = (
            self.clip_text_encoder(clip_texts),
            self.clip_image_encoder(batch["image_path"]),
        )
        text_att, image_att = self.cross_transformer(
            self.text_mapping(text), self.image_mapping(image), mask
        )
        fine = self.fine(torch.cat((_mean(text_att, mask), image_att.mean(1)), -1))
        coarse = self.coarse(torch.cat((clip_text, clip_image), -1))
        multimodal = self.multimodal_projection(torch.cat((fine, coarse), -1))
        multimodal = multimodal * F.cosine_similarity(clip_text, clip_image, dim=-1)[:, None]
        text_prime = self.text_projection(torch.cat((_mean(text, mask), clip_text), -1))
        image_prime = self.image_projection(torch.cat((image.mean(1), clip_image), -1))
        return self.classifier(torch.cat((multimodal, text_prime, image_prime), -1))

    def calculate_loss(self, batch):
        return F.cross_entropy(self(batch), batch["labels"])

    def predict(self, batch):
        was_training = self.training
        try:
            self.eval()
            return self(batch)
        finally:
            self.train(was_training)
