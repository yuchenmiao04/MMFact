"""HMCAN: hierarchical BERT semantics and bidirectional contextual attention."""

import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import hub_inputs, material_encoder, material_mask
from mmfact.models_classification._encoders import BertEncoder, ResNetEncoder
from mmfact.models_classification._layers import SingletonBatchNorm
from mmfact.models_classification._transformers import sinusoidal_positions
from mmfact.registry import register_model


def _mean(values, mask):
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(1) / weights.sum(1).clamp_min(1)


class _AuthorNorm(nn.Module):
    """The selected author implementation uses sample standard deviation."""

    def __init__(self, dimension, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.bias = nn.Parameter(torch.zeros(dimension))
        self.eps = eps

    def forward(self, x):
        return (
            self.weight * (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + self.eps)
            + self.bias
        )


class _AttentionBlock(nn.Module):
    """Paper Equations 3-8: query residuals, post-norm attention and FFN."""

    def __init__(self, dimension, heads, ffn_dim, dropout, eps):
        super().__init__()
        self.heads, self.head_dim = heads, dimension // heads
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.projection = nn.Linear(dimension, dimension)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(dimension, eps=eps)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, ffn_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(ffn_dim, dimension),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(dimension, eps=eps)

    def forward(self, query, context, key_mask, query_mask):
        if (
            query.ndim != 3
            or context.ndim != 3
            or query.shape[0] != context.shape[0]
            or query.shape[2] != context.shape[2]
            or key_mask.shape != context.shape[:2]
            or query_mask.shape != query.shape[:2]
            or key_mask.dtype != torch.bool
            or query_mask.dtype != torch.bool
            or not key_mask.any(-1).all()
        ):
            raise ValueError("HMCAN attention requires valid per-news token masks and features.")
        batch, length, dimension = query.shape

        def heads(values):
            return values.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(self.query(query)), heads(self.key(context)), heads(self.value(context))
        scores = (q @ k.transpose(-1, -2)) * self.head_dim**-0.5
        scores = scores.masked_fill(~key_mask[:, None, None, :], -torch.inf)
        weights = self.attention_dropout(scores.softmax(-1))
        attended = (weights @ v).transpose(1, 2).reshape(batch, length, dimension)
        values = self.dropout(self.attention_norm(query + self.projection(attended)))
        values = self.ffn_norm(values + self.ffn(values))
        return values * query_mask.unsqueeze(-1).to(values.dtype)


class _ContextualTransformer(nn.Module):
    """Self-attend input1, then query it with input2; concatenate both means."""

    def __init__(
        self,
        dimension,
        self_layers,
        context_layers,
        heads,
        ffn_dim,
        dropout,
        eps,
        init_std,
        implementation="paper",
    ):
        super().__init__()
        self.implementation = implementation
        self.input_norm = nn.LayerNorm(dimension, eps=eps)
        self.input_dropout = nn.Dropout(dropout)
        self.self_layers = nn.ModuleList(
            [_AttentionBlock(dimension, heads, ffn_dim, dropout, eps) for _ in range(self_layers)]
        )
        self.context_layers = nn.ModuleList(
            [
                _AttentionBlock(dimension, heads, ffn_dim, dropout, eps)
                for _ in range(context_layers)
            ]
        )
        if implementation == "author":
            self.input_norm = _AuthorNorm(dimension, eps)
            for block in [*self.self_layers, *self.context_layers]:
                block.attention_norm = _AuthorNorm(dimension, eps)
                block.ffn_norm = _AuthorNorm(dimension, eps)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=init_std, a=-2 * init_std, b=2 * init_std)
                nn.init.trunc_normal_(module.bias, std=init_std, a=-2 * init_std, b=2 * init_std)

        if implementation == "author":
            for param in self.parameters():
                nn.init.trunc_normal_(param, std=init_std, a=-2 * init_std, b=2 * init_std)

    def forward(self, first, second, first_mask, second_mask):
        if self.implementation == "author":
            positions = torch.arange(first.shape[1], device=first.device, dtype=first.dtype)[
                :, None
            ]
            dims = torch.arange(first.shape[2], device=first.device, dtype=first.dtype)
            pe = torch.zeros(first.shape[1:], device=first.device, dtype=first.dtype)
            divisor = 10000 ** (2 * dims / first.shape[2])
            pe[:, 0::2] = torch.sin(positions / divisor[0::2])
            pe[:, 1::2] = torch.cos(positions / divisor[1::2])
            source = self.input_dropout(self.input_norm(first) + pe)
            # Original encoder loop reuses unchanged Q/K/V; only last layer output reaches loss.
            for layer in self.self_layers:
                sources = layer(source, source, first_mask, first_mask)
            for layer in self.context_layers:
                context = layer(second, sources, first_mask, second_mask)
            return torch.cat((sources.mean(1), context.mean(1)), -1)
        sources = self.input_dropout(self.input_norm(first) + sinusoidal_positions(first))
        sources = sources * first_mask.unsqueeze(-1).to(sources.dtype)
        for layer in self.self_layers:
            sources = layer(sources, sources, first_mask, first_mask)
        context = second
        for layer in self.context_layers:
            context = layer(context, sources, first_mask, second_mask)
        return torch.cat((_mean(sources, first_mask), _mean(context, second_mask)), -1)


@register_model("HMCAN")
class HMCAN(AbstractDetector):
    """Hierarchical contextual attention with explicit paper and author variants."""

    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        for name in ("self_layers", "context_layers", "num_heads", "ffn_dim", "classifier_hidden"):
            if not isinstance(p[name], int) or isinstance(p[name], bool) or p[name] < 1:
                raise ValueError(f"HMCAN {name} must be a positive integer.")
        for name in ("dropout", "alpha", "layer_norm_eps", "transformer_init_std"):
            value = p[name]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or (
                    not 0 <= value < 1
                    if name == "dropout"
                    else not 0 <= value <= 1 if name == "alpha" else value <= 0
                )
            ):
                raise ValueError(f"HMCAN {name} is outside its finite permitted range.")
        text, image = p["text"], p["image"]
        if not isinstance(text, dict) or not isinstance(image, dict):
            raise ValueError("HMCAN text and image settings must be dictionaries.")
        for name in ("model_name", "revision"):
            value = text.get(name, "main" if name == "revision" else None)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"HMCAN text.{name} must be a nonempty string.")
        length = text.get("max_length", 200)
        if not isinstance(length, int) or isinstance(length, bool) or length < 2:
            raise ValueError("HMCAN text.max_length must be an integer >= 2.")
        if (
            text.get("freeze", True) is not True
            or image.get("freeze", True) is not True
            or text.get("mask_padding", False) is not True
            or text.get("output") != "layer_groups"
            or image.get("output") != "spatial"
        ):
            raise ValueError("HMCAN requires frozen BERT layer groups and spatial ResNet features.")
        groups = text.get("layer_groups")
        if (
            not isinstance(groups, list)
            or len(groups) != 3
            or any(not isinstance(group, list) or not group for group in groups)
            or len({len(group) for group in groups}) != 1
            or any(
                not isinstance(index, int) or isinstance(index, bool) or index < 1
                for group in groups
                for index in group
            )
            or [index for group in groups for index in group]
            != list(range(1, sum(map(len, groups)) + 1))
        ):
            raise ValueError("HMCAN needs three equal, contiguous, nonoverlapping BERT groups.")
        grid = image.get("grid_size", 4)
        if not isinstance(grid, int) or isinstance(grid, bool) or grid < 2:
            raise ValueError("HMCAN image.grid_size must be an integer >= 2.")
        architecture = image.get("architecture", "resnet34")
        if not isinstance(architecture, str) or architecture not in {
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
        }:
            raise ValueError("HMCAN requires a supported ResNet architecture.")
        weights = image.get("weights", "IMAGENET1K_V1")
        if weights is not None and (
            not isinstance(weights, str) or weights not in {"IMAGENET1K_V1", "IMAGENET1K_V2"}
        ):
            raise ValueError("HMCAN image.weights must be a supported ImageNet preset or null.")
        if weights == "IMAGENET1K_V2" and architecture in {"resnet18", "resnet34"}:
            raise ValueError("HMCAN this ResNet architecture has no V2 weights.")
        for name in ("stem_maxpool", "center_crop"):
            if name in image and not isinstance(image[name], bool):
                raise ValueError(f"HMCAN image.{name} must be boolean.")

        self.implementation = p.get("implementation", "paper")
        if self.implementation not in {"paper", "author"}:
            raise ValueError("HMCAN implementation must be paper or author")
        self.parameters_spec = deepcopy(p)
        self.alpha = p["alpha"]
        self.text_encoder = material_encoder(config, "text_encoder", BertEncoder, **text)
        blocks = (
            sum(map(len, self.text_encoder.encoding_spec["layer_groups"]))
            if hub_inputs(config)
            else self.text_encoder.network.config.num_hidden_layers
        )
        if sum(map(len, groups)) != blocks:
            raise ValueError("HMCAN layer groups must cover all BERT Transformer blocks.")
        dimension = self.text_encoder.output_dim
        if dimension % p["num_heads"]:
            raise ValueError("HMCAN BERT dimension must divide evenly among attention heads.")
        self.image_encoder = material_encoder(config, "image_encoder", ResNetEncoder, **image)
        self.image_projection = nn.Sequential(
            nn.Conv2d(self.image_encoder.output_dim, dimension, kernel_size=1),
            nn.BatchNorm2d(dimension),
            nn.ReLU(),
        )
        options = dict(
            dimension=dimension,
            self_layers=p["self_layers"],
            context_layers=p["context_layers"],
            heads=p["num_heads"],
            ffn_dim=p["ffn_dim"],
            dropout=p["dropout"],
            eps=p["layer_norm_eps"],
            init_std=p["transformer_init_std"],
            implementation=self.implementation,
        )
        self.text_image = _ContextualTransformer(**options)
        self.image_text = _ContextualTransformer(**options)
        self.classifier = nn.Sequential(
            nn.Linear(6 * dimension, p["classifier_hidden"]),
            nn.ReLU(),
            SingletonBatchNorm(p["classifier_hidden"]),
            nn.Linear(p["classifier_hidden"], 2),
        )

    def forward(self, batch):
        texts, paths = batch["text"], batch["image_path"]
        if not texts or len(texts) != len(paths):
            raise ValueError("HMCAN needs one text and image per item in a nonempty batch.")
        text = self.text_encoder(texts)
        text_mask = material_mask(self.text_encoder, texts).to(text.device, dtype=torch.bool)
        if text_mask.ndim == 3:
            if not torch.equal(text_mask, text_mask[:, :1].expand_as(text_mask)):
                raise ValueError("HMCAN layer groups must share one input token mask.")
            text_mask = text_mask[:, 0]
        if self.implementation == "author":
            text_mask = torch.ones_like(text_mask)
        image = self.image_projection(self.image_encoder(paths)).flatten(2).transpose(1, 2)
        image_mask = torch.ones(image.shape[:2], dtype=torch.bool, device=image.device)
        groups = []
        for values in text.unbind(1):
            text_image = self.text_image(values, image, text_mask, image_mask)
            image_text = self.image_text(image, values, image_mask, text_mask)
            groups.append(self.alpha * text_image + (1 - self.alpha) * image_text)
        return self.classifier(torch.cat(groups, -1))

    def calculate_loss(self, batch):
        return {"classification": F.cross_entropy(self(batch), batch["labels"])}

    def predict(self, batch):
        modes = {module: module.training for module in self.modules()}
        self.eval()
        try:
            return self(batch)
        finally:
            for module, mode in modes.items():
                module.training = mode

    def get_extra_state(self):
        return {"format_version": 1, "parameters": deepcopy(self.parameters_spec)}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state != self.get_extra_state():
            raise ValueError("HMCAN checkpoint parameters do not match this configuration.")
