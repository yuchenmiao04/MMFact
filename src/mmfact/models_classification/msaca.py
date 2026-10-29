"""MSACA-PaperLed/Corrected: three-scale semantic alignment and attention."""

import math
from copy import deepcopy

import torch
from PIL import Image
from torch import nn
from torch.distributions import Independent, Normal
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder
from mmfact.models_classification._encoders import BertEncoder, ResNetEncoder
from mmfact.models_classification._supervised_vision import ViTMultiScaleEncoder
from mmfact.models_classification._layers import SingletonBatchNorm
from mmfact.registry import register_model


def _positive_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate(parameters):
    for key in (
        "cnn_channels",
        "alignment_dim",
        "image_hidden_dim",
        "unimodal_dim",
        "classifier_dim",
    ):
        if not _positive_integer(parameters[key]):
            raise ValueError(f"MSACA {key} must be a positive integer.")
    kernels = parameters["cnn_kernels"]
    hidden = parameters["attention_hidden_dims"]
    if (
        not isinstance(kernels, (list, tuple))
        or not kernels
        or not all(_positive_integer(value) for value in kernels)
        or len(set(kernels)) != len(kernels)
    ):
        raise ValueError("MSACA cnn_kernels must be distinct positive integers.")
    if (
        not isinstance(hidden, (list, tuple))
        or len(hidden) != 2
        or not all(_positive_integer(value) for value in hidden)
    ):
        raise ValueError("MSACA attention_hidden_dims requires two positive integers.")
    for key in ("dropout", "alpha"):
        value = parameters[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"MSACA {key} must be finite.")
    if not 0 <= parameters["dropout"] < 1 or parameters["alpha"] <= 0:
        raise ValueError("MSACA requires dropout in [0,1) and alpha > 0.")
    text, image = parameters["text"], parameters["image"]
    if (
        text.get("freeze") is not True
        or text.get("output") != "tokens"
        or text.get("mask_padding") is not False
        or text.get("normalize", False) is not False
        or not _positive_integer(text.get("max_length"))
        or text["max_length"] < max(kernels)
    ):
        raise ValueError(
            "MSACA needs frozen unnormalized tokens with retained pads and L>=kernels."
        )
    if (
        set(image) - {
            "freeze", "weights", "scales", "architecture", "backbone",
            "model_name", "revision", "center_crop",
        }
        or image.get("freeze") is not True
    ):
        raise ValueError("MSACA image settings are freeze=true, weights and three scales.")
    if image.get("weights") not in (None, "IMAGENET1K_V1"):
        raise ValueError("MSACA ResNet weights must be IMAGENET1K_V1 or null.")
    if image.get("backbone") != "vit" and image.get("architecture", "resnet50") not in {
        "resnet34", "resnet50"
    }:
        raise ValueError("MSACA requires a supervised ViT or ResNet image backbone.")
    _validate_scales(image.get("scales"))


def _validate_scales(scales):
    if (
        not isinstance(scales, (list, tuple))
        or len(scales) != 3
        or not all(_positive_integer(value) for value in scales)
        or any(first <= second for first, second in zip(scales, scales[1:]))
    ):
        raise ValueError("MSACA needs exactly three strictly decreasing positive image scales.")


class _MultiScaleResNet(ResNetEncoder):
    """One frozen ResNet, with the method's three square RGB/ImageNet transforms."""

    def __init__(
        self, weights="IMAGENET1K_V1", freeze=True, scales=(224, 112, 56), architecture="resnet50"
    ):
        _validate_scales(scales)
        if freeze is not True:
            raise ValueError("MSACA ResNet must be frozen.")
        super().__init__(freeze=True, weights=weights, architecture=architecture, center_crop=False)
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        self.scales = tuple(scales)
        self.transforms = tuple(
            transforms.Compose(
                [
                    transforms.Resize((scale, scale), interpolation=InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            for scale in scales
        )
        self.encoding_spec.update(
            encoder=f"msaca-three-scale-{architecture}",
            scales=list(scales),
            transform=[repr(transform) for transform in self.transforms],
            output="ordered-scale-avgpool",
        )

    def _encode(self, paths):
        pixels = [[] for _ in self.scales]
        for path in paths:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                for values, transform in zip(pixels, self.transforms):
                    values.append(transform(rgb))
        with torch.no_grad():
            return torch.stack(
                [self.network(torch.stack(values).to(self.device)) for values in pixels], dim=1
            )


def _multiscale_encoder(**options):
    if options.get("backbone") == "vit":
        return ViTMultiScaleEncoder(**options)
    return _MultiScaleResNet(**options)


def _mlp(input_dim, dimensions, dropout=0.0):
    layers = []
    for index, dimension in enumerate(dimensions):
        layers.extend((nn.Linear(input_dim, dimension), SingletonBatchNorm(dimension), nn.ReLU()))
        if dropout and index < len(dimensions) - 1:
            layers.append(nn.Dropout(dropout))
        input_dim = dimension
    return nn.Sequential(*layers)


class _TextCNN(nn.Module):
    def __init__(self, input_dim, channels, kernels):
        super().__init__()
        self.convolutions = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(input_dim, channels, kernel),
                SingletonBatchNorm(channels),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
            )
            for kernel in kernels
        )

    def forward(self, tokens):
        return torch.cat(
            [layer(tokens.transpose(1, 2)).squeeze(-1) for layer in self.convolutions], dim=-1
        )


class _Gaussian(nn.Module):
    """Three FC layers per parameter path, shared visual parameters across scales."""

    def __init__(self, dimension):
        super().__init__()
        self.hidden = _mlp(dimension, [dimension, dimension])
        self.mean = nn.Linear(dimension, dimension)
        self.rho = nn.Linear(dimension, dimension)

    def forward(self, values):
        hidden = self.hidden(values)
        return Independent(Normal(self.mean(hidden), F.softplus(self.rho(hidden)) + 1e-7), 1)


def _cross_fusion(text, image):
    """Eq3--4: explicit coordinate alignment and source-aided max pooling."""
    scores = text.unsqueeze(-1) * image.unsqueeze(-2) / math.sqrt(text.shape[-1])
    forward = scores.softmax(-1)
    backward = scores.transpose(-1, -2).softmax(-1).transpose(-1, -2)
    return (forward * backward).amax(-1)


def _prior_kl(distribution):
    mean, scale = distribution.base_dist.loc, distribution.base_dist.scale
    return 0.5 * (mean.square() + scale.square() - 1 - 2 * scale.log()).sum(-1)


def _alignment_loss(text, images, labels):
    """Eq11: one textual KL plus three visual KLs, mean over real news only."""
    values = _prior_kl(text) + sum(_prior_kl(image) for image in images)
    real = labels.eq(0)
    return values[real].mean() if real.any() else values.sum() * 0


@register_model("MSACA")
class MSACA(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        parameters = config["model"]["parameters"]
        _validate(parameters)
        self.parameters_config = deepcopy(parameters)
        self.text_encoder = material_encoder(
            config, "text_encoder", BertEncoder, **parameters["text"]
        )
        self.image_encoder = material_encoder(
            config, "image_encoder", _multiscale_encoder, **parameters["image"]
        )
        dimension, unimodal = parameters["alignment_dim"], parameters["unimodal_dim"]
        self.alpha = parameters["alpha"]
        self.text_cnn = _TextCNN(
            self.text_encoder.output_dim, parameters["cnn_channels"], parameters["cnn_kernels"]
        )
        self.text_projection = _mlp(
            parameters["cnn_channels"] * len(parameters["cnn_kernels"]),
            [dimension, dimension],
            parameters["dropout"],
        )
        self.image_projection = _mlp(
            self.image_encoder.output_dim,
            [parameters["image_hidden_dim"], dimension],
            parameters["dropout"],
        )
        self.text_distribution, self.image_distribution = _Gaussian(dimension), _Gaussian(dimension)
        self.text_residual = nn.Linear(dimension, dimension)
        self.image_residual = nn.Linear(dimension, dimension)
        self.cross_residual = nn.Linear(dimension, dimension)
        self.text_head = _mlp(dimension, [dimension, unimodal])
        self.image_head = _mlp(dimension, [dimension, unimodal])
        self.cross_head = _mlp(dimension, [dimension])
        features = 4 * unimodal + 3 * dimension
        self.attention = _mlp(features, [*parameters["attention_hidden_dims"], 7])
        self.classifier = nn.Sequential(
            _mlp(features, [parameters["classifier_dim"]] * 2, parameters["dropout"]),
            nn.Dropout(parameters["dropout"]),
            nn.Linear(parameters["classifier_dim"], 2),
        )

    def _features(self, batch):
        text = self.text_projection(self.text_cnn(self.text_encoder(batch["text"])))
        images = [
            self.image_projection(image)
            for image in self.image_encoder(batch["image_path"]).unbind(1)
        ]
        text_distribution = self.text_distribution(text)
        image_distributions = [self.image_distribution(image) for image in images]
        z_text = text_distribution.rsample() if self.training else text_distribution.mean
        z_images = [q.rsample() if self.training else q.mean for q in image_distributions]
        text_view = self.text_head(self.text_residual(z_text) + text)
        image_views = [
            self.image_head(self.image_residual(latent) + image)
            for latent, image in zip(z_images, images)
        ]
        cross_views = [
            self.cross_head(
                self.cross_residual(_cross_fusion(z_text, latent)) + _cross_fusion(text, image)
            )
            for latent, image in zip(z_images, images)
        ]
        views = [text_view, *image_views, *cross_views]
        attention = self.attention(torch.cat(views, -1)).softmax(-1)
        weighted = torch.cat(
            [value * attention[:, index : index + 1] for index, value in enumerate(views)], -1
        )
        return self.classifier(weighted), text_distribution, image_distributions, attention, views

    def forward(self, batch):
        return self._features(batch)[0]

    def calculate_loss(self, batch):
        logits, text, images, _, _ = self._features(batch)
        return {
            "classification": F.cross_entropy(logits, batch["labels"]),
            "alignment": self.alpha * _alignment_loss(text, images, batch["labels"]),
        }

    def predict(self, batch):
        modes = tuple((module, module.training) for module in self.modules())
        self.eval()
        try:
            return self(batch)
        finally:
            for module, training in modes:
                module.training = training

    def get_extra_state(self):
        return {"parameters": deepcopy(self.parameters_config)}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state != self.get_extra_state():
            raise ValueError("MSACA checkpoint parameters do not match this configuration.")
