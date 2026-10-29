"""Shared supervised visual backbones, with unchanged method-specific outputs.

ViT uses the same RGB views and final-layer representations as the published
MMFact feature matrices. The ResNet branch returns the original encoder, so
its transforms, parameter names and historical checkpoints remain unchanged.
"""

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from mmfact.models_classification._encoders import (
    ResNetEncoder,
    _PretrainedEncoder,
    _load_pretrained_network,
)


MODEL_NAME = "google/vit-base-patch16-224"
MODEL_REVISION = "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3"
ARCHITECTURE = "vit_base_patch16_224"


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"Supervised ViT {name} must be a positive integer.")


def _pixels(image, size=224):
    """The extraction recipe, including PIL rather than tensor interpolation."""
    image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    values = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True))
    return values.permute(2, 0, 1).contiguous().float().div_(255).sub_(0.5).div_(0.5)


def supervised_encoder(*, backbone="resnet", **options):
    """Return the selected native encoder without wrapping the legacy ResNet."""
    if backbone == "resnet":
        return ResNetEncoder(**options)
    if backbone == "vit":
        return SupervisedViTEncoder(backbone=backbone, **options)
    raise ValueError("Supervised visual backbone must be vit or resnet.")


def imagenet_classifier(encoder):
    """Return the actual pretrained classifier, never a newly initialized head."""
    if isinstance(encoder, SupervisedViTEncoder):
        classifier = encoder.network.classifier
    else:
        classifier = getattr(encoder.network, "fc", None)
    if not isinstance(classifier, nn.Linear) or classifier.out_features != 1000:
        raise ValueError("This encoder does not retain its trained ImageNet1000 classifier.")
    return classifier


class SupervisedViTEncoder(_PretrainedEncoder):
    """ViT-B/16 final-LN CLS, trained ImageNet logits or pooled spatial cells."""

    input_kind = "image"
    packages = ("torch", "transformers", "Pillow", "numpy")

    def __init__(
        self,
        *,
        backbone="vit",
        architecture=ARCHITECTURE,
        model_name=MODEL_NAME,
        revision=MODEL_REVISION,
        freeze=True,
        output="avgpool",
        center_crop=False,
        grid_size=4,
    ):
        super().__init__()
        if backbone != "vit" or architecture != ARCHITECTURE:
            raise ValueError("Supervised ViT requires the shared ViT-B/16 architecture.")
        if not isinstance(model_name, (str, Path)) or not str(model_name):
            raise ValueError("Supervised ViT requires a model identifier or local model directory.")
        if str(model_name) != MODEL_NAME and not Path(model_name).is_dir():
            raise ValueError("Supervised ViT requires the shared checkpoint or a local copy.")
        if revision != MODEL_REVISION:
            raise ValueError("Supervised ViT requires the fixed shared checkpoint revision.")
        if type(freeze) is not bool or center_crop is not False:
            raise ValueError("Supervised ViT requires boolean freeze and direct square resizing.")
        if output not in {"avgpool", "cls", "logits", "spatial"}:
            raise ValueError("Supervised ViT output must be avgpool/cls, logits or spatial.")
        _positive_integer(grid_size, "grid_size")

        from transformers import ViTForImageClassification

        self.network = _load_pretrained_network(
            ViTForImageClassification,
            str(model_name),
            revision,
            attn_implementation="eager",
        )
        config = self.network.config
        expected = {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "patch_size": 16,
            "image_size": 224,
            "num_channels": 3,
            "num_labels": 1000,
        }
        if any(getattr(config, key, None) != value for key, value in expected.items()):
            raise ValueError("Loaded supervised ViT does not match ViT-B/16 ImageNet1000.")
        if (
            not isinstance(self.network.classifier, nn.Linear)
            or self.network.classifier.in_features != 768
            or self.network.classifier.out_features != 1000
        ):
            raise ValueError("Supervised ViT requires its trained ImageNet classification head.")
        self.output, self.grid_size = output, grid_size
        self.output_dim = 1000 if output == "logits" else config.hidden_size
        self.transform = _pixels
        self.encoding_spec = {
            "encoder": "supervised-vit-b16",
            "model_name": MODEL_NAME,
            "revision": revision,
            "architecture": architecture,
            "output": "cls" if output == "avgpool" else output,
            "layer": "final-layernorm",
            "attention": "eager",
            "image_size": 224,
            "resize": "PIL-BILINEAR-square",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "postprocessing": "none",
        }
        if output == "spatial":
            self.encoding_spec.update(
                pooling="patch-grid-adaptive-mean", grid_size=grid_size, layout="CHW"
            )
        self._set_frozen(freeze)

    def _set_frozen(self, freeze):
        super()._set_frozen(freeze)
        # An unused classifier must not appear as a trainable component of a
        # CLS/spatial consumer, including after a staged unfreeze operation.
        if self.output != "logits":
            self.network.classifier.requires_grad_(False)

    def encode_pixels(self, pixels):
        """Encode already normalized BCHW pixels, preserving the gradient policy."""
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError("Supervised ViT pixels must have shape [batch, 3, height, width].")
        height, width = pixels.shape[-2:]
        if height < 16 or width < 16 or height % 16 or width % 16:
            raise ValueError("Supervised ViT pixel sizes must be positive multiples of 16.")
        pixels = pixels.to(self.device)
        # Published frozen features were encoded in FP32; a caller's training
        # autocast must not silently change their recipe. Trainable ViT keeps AMP.
        precision = (
            torch.autocast(device_type=self.device.type, enabled=False)
            if self.freeze else nullcontext()
        )
        if self.freeze:
            pixels = pixels.float()
        with precision, torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze):
            tokens = self.network.vit(
                pixel_values=pixels,
                interpolate_pos_encoding=(height, width) != (224, 224),
                return_dict=True,
            ).last_hidden_state
            if self.output == "logits":
                return self.network.classifier(tokens[:, 0])
            if self.output == "spatial":
                cells = tokens[:, 1:].transpose(1, 2).reshape(
                    len(pixels), self.network.config.hidden_size, height // 16, width // 16
                )
                return F.adaptive_avg_pool2d(cells, (self.grid_size, self.grid_size))
            return tokens[:, 0]

    def _encode(self, paths):
        pixels = []
        for path in paths:
            with Image.open(path) as image:
                pixels.append(self.transform(image))
        return self.encode_pixels(torch.stack(pixels))


class ViTIndependentPatches(SupervisedViTEncoder):
    """Encode each source-image crop independently with the same supervised ViT."""

    def __init__(
        self,
        *,
        image_size,
        grid_size,
        patch_batch_size,
        resize="bilinear",
        freeze=True,
        output="avgpool",
        **options,
    ):
        for name, value in (
            ("image_size", image_size),
            ("grid_size", grid_size),
            ("patch_batch_size", patch_batch_size),
        ):
            _positive_integer(value, name)
        if image_size % grid_size:
            raise ValueError("Independent source crops must evenly divide the resized image.")
        if freeze is not True or output not in {"avgpool", "cls"}:
            raise ValueError("Independent ViT patches require frozen CLS representations.")
        if resize not in {"bilinear", "lanczos"}:
            raise ValueError("Independent source resize must be bilinear or lanczos.")
        super().__init__(freeze=True, output=output, grid_size=grid_size, **options)
        self.image_size, self.patch_batch_size = image_size, patch_batch_size
        self.resize = resize
        self.encoding_spec.update(
            encoder="supervised-vit-independent-patches",
            source_image_size=image_size,
            source_resize=f"PIL-{resize.upper()}",
            grid_size=grid_size,
            patch_size=image_size // grid_size,
            order="row-major",
            patch_batch_size=patch_batch_size,
        )

    def _encode(self, paths):
        interpolation = getattr(Image.Resampling, self.resize.upper())
        results = []
        side = self.image_size // self.grid_size
        for path in paths:
            with Image.open(path) as source:
                image = source.convert("RGB").resize(
                    (self.image_size, self.image_size), interpolation
                )
                pixels = torch.stack(
                    [
                        self.transform(image.crop((x * side, y * side, (x + 1) * side, (y + 1) * side)))
                        for y in range(self.grid_size)
                        for x in range(self.grid_size)
                    ]
                )
            results.append(
                torch.cat(
                    [
                        self.encode_pixels(pixels[start : start + self.patch_batch_size])
                        for start in range(0, len(pixels), self.patch_batch_size)
                    ]
                )
            )
        return torch.stack(results)


class ViTMultiScaleEncoder(SupervisedViTEncoder):
    """Three independently resized views, with positional interpolation below224."""

    def __init__(self, *, scales=(224, 112, 64), freeze=True, output="avgpool", **options):
        if tuple(scales) != (224, 112, 64):
            raise ValueError("The shared supervised ViT scale recipe is [224, 112, 64].")
        if freeze is not True or output not in {"avgpool", "cls"}:
            raise ValueError("Multi-scale ViT requires frozen CLS representations.")
        super().__init__(freeze=True, output=output, **options)
        self.scales = tuple(scales)
        self.encoding_spec.update(
            encoder="supervised-vit-three-scales",
            scales=list(scales),
            position_interpolation="non224-bicubic",
            order="224-112-64",
        )

    def _encode(self, paths):
        pixels = [[] for _ in self.scales]
        for path in paths:
            with Image.open(path) as source:
                image = source.convert("RGB")
                for values, scale in zip(pixels, self.scales):
                    values.append(_pixels(image, scale))
        return torch.stack(
            [self.encode_pixels(torch.stack(values)) for values in pixels], dim=1
        )
