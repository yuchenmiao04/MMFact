"""Shared supervised heads for the five neural reference baselines."""

import json
import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder
from mmfact.models_classification._encoders import BertEncoder, CLIPImageEncoder, CLIPTextEncoder
from mmfact.models_classification._supervised_vision import supervised_encoder


def _signature(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Neural baseline parameters must contain finite basic values.") from error


def _head(input_dim, hidden, dropout):
    layers = []
    for dimension in hidden:
        layers.extend((nn.Linear(input_dim, dimension), nn.ReLU(), nn.Dropout(dropout)))
        input_dim = dimension
    if not hidden and dropout:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(input_dim, 2))
    return nn.Sequential(*layers)


class _NeuralBaseline(AbstractDetector):
    """One pooled representation per permitted modality and an ordinary CE head."""

    modalities = ("text", "image")
    clip = False
    head_kind = None

    def __init__(self, config):
        super().__init__(config)
        p = deepcopy(config["model"]["parameters"])
        self._validate(p)
        self.parameters_spec = p
        self._parameters_signature = _signature(p)
        dimension = 0
        encoder_types = {
            "text": CLIPTextEncoder if self.clip else BertEncoder,
            "image": CLIPImageEncoder if self.clip else supervised_encoder,
        }
        for modality in self.modalities:
            encoder = material_encoder(
                config, f"{modality}_encoder", encoder_types[modality], **p[modality]
            )
            setattr(self, f"{modality}_encoder", encoder)
            dimension += encoder.output_dim
        if self.clip and (
            self.text_encoder.output_dim != self.image_encoder.output_dim
            or self.text_encoder.encoding_spec.get("revision")
            != self.image_encoder.encoding_spec.get("revision")
        ):
            raise ValueError(
                "CLIP baselines require matching resolved checkpoints and projections."
            )
        self.head = _head(dimension, p["head_hidden"], p["head_dropout"])

    @classmethod
    def _validate(cls, p):
        if set(p) != {*cls.modalities, "head_hidden", "head_dropout"}:
            raise ValueError(
                "Neural baseline parameters must specify only its modalities and head."
            )
        hidden, dropout = p["head_hidden"], p["head_dropout"]
        if not isinstance(hidden, list) or any(
            type(size) is not int or size < 1 for size in hidden
        ):
            raise ValueError("head_hidden must be a list of positive integer widths.")
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("head_dropout must be finite and in [0,1).")
        if cls.head_kind == "linear" and (hidden or dropout != 0):
            raise ValueError("CLIPLinear requires head_hidden=[] and head_dropout=0.")
        if cls.head_kind == "mlp" and not hidden:
            raise ValueError("CLIPMLP requires at least one hidden layer.")
        for modality in cls.modalities:
            options = p[modality]
            if not isinstance(options, dict) or type(options.get("freeze")) is not bool:
                raise ValueError(f"{modality} must specify a boolean freeze setting.")
            if modality == "text" or cls.clip:
                for name in ("model_name", "revision"):
                    if not isinstance(options.get(name), str) or not options[name].strip():
                        raise ValueError(f"{modality}.{name} must be a nonempty string.")
            if modality == "text":
                if type(options.get("max_length")) is not int or options["max_length"] < 2:
                    raise ValueError("text.max_length must be an integer >= 2.")
                if options.get("output") not in (("pooled",) if cls.clip else ("pooled", "cls")):
                    raise ValueError(
                        "Neural baselines require pooled or explicit PLM CLS features."
                    )
            elif options.get("output") != ("pooled" if cls.clip else "avgpool"):
                raise ValueError("Neural baselines require pooled image features.")
            if cls.clip and (options["freeze"] is not True or options.get("normalize") is not True):
                raise ValueError("CLIP baseline towers must be frozen and individually normalized.")
        if cls.clip:
            if p["image"].get("square_resize", False) is not False:
                raise ValueError("CLIP baselines require the checkpoint's native image processor.")
            if any(p["text"][key] != p["image"][key] for key in ("model_name", "revision")):
                raise ValueError(
                    "CLIP baselines require the same text/image checkpoint and revision."
                )

    def _features(self, batch):
        values = [
            getattr(self, f"{modality}_encoder")(
                batch["text" if modality == "text" else "image_path"]
            )
            for modality in self.modalities
        ]
        if any(value.ndim != 2 or value.shape[0] != values[0].shape[0] for value in values):
            raise ValueError("Neural baselines require aligned batches of pooled features.")
        return values[0] if len(values) == 1 else torch.cat(values, dim=-1)

    def forward(self, batch):
        logits = self.head(self._features(batch))
        if not torch.isfinite(logits).all():
            raise ValueError("Neural baseline logits must be finite.")
        return logits

    def calculate_loss(self, batch):
        return F.cross_entropy(self(batch), batch["labels"])

    @torch.no_grad()
    def predict(self, batch):
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            return self(batch)
        finally:
            for module, mode in modes:
                module.training = mode

    def optimizer_parameter_groups(self):
        if self.clip:
            return super().optimizer_parameter_groups()
        modules = {
            **{name: getattr(self, f"{name}_encoder") for name in self.modalities},
            "head": self.head,
        }
        groups = {
            name: [parameter for parameter in module.parameters() if parameter.requires_grad]
            for name, module in modules.items()
        }
        return {name: parameters for name, parameters in groups.items() if parameters}

    def get_extra_state(self):
        return {
            "version": 1,
            "model": type(self).__name__,
            "parameters": deepcopy(self.parameters_spec),
        }

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or set(state) != {"version", "model", "parameters"}
            or type(state["version"]) is not int
            or state["version"] != 1
            or state["model"] != type(self).__name__
            or _signature(state["parameters"]) != self._parameters_signature
        ):
            raise ValueError("Neural baseline checkpoint parameters differ from the configuration.")
