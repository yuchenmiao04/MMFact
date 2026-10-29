"""ERIC-FND: fixed entity knowledge, contrastive alignment and adaptive fusion."""

import json
import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder
from mmfact.materials.resources import material_resources
from mmfact.models_classification._encoders import BertEncoder, _PretrainedEncoder
from mmfact.models_classification._supervised_vision import supervised_encoder
from mmfact.models_classification._entities import EntityDescriptionStore
from mmfact.models_classification._eric_nlin import ERICNLINResources
from mmfact.models_classification._layers import SingletonBatchNorm
from mmfact.registry import register_model


class _EnglishNewsEncoder(BertEncoder):
    """Encode the same frozen English news that supplied NLIN's text entities."""

    def __init__(self, lookup, **parameters):
        super().__init__(**parameters)
        self.lookup = lookup

    def inputs_from_rows(self, rows):
        return [self.lookup(row["text"], row["image_path"])["news_english"] for row in rows]


class _DescriptionEncoder(BertEncoder):
    """Fixed-size description groups, sharing the news tower's sole weight owner."""

    def __init__(self, source, lookup, max_length, max_entities):
        _PretrainedEncoder.__init__(self)
        object.__setattr__(self, "network", source.network)
        for name in (
            "tokenizer",
            "output_dim",
            "output",
            "mask_padding",
            "freeze",
            "global_attention",
            "layer_groups",
        ):
            setattr(self, name, getattr(source, name))
        self.lookup, self.max_length, self.max_entities = lookup, max_length, max_entities
        self.sequence_length = max_length
        self.encoding_spec = deepcopy(source.encoding_spec)
        self.encoding_spec.update(
            max_length=max_length,
            input_format="json:entity-descriptions",
            max_entities=max_entities,
        )
        source.register_load_state_dict_post_hook(self._clear_shared_cache)
        self.train(self.training)

    def _clear_shared_cache(self, _owner, incompatible_keys):
        self._clear_cache(self, incompatible_keys)

    def inputs_from_rows(self, rows):
        return [
            json.dumps(self.lookup(row), ensure_ascii=False, separators=(",", ":")) for row in rows
        ]

    def _encode(self, inputs):
        groups = [json.loads(value) for value in inputs]
        if any(
            not isinstance(group, list)
            or len(group) > self.max_entities
            or any(not isinstance(text, str) or not text.strip() for text in group)
            for group in groups
        ):
            raise ValueError("ERIC-FND needs JSON lists of nonempty entity descriptions.")
        dtype = next(self.network.parameters()).dtype
        output = torch.zeros(
            len(groups), self.max_entities, self.output_dim, device=self.device, dtype=dtype
        )
        positions = [(i, j) for i, group in enumerate(groups) for j in range(len(group))]
        if positions:
            texts = [text for group in groups for text in group]
            rows, columns = zip(*positions)
            output[list(rows), list(columns)] = BertEncoder._encode(self, texts)
        return output


def _mlp(input_dim, hidden_dim, output_dim, dropout=0.0):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        SingletonBatchNorm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
        SingletonBatchNorm(output_dim),
        nn.ReLU(),
    )


def _entity_mean(text, descriptions, mask):
    """Eq.7–8: entity-axis attention followed by mean over valid entities."""
    scores = torch.einsum("bd,bkd->bk", text, descriptions)
    scores = scores.masked_fill(~mask, -torch.inf)
    scores = torch.where(mask.any(1, keepdim=True), scores, torch.zeros_like(scores))
    weights = scores.softmax(1) * mask.to(scores.dtype)
    weighted = torch.einsum("bk,bkd->bd", weights, descriptions)
    return weighted / mask.sum(1, keepdim=True).clamp_min(1)


def _interaction(image, text):
    """Eq.17–22: pooled one-key attention is one; image rows, text columns."""
    return (image.unsqueeze(2) * text.unsqueeze(1)).max(-1).values


@register_model("ERIC-FND")
class ERICFND(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        for name in ("feature_dim", "alignment_dim", "classifier_dim", "description_length"):
            if type(p[name]) is not int or p[name] < (2 if name == "description_length" else 1):
                raise ValueError(f"ERIC-FND {name} must be a positive integer of sufficient size.")
        for name in ("dropout", "temperature", "contrastive_weight"):
            value = p[name]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or (not 0 <= value < 1 if name == "dropout" else value <= 0)
            ):
                raise ValueError(f"ERIC-FND {name} is outside its permitted finite range.")
        if p["text"].get("output") != "cls" or p["image"].get("output") != "avgpool":
            raise ValueError("ERIC-FND requires last-layer BERT CLS and global image features.")
        for name in ("text", "image"):
            if type(p[name].get("freeze", True)) is not bool:
                raise ValueError(f"ERIC-FND {name}.freeze must be boolean.")
        for name in ("model_name", "revision"):
            if not isinstance(p["text"].get(name), str) or not p["text"][name].strip():
                raise ValueError(f"ERIC-FND text.{name} must be a nonempty string.")
        if type(p["text"].get("max_length")) is not int or p["text"]["max_length"] < 2:
            raise ValueError("ERIC-FND text.max_length must be an integer >= 2.")

        self.parameters_spec = deepcopy(p)
        resources = deepcopy(p["resources"])
        self.resource_mode = resources.pop("mode", "wikipedia")
        if self.resource_mode == "nlin":
            self.entities = material_resources(config, "eric_fnd", ERICNLINResources, **resources)
            self.text_encoder = material_encoder(
                config, "text_encoder", _EnglishNewsEncoder, self.entities.lookup, **p["text"]
            )
        elif self.resource_mode == "wikipedia":
            self.entities = EntityDescriptionStore(**resources)
            self.text_encoder = material_encoder(config, "text_encoder", BertEncoder, **p["text"])
        else:
            raise ValueError("ERIC-FND resources.mode must be nlin or wikipedia.")
        capacity = (
            512
            if hasattr(self.text_encoder, "matrix")
            else self.text_encoder.network.config.max_position_embeddings
        )
        if p["description_length"] > capacity:
            raise ValueError("ERIC-FND description_length exceeds BERT position capacity.")
        self.description_encoder = material_encoder(
            config,
            "description_encoder",
            _DescriptionEncoder,
            self.text_encoder,
            self._descriptions,
            p["description_length"],
            self.entities.max_entities,
        )
        self.image_encoder = material_encoder(config, "image_encoder", supervised_encoder, **p["image"])
        feature, aligned, hidden = p["feature_dim"], p["alignment_dim"], p["classifier_dim"]
        self.text_projection = nn.Linear(self.text_encoder.output_dim, feature)
        self.image_projection = nn.Linear(self.image_encoder.output_dim, feature)
        self.description_projection = nn.Linear(self.text_encoder.output_dim, feature)
        self.enhance_text = nn.Linear(feature, feature, bias=False)
        self.enhance_description = nn.Linear(feature, feature, bias=False)
        self.align_text, self.align_image = nn.Linear(feature, aligned), nn.Linear(feature, aligned)
        self.interaction = nn.Sequential(
            nn.Linear(aligned, aligned), SingletonBatchNorm(aligned), nn.ReLU()
        )
        self.gate_text = _mlp(feature, feature, aligned)
        self.gate_image = _mlp(feature, feature, aligned)
        self.gate = nn.Linear(aligned, 3)
        self.classifier = nn.Sequential(
            _mlp(2 * feature + aligned, hidden, hidden, p["dropout"]),
            nn.Dropout(p["dropout"]),
            nn.Linear(hidden, 2),
        )
        self.temperature, self.contrastive_weight = p["temperature"], p["contrastive_weight"]

    def _descriptions(self, row):
        if self.resource_mode == "nlin":
            return self.entities.lookup(row["text"], row["image_path"])["descriptions"]
        return self.entities.lookup(row["text"])

    def _features(self, batch):
        if not batch["text"] or len(batch["text"]) != len(batch["image_path"]):
            raise ValueError("ERIC-FND needs one text and image per item in a nonempty batch.")
        rows = [
            {"text": text, "image_path": path}
            for text, path in zip(batch["text"], batch["image_path"], strict=True)
        ]
        text = self.text_projection(self.text_encoder(self.text_encoder.inputs_from_rows(rows)))
        image = self.image_projection(self.image_encoder(batch["image_path"]))
        inputs = self.description_encoder.inputs_from_rows(rows)
        counts = torch.tensor([len(json.loads(value)) for value in inputs], device=text.device)
        mask = torch.arange(self.entities.max_entities, device=text.device)[None] < counts[:, None]
        descriptions = self.description_projection(self.description_encoder(inputs))
        evidence = _entity_mean(text, descriptions, mask)
        enhanced = self.enhance_text(text) + self.enhance_description(evidence)
        aligned_text = F.normalize(self.align_text(text), dim=-1)
        aligned_image = F.normalize(self.align_image(image), dim=-1)
        interaction = self.interaction(_interaction(aligned_image, aligned_text))
        gate_inputs = torch.stack(
            (self.gate_text(enhanced), self.gate_image(image), interaction), dim=-1
        )
        weights = self.gate(gate_inputs.mean(-1)).sigmoid()
        fused = torch.cat(
            [
                value * weights[:, index : index + 1]
                for index, value in enumerate((enhanced, image, interaction))
            ],
            dim=-1,
        )
        return self.classifier(fused), aligned_text, aligned_image

    def forward(self, batch):
        return self._features(batch)[0]

    def calculate_loss(self, batch):
        logits, text, image = self._features(batch)
        similarity = text @ image.T / self.temperature
        targets = torch.arange(len(text), device=text.device)
        contrastive = (
            F.cross_entropy(similarity, targets) + F.cross_entropy(similarity.T, targets)
        ) / 2
        return {
            "classification": F.cross_entropy(logits, batch["labels"]),
            "contrastive": self.contrastive_weight * contrastive,
        }

    def predict(self, batch):
        was_training = self.training
        self.eval()
        try:
            return self(batch)
        finally:
            self.train(was_training)

    def get_extra_state(self):
        return {"format_version": 1, "parameters": deepcopy(self.parameters_spec)}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state != self.get_extra_state():
            raise ValueError("ERIC-FND checkpoint parameters differ from the model configuration.")
