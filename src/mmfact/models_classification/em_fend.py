"""EM-FEND (MM2021): entity/OCR evidence and serial bidirectional transformers."""

import json
import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder
from mmfact.materials.resources import material_resources
from mmfact.models_classification._em_fend_resources import TYPES, EntityResources
from mmfact.models_classification._encoders import BertEncoder
from mmfact.models_classification._supervised_vision import supervised_encoder
from mmfact.registry import register_model


class _PairBert(BertEncoder):
    cache_features = False

    def _tokenize(self, inputs):
        pairs = [json.loads(value) for value in inputs]
        return self.tokenizer(
            [value[0] for value in pairs],
            text_pair=[value[1] for value in pairs],
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt",
        )

    def entities(self, names, max_length):
        inputs = self.tokenizer(
            names,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
            return_tensors="pt",
        ).to(self.device)
        special = inputs.pop("special_tokens_mask").bool()
        mask = inputs["attention_mask"].bool() & ~special
        # Reject names erased by tokenization rather than silently inventing an entity.
        if not mask.any(1).all():
            raise ValueError("EM-FEND entity names must retain at least one content token.")
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze):
            tokens = self.network(**inputs).last_hidden_state
        return _mean(tokens, mask)


def _mean(values, mask):
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(1) / weights.sum(1).clamp_min(1)


class _Cross(nn.Module):
    def __init__(self, dimension, heads, ffn_dim, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(dimension), nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dimension),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, query_mask, key_mask):
        valid = key_mask.any(1)
        # Empty visual evidence has an explicit identity text path, no artificial key.
        # LayerNorm returns float32 under autocast; keep indexed updates in that dtype.
        result = query.float().clone()
        if valid.any():
            q, k = query[valid], key[valid]
            mixed, _ = self.attention(
                q, k, k, key_padding_mask=~key_mask[valid], need_weights=False
            )
            output = self.norm1(q + self.dropout(mixed))
            result[valid] = self.norm2(output + self.dropout(self.ffn(output)))
        return result * query_mask.unsqueeze(-1).to(result.dtype)


class _MCT(nn.Module):
    def __init__(self, dimension, heads, ffn_dim, dropout):
        super().__init__()
        self.text = _Cross(dimension, heads, ffn_dim, dropout)
        self.visual = _Cross(dimension, heads, ffn_dim, dropout)

    def forward(self, text, visual, text_mask, visual_mask):
        return (
            self.text(text, visual, text_mask, visual_mask),
            self.visual(visual, text, visual_mask, text_mask),
        )


def _consistency(textual, visual, probabilities):
    """Eq2: confidence-weighted visual sum, then max over textual entities."""
    if not len(textual) or not len(visual):
        source = textual if len(textual) else visual
        return source.new_ones(())
    similarity = F.normalize(textual, dim=-1) @ F.normalize(visual, dim=-1).T
    return (similarity * probabilities[None, :]).sum(-1).max()


@register_model("EM-FEND")
class EMFEND(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        for name in ("hidden_dim", "num_heads", "ffn_dim", "entity_max_length"):
            if type(p[name]) is not int or p[name] < 1:
                raise ValueError(f"EM-FEND {name} must be a positive integer.")
        if p["hidden_dim"] % p["num_heads"] or p["entity_max_length"] < 3:
            raise ValueError("EM-FEND heads must divide hidden_dim; entity length must be >=3.")
        if (
            type(p["dropout"]) not in (int, float)
            or not math.isfinite(p["dropout"])
            or not 0 <= p["dropout"] < 1
        ):
            raise ValueError("EM-FEND dropout must be finite and in [0,1).")
        text = p["text"]
        if (
            text.get("freeze") is not False
            or text.get("output") != "tokens"
            or text.get("mask_padding") is not True
            or text.get("max_length", 0) < 4
        ):
            raise ValueError("EM-FEND requires trainable paired-BERT token features with masks.")
        self.resources = material_resources(config, "em_fend", EntityResources, **p["resources"])
        self.recipe = {**deepcopy(p), "resources": deepcopy(self.resources.protocol)}
        self.text_encoder = material_encoder(config, "text_encoder", _PairBert, **text)
        self.text_encoder.network.pooler = None  # Token/mean paths never use a pooled CLS head.
        self.image_encoder = material_encoder(config, "image_encoder", supervised_encoder, **p["image"])
        if self.image_encoder.freeze or p["image"].get("output") != "spatial":
            raise ValueError("EM-FEND requires trainable spatial image features.")
        if p["image"].get("grid_size") != 7:
            raise ValueError("EM-FEND requires a 7 x 7 image grid.")
        self.identity = {
            "text": deepcopy(self.text_encoder.encoding_spec),
            "image": deepcopy(self.image_encoder.encoding_spec),
        }
        if p["entity_max_length"] > self.text_encoder.network.config.max_position_embeddings:
            raise ValueError("EM-FEND entity length exceeds BERT position capacity.")
        hidden = p["hidden_dim"]
        self.text_projection = nn.Linear(self.text_encoder.output_dim, hidden)
        self.image_projection = nn.Linear(self.image_encoder.output_dim, hidden)
        self.entity_mct = _MCT(hidden, p["num_heads"], p["ffn_dim"], p["dropout"])
        self.image_mct = _MCT(hidden, p["num_heads"], p["ffn_dim"], p["dropout"])
        self.classifier = nn.Linear(hidden * 3 + 3, 2)
        self.entity_max_length = p["entity_max_length"]

    def prepare(self, train_dataset):
        # Load fixed evidence, never fit or supervise on validation/test inputs.
        for row in train_dataset:
            self.resources.lookup(row["text"], row["image_path"])

    def optimizer_parameter_groups(self):
        backbone = [*self.text_encoder.parameters(), *self.image_encoder.parameters()]
        ids = {id(parameter) for parameter in backbone}
        return {
            "backbone": backbone,
            "default": [parameter for parameter in self.parameters() if id(parameter) not in ids],
        }

    def forward(self, batch):
        texts, paths = batch["text"], batch["image_path"]
        if not texts or len(texts) != len(paths):
            raise ValueError("EM-FEND needs one text and image per news item.")
        records = [self.resources.lookup(text, path) for text, path in zip(texts, paths)]
        pairs = [
            json.dumps([text, record["ocr"]], ensure_ascii=False)
            for text, record in zip(texts, records)
        ]
        text_mask = self.text_encoder.token_mask(pairs)
        text = self.text_projection(self.text_encoder(pairs))
        names = list(
            dict.fromkeys(
                name
                for record in records
                for name in (
                    [name for kind in TYPES for name in record["textual"][kind]]
                    + [entity["name"] for entity in record["visual"]]
                )
            )
        )
        vectors = (
            self.text_encoder.entities(names, self.entity_max_length)
            if names
            else text.new_zeros((0, self.text_encoder.output_dim))
        )
        index = {name: position for position, name in enumerate(names)}
        count = max(1, max(len(record["visual"]) for record in records))
        visual = text.new_zeros((len(text), count, self.text_encoder.output_dim))
        visual_mask = torch.zeros(len(text), count, dtype=torch.bool, device=text.device)
        similarities = []
        for row, record in enumerate(records):
            if record["visual"]:
                visual[row, : len(record["visual"])] = torch.stack(
                    [vectors[index[e["name"]]] for e in record["visual"]]
                )
                visual_mask[row, : len(record["visual"])] = True
            scores = []
            for kind in TYPES:
                t = [index[name] for name in record["textual"][kind]]
                v = [entity for entity in record["visual"] if entity["type"] == kind]
                probability = text.new_tensor([entity["confidence"] for entity in v])
                scores.append(
                    _consistency(vectors[t], vectors[[index[e["name"]] for e in v]], probability)
                )
            similarities.append(torch.stack(scores))
        entities = self.text_projection(visual) * visual_mask.unsqueeze(-1).to(text.dtype)
        text, entities = self.entity_mct(text, entities, text_mask, visual_mask)
        regions = self.image_encoder(paths).flatten(2).transpose(1, 2)
        image = self.image_projection(regions)
        image_mask = torch.ones(image.shape[:2], dtype=torch.bool, device=image.device)
        text, image = self.image_mct(text, image, text_mask, image_mask)
        fused = torch.cat(
            (
                _mean(text, text_mask),
                _mean(entities, visual_mask),
                image.mean(1),
                torch.stack(similarities),
            ),
            dim=-1,
        )
        return self.classifier(fused)

    def calculate_loss(self, batch):
        return {"classification": F.cross_entropy(self(batch), batch["labels"])}

    def predict(self, batch):
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            with torch.no_grad():
                return self(batch)
        finally:
            for module, mode in modes:
                module.training = mode

    def get_extra_state(self):
        return {
            "format_version": 1,
            "recipe": deepcopy(self.recipe),
            "identity": deepcopy(self.identity),
        }

    def set_extra_state(self, state):
        if state != {"format_version": 1, "recipe": self.recipe, "identity": self.identity}:
            raise ValueError("EM-FEND checkpoint has an incompatible computational recipe.")
