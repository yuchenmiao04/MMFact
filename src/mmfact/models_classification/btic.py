"""BTIC: image-patch Transformer and training-only historical contrastive memory."""

import hashlib
import math
from copy import deepcopy
from datetime import datetime

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.data.features import _value_digest
from mmfact.materials.encoders import material_encoder
from mmfact.materials.training import (
    hub_training_resource,
    resource_rows,
    resource_tensor,
)
from mmfact.models_classification._encoders import BertEncoder, ResNetEncoder
from mmfact.models_classification._supervised_vision import ViTIndependentPatches
from mmfact.registry import register_model


def _headline(row, key):
    value = row.get("metadata", {}).get(key)
    if value is None:
        return row["text"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("BTIC headline must be a nonempty string when provided.")
    return value


def _timestamp(value):
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError("timestamp has no timezone")
            value = parsed.timestamp()
        except (ValueError, OverflowError) as error:
            raise ValueError("BTIC requires timezone-aware ISO8601 or Unix seconds.") from error
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("BTIC historical references require a reliable timestamp for every row.")
    return float(value)


def _validate(p):
    integers = (
        "projection_dim",
        "attention_heads",
        "transformer_layers",
        "feedforward_dim",
        "pool_size",
        "neighbors",
        "reference_chunk_size",
    )
    image, text = p["image"], p["text"]
    if any(type(p[key]) is not int or p[key] < 1 for key in integers) or any(
        type(image[key]) is not int or image[key] < 1
        for key in ("image_size", "grid_size", "patch_batch_size")
    ):
        raise ValueError("BTIC dimensions and chunk sizes must be positive integers.")
    if (
        type(text["max_length"]) is not int
        or text["max_length"] < 2
        or text["freeze"] is not False
        or text["output"] != "tokens"
        or text["mask_padding"] is not True
        or image["image_size"] % image["grid_size"]
        or p["reference_mode"] not in {"time_free", "historical"}
        or any(
            not isinstance(p[key], str) or not p[key].strip()
            for key in ("headline_key", "timestamp_key")
        )
    ):
        raise ValueError("BTIC requires masked trainable BERT tokens and an explicit time mode.")
    for key in ("dropout", "transformer_dropout", "contrastive_weight"):
        if type(p[key]) not in (int, float) or not math.isfinite(p[key]) or not 0 <= p[key] < 1:
            raise ValueError(f"BTIC {key} must be finite in [0, 1).")


class _HeadlineEncoder(BertEncoder):
    def __init__(self, headline_key, **options):
        self.headline_key = headline_key
        super().__init__(**options)

    def inputs_from_rows(self, rows):
        return [_headline(row, self.headline_key) for row in rows]


class _PatchEncoder(ResNetEncoder):
    """Independent patches with the common pretrained ResNet input transform."""

    def __init__(
        self,
        weights,
        image_size,
        grid_size,
        patch_batch_size,
        architecture="resnet50",
        center_crop=True,
        stem_maxpool=True,
    ):
        super().__init__(
            architecture=architecture,
            weights=weights,
            freeze=True,
            output="avgpool",
            center_crop=center_crop,
            stem_maxpool=stem_maxpool,
        )
        self.image_size, self.grid_size = image_size, grid_size
        self.patch_batch_size = patch_batch_size
        self.encoding_spec = {
            "encoder": "btic-resnet50-independent-patches",
            "architecture": architecture,
            "weights": weights,
            "image_size": image_size,
            "grid_size": grid_size,
            "order": "row-major",
            "resize": "PIL-LANCZOS",
            "patch_transform": repr(self.transform),
            "output": "avgpool",
            "patch_batch_size": patch_batch_size,
        }

    def _encode(self, paths):
        results = []
        size, grid = self.image_size, self.grid_size
        for path in paths:
            with Image.open(path) as image:
                image = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
                patch = size // grid
                pixels = torch.stack(
                    [
                        self.transform(image.crop((x, y, x + patch, y + patch)))
                        for y in range(0, size, patch)
                        for x in range(0, size, patch)
                    ]
                )
            with torch.no_grad():
                encoded = [
                    self.network(pixels[start : start + self.patch_batch_size].to(self.device))
                    for start in range(0, len(pixels), self.patch_batch_size)
                ]
            results.append(torch.cat(encoded))
        return torch.stack(results)


def _patch_encoder(**options):
    if options.get("backbone") == "vit":
        return ViTIndependentPatches(resize="lanczos", **options)
    return _PatchEncoder(**options)


def _contrastive(features, memory, indices):
    """Paper Eq.(3), with actual positive count and a negative-only denominator."""
    # Retrieval state stays on CPU; masks participate in the feature-device loss.
    valid = indices.ge(0).to(device=features.device)
    references = memory[indices.clamp_min(0)].to(features)
    similarities = F.cosine_similarity(features[:, None, None], references, dim=-1)
    positives, negatives = valid[:, 0], valid[:, 1]
    usable = positives.any(-1) & negatives.any(-1)
    if not usable.any():
        return features.sum() * 0
    similarities, positives, negatives = similarities[usable], positives[usable], negatives[usable]
    positive_mean = (similarities[:, 0] * positives).sum(-1) / positives.sum(-1)
    negative_logsum = similarities[:, 1].masked_fill(~negatives, -torch.inf).logsumexp(-1)
    return 0.5 * (negative_logsum - positive_mean).mean()


@register_model("BTIC")
class BTIC(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        _validate(p)
        self.parameters_spec = deepcopy(p)
        self.text_encoder = material_encoder(config, "text_encoder", BertEncoder, **p["text"])
        d = self.text_encoder.output_dim
        if d % p["attention_heads"] or d % p["pool_size"]:
            raise ValueError(
                "BTIC BERT dimension must divide into attention heads and pooling bins."
            )
        self.reference_encoder = material_encoder(
            config,
            "reference_encoder",
            _HeadlineEncoder,
            headline_key=p["headline_key"],
            **{**p["text"], "freeze": True, "output": "cls", "mask_padding": False},
        )
        self.image_encoder = material_encoder(config, "image_encoder", _patch_encoder, **p["image"])
        self.image_projection = nn.Sequential(
            nn.Linear(self.image_encoder.output_dim, p["projection_dim"]),
            nn.Linear(p["projection_dim"], d),
        )
        self.image_transformer = nn.Transformer(
            d_model=d,
            nhead=p["attention_heads"],
            num_encoder_layers=p["transformer_layers"],
            num_decoder_layers=p["transformer_layers"],
            dim_feedforward=p["feedforward_dim"],
            dropout=p["transformer_dropout"],
            batch_first=True,
        )
        self.attention = nn.MultiheadAttention(d, p["attention_heads"], batch_first=True)
        self.classifier = nn.Sequential(nn.Dropout(p["dropout"]), nn.Linear(d // p["pool_size"], 2))
        self.ids, self.text_digests, self.audit, self.positions = [], [], {}, {}
        self.labels = torch.empty(0, dtype=torch.long)
        self.times = torch.empty(0, dtype=torch.float64)
        self.neighbors = torch.empty(0, 2, p["neighbors"], dtype=torch.long)
        self.memory = torch.empty(0, d // p["pool_size"])
        self.pending = torch.empty_like(self.memory)
        self.pending_seen = torch.empty(0, dtype=torch.bool)
        self.epoch = 0

    def prepare(self, train_dataset):
        p = self.parameters_spec
        rows = sorted(list(train_dataset), key=lambda row: row["sample_id"])
        ids = [row["sample_id"] for row in rows]
        labels = torch.tensor([row["label"] for row in rows], dtype=torch.long)
        if (
            not ids
            or len(ids) != len(set(ids))
            or set(labels.tolist()) != {0, 1}
            or any(not isinstance(key, str) or not key.strip() for key in ids)
        ):
            raise ValueError("BTIC requires unique training IDs and both binary labels.")
        times = torch.tensor(
            [_timestamp(row.get("metadata", {}).get(p["timestamp_key"])) for row in rows]
            if p["reference_mode"] == "historical"
            else [0.0] * len(rows),
            dtype=torch.float64,
        )
        resource = hub_training_resource(
            self.config, "btic_reference_neighbors", expected={"neighbors": p["neighbors"]}
        )
        if resource is not None:
            if p["reference_mode"] != "time_free":
                raise ValueError("Exported BTIC neighbors require the fixed time_free protocol.")
            indexed = resource_rows(resource, "id")
            source = {row["id"]: row["row"] for row in indexed}
            if set(source) != set(ids):
                raise ValueError("BTIC exported references differ from the complete train split.")
            neighbors = resource_tensor(resource, "neighbors")
            if neighbors.dtype != torch.long or neighbors.shape != (len(ids), 2, p["neighbors"]):
                raise ValueError("BTIC exported neighbors have an invalid shape or dtype.")
            if (neighbors < -1).any() or (neighbors >= len(ids)).any():
                raise ValueError("BTIC exported neighbor points outside the training rows.")
            order = torch.tensor([source[key] for key in ids])
            inverse = order.argsort()
            neighbors = neighbors[order]
            neighbors = inverse[neighbors.clamp_min(0)].masked_fill(neighbors.lt(0), -1)
            self.ids, self.labels, self.times, self.neighbors = ids, labels, times, neighbors
            self.text_digests = [hashlib.sha256(row["text"].encode()).hexdigest() for row in rows]
            self.positions = {key: index for index, key in enumerate(ids)}
            self.memory = torch.zeros(len(rows), self.classifier[-1].in_features)
            self.pending = torch.zeros_like(self.memory)
            self.pending_seen = torch.zeros(len(rows), dtype=torch.bool)
            self.epoch = 0
            self.audit = {"training_rows": len(rows), "reference_mode": "time_free"}
            self.set_extra_state(self.get_extra_state())
            return
        chunk = p["reference_chunk_size"]
        with torch.no_grad():
            vectors = F.normalize(
                torch.cat(
                    [
                        self.reference_encoder(
                            [_headline(row, p["headline_key"]) for row in rows[i : i + chunk]]
                        )
                        .detach()
                        .cpu()
                        for i in range(0, len(rows), chunk)
                    ]
                ),
                dim=-1,
            )
        if not torch.isfinite(vectors).all() or (vectors.norm(dim=-1) == 0).any():
            raise ValueError("BTIC reference embeddings must be finite and nonzero.")
        neighbors = torch.full((len(rows), 2, p["neighbors"]), -1, dtype=torch.long)
        # Block both axes: complete training retrieval without an N-by-N allocation.
        for start in range(0, len(rows), chunk):
            stop = min(start + chunk, len(rows))
            count = stop - start
            best_scores = torch.full((count, 2, p["neighbors"]), -torch.inf)
            best_indices = neighbors[start:stop].clone()
            query_ids = torch.arange(start, stop)
            for ref in range(0, len(rows), chunk):
                end = min(ref + chunk, len(rows))
                scores = vectors[start:stop] @ vectors[ref:end].T
                allowed = query_ids[:, None].ne(torch.arange(ref, end)[None])
                if p["reference_mode"] == "historical":
                    allowed &= times[ref:end][None] < times[start:stop, None]
                for category in range(2):
                    same = labels[start:stop, None].eq(labels[ref:end][None])
                    valid = allowed & (same if category == 0 else ~same)
                    candidate = scores.masked_fill(~valid, -torch.inf)
                    combined = torch.cat((best_scores[:, category], candidate), -1)
                    indices = torch.cat(
                        (best_indices[:, category], torch.arange(ref, end).expand(count, -1)), -1
                    )
                    order = combined.argsort(dim=-1, descending=True, stable=True)[
                        :, : p["neighbors"]
                    ]
                    best_scores[:, category] = combined.gather(1, order)
                    best_indices[:, category] = indices.gather(1, order)
            neighbors[start:stop] = best_indices.masked_fill(~torch.isfinite(best_scores), -1)
        self.ids, self.labels, self.times, self.neighbors = ids, labels, times, neighbors
        self.text_digests = [hashlib.sha256(row["text"].encode()).hexdigest() for row in rows]
        self.positions = {key: index for index, key in enumerate(ids)}
        self.memory = torch.zeros(len(rows), self.classifier[-1].in_features)
        self.pending, self.pending_seen = (
            torch.zeros_like(self.memory),
            torch.zeros(len(rows), dtype=torch.bool),
        )
        self.epoch = 0
        self.audit = {
            "training_rows": len(rows),
            "headline_fallback_rows": sum(
                row.get("metadata", {}).get(p["headline_key"]) is None for row in rows
            ),
            "reference_mode": p["reference_mode"],
            "headline_vectors_sha256": _value_digest(vectors),
        }

    def _features(self, batch):
        text = self.text_encoder(batch["text"])
        text_mask = self.text_encoder.token_mask(batch["text"])
        image = self.image_projection(self.image_encoder(batch["image_path"]))
        image = self.image_transformer(image, image)
        tokens = torch.cat((text, image), 1)
        valid = torch.cat(
            (text_mask, torch.ones(image.shape[:2], dtype=torch.bool, device=image.device)), 1
        )
        tokens = self.attention(
            tokens, tokens, tokens, key_padding_mask=~valid, need_weights=False
        )[0]
        size = self.parameters_spec["pool_size"]
        padding = (-tokens.shape[1]) % size
        tokens = F.pad(
            tokens.masked_fill(~valid[..., None], -torch.inf), (0, 0, 0, padding), value=-torch.inf
        )
        valid = F.pad(valid, (0, padding))
        bins = tokens.reshape(len(tokens), -1, size, tokens.shape[-1] // size, size).amax((2, 4))
        occupied = valid.reshape(len(valid), -1, size).any(-1)
        bins = bins.masked_fill(~occupied[..., None], 0)
        return bins.sum(1) / occupied.sum(1, keepdim=True).clamp_min(1)

    def forward(self, batch):
        return self.classifier(self._features(batch))

    def calculate_loss(self, batch):
        if not self.training:
            return self.validation_loss(batch)
        if not self.positions:
            raise ValueError("Prepare BTIC training resources or restore a checkpoint first.")
        try:
            indices = torch.tensor([self.positions[key] for key in batch["sample_id"]])
        except KeyError as error:
            raise ValueError("BTIC loss accepts training IDs only.") from error
        digests = [hashlib.sha256(text.encode()).hexdigest() for text in batch["text"]]
        if digests != [self.text_digests[i] for i in indices] or not torch.equal(
            batch["labels"].detach().cpu(), self.labels[indices]
        ):
            raise ValueError("BTIC training inputs/labels differ from prepared resources.")
        features = self._features(batch)
        alpha = self.parameters_spec["contrastive_weight"]
        contrastive = _contrastive(features, self.memory, self.neighbors[indices])
        # Reads above use the previous epoch; pending is detached and never a loss target.
        self.pending[indices] = features.detach().float().cpu()
        self.pending_seen[indices] = True
        return {
            "classification": (1 - alpha)
            * F.cross_entropy(self.classifier(features), batch["labels"]),
            "contrastive": alpha * contrastive,
        }

    def validation_loss(self, batch):
        return {"classification": F.cross_entropy(self.predict(batch), batch["labels"])}

    def predict(self, batch):
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            return self(batch)
        finally:
            for module, mode in modes:
                module.training = mode

    def on_epoch_start(self, epoch, total_epochs):
        if not self.ids:
            raise ValueError("Prepare BTIC training resources or restore a checkpoint first.")
        self.epoch = epoch
        self.pending.zero_()
        self.pending_seen.zero_()

    def on_epoch_end(self, epoch):
        if epoch != self.epoch:
            raise ValueError("BTIC memory epoch does not match the active epoch.")
        self.memory[self.pending_seen] = self.pending[self.pending_seen]
        self.pending.zero_()
        self.pending_seen.zero_()

    def get_extra_state(self):
        resources = {
            "ids": list(self.ids),
            "text_digests": list(self.text_digests),
            "labels": self.labels,
            "times": self.times,
            "neighbors": self.neighbors,
            "memory": self.memory,
            "pending": self.pending,
            "pending_seen": self.pending_seen,
            "epoch": self.epoch,
            "audit": dict(self.audit),
        }
        return {
            "format_version": 1,
            "parameters": deepcopy(self.parameters_spec),
            "resources": resources,
            "sha256": _value_digest(resources),
        }

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or state.get("format_version") != 1
            or state.get("parameters") != self.parameters_spec
            or not isinstance(state.get("resources"), dict)
        ):
            raise ValueError("BTIC checkpoint parameters/resources are incompatible.")
        r = state["resources"]
        if state.get("sha256") != _value_digest(r):
            raise ValueError("BTIC checkpoint resource digest does not match.")
        required = {
            "ids",
            "text_digests",
            "labels",
            "times",
            "neighbors",
            "memory",
            "pending",
            "pending_seen",
            "epoch",
            "audit",
        }
        if set(r) != required or not isinstance(r["ids"], list):
            raise ValueError("BTIC checkpoint resource fields are invalid.")
        ids, digests, n = r["ids"], r["text_digests"], len(r["ids"])
        p, width = self.parameters_spec, self.classifier[-1].in_features
        if (
            any(not isinstance(key, str) or not key.strip() for key in ids)
            or ids != sorted(set(ids))
            or not isinstance(digests, list)
            or len(digests) != n
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
                for value in digests
            )
            or type(r["epoch"]) is not int
            or r["epoch"] < 0
            or not isinstance(r["audit"], dict)
        ):
            raise ValueError("BTIC checkpoint identities/epoch are invalid.")
        for key, shape, dtype in (
            ("labels", (n,), torch.long),
            ("times", (n,), torch.float64),
            ("neighbors", (n, 2, p["neighbors"]), torch.long),
            ("memory", (n, width), torch.float32),
            ("pending", (n, width), torch.float32),
            ("pending_seen", (n,), torch.bool),
        ):
            if (
                not isinstance(r[key], torch.Tensor)
                or tuple(r[key].shape) != shape
                or r[key].dtype != dtype
                or not torch.isfinite(r[key]).all()
            ):
                raise ValueError(f"BTIC checkpoint {key} has invalid shape/type/values.")
        r = {
            key: value.detach().cpu().clone()
            if isinstance(value, torch.Tensor)
            else deepcopy(value)
            for key, value in r.items()
        }
        if n:
            if (
                set(r["labels"].tolist()) != {0, 1}
                or r["audit"].get("training_rows") != n
                or r["audit"].get("reference_mode") != p["reference_mode"]
                or (r["neighbors"] < -1).any()
                or (r["neighbors"] >= n).any()
            ):
                raise ValueError("BTIC checkpoint training resources are inconsistent.")
            for category in range(2):
                selected = r["neighbors"][:, category]
                valid = selected.ge(0)
                own = torch.arange(n)[:, None].expand_as(selected)
                same = r["labels"][:, None].eq(r["labels"][selected.clamp_min(0)])
                if (
                    (valid & selected.eq(own)).any()
                    or (valid & (same if category else ~same)).any()
                    or any(
                        len(values[values >= 0]) != len(set(values[values >= 0].tolist()))
                        for values in selected
                    )
                ):
                    raise ValueError("BTIC checkpoint neighbors violate training class/self rules.")
                if (
                    p["reference_mode"] == "historical"
                    and (valid & (r["times"][selected.clamp_min(0)] >= r["times"][:, None])).any()
                ):
                    raise ValueError("BTIC checkpoint neighbors violate historical filtering.")
            if p["reference_mode"] == "time_free" and r["times"].ne(0).any():
                raise ValueError("BTIC TimeFree checkpoint cannot contain synthetic timestamps.")
        elif r["audit"] or r["epoch"]:
            raise ValueError("BTIC checkpoint has an audit/epoch without training resources.")
        self.ids, self.text_digests, self.audit = r["ids"], r["text_digests"], r["audit"]
        self.positions = {key: index for index, key in enumerate(self.ids)}
        for key in ("labels", "times", "neighbors", "memory", "pending", "pending_seen", "epoch"):
            setattr(self, key, r[key])
