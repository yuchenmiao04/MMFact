"""CSFND: train-only semantic contexts, gated representations and local classifiers."""

import hashlib
import math
import re
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from mmfact.base import AbstractDetector, TrainingStage
from mmfact.data.collate import collate_news
from mmfact.data.features import _value_digest
from mmfact.materials.encoders import hub_inputs, material_encoder
from mmfact.materials.training import (
    encoder_identity,
    hub_training_resource,
    load_prepared_resource,
    resource_rows,
    resource_tensor,
)
from mmfact.models_classification._encoders import BertEncoder
from mmfact.models_classification._supervised_vision import supervised_encoder
from mmfact.models_classification._generated import pair_identity
from mmfact.models_classification._layers import SingletonBatchNorm
from mmfact.registry import register_model


def _inputs(rows):
    return {key: [row[key] for row in rows] for key in ("text", "image_path")}


def _clean_text(text, language):
    """Fixed author text_process rule order; no data-dependent filtering or fitting."""
    rules = [("[’‘]", "'"), ("[“”]", '"')]
    if language == "english":
        rules += [("[\u2000-\u206f\u20a0-\u20cf\u2100-\u214f]", " "), (r"http:\ ", "http:")]
    rules += [(r"http[s]?:[^\ ]+", " "), ("&gt;|&lt;|&quot;", " ")]
    if language == "chinese":
        rules.append(("&nbsp;", " "))
    rules += [('"', " "), (r"#\ ", "#"), (r"\\n|\\", " "), (r"[()\[\]{}]", " ")]
    if language == "english":
        rules += [
            ("[\U0001f300-\U0001f64f\U0001f680-\U0001f6ff\u2600-\u26ff\u2700-\u27bf]+", " "),
            ("'s", " is "),
            ("'ve", " have "),
            ("'re", " are "),
            ("'d", " had "),
            ("'ll", " will "),
            ("'m", " am"),
            ("n't", " not"),
        ]
    rules += [("#", " #"), ("@", r" \@"), (r"[!?.,+\-$%\^><=:;*(){}\[\]/~&'|]", " ")]
    result = text.lower()
    for pattern, replacement in rules:
        result = re.sub(pattern, replacement, result)
    return result


class _ContentBertEncoder(BertEncoder):
    """Cache CLS plus N valid body tokens; added specials and padding are excluded."""

    def __init__(self, cleaning, **options):
        if cleaning not in ("english", "chinese"):
            raise ValueError("CSFND text cleaning must be english or chinese.")
        self.cleaning = cleaning
        super().__init__(**options)
        self.body_length = self.max_length - 2
        self.sequence_length = self.body_length + 1
        self.encoding_spec.update(
            pooling="cls-and-valid-body-tokens",
            body_length=self.body_length,
            body_padding="zero-excluding-added-specials",
            cleaning={"language": cleaning, "recipe": "csfnd-author-text-process-v1"},
        )

    def _encode(self, texts):
        inputs = self.tokenizer(
            [_clean_text(text, self.cleaning) for text in texts],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        ).to(self.device)
        special = inputs.pop("special_tokens_mask").bool()
        with torch.no_grad():
            tokens = self.network(**inputs).last_hidden_state
        valid = inputs["attention_mask"].bool() & ~special
        cls = (
            (inputs["input_ids"].eq(self.tokenizer.cls_token_id) & inputs["attention_mask"].bool())
            .int()
            .argmax(-1)
        )
        output = tokens.new_zeros(len(texts), self.body_length + 1, tokens.shape[-1])
        output[:, 0] = tokens[torch.arange(len(texts), device=tokens.device), cls]
        for index, (row, mask) in enumerate(zip(tokens, valid)):
            body = row[mask][: self.body_length]
            output[index, 1 : len(body) + 1] = body
        return output


def _triplet(features, clusters, margin, labels=None, warmup=False):
    """Eq1/8 squared-distance sum over active distinct anchor/positive/negative triples."""
    distance = (features[:, None] - features[None]).square().sum(-1)
    same = clusters[:, None].eq(clusters[None])
    identity = torch.eye(len(features), device=features.device, dtype=torch.bool)
    positive = same & ~identity
    negative = ~same
    if labels is not None:
        equal = labels[:, None].eq(labels[None])
        positive &= equal
        negative = same & ~equal
    # [anchor, negative, positive] evaluates the same ordered triplets without
    # per-anchor dynamic CUDA indexing (and its repeated host synchronization).
    difference = distance[:, :, None] - distance[:, None, :]
    active = negative[:, :, None] & positive[:, None, :] & difference.lt(margin)
    if warmup:
        active &= difference.gt(0)
    return torch.where(active, margin - difference, 0.0).sum()


def _intra(features, clusters):
    distance = (features[:, None] - features[None]).square().sum(-1)
    valid = clusters[:, None].eq(clusters[None])
    valid &= ~torch.eye(len(features), device=features.device, dtype=torch.bool)
    return distance[valid].sum() / len(features)


def _mlp(input_dim, hidden_dim, output_dim=None):
    layers = [nn.Linear(input_dim, hidden_dim), SingletonBatchNorm(hidden_dim), nn.ReLU()]
    if output_dim is not None:
        layers += [nn.Linear(hidden_dim, output_dim), SingletonBatchNorm(output_dim), nn.ReLU()]
    return nn.Sequential(*layers)


class _Aggregation(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(2 * dimension, dimension, bias=False),
            nn.Tanh(),
            nn.Linear(dimension, 1, bias=False),
        )

    def forward(self, query, neighbors, mask):
        repeated = query[:, None].expand(-1, neighbors.shape[1], -1)
        scores = self.score(torch.cat((repeated, neighbors), -1)).squeeze(-1)
        scores = scores.masked_fill(~mask, -torch.inf).masked_fill(~mask.any(-1, keepdim=True), 0)
        weights = scores.softmax(-1) * mask
        return (weights[..., None] * neighbors).sum(1)


class _CrossAttention(nn.Module):
    """Source coordinate tokens, with standard heads, key softmax and scaling."""

    def __init__(self, dimension, heads, feedforward_dim, dropout):
        super().__init__()
        self.heads, self.dimension = heads, dimension
        self.query, self.key, self.value = (nn.Linear(1, dimension, bias=False) for _ in range(3))
        self.output = nn.Linear(dimension, 1, bias=False)
        self.attention_dropout, self.dropout = nn.Dropout(dropout), nn.Dropout(dropout)
        self.norm1, self.norm2 = nn.LayerNorm(dimension), nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, feedforward_dim), nn.ReLU(), nn.Linear(feedforward_dim, dimension)
        )

    def forward(self, query, context):
        def heads(layer, values):
            projected = layer(values[..., None])
            return projected.reshape(len(values), self.dimension, self.heads, -1).transpose(1, 2)

        q, k, v = heads(self.query, query), heads(self.key, context), heads(self.value, context)
        weights = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(-1)
        combined = (
            (self.attention_dropout(weights) @ v)
            .transpose(1, 2)
            .reshape(len(query), self.dimension, self.dimension)
        )
        values = self.norm1(query + self.dropout(self.output(combined).squeeze(-1)))
        return self.norm2(values + self.dropout(self.ffn(values)))


class _Representation(nn.Module):
    def __init__(self, text_dim, image_dim, p):
        super().__init__()
        d = p["dimension"]
        self.global_learners = nn.ModuleList((_mlp(text_dim, d), _mlp(image_dim, d)))
        self.aggregators = nn.ModuleList(_Aggregation(d) for _ in range(2))
        self.gates = nn.ModuleList(nn.GRUCell(d, d) for _ in range(2))
        self.cross = nn.ModuleList(
            _CrossAttention(d, p["heads"], p["feedforward_dim"], p["dropout"]) for _ in range(2)
        )
        self.fusion = nn.Linear(2 * d, d)
        self.classifier = nn.Linear(d, 2)

    def forward(self, features, contexts, neighbors, masks):
        values = [
            gate(aggregator(context, pool, mask), learner(feature))
            for feature, context, pool, mask, learner, aggregator, gate in zip(
                features,
                contexts,
                neighbors,
                masks,
                self.global_learners,
                self.aggregators,
                self.gates,
            )
        ]
        fused = self.fusion(
            torch.cat(
                (self.cross[0](values[0], values[1]), self.cross[1](values[1], values[0])), -1
            )
        )
        return fused, self.classifier(fused)


def _validate(p):
    for key in (
        "clusters",
        "dimension",
        "context_hidden",
        "heads",
        "feedforward_dim",
        "resource_batch_size",
        "kmeans_n_init",
    ):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError(f"CSFND {key} must be a positive integer.")
    if p["neighbors"] != "all" and (
        type(p["neighbors"]) is not int or p["neighbors"] < 1
    ):
        raise ValueError("CSFND neighbors must be 'all' or a positive integer.")
    if p["clusters"] < 2 or p["dimension"] % p["heads"]:
        raise ValueError("CSFND needs K>=2 and a dimension divisible by heads.")
    for key in ("warmup_epochs", "kmeans_seed"):
        if type(p[key]) is not int or p[key] < 0:
            raise ValueError(f"CSFND {key} must be a nonnegative integer.")
    for key in ("dropout", "context_margin", "class_margin", "triplet_weight", "intra_weight"):
        if type(p[key]) not in (int, float) or not math.isfinite(p[key]) or p[key] < 0:
            raise ValueError(f"CSFND {key} must be finite and nonnegative.")
    if p["dropout"] >= 1 or p["context_margin"] <= 0 or p["class_margin"] <= 0:
        raise ValueError("CSFND dropout/margins are invalid.")
    if (
        p["text"]["freeze"] is not True
        or p["text"]["output"] != "tokens"
        or p["text"]["mask_padding"] is not True
        or p["text"]["max_length"] < 3
        or p["image"]["freeze"] is not True
        or p["image"]["output"] != "avgpool"
        or (p["image"].get("backbone") != "vit" and p["image"]["center_crop"] is not True)
    ):
        raise ValueError(
            "CSFND requires frozen masked BERT body tokens and frozen supervised global image features."
        )


@register_model("CSFND")
class CSFND(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        _validate(p)
        self.parameters_spec = deepcopy(p)
        self.text_encoder, self.image_encoder = (
            material_encoder(config, "text_encoder", _ContentBertEncoder, **p["text"]),
            material_encoder(config, "image_encoder", supervised_encoder, **p["image"]),
        )
        td, vd, d = self.text_encoder.output_dim, self.image_encoder.output_dim, p["dimension"]
        self.context_learners = nn.ModuleList(
            (_mlp(td, p["context_hidden"], d), _mlp(vd, p["context_hidden"], d))
        )
        self.representation = _Representation(td * (p["text"]["max_length"] - 2), vd, p)
        self.local_heads = nn.ModuleList(
            nn.Sequential(nn.Linear(d, 2), nn.Tanh()) for _ in range(p["clusters"])
        )
        self.resources, self.stage_records = {}, {}
        self._train_rows, self._pool_rows = None, None
        self._device_resources, self._indices = {}, {}
        self._members, self._validation_routes = {}, None
        self._local_training_features = {}
        self._epoch, self.stage = 1, None
        self.set_training_stage(None)

    @property
    def device(self):
        return next(self.parameters()).device

    def _keys(self, batch):
        if hub_inputs(self.config):
            return [
                hashlib.sha256(
                    (hashlib.sha256(text.encode()).hexdigest() + Path(path).stem).encode()
                ).hexdigest()
                for text, path in zip(batch["text"], batch["image_path"], strict=True)
            ]
        return [
            pair_identity(text, path)["pair_sha256"]
            for text, path in zip(batch["text"], batch["image_path"])
        ]

    def _tensor(self, key):
        identity = (key, str(self.device))
        if identity not in self._device_resources:
            self._device_resources[identity] = self.resources[key].to(self.device)
        return self._device_resources[identity]

    def _features(self, batch):
        text = self.text_encoder(batch["text"])
        return text[:, 1:], self.image_encoder(batch["image_path"]), text[:, 0]

    def _encode_rows(self, rows, kind):
        output = []
        size = self.parameters_spec["resource_batch_size"]
        modes = [(m, m.training) for m in self.modules()]
        try:
            self.eval()
            with torch.no_grad():
                for start in range(0, len(rows), size):
                    batch = _inputs(rows[start : start + size])
                    if kind == "representation":
                        values = (self._representation(batch)[0],)
                    else:
                        body, image, cls = self._features(batch)
                        values = (
                            (cls, image)
                            if kind == "initial"
                            else tuple(
                                learner(value)
                                for learner, value in zip(
                                    self.context_learners, (body.sum(1), image)
                                )
                            )
                        )
                    output.append(tuple(value.detach().cpu() for value in values))
        finally:
            for module, mode in modes:
                module.training = mode
        return tuple(torch.cat([values[i] for values in output]) for i in range(len(output[0])))

    def _kmeans(self, features):
        from sklearn.cluster import KMeans

        p = self.parameters_spec
        if len(features) <= p["clusters"]:
            raise ValueError("CSFND K must be smaller than the number of training news.")
        fitted = KMeans(
            n_clusters=p["clusters"],
            init="k-means++",
            n_init=p["kmeans_n_init"],
            random_state=p["kmeans_seed"],
        ).fit(features.numpy())
        labels = torch.tensor(fitted.labels_, dtype=torch.long)
        if labels.unique().numel() != p["clusters"]:
            raise ValueError(
                "CSFND KMeans produced empty clusters; adjust explicit K or inspect features."
            )
        return labels

    def prepare(self, train_dataset):
        self._local_training_features.clear()
        rows = list(train_dataset)
        keys = self._keys(_inputs(rows))
        order = sorted(range(len(keys)), key=keys.__getitem__)
        unique = {}
        for i in order:
            if keys[i] in unique and rows[i]["label"] != unique[keys[i]]["label"]:
                raise ValueError("CSFND identical training content has conflicting labels.")
            unique.setdefault(keys[i], rows[i])
        # Resource membership is content-based; every original row still trains.
        self._train_rows = [rows[i] for i in order]
        self._pool_rows = list(unique.values())
        self.resources = {"keys": list(unique)}
        self._indices = {key: i for i, key in enumerate(self.resources["keys"])}
        resource = hub_training_resource(
            self.config,
            "csfnd_initial_clusters",
        )
        initialization = {"mode": "native_train_only", "clusters": self.parameters_spec["clusters"]}
        if resource is not None:
            indexed = resource_rows(resource, "pair_id")
            positions = {row["pair_id"]: row["row"] for row in indexed}
            pool_keys = [
                hashlib.sha256(
                    (
                        hashlib.sha256(row["text"].encode()).hexdigest()
                        + ":"
                        + Path(row["image_path"]).stem
                    ).encode()
                ).hexdigest()
                for row in self._pool_rows
            ]
            if set(positions) != set(pool_keys):
                raise ValueError("CSFND exported clusters differ from unique training content.")
            assignments = resource_tensor(resource, "assignments")
            if assignments.dtype != torch.long or assignments.shape != (len(indexed), 2):
                raise ValueError("CSFND exported cluster assignments have invalid shape or dtype.")
            provided_clusters = resource.settings.get("clusters")
            if type(provided_clusters) is not int or provided_clusters < 2:
                raise ValueError("CSFND exported cluster count is invalid.")
            if assignments.min() < 0 or assignments.max() >= provided_clusters:
                raise ValueError("CSFND exported assignments disagree with their cluster count.")
            if provided_clusters == self.parameters_spec["clusters"]:
                saved = {"assignments": assignments[[positions[key] for key in pool_keys]]}
                mode = "fixed_release_train_only_assignments"
            else:
                # K is a defining tunable parameter (paper Sec5.6.1). The fixed
                # release contains one K, so an alternate K must actually fit
                # KMeans on the same immutable training CLS/image features.
                # Never relabel or truncate the released assignments.
                saved = {"assignments": self._fit_initial_assignments()}
                mode = "requested_K_fit_on_fixed_training_features"
            initialization.update(
                mode=mode,
                provided_clusters=provided_clusters,
                provided_settings=deepcopy(resource.settings),
                initialization_kmeans_parameters=(
                    {
                        "seed": self.parameters_spec["kmeans_seed"],
                        "n_init": self.parameters_spec["kmeans_n_init"],
                    }
                    if mode == "requested_K_fit_on_fixed_training_features"
                    else None
                ),
                quality_clustering_parameters={
                    "seed": self.parameters_spec["kmeans_seed"],
                    "n_init": self.parameters_spec["kmeans_n_init"],
                },
                training_content_sha256=_value_digest(self.resources["keys"]),
            )
        else:
            saved = self._initial_assignments()
        assignments = saved.get("assignments") if isinstance(saved, dict) else None
        count = self.parameters_spec["clusters"]
        if (
            not isinstance(assignments, torch.Tensor)
            or assignments.dtype != torch.long
            or tuple(assignments.shape) != (len(self._pool_rows), 2)
            or assignments.min() < 0
            or assignments.max() >= count
            or any(assignments[:, index].unique().numel() != count for index in range(2))
        ):
            raise ValueError("CSFND initial cluster cache is invalid.")
        self.resources["assignments"] = assignments.clone()
        self.stage_records, self._device_resources = {"initialization": initialization}, {}
        self._members, self._validation_routes = {}, None

    def _initial_assignments(self):
        return load_prepared_resource(
            self,
            "csfnd_initial_clusters",
            lambda: {
                "training_keys": self.resources["keys"],
                "text": encoder_identity(self.text_encoder),
                "image": encoder_identity(self.image_encoder),
                "parameters": {
                    key: self.parameters_spec[key]
                    for key in ("clusters", "kmeans_n_init", "kmeans_seed", "resource_batch_size")
                },
            },
            lambda: {"assignments": self._fit_initial_assignments()},
            encoders=(self.text_encoder, self.image_encoder),
        )

    def _fit_initial_assignments(self):
        return torch.stack(
            [self._kmeans(value) for value in self._encode_rows(self._pool_rows, "initial")],
            -1,
        )

    def training_stages(self):
        if not self.resources:
            raise ValueError("Prepare CSFND training-only clusters before fitting stages.")
        return (
            TrainingStage("context", module="context_learners"),
            TrainingStage("representation", module="representation"),
            *(
                TrainingStage(f"local_{i:03d}", "local", f"local_heads.{i}")
                for i in range(len(self.local_heads))
            ),
        )

    def set_training_stage(self, stage):
        if stage not in (None, "context", "representation") and not (
            isinstance(stage, str)
            and stage.startswith("local_")
            and stage[6:].isdigit()
            and 0 <= int(stage[6:]) < len(self.local_heads)
        ):
            raise ValueError("Unknown CSFND training stage.")
        self.requires_grad_(False)
        if stage in ("context", "representation"):
            self._local_training_features.clear()
        self.stage = stage
        if stage is not None:
            target = (
                self.local_heads[int(stage[6:])]
                if stage.startswith("local_")
                else getattr(self, "context_learners" if stage == "context" else stage)
            )
            target.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.stage != "context":
            self.context_learners.eval()
        if self.stage != "representation":
            self.representation.eval()
        return self

    def _centers(self, features, assignments):
        return torch.stack(
            [features[assignments.eq(i)].mean(0) for i in range(len(self.local_heads))]
        )

    def _build_context(self):
        from sklearn.metrics import calinski_harabasz_score

        contexts = self._encode_rows(self._pool_rows, "context")
        quality = []
        for values in contexts:
            labels = self._kmeans(values)
            score = float(calinski_harabasz_score(values.numpy(), labels.numpy()))
            if not math.isfinite(score):
                raise ValueError("CSFND context quality is nonfinite.")
            quality.append(score)
        self.resources.update(
            context_pool=torch.stack(contexts, 1),
            context_centers=torch.stack(
                [
                    self._centers(value, self.resources["assignments"][:, i])
                    for i, value in enumerate(contexts)
                ],
                1,
            ),
            quality=quality,
            primary=0 if quality[0] > quality[1] else 1,
        )
        self._device_resources.clear()

    def _context_clusters(self, contexts, keys):
        centers = self._tensor("context_centers")
        assignments = torch.stack(
            [
                torch.cdist(value.float(), centers[:, i].float()).argmin(-1)
                for i, value in enumerate(contexts)
            ],
            -1,
        )
        for index, key in enumerate(keys):
            if key in self._indices:
                assignments[index] = self._tensor("assignments")[self._indices[key]]
        return assignments

    def _neighbors(self, contexts, assignments, keys):
        p, pools, masks = self.parameters_spec, [], []
        for modality in range(2):
            selected_rows = []
            for key, cluster in zip(keys, assignments[:, modality].cpu().tolist(), strict=True):
                membership = (modality, cluster)
                if membership not in self._members:
                    self._members[membership] = (
                        self.resources["assignments"][:, modality]
                        .eq(membership[1])
                        .nonzero(as_tuple=True)[0]
                    )
                candidates = self._members[membership]
                if key in self._indices:
                    candidates = candidates[candidates.ne(self._indices[key])]
                if p["neighbors"] != "all":
                    candidates = candidates[: p["neighbors"]]
                selected_rows.append(candidates)
            width = max(1, max(map(len, selected_rows), default=0))
            indices = torch.zeros(len(keys), width, dtype=torch.long)
            valid = torch.zeros(len(keys), width, dtype=torch.bool)
            for row, selected in enumerate(selected_rows):
                indices[row, : len(selected)] = selected
                valid[row, : len(selected)] = True
            valid = valid.to(self.device)
            values = self._tensor("context_pool")[indices.to(self.device), modality]
            values = values.masked_fill(~valid[..., None], 0)
            pools.append(values)
            masks.append(valid)
        return pools, masks

    def _representation(self, batch):
        if "context_pool" not in self.resources:
            raise ValueError("CSFND context training must finish before representation learning.")
        body, image, _ = self._features(batch)
        with torch.no_grad():
            contexts = tuple(
                learner(value)
                for learner, value in zip(self.context_learners, (body.sum(1), image))
            )
        keys = self._keys(batch)
        assignments = self._context_clusters(contexts, keys)
        neighbors, masks = self._neighbors(contexts, assignments, keys)
        features, logits = self.representation((body.flatten(1), image), contexts, neighbors, masks)
        return features, logits, assignments

    def _final_clusters(self, features):
        if "fusion_centers" not in self.resources:
            raise ValueError("CSFND local classifiers are not ready.")
        return torch.cdist(features.float(), self._tensor("fusion_centers").float()).argmin(-1)

    def _loader(self, rows, options, training):
        return DataLoader(
            rows,
            batch_size=options["batch_size" if training else "eval_batch_size"],
            shuffle=training,
            collate_fn=collate_news,
            generator=torch.Generator().manual_seed(torch.initial_seed()),
            num_workers=0,
        )

    def training_stage_data(self, stage, train_data, val_data, options):
        if self._train_rows is None:
            raise ValueError("CSFND training stages require their original training rows.")
        train, val, origin = self._train_rows, list(val_data.dataset), "validation"
        if stage == "context":
            options["selection_metric"], options["min_delta"] = "last", 0.0
            val, origin = [], "fixed-unsupervised-budget"
        elif stage == "representation":
            self._build_context()
        else:
            if "fusion_centers" not in self.resources or not self._local_training_features:
                values = self._encode_rows(self._pool_rows, "representation")[0]
                # Eq12 trains only the local head. The complete representation
                # is frozen and in eval mode throughout all local stages, so
                # reuse these genuine training features instead of recomputing
                # the same network on every one of the 301 local epochs.
                self._local_training_features = dict(zip(self.resources["keys"], values, strict=True))
                if "fusion_centers" not in self.resources:
                    assigned = self.resources["assignments"][:, self.resources["primary"]]
                    self.resources["fusion_centers"] = self._centers(values, assigned)
                    self._device_resources.clear()
            index = int(stage[6:])
            training_clusters = self.resources["assignments"][:, self.resources["primary"]]
            train = [
                row
                for row, key in zip(train, self._keys(_inputs(train)))
                if int(training_clusters[self._indices[key]]) == index
            ]
            validation_keys = self._keys(_inputs(val))
            if self._validation_routes is None or self._validation_routes[0] != validation_keys:
                val_features = self._encode_rows(val, "representation")[0]
                self._validation_routes = (
                    validation_keys,
                    self._final_clusters(val_features.to(self.device)).cpu(),
                )
            val_clusters = self._validation_routes[1]
            val = [row for row, cluster in zip(val, val_clusters) if int(cluster) == index]
            if not val:
                options["selection_metric"], options["min_delta"] = "last", 0.0
                origin = "fixed-local-budget-no-validation-cluster"
        if not train:
            raise ValueError("CSFND local training cluster unexpectedly has no training news.")
        self.stage_records[stage] = {
            "training_rows": len(train),
            "validation_rows": len(val),
            "selection_data": origin,
            "selection_metric": options["selection_metric"],
            "epochs": options["epochs"],
            "training_content_sha256": _value_digest(self._keys(_inputs(train))),
            "validation_content_sha256": _value_digest(self._keys(_inputs(val))) if val else None,
            "single_training_class": len({row["label"] for row in train}) == 1
            if stage.startswith("local_")
            else None,
        }
        return self._loader(train, options, True), self._loader(
            val, options, False
        ) if val else None

    def on_epoch_start(self, epoch, total_epochs):
        self._epoch = epoch

    def calculate_loss(self, batch):
        p = self.parameters_spec
        if self.stage == "context":
            body, image, _ = self._features(batch)
            keys = self._keys(batch)
            if any(key not in self._indices for key in keys):
                raise ValueError("CSFND context triplets may only use training news.")
            labels = self._tensor("assignments")[[self._indices[key] for key in keys]]
            values = [
                learner(feature)
                for learner, feature in zip(self.context_learners, (body.sum(1), image))
            ]
            return {
                name: _triplet(
                    value,
                    labels[:, i],
                    p["context_margin"],
                    warmup=self._epoch <= p["warmup_epochs"],
                )
                for i, (name, value) in enumerate(zip(("context_text", "context_image"), values))
            }
        if self.stage == "representation":
            features, logits, clusters = self._representation(batch)
            primary = clusters[:, self.resources["primary"]]
            return {
                "classification": F.cross_entropy(logits, batch["labels"]),
                "context_triplet": p["triplet_weight"]
                * _triplet(features, primary, p["class_margin"], batch["labels"]),
                "intra_context": p["intra_weight"] * _intra(features, primary),
            }
        if self.stage is not None and self.stage.startswith("local_"):
            with torch.no_grad():
                keys = self._keys(batch)
                features = (
                    torch.stack([self._local_training_features[key] for key in keys]).to(self.device)
                    if all(key in self._local_training_features for key in keys)
                    else self._representation(batch)[0]
                )
            return {
                "classification": F.cross_entropy(
                    self.local_heads[int(self.stage[6:])](features), batch["labels"]
                )
            }
        raise ValueError("Select a CSFND training stage before calculating its loss.")

    def predict(self, batch):
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            features, logits, _ = self._representation(batch)
            if self.stage == "representation":
                return logits
            if self.stage is not None and self.stage.startswith("local_"):
                return self.local_heads[int(self.stage[6:])](features)
            clusters = self._final_clusters(features)
            predictions = features.new_empty(len(features), 2)
            for index, head in enumerate(self.local_heads):
                selected = clusters.eq(index)
                if selected.any():
                    predictions[selected] = head(features[selected])
            return predictions
        finally:
            for module, mode in modes:
                module.training = mode

    def get_extra_state(self):
        return {
            "format_version": 1,
            "parameters": deepcopy(self.parameters_spec),
            "resources": deepcopy(self.resources),
            "sha256": _value_digest(self.resources),
            "stage_records": deepcopy(self.stage_records),
        }

    def set_extra_state(self, state):
        self._local_training_features.clear()
        if (
            not isinstance(state, dict)
            or set(state)
            != {"format_version", "parameters", "resources", "sha256", "stage_records"}
            or state["format_version"] != 1
            or state["parameters"] != self.parameters_spec
            or not isinstance(state["resources"], dict)
            or state["sha256"] != _value_digest(state["resources"])
            or not isinstance(state["stage_records"], dict)
        ):
            raise ValueError("CSFND checkpoint parameters/resources/digest are incompatible.")
        r, p = state["resources"], self.parameters_spec
        if r:
            required = {"keys", "assignments"}
            if "context_pool" in r:
                required |= {"context_pool", "context_centers", "quality", "primary"}
            if "fusion_centers" in r:
                required.add("fusion_centers")
            if (
                set(r) != required
                or ("fusion_centers" in r and "context_pool" not in r)
                or not isinstance(r["keys"], list)
                or not r["keys"]
                or r["keys"] != sorted(set(r["keys"]))
            ):
                raise ValueError("CSFND checkpoint resources have invalid fields/identities.")
            for key in r["keys"]:
                if (
                    not isinstance(key, str)
                    or len(key) != 64
                    or any(c not in "0123456789abcdef" for c in key)
                ):
                    raise ValueError("CSFND checkpoint content hash is invalid.")
            n, k, d = len(r["keys"]), p["clusters"], p["dimension"]
            expected = {
                "assignments": (n, 2),
                "context_pool": (n, 2, d),
                "context_centers": (k, 2, d),
                "fusion_centers": (k, d),
            }
            for key, shape in expected.items():
                if key in r and (
                    not isinstance(r[key], torch.Tensor)
                    or tuple(r[key].shape) != shape
                    or not torch.isfinite(r[key]).all()
                    or (key != "assignments" and not r[key].is_floating_point())
                ):
                    raise ValueError("CSFND checkpoint tensor shape/value is invalid.")
            labels = r["assignments"]
            if (
                labels.dtype != torch.long
                or labels.min() < 0
                or labels.max() >= k
                or any(labels[:, i].unique().numel() != k for i in range(2))
            ):
                raise ValueError("CSFND checkpoint cluster labels are invalid.")
            if "context_pool" in r and (
                type(r["primary"]) is not int
                or r["primary"] not in (0, 1)
                or not isinstance(r["quality"], list)
                or len(r["quality"]) != 2
                or any(
                    type(q) not in (int, float) or not math.isfinite(q) or q < 0
                    for q in r["quality"]
                )
            ):
                raise ValueError("CSFND checkpoint context quality/primary modality is invalid.")
            if "context_pool" in r:
                centers = torch.stack(
                    [self._centers(r["context_pool"][:, i], labels[:, i]) for i in range(2)],
                    1,
                )
                primary = 0 if r["quality"][0] > r["quality"][1] else 1
                if primary != r["primary"] or not torch.allclose(centers, r["context_centers"]):
                    raise ValueError("CSFND checkpoint context centers/primary are inconsistent.")
        self.resources, self.stage_records = deepcopy(r), deepcopy(state["stage_records"])
        self._indices = {key: i for i, key in enumerate(r.get("keys", []))}
        self._device_resources, self._train_rows, self._pool_rows = {}, None, None
        self._members, self._validation_routes = {}, None
        self.set_training_stage(None)
