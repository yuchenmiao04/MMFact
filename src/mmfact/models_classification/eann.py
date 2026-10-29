"""EANN (KDD 2018): TextCNN/ResNet features with an adversarial event head."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder
from mmfact.models_classification._encoders import WordEncoder
from mmfact.models_classification._supervised_vision import supervised_encoder
from mmfact.registry import register_model


class GradientReversal(torch.autograd.Function):
    """Keep features unchanged, reversing only their backward gradient."""

    @staticmethod
    def forward(ctx, features: Tensor, strength: float) -> Tensor:
        ctx.strength = strength
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, None]:
        return -ctx.strength * gradient, None


@register_model("EANN")
class EANN(AbstractDetector):
    """Learn event-invariant features; event targets are needed only to train."""

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        parameters = config["model"]["parameters"]
        self.text = material_encoder(config, "text", WordEncoder, **parameters["text"])
        self.image = material_encoder(config, "image", supervised_encoder, **parameters["image"])
        hidden = parameters["hidden_dim"]
        self.kernels = tuple(parameters["kernel_sizes"])
        if not self.kernels or any(type(size) is not int or size < 1 for size in self.kernels):
            raise ValueError("EANN kernel_sizes must contain positive integers.")
        self.reverse_lambda = float(parameters["reverse_lambda"])
        if not torch.isfinite(torch.tensor(self.reverse_lambda)) or self.reverse_lambda < 0:
            raise ValueError("EANN reverse_lambda must be finite and nonnegative.")
        channels = parameters["filter_num"]
        self.convolutions = nn.ModuleList(
            nn.Conv1d(self.text.dimension, channels, size) for size in self.kernels
        )
        self.text_projection = nn.Linear(channels * len(self.kernels), hidden)
        self.image_projection = nn.Linear(self.image.output_dim, hidden)
        self.classifier = nn.Linear(hidden * 2, 2)
        self.event_classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 2)
        )
        self.event_source: str | None = None
        self.event_names: list[str] = []
        self.event_targets: dict[str, int] = {}

    def prepare(self, train_dataset) -> None:
        """Fit the vocabulary and event assignment on the training split only."""
        rows = list(train_dataset)
        if len(rows) < 2:
            raise ValueError("EANN requires at least two training samples and events.")
        self.text.prepare([row["text"] for row in rows])
        options = self.config["model"]["parameters"]
        source = options["event_source"]
        if source not in {"auto", "metadata", "cluster"}:
            raise ValueError("EANN event_source must be auto, metadata, or cluster.")
        event_ids = [row["metadata"].get("event_id") for row in rows]
        available = [event_id is not None for event_id in event_ids]
        if source == "auto":
            if any(available) and not all(available):
                raise ValueError(
                    "EANN found partially missing metadata.event_id; complete the IDs "
                    "or explicitly use event_source=cluster."
                )
            source = "metadata" if all(available) else "cluster"
        if source == "metadata":
            if not all(available) or any(
                isinstance(event_id, bool)
                or not isinstance(event_id, (str, int))
                or not str(event_id).strip()
                for event_id in event_ids
            ):
                raise ValueError(
                    "EANN requires precomputed metadata.event_id for every training sample "
                    "(nonempty string or integer); prepare fixed event labels before training."
                )
            names = sorted({str(event_id) for event_id in event_ids})
            indices = {name: index for index, name in enumerate(names)}
            targets = [indices[str(event_id)] for event_id in event_ids]
        else:
            from sklearn.cluster import KMeans
            from sklearn.feature_extraction.text import TfidfVectorizer
            from threadpoolctl import threadpool_limits

            count = options["event_clusters"]
            if type(count) is not int or count < 2:
                raise ValueError("EANN event_clusters must be an integer >= 2.")
            count = min(count, len(rows))
            # Character n-grams work in both languages without using class labels.
            features = TfidfVectorizer(
                analyzer="char", ngram_range=(1, 3), max_features=5000, sublinear_tf=True
            ).fit_transform([row["text"] for row in rows])
            clusters = KMeans(n_clusters=count, random_state=options["event_seed"], n_init=10)
            # Avoid competing OpenMP pools from PyTorch and scikit-learn.
            with threadpool_limits(limits=1, user_api="openmp"):
                targets = clusters.fit_predict(features).tolist()
            unique = sorted(set(targets))
            indices = {target: index for index, target in enumerate(unique)}
            targets = [indices[target] for target in targets]
            names = [f"cluster_{target}" for target in unique]
        if len(names) < 2:
            raise ValueError("EANN requires at least two distinct training events.")
        self.event_source = source
        self.event_names = names
        self.event_targets = {row["sample_id"]: target for row, target in zip(rows, targets)}
        self._resize_event_head(len(names))

    def _resize_event_head(self, count: int) -> None:
        previous = self.event_classifier[-1]
        if previous.out_features != count:
            self.event_classifier[-1] = nn.Linear(
                previous.in_features,
                count,
                device=previous.weight.device,
                dtype=previous.weight.dtype,
            )

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "event_source": self.event_source,
            "event_names": self.event_names,
            "event_targets": self.event_targets,
        }

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        self.event_source = state["event_source"]
        self.event_names = list(state["event_names"])
        self.event_targets = dict(state["event_targets"])
        self._resize_event_head(max(2, len(self.event_names)))

    def _features(self, batch: Mapping[str, Any]) -> Tensor:
        embedded, _, mask = self.text(batch["text"])
        embedded = embedded * mask.unsqueeze(-1)
        sequence = embedded.transpose(1, 2)
        if sequence.shape[-1] < max(self.kernels):
            sequence = F.pad(sequence, (0, max(self.kernels) - sequence.shape[-1]))
        pooled = [F.relu(layer(sequence)).amax(dim=-1) for layer in self.convolutions]
        text = F.relu(self.text_projection(torch.cat(pooled, dim=-1)))
        image = F.relu(self.image_projection(self.image(batch["image_path"])))
        return torch.cat((text, image), dim=-1)

    def calculate_loss(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        if not self.event_targets:
            raise RuntimeError("Call EANN.prepare(training_dataset) before calculating loss.")
        try:
            events = [self.event_targets[sample_id] for sample_id in batch["sample_id"]]
        except KeyError as error:
            raise ValueError(
                "EANN event loss only accepts samples from its training split."
            ) from error
        features = self._features(batch)
        event_logits = self.event_classifier(GradientReversal.apply(features, self.reverse_lambda))
        targets = torch.tensor(events, device=features.device, dtype=torch.long)
        return {
            "classification": F.cross_entropy(self.classifier(features), batch["labels"]),
            "event": F.cross_entropy(event_logits, targets),
        }

    def predict(self, batch: Mapping[str, Any]) -> Tensor:
        return self.classifier(self._features(batch))
