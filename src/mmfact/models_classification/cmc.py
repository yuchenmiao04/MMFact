"""CMC author-code mutual CRD, independent peers and frozen BLOCK/FC fusion."""

import math
import re
from collections.abc import Mapping
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from mmfact.base import AbstractDetector, TrainingStage
from mmfact.data.features import file_digest, fingerprint
from mmfact.data.preprocessing import configure_preprocessing
from mmfact.models_classification._cmc_inputs import CMCImageEncoder, CMCTextEncoder
from mmfact.registry import register_model


def _positive(value, name, *, integer=False, zero=False):
    valid = type(value) is int if integer else type(value) in (int, float)
    if not valid or not math.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError(
            f"CMC {name} must be a finite {'nonnegative' if zero else 'positive'} number."
        )


def _validate(p):
    for key in ("text", "image", "crd", "fusion"):
        if not isinstance(p.get(key), Mapping):
            raise ValueError(f"CMC {key} must be a mapping.")
    if p["text"].get("language") not in ("zh", "en"):
        raise ValueError("CMC text.language must be zh or en.")
    if p["text"].get("freeze", False) is not False or p["image"].get("freeze", False) is not False:
        raise ValueError("CMC peers must be trainable during mutual distillation.")
    if p["image"].get("weights") not in (None, "IMAGENET1K_V1"):
        raise ValueError("CMC image weights must be IMAGENET1K_V1 or null.")
    if p["image"].get("backbone") != "vit" and p["image"].get("architecture") != "resnet50":
        raise ValueError("CMC common image peer requires supervised ViT or ResNet50.")
    hidden = p.get("image_classifier_dims")
    if not isinstance(hidden, list):
        raise ValueError("CMC image_classifier_dims must be a list.")
    for value in hidden:
        _positive(value, "image classifier width", integer=True)
    for value, name in (
        (p.get("image_classifier_dropout"), "image classifier dropout"),
        (p["fusion"].get("input_dropout"), "fusion input dropout"),
    ):
        _positive(value, name, zero=True)
        if value >= 1:
            raise ValueError("CMC dropouts must be less than one.")
    crd, fusion = p["crd"], p["fusion"]
    for key in ("dim", "num_negatives"):
        _positive(crd.get(key), f"crd.{key}", integer=True)
    for key in ("temperature", "epsilon"):
        _positive(crd.get(key), f"crd.{key}")
    for key in ("weight", "momentum"):
        _positive(crd.get(key), f"crd.{key}", zero=True)
    if crd["momentum"] > 1:
        raise ValueError("CMC memory momentum must not exceed one.")
    if fusion.get("name") not in ("block", "fc"):
        raise ValueError("CMC fusion.name must be block or fc.")
    for key in ("mm_dim", "chunks", "rank"):
        _positive(fusion.get(key), f"fusion.{key}", integer=True)
    if fusion["chunks"] > fusion["mm_dim"]:
        raise ValueError("CMC BLOCK chunks must not exceed mm_dim.")


class _SignedSqrt(torch.autograd.Function):
    """Exact signed square root, with the undefined derivative at zero set to zero."""

    @staticmethod
    def forward(ctx, values):
        ctx.save_for_backward(values)
        return values.sign() * values.abs().sqrt()

    @staticmethod
    def backward(ctx, gradient):
        (values,) = ctx.saved_tensors
        nonzero = values != 0
        derivative = torch.zeros_like(values)
        derivative[nonzero] = 0.5 / values[nonzero].abs().sqrt()
        return gradient * derivative


def _chunk_sizes(dimension, chunks):
    """Author BLOCK ceil splits, redistributing only a negative final remainder."""
    size = math.ceil(dimension / chunks)
    sizes = [size] * chunks
    sizes[-1] -= sum(sizes) - dimension
    if sizes[-1] <= 0:
        # Match the source redistribution while avoiding an empty chunk.
        missing = size - sizes[-1]
        sizes[-1] = size
        for index in range(missing):
            sizes[-index - 1] -= 1
    if sum(sizes) != dimension or min(sizes) < 1:
        raise ValueError("CMC BLOCK dimensions do not admit positive source chunks.")
    return sizes


class _Block(nn.Module):
    def __init__(self, image_dim, text_dim, mm_dim, chunks, rank, input_dropout):
        super().__init__()
        self.sizes = _chunk_sizes(mm_dim, chunks)
        self.rank = rank
        self.image = nn.Linear(image_dim, mm_dim)
        self.text = nn.Linear(text_dim, mm_dim)
        self.image_chunks = nn.ModuleList(nn.Linear(size, size * rank) for size in self.sizes)
        self.text_chunks = nn.ModuleList(nn.Linear(size, size * rank) for size in self.sizes)
        self.dropout = nn.Dropout(input_dropout)
        self.classifier = nn.Linear(mm_dim, 2)

    def forward(self, image, text):
        image = self.dropout(self.image(image)).split(self.sizes, dim=1)
        text = self.dropout(self.text(text)).split(self.sizes, dim=1)
        chunks = []
        for left, right, image_layer, text_layer in zip(
            image, text, self.image_chunks, self.text_chunks, strict=True
        ):
            product = image_layer(left) * text_layer(right)
            merged = product.reshape(len(product), self.rank, -1).sum(dim=1)
            chunks.append(F.normalize(_SignedSqrt.apply(merged), dim=1, eps=1e-12))
        return self.classifier(torch.cat(chunks, dim=1))


class _FC(nn.Module):
    def __init__(self, image_dim, text_dim):
        super().__init__()
        self.classifier = nn.Linear(image_dim + text_dim, 2)

    def forward(self, image, text):
        return self.classifier(torch.cat((image, text), dim=1))


class _Peer(nn.Module):
    def __init__(self, encoder, projection_dim, epsilon, hidden=(), dropout=0):
        super().__init__()
        self.encoder, self.epsilon = encoder, epsilon
        layers, width = [], encoder.output_dim
        for output in hidden:
            layers.extend((nn.Linear(width, output), nn.ReLU(), nn.Dropout(dropout)))
            width = output
        layers.append(nn.Linear(width, 2))
        self.classifier = nn.Sequential(*layers)
        self.projection = nn.Linear(encoder.output_dim, projection_dim)
        # The source VGG head explicitly initializes all its linear layers this way.
        if hidden:
            for layer in self.classifier:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.01)
                    nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        features = self.encoder(inputs)
        return features, self.classifier(features)

    def project(self, features):
        return F.normalize(self.projection(features), dim=1, eps=self.epsilon)


class _ContrastMemory(nn.Module):
    """Train-only opposite-label NCE and two momentum banks; no inference lookup."""

    def __init__(self, dim, num_negatives, temperature, momentum, epsilon, seed):
        super().__init__()
        self.options = dict(
            dim=dim,
            num_negatives=num_negatives,
            temperature=temperature,
            momentum=momentum,
            epsilon=epsilon,
        )
        self.seed = seed
        self.sample_ids, self.row_sha256, self.training_sha256 = [], [], None
        self._indices = {}
        self.register_buffer("image_bank", torch.empty(0, dim))
        self.register_buffer("text_bank", torch.empty(0, dim))
        self.register_buffer("normalizers", torch.full((2,), -1.0))
        self.register_buffer("labels", torch.empty(0, dtype=torch.long))
        self.register_buffer("generator_state", torch.Generator().manual_seed(seed).get_state())

    def prepare(self, train_rows):
        rows = list(train_rows)
        if not rows or any(not isinstance(row, Mapping) for row in rows):
            raise ValueError("CMC needs nonempty canonical training rows.")
        ids = [row.get("sample_id") for row in rows]
        if any(not isinstance(value, str) or not value.strip() for value in ids) or len(
            set(ids)
        ) != len(ids):
            raise ValueError("CMC training IDs must be unique nonempty strings.")
        rows = sorted(rows, key=lambda row: row["sample_id"])
        labels = [row.get("label") for row in rows]
        if any(type(label) is not int or label not in (0, 1) for label in labels) or set(
            labels
        ) != {0, 1}:
            raise ValueError("CMC training requires both binary classes for negative sampling.")
        images, hashes = {}, []
        for row in rows:
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError("CMC training text must be nonempty.")
            path = row.get("image_path")
            if not isinstance(path, str) or not path.strip():
                raise ValueError("CMC training image paths must be nonempty strings.")
            if path not in images:
                images[path] = file_digest(path)
            hashes.append(fingerprint([row["sample_id"], row["text"], images[path], row["label"]]))
        ids = [row["sample_id"] for row in rows]
        identity = fingerprint(list(zip(ids, hashes, strict=True)))
        if self.sample_ids:
            if identity != self.training_sha256:
                raise ValueError("CMC memory was already fitted to a different training set.")
            return
        self.sample_ids, self.row_sha256, self.training_sha256 = ids, hashes, identity
        self._indices = {sample: index for index, sample in enumerate(ids)}
        generator = torch.Generator().manual_seed(self.seed)
        bound = math.sqrt(3 / self.options["dim"])
        shape = (len(rows), self.options["dim"])
        for name in ("image_bank", "text_bank"):
            values = torch.empty(shape).uniform_(-bound, bound, generator=generator)
            setattr(self, name, values.to(getattr(self, name)))
        self.labels = torch.tensor(labels, dtype=torch.long, device=self.labels.device)
        self.generator_state.copy_(generator.get_state().to(self.generator_state.device))

    def indices(self, sample_ids, labels):
        if not self.sample_ids:
            raise RuntimeError("Prepare CMC training memory before mutual distillation.")
        if (
            not isinstance(sample_ids, (list, tuple))
            or not sample_ids
            or any(not isinstance(sample, str) for sample in sample_ids)
            or len(set(sample_ids)) != len(sample_ids)
            or any(sample not in self._indices for sample in sample_ids)
        ):
            raise ValueError("CMC mutual batches require distinct prepared training IDs.")
        indices = torch.tensor(
            [self._indices[sample] for sample in sample_ids], device=self.labels.device
        )
        if (
            not isinstance(labels, torch.Tensor)
            or labels.dtype != torch.long
            or labels.shape != indices.shape
            or not torch.equal(labels.to(self.labels.device), self.labels[indices])
        ):
            raise ValueError("CMC batch labels must match the prepared training labels.")
        return indices

    def sample(self, indices):
        generator = torch.Generator().set_state(self.generator_state.cpu())
        count = self.options["num_negatives"]
        rows = []
        for positive in indices.cpu().tolist():
            candidates = torch.where(self.labels.cpu() != self.labels[positive].cpu())[0]
            selected = (
                torch.randint(len(candidates), (count,), generator=generator)
                if count > len(candidates)
                else torch.randperm(len(candidates), generator=generator)[:count]
            )
            rows.append(torch.cat((torch.tensor([positive]), candidates[selected])))
        self.generator_state.copy_(generator.get_state().to(self.generator_state.device))
        return torch.stack(rows).to(self.labels.device)

    def _nce(self, scores, direction):
        scores = scores / self.normalizers[direction].detach().clone()
        noise = self.options["num_negatives"] / len(self.labels)
        positive = torch.log(scores[:, 0] / (scores[:, 0] + noise + 1e-7))
        negative = torch.log(noise / (scores[:, 1:] + noise + 1e-7)).sum(dim=1)
        return -(positive + negative).mean()

    def forward(self, image, text, indices, sampled=None):
        if not self.training:
            raise RuntimeError("CMC CRD memory is a training-only operation.")
        if (
            not isinstance(indices, torch.Tensor)
            or indices.dtype != torch.long
            or indices.ndim != 1
            or not len(indices)
            or indices.unique().numel() != len(indices)
            or (indices < 0).any()
            or (indices >= len(self.labels)).any()
            or any(
                not isinstance(query, torch.Tensor)
                or query.shape != (len(indices), self.options["dim"])
                or not query.is_floating_point()
                or not query.isfinite().all()
                for query in (image, text)
            )
        ):
            raise ValueError("CMC memory requires finite peer features and valid training indices.")
        sampled = self.sample(indices) if sampled is None else sampled
        expected = (len(indices), self.options["num_negatives"] + 1)
        if (
            not isinstance(sampled, torch.Tensor)
            or sampled.dtype != torch.long
            or sampled.shape != expected
            or not torch.equal(sampled[:, 0], indices)
            or (sampled < 0).any()
            or (sampled >= len(self.labels)).any()
            or (self.labels[sampled[:, 1:]] == self.labels[indices, None]).any()
        ):
            raise ValueError("CMC negatives must belong to opposite training classes.")
        scores = [
            torch.exp(
                torch.einsum("bkd,bd->bk", bank[sampled].detach(), query)
                / self.options["temperature"]
            )
            for query, bank in ((image, self.text_bank), (text, self.image_bank))
        ]
        if any(not value.isfinite().all() or (value <= 0).any() for value in scores):
            raise ValueError(
                "CMC NCE scores overflowed or underflowed; check temperature/features."
            )
        with torch.no_grad():
            for direction, values in enumerate(scores):
                if self.normalizers[direction] < 0:
                    normalizer = values.detach().mean() * len(self.labels)
                    if not normalizer.isfinite() or normalizer <= 0:
                        raise ValueError(
                            "CMC NCE normalization constant is not finite and positive."
                        )
                    self.normalizers[direction].copy_(normalizer)
        image_loss, text_loss = [self._nce(value, i) for i, value in enumerate(scores)]
        with torch.no_grad():
            for name, query in (("image_bank", image), ("text_bank", text)):
                bank = getattr(self, name)
                updated = self.options["momentum"] * bank[indices]
                updated += (1 - self.options["momentum"]) * query.detach()
                bank.index_copy_(
                    0, indices, F.normalize(updated, dim=1, eps=self.options["epsilon"])
                )
        return image_loss, text_loss

    def get_extra_state(self):
        state = {
            "version": 1,
            "options": deepcopy(self.options),
            "seed": self.seed,
            "sample_ids": list(self.sample_ids),
            "row_sha256": list(self.row_sha256),
            "training_sha256": self.training_sha256,
            "labels": self.labels.cpu().tolist(),
        }
        return {**state, "sha256": fingerprint(state)}

    def set_extra_state(self, state):
        keys = {
            "version",
            "options",
            "seed",
            "sample_ids",
            "row_sha256",
            "training_sha256",
            "labels",
            "sha256",
        }
        if not isinstance(state, dict) or set(state) != keys:
            raise ValueError("Invalid CMC memory state fields.")
        payload = {key: value for key, value in state.items() if key != "sha256"}
        try:
            valid_hash = state["sha256"] == fingerprint(payload)
        except (TypeError, ValueError):
            valid_hash = False
        ids, hashes = state["sample_ids"], state["row_sha256"]
        if (
            not valid_hash
            or type(state["version"]) is not int
            or state["version"] != 1
            or not isinstance(state["options"], dict)
            or fingerprint(state["options"]) != fingerprint(self.options)
            or type(state["seed"]) is not int
            or not 0 <= state["seed"] < 2**64
            or not isinstance(ids, list)
            or not isinstance(hashes, list)
            or len(ids) != len(hashes)
            or any(not isinstance(sample, str) or not sample.strip() for sample in ids)
            or ids != sorted(set(ids))
            or any(
                not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in hashes
            )
            or not isinstance(state["labels"], list)
            or any(type(label) is not int for label in state["labels"])
            or state["labels"] != self.labels.cpu().tolist()
            or len(ids) != len(self.labels)
            or (
                ids and state["training_sha256"] != fingerprint(list(zip(ids, hashes, strict=True)))
            )
            or (not ids and state["training_sha256"] is not None)
        ):
            raise ValueError("CMC training identity, recipe or labels are inconsistent.")
        self.seed = state["seed"]
        self.sample_ids, self.row_sha256 = list(ids), list(hashes)
        self.training_sha256 = state["training_sha256"]
        self._indices = {sample: index for index, sample in enumerate(ids)}

    def _load_from_state_dict(self, state, prefix, *args, **kwargs):
        names = ("image_bank", "text_bank", "normalizers", "labels", "generator_state")
        if all(prefix + name in state for name in names):
            image, text, normalizers, labels, generator = [state[prefix + name] for name in names]
            if (
                not all(
                    isinstance(value, torch.Tensor)
                    for value in (image, text, normalizers, labels, generator)
                )
                or image.ndim != 2
                or image.shape[1] != self.options["dim"]
                or text.shape != image.shape
                or not image.is_floating_point()
                or not text.is_floating_point()
                or not image.isfinite().all()
                or not text.isfinite().all()
                or normalizers.shape != (2,)
                or not normalizers.is_floating_point()
                or not normalizers.isfinite().all()
                or not ((normalizers == -1).all() or (normalizers > 0).all())
                or labels.dtype != torch.long
                or labels.shape != (len(image),)
                or (len(labels) and set(labels.cpu().tolist()) != {0, 1})
                or generator.dtype != torch.uint8
                or generator.ndim != 1
                or (not len(labels) and not (normalizers == -1).all())
            ):
                raise ValueError("Invalid CMC memory tensors, classes or normalization constants.")
            try:
                torch.Generator().set_state(generator.cpu())
            except RuntimeError as error:
                raise ValueError("Invalid CMC negative-sampling generator state.") from error
            for name in ("image_bank", "text_bank", "labels", "generator_state"):
                current = getattr(self, name)
                setattr(
                    self,
                    name,
                    torch.empty_like(
                        state[prefix + name], device=current.device, dtype=current.dtype
                    ),
                )
        super()._load_from_state_dict(state, prefix, *args, **kwargs)


@register_model("CMC")
class CMC(AbstractDetector):
    """CodeBased/Corrected two-stage CMC; no claim of unverified paper-best settings."""

    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        _validate(p)
        self.parameters_spec = deepcopy(p)
        crd = p["crd"]
        self.image_peer = _Peer(
            CMCImageEncoder(**p["image"]),
            crd["dim"],
            crd["epsilon"],
            p["image_classifier_dims"],
            p["image_classifier_dropout"],
        )
        self.text_peer = _Peer(CMCTextEncoder(**p["text"]), crd["dim"], crd["epsilon"])
        self.memory = _ContrastMemory(
            **{key: value for key, value in crd.items() if key != "weight"},
            seed=torch.initial_seed(),
        )
        self.crd_weight = crd["weight"]
        image_dim, text_dim = self.image_encoder.output_dim, self.text_encoder.output_dim
        fusion = p["fusion"]
        self.fusion = (
            _Block(
                image_dim,
                text_dim,
                **{key: value for key, value in fusion.items() if key != "name"},
            )
            if fusion["name"] == "block"
            else _FC(image_dim, text_dim)
        )
        self._stage = None
        self.set_training_stage("mutual_distillation")
        self.register_load_state_dict_post_hook(self._validate_loaded_stage)

    @staticmethod
    def _validate_loaded_stage(model, _incompatible_keys):
        mutual = model._stage == "mutual_distillation"
        if (
            any(peer.encoder.freeze != (not mutual) for peer in (model.image_peer, model.text_peer))
            or model.image_encoder.augment != mutual
            or any(
                parameter.requires_grad != mutual
                for peer in (model.image_peer, model.text_peer)
                for parameter in peer.parameters()
            )
            or any(
                parameter.requires_grad != (model._stage == "fusion")
                for parameter in model.fusion.parameters()
            )
        ):
            raise ValueError("CMC encoder freezing/augmentation does not match the saved stage.")

    @property
    def image_encoder(self):
        return self.image_peer.encoder

    @property
    def text_encoder(self):
        return self.text_peer.encoder

    def prepare(self, train_dataset):
        self.memory.prepare(train_dataset)

    def training_stages(self):
        return (
            TrainingStage("mutual_distillation", heads=("image_peer", "text_peer")),
            TrainingStage("fusion", module="fusion"),
        )

    def set_training_stage(self, stage):
        if stage not in (None, "mutual_distillation", "fusion"):
            raise ValueError("Unknown CMC training stage.")
        self._stage = stage
        mutual = stage == "mutual_distillation"
        for peer in (self.image_peer, self.text_peer):
            peer.requires_grad_(mutual)
            peer.encoder._set_frozen(not mutual)
            if not mutual:
                peer.eval()
            if peer.encoder.preprocessing is not None:
                peer.encoder.preprocessing.reset()
        self.image_encoder.set_augmentation(mutual)
        self.fusion.requires_grad_(stage == "fusion")

    def train(self, mode=True):
        super().train(mode)
        if self._stage != "mutual_distillation":
            self.image_peer.eval()
            self.text_peer.eval()
        return self

    def optimizer_parameter_groups(self):
        if self._stage == "mutual_distillation":
            return {"image": self.image_peer.parameters(), "text": self.text_peer.parameters()}
        return {
            "default": (
                parameter for parameter in self.fusion.parameters() if parameter.requires_grad
            )
        }

    def get_extra_state(self):
        return {"version": 1, "parameters": deepcopy(self.parameters_spec), "stage": self._stage}

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or set(state) != {"version", "parameters", "stage"}
            or type(state["version"]) is not int
            or state["version"] != 1
            or state["parameters"] != self.parameters_spec
            or state["stage"] not in (None, "mutual_distillation", "fusion")
        ):
            raise ValueError("Invalid CMC training stage state.")
        self.set_training_stage(state["stage"])

    def training_stage_data(self, stage, train_data, val_data, options):
        if stage != self._stage:
            raise ValueError("CMC stage loaders must match the active training stage.")
        loaders = []
        for original, key, shuffle in (
            (train_data, "batch_size", True),
            (val_data, "eval_batch_size", False),
        ):
            if (
                not isinstance(original, DataLoader)
                or original.batch_size is None
                or not isinstance(original.sampler, (RandomSampler, SequentialSampler))
            ):
                raise ValueError("CMC stages require ordinary nonempty news loaders.")
            loaders.append(
                DataLoader(
                    original.dataset,
                    batch_size=options[key],
                    shuffle=shuffle,
                    collate_fn=original.collate_fn,
                    num_workers=original.num_workers,
                    pin_memory=original.pin_memory,
                    worker_init_fn=original.worker_init_fn,
                    generator=torch.Generator().manual_seed(
                        (self.memory.seed + (100000 if stage == "fusion" else 0) + int(not shuffle))
                        % 2**64
                    ),
                )
            )
        if stage == "fusion":
            configure_preprocessing(
                self,
                {"train": train_data.dataset, "val": val_data.dataset},
                self.config["preprocessing"],
            )
        return tuple(loaders)

    def calculate_loss(self, batch):
        if self._stage == "mutual_distillation":
            indices = self.memory.indices(batch["sample_id"], batch["labels"])
            image, image_logits = self.image_peer(batch["image_path"])
            text, text_logits = self.text_peer(batch["text"])
            image_nce, text_nce = self.memory(
                self.image_peer.project(image), self.text_peer.project(text), indices
            )
            return {
                "image_classification": F.cross_entropy(image_logits, batch["labels"]),
                "text_classification": F.cross_entropy(text_logits, batch["labels"]),
                "image_contrastive": self.crd_weight * image_nce,
                "text_contrastive": self.crd_weight * text_nce,
            }
        if self._stage == "fusion":
            with torch.no_grad():
                image, text = (
                    self.image_encoder(batch["image_path"]),
                    self.text_encoder(batch["text"]),
                )
            return F.cross_entropy(self.fusion(image, text), batch["labels"])
        raise RuntimeError("Select a CMC training stage before calculating loss.")

    def _predict(self, batch, heads):
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            with torch.no_grad():
                if heads:
                    return {
                        "image_peer": self.image_peer(batch["image_path"])[1],
                        "text_peer": self.text_peer(batch["text"])[1],
                    }
                return self.fusion(
                    self.image_encoder(batch["image_path"]), self.text_encoder(batch["text"])
                )
        finally:
            for module, mode in modes:
                module.training = mode

    def predict_heads(self, batch):
        return self._predict(batch, heads=True)

    def predict(self, batch):
        return self._predict(batch, heads=False)
