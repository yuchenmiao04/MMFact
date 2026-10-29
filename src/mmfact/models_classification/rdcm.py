"""RDCM-DG: joint kernel alignment and real-anchor cross-modal contrast."""

import math
from copy import deepcopy
from itertools import combinations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import hub_inputs, material_encoder
from mmfact.models_classification._domains import DomainTargets
from mmfact.models_classification._encoders import BertEncoder
from mmfact.models_classification._supervised_vision import imagenet_classifier, supervised_encoder
from mmfact.registry import register_model


def _gaussian_kernel(left, right, sigmas):
    distance = torch.cdist(left, right).square()
    return sum(torch.exp(-distance / (2 * sigma**2)) for sigma in sigmas)


def _joint_discrepancy(text, image, domains, sigmas):
    """Equations 6--7: biased joint MMD, summed domain pairs scaled by 2/M."""
    groups = [domains.eq(domain).nonzero().flatten() for domain in domains.unique()]
    losses = []
    for first, second in combinations(groups, 2):

        def joint(left, right):
            return _gaussian_kernel(text[left], text[right], sigmas) * _gaussian_kernel(
                image[left], image[right], sigmas
            )

        losses.append(
            joint(first, first).mean()
            + joint(second, second).mean()
            - 2 * joint(first, second).mean()
        )
    return (
        torch.stack(losses).sum() * (2.0 / len(groups))
        if losses
        else (text.sum() + image.sum()) * 0
    )


def _cross_modal(text, image, instances, labels, temperature, threshold):
    """Equations 9--10 with the paper's ImageNet softmax instance vectors."""
    if not labels.eq(0).any():
        return (text.sum() + image.sum()) * 0
    instances = F.normalize(instances.detach().softmax(dim=-1), dim=-1)
    similarity = ((instances @ instances.T).clamp(-1, 1) + 1) / 2
    weights = (threshold - similarity).clamp_min(0)
    weights.fill_diagonal_(0)
    scores = F.normalize(text, dim=-1) @ F.normalize(image, dim=-1).T / temperature
    positive = scores.diagonal()
    negative = scores + weights.masked_fill(weights.eq(0), 1).log()
    negative = negative.masked_fill(weights.eq(0), -torch.inf)
    denominator = torch.logsumexp(torch.cat((positive[:, None], negative), dim=-1), dim=-1)
    return (denominator - positive)[labels.eq(0)].mean()


class _DomainBatches(Sampler):
    """Visit every source row, cycling smaller domains to an equal batch budget."""

    def __init__(self, domains, num_domains, batch_size, seed):
        if type(batch_size) is not int or batch_size < num_domains or batch_size % num_domains:
            raise ValueError("RDCM batch_size must be positive and divisible by num_domains.")
        self.groups = [domains.eq(domain).nonzero().flatten() for domain in range(num_domains)]
        if any(not len(group) for group in self.groups):
            raise ValueError("RDCM requires every source domain to be occupied.")
        self.per_domain, self.seed, self.epoch = batch_size // num_domains, seed, 0

    def __len__(self):
        return math.ceil(max(map(len, self.groups)) / self.per_domain)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        budget = len(self) * self.per_domain
        indices = []
        for group in self.groups:
            permutations = []
            for _ in range(math.ceil(budget / len(group))):
                permutations.append(group[torch.randperm(len(group), generator=generator)])
            indices.append(torch.cat(permutations)[:budget])
        for start in range(0, budget, self.per_domain):
            yield torch.cat([group[start : start + self.per_domain] for group in indices]).tolist()


@register_model("RDCM")
class RDCM(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        self._validate(p)
        self.parameters_spec = deepcopy(p)
        self.domains = DomainTargets(**p["domains"])
        self.register_load_state_dict_post_hook(lambda *_: self._check_domains())
        self._sampler = None
        self._seed = torch.initial_seed() % 2**32
        self.text_encoder = material_encoder(config, "text_encoder", BertEncoder, **p["text"])
        # MLM checkpoints may synthesize an unused random pooler; omit its
        # weights so fixed word features retain the same cache identity by seed.
        if not hub_inputs(config):
            self.text_encoder.network.pooler = None
        self.instance_encoder = None
        if p["image"].get("backbone") == "vit" and hub_inputs(config):
            # Both immutable views were produced by the same ViT checkpoint.
            self.image_encoder = material_encoder(
                config, "image_encoder", supervised_encoder, **p["image"]
            )
            self.instance_encoder = material_encoder(
                config, "instance_encoder", supervised_encoder,
                **{**p["image"], "output": "logits"},
            )
            self.instance_head = nn.Identity()
        else:
            # Keep the actual ImageNet head, not a new 1000-way projection.
            vit = p["image"].get("backbone") == "vit"
            self.image_encoder = supervised_encoder(
                **{**p["image"], "output": "avgpool" if vit else "logits"}
            )
            self.instance_head = imagenet_classifier(self.image_encoder)
            self.image_encoder.output_dim = self.instance_head.in_features
            if vit:
                self.image_encoder.network.classifier = nn.Identity()
            else:
                self.image_encoder.network.fc = nn.Identity()
                self.image_encoder.encoding_spec["output"] = "avgpool"
            if hub_inputs(config):
                self.image_encoder = material_encoder(
                    config, "image_encoder", supervised_encoder, **p["image"]
                )
        dimension = p["hidden_dim"]
        self.convolutions = nn.ModuleList(
            nn.Conv1d(self.text_encoder.output_dim, p["filter_num"], kernel)
            for kernel in p["kernel_sizes"]
        )
        self.text_projection = nn.Sequential(
            nn.Dropout(p["dropout"]),
            nn.Linear(p["filter_num"] * len(p["kernel_sizes"]), p["text_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(p["text_hidden_dim"], dimension),
            nn.LayerNorm(dimension),
        )
        self.image_projection = nn.Sequential(
            nn.Linear(self.image_encoder.output_dim, p["image_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(p["image_hidden_dim"], dimension),
            nn.ReLU(),
            nn.Dropout(p["dropout"]),
            nn.LayerNorm(dimension),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * dimension, p["classifier_dim"]),
            nn.ReLU(),
            nn.Linear(p["classifier_dim"], 2),
        )

    @staticmethod
    def _validate(p):
        for key in (
            "hidden_dim",
            "text_hidden_dim",
            "image_hidden_dim",
            "classifier_dim",
            "filter_num",
        ):
            if type(p[key]) is not int or p[key] < 1:
                raise ValueError(f"RDCM {key} must be a positive integer.")
        kernels = p["kernel_sizes"]
        if (
            not isinstance(kernels, (list, tuple))
            or not kernels
            or any(type(kernel) is not int or kernel < 1 for kernel in kernels)
        ):
            raise ValueError("RDCM kernel_sizes must contain positive integers.")
        text, image = p["text"], p["image"]
        if (
            text.get("output") != "embeddings"
            or text.get("freeze") is not True
            or text.get("mask_padding") is not False
            or type(text.get("max_length")) is not int
            or text["max_length"] < max(kernels)
            or image.get("output") != "avgpool"
            or image.get("freeze") is not True
            or (
                image.get("backbone") != "vit"
                and (image.get("architecture") != "resnet50" or image.get("center_crop") is not True)
            )
            or image.get("stem_maxpool", True) is not True
            or p["domains"].get("max_labels") != 1
        ):
            raise ValueError(
                "RDCM requires frozen word/supervised image features and single source domains."
            )
        sigmas = p["kernel_sigmas"]
        if (
            not isinstance(sigmas, (list, tuple))
            or not sigmas
            or any(type(s) not in (int, float) or not math.isfinite(s) or s <= 0 for s in sigmas)
        ):
            raise ValueError("RDCM kernel_sigmas must contain finite positive bandwidths.")
        for key in ("dropout", "temperature", "threshold", "inter_weight", "contrastive_weight"):
            value = p[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"RDCM {key} must be finite.")
            if (
                (key == "dropout" and not 0 <= value < 1)
                or (key == "temperature" and value <= 0)
                or (key == "threshold" and not 0 < value <= 1)
                or (key.endswith("weight") and value < 0)
            ):
                raise ValueError(f"RDCM {key} is outside its valid range.")

    def prepare(self, train_dataset):
        self.domains.prepare(train_dataset)
        self._check_domains()

    def _check_domains(self):
        labels = self.domains.get_extra_state()["labels"]
        if not labels.sum(-1).eq(1).all():
            raise ValueError("RDCM requires exactly one domain per training row.")

    def training_stage_data(self, stage, train_data, val_data, options):
        labels = self.domains.targets([row["sample_id"] for row in train_data.dataset]).argmax(-1)
        self._sampler = _DomainBatches(
            labels, self.domains.num_domains, options["batch_size"], self._seed
        )
        train = DataLoader(
            train_data.dataset,
            batch_sampler=self._sampler,
            collate_fn=train_data.collate_fn,
            num_workers=train_data.num_workers,
            pin_memory=train_data.pin_memory,
            worker_init_fn=train_data.worker_init_fn,
            generator=train_data.generator,
        )
        if val_data is not None and val_data.batch_size != options["eval_batch_size"]:
            val_data = DataLoader(
                val_data.dataset,
                batch_size=options["eval_batch_size"],
                collate_fn=val_data.collate_fn,
                num_workers=val_data.num_workers,
                pin_memory=val_data.pin_memory,
                worker_init_fn=val_data.worker_init_fn,
                generator=val_data.generator,
                shuffle=False,
            )
        return train, val_data

    def on_epoch_start(self, epoch, total_epochs):
        self._sampler.epoch = epoch

    def features(self, batch):
        embeddings = self.text_encoder(batch["text"]).transpose(1, 2)
        pooled = torch.cat([conv(embeddings).relu().amax(-1) for conv in self.convolutions], -1)
        text = self.text_projection(pooled)
        visual = self.image_encoder(batch["image_path"])
        image = self.image_projection(visual)
        return text, image, visual

    def forward(self, batch):
        text, image, _visual = self.features(batch)
        return self.classifier(torch.cat((text, image), dim=-1))

    def calculate_loss(self, batch):
        p = self.parameters_spec
        text, image, visual = self.features(batch)
        labels = batch["labels"]
        if (
            labels.dtype != torch.long
            or labels.shape != (len(text),)
            or not labels.ge(0).all()
            or not labels.le(1).all()
        ):
            raise ValueError("RDCM labels must be a binary long batch vector.")
        domains = self.domains.targets(batch["sample_id"], text.device).argmax(-1)
        logits = self.classifier(torch.cat((text, image), dim=-1))
        return {
            "classification": F.cross_entropy(logits, labels),
            "inter_domain": p["inter_weight"]
            * _joint_discrepancy(text, image, domains, p["kernel_sigmas"]),
            "cross_modal": p["contrastive_weight"]
            * _cross_modal(
                text,
                image,
                self.instance_encoder(batch["image_path"])
                if self.instance_encoder is not None else self.instance_head(visual),
                labels,
                p["temperature"],
                p["threshold"],
            ),
        }

    def predict(self, batch):
        was_training = self.training
        self.eval()
        try:
            return self(batch)
        finally:
            self.train(was_training)

    def get_extra_state(self):
        return {"version": 1, "parameters": deepcopy(self.parameters_spec)}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state != self.get_extra_state():
            raise ValueError("RDCM checkpoint parameters differ from the configured model.")
