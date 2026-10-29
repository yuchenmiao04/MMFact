"""NSLM-Binary: three variational deceptive patterns and product-rule supervision."""

import hashlib
import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.data.features import file_digest
from mmfact.materials.encoders import material_encoder
from mmfact.materials.resources import material_resources
from mmfact.models_classification._encoders import BertEncoder, _PretrainedEncoder
from mmfact.models_classification._supervised_vision import supervised_encoder
from mmfact.models_classification._nslm_resources import ReverseImageEvidence
from mmfact.models_classification.mcan import DCTEncoder
from mmfact.registry import register_model


class _MeanText(BertEncoder):
    def __init__(self, source=None, lookup=None, max_length=45, **options):
        if source is None:
            super().__init__(max_length=max_length, **options)
            self.network.pooler = None
        else:
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
            self.max_length = self.sequence_length = max_length
            self.encoding_spec = deepcopy(source.encoding_spec)
            self.encoding_spec["max_length"] = max_length
            source.register_load_state_dict_post_hook(self._clear_shared_cache)
            self.train(self.training)
        self.lookup = lookup
        self.encoding_spec["pooling"] = "last-layer-mask-mean-including-special-tokens"

    def _clear_shared_cache(self, _owner, incompatible_keys):
        self._clear_cache(self, incompatible_keys)

    def inputs_from_rows(self, rows):
        return (
            self.lookup([row["image_path"] for row in rows])
            if self.lookup
            else super().inputs_from_rows(rows)
        )

    def _encode(self, texts):
        tokens, mask = super()._encode(texts), self.token_mask(texts)
        return tokens.sum(1) / mask.sum(1, keepdim=True)


class _FrequencyInception(nn.Module):
    def __init__(self, size):
        super().__init__()
        from torchvision.models import inception_v3

        self.size = size
        self.network = inception_v3(weights=None, aux_logits=False, init_weights=False)

    def forward(self, spectrum):
        pixels = F.interpolate(
            spectrum[:, None], size=(self.size, self.size), mode="bilinear", align_corners=False
        )
        return self.network(pixels.expand(-1, 3, -1, -1))


def _logic(posterior):
    real = posterior[..., 0].prod(-1)
    return torch.stack((real, 1 - real), dim=-1)


def _categorical_kl(left, right):
    # xlogy handles exact zero targets while retaining the explicit KL direction.
    tiny = torch.finfo(left.dtype).tiny
    return (torch.xlogy(left, left.clamp_min(tiny)) - left * right.clamp_min(tiny).log()).sum(-1)


@register_model("NSLM")
class NSLM(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        for key in (
            "feature_dim",
            "latent_dim",
            "context_length",
            "inference_steps",
            "frequency_size",
        ):
            if type(p[key]) is not int or p[key] < (2 if key == "context_length" else 1):
                raise ValueError(f"NSLM {key} must be a positive integer of sufficient size.")
        if p["frequency_size"] < 75:
            raise ValueError("NSLM Inception frequency_size must be at least 75.")
        for key in ("dropout", "logic_weight", "temperature"):
            value = p[key]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or (
                    not 0 <= value < 1
                    if key == "dropout"
                    else not 0 < value < 1
                    if key == "logic_weight"
                    else value <= 0
                )
            ):
                raise ValueError(f"NSLM {key} is outside its finite permitted range.")
        if type(p["inference_seed"]) is not int or not 0 <= p["inference_seed"] < 2**32:
            raise ValueError("NSLM inference_seed must be an integer in [0,2**32).")
        if (
            p["text"].get("output") != "tokens"
            or p["text"].get("mask_padding") is not True
            or (
                p["image"].get("backbone") != "vit"
                and p["image"].get("architecture") != "resnet50"
            )
            or p["image"].get("output") != "avgpool"
        ):
            raise ValueError("NSLM requires masked contextual text and supervised global image semantics.")
        self.parameters_spec = deepcopy(p)
        self.evidence = material_resources(config, "nslm", ReverseImageEvidence, **p["resources"])
        self.text_encoder = material_encoder(config, "text_encoder", _MeanText, **p["text"])
        if p["context_length"] > self.text_encoder.network.config.max_position_embeddings:
            raise ValueError("NSLM context_length exceeds text backbone position capacity.")
        self.context_encoder = _MeanText(
            self.text_encoder, self.evidence.contexts, p["context_length"]
        )
        self.image_encoder = material_encoder(config, "image_encoder", supervised_encoder, **p["image"])
        self.frequency_encoder = material_encoder(
            config, "frequency_encoder", DCTEncoder, **p["frequency"]
        )
        self.frequency_inception = _FrequencyInception(p["frequency_size"])
        d, h = p["feature_dim"], p["latent_dim"]
        self.text_projection = nn.Linear(self.text_encoder.output_dim, d)
        self.context_projection = nn.Linear(self.text_encoder.output_dim, d)
        self.image_projection = nn.Linear(self.image_encoder.output_dim, d)
        self.frequency_projection = nn.Linear(1000, d)
        self.ci, self.ir = nn.Bilinear(d, d, d), nn.Bilinear(d, d, d)
        self.posteriors = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d + h, h),
                nn.Tanh(),
                nn.Dropout(p["dropout"]),
                nn.Linear(h, 2, bias=False),
            )
            for _ in range(3)
        )
        self.decoder = nn.Sequential(
            nn.Dropout(p["dropout"]), nn.Linear(2 * d + 6, h), nn.Tanh(), nn.Dropout(p["dropout"])
        )
        self.classifier = nn.Linear(h, 2, bias=False)

    def _features(self, batch):
        paths, texts = batch["image_path"], batch["text"]
        if not texts or len(paths) != len(texts):
            raise ValueError("NSLM needs a nonempty aligned text/image batch.")
        contexts = self.evidence.contexts(paths)
        text = self.text_projection(self.text_encoder(texts))
        image = self.image_projection(self.image_encoder(paths))
        observed = torch.tensor([bool(context) for context in contexts], device=text.device)
        context = text.new_zeros(len(text), self.context_projection.out_features)
        indices = observed.nonzero().flatten()
        if len(indices):
            context[indices] = self.context_projection(
                self.context_encoder([contexts[i] for i in indices.tolist()])
            )
        manipulation = self.frequency_projection(
            self.frequency_inception(self.frequency_encoder(paths))
        )
        patterns = torch.stack((manipulation, self.ci(image, text), self.ir(text, context)), dim=1)
        mask = torch.ones(len(text), 3, dtype=torch.bool, device=text.device)
        mask[:, 2] = observed
        return torch.cat((text, image), -1), F.normalize(patterns, dim=-1), mask

    def _posterior(self, patterns, label_probabilities, mask):
        embedding = label_probabilities @ self.classifier.weight
        logits = torch.stack(
            [
                network(torch.cat((patterns[:, i], embedding), -1))
                for i, network in enumerate(self.posteriors)
            ],
            dim=1,
        )
        probabilities = logits.softmax(-1)
        return logits, torch.where(
            mask[..., None], probabilities, torch.full_like(probabilities, 0.5)
        )

    def _decode(self, global_features, latent):
        return self.classifier(self.decoder(torch.cat((global_features, latent.flatten(1)), -1)))

    def calculate_loss(self, batch):
        p = self.parameters_spec
        features, patterns, mask = self._features(batch)
        labels = batch["labels"]
        if (
            labels.dtype != torch.long
            or labels.shape != (len(features),)
            or not ((labels == 0) | (labels == 1)).all()
        ):
            raise ValueError("NSLM labels must be a binary long batch vector.")
        logits, posterior = self._posterior(patterns, F.one_hot(labels, 2).to(features), mask)
        latent = F.gumbel_softmax(logits, tau=p["temperature"], hard=True, dim=-1)
        latent = torch.where(mask[..., None], latent, torch.full_like(latent, 0.5))
        prediction = self._decode(features, latent)
        prior = torch.randn_like(posterior).softmax(-1)
        mu = p["logic_weight"]
        return {
            "reconstruction": (1 - mu) * F.cross_entropy(prediction, labels),
            "latent_kl": (1 - mu) * (_categorical_kl(posterior, prior) * mask).sum(-1).mean(),
            "logic": mu * _categorical_kl(prediction.softmax(-1), _logic(posterior)).mean(),
        }

    def forward(self, batch):
        features, patterns, mask = self._features(batch)
        p = self.parameters_spec
        initial = []
        for text, path in zip(batch["text"], batch["image_path"], strict=True):
            identity = f"{p['inference_seed']}:{text}:{file_digest(path)}"
            seed = int(hashlib.sha256(identity.encode()).hexdigest()[:15], 16)
            initial.append(torch.randn(3, 2, generator=torch.Generator().manual_seed(seed)))
        posterior = torch.stack(initial).to(features).softmax(-1)
        for _ in range(p["inference_steps"]):
            latent = F.one_hot(posterior.argmax(-1), 2).to(features)
            latent = torch.where(mask[..., None], latent, torch.full_like(latent, 0.5))
            prediction = self._decode(features, latent)
            _logits, posterior = self._posterior(patterns, prediction.softmax(-1), mask)
        return prediction

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
        if state != self.get_extra_state():
            raise ValueError("NSLM checkpoint parameters differ from this configuration.")
