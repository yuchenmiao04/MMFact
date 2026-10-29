"""att-RNN (MM2017): the paper's content-only visual-neuron attention variant."""

import hashlib
import json
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder
from mmfact.models_classification._encoders import WordEncoder
from mmfact.models_classification._supervised_vision import supervised_encoder
from mmfact.registry import register_model


class _Words(WordEncoder):
    """Keep shared Word2Vec caching, validating vocabulary on independent recovery."""

    def set_extra_state(self, state):
        vocabulary = state.get("vocabulary") if isinstance(state, dict) else None
        if (
            not isinstance(vocabulary, dict)
            or type(state.get("ready")) is not bool
            or not 2 <= len(vocabulary) <= self.max_vocab_size
            or any(
                not isinstance(key, str) or type(value) is not int
                for key, value in vocabulary.items()
            )
            or vocabulary.get("<pad>") != 0
            or vocabulary.get("<unk>") != 1
            or set(vocabulary.values()) != set(range(len(vocabulary)))
        ):
            raise ValueError("att-RNN checkpoint has an invalid word vocabulary.")
        super().set_extra_state(state)


@register_model("att-RNN")
class AttRNN(AbstractDetector):
    """Literal Eq10 scalar-per-word fusion; no social inputs or pooled substitute."""

    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        for name in ("hidden_dim", "visual_dim", "attention_hidden"):
            if type(p[name]) is not int or p[name] < 1:
                raise ValueError(f"att-RNN {name} must be a positive integer.")
        if p["image"].get("freeze") is not True or p["image"].get("output") != "avgpool":
            raise ValueError("att-RNN requires frozen global image features.")
        if type(p["text"].get("trainable")) is not bool:
            raise ValueError("att-RNN text.trainable must be boolean.")
        self.recipe = deepcopy(p)
        self.training_texts_sha256 = None
        self.text = material_encoder(config, "text", _Words, **p["text"])
        self.image = material_encoder(config, "image", supervised_encoder, **p["image"])
        hidden, visual = p["hidden_dim"], p["visual_dim"]
        self.lstm = nn.LSTM(self.text.dimension, hidden, batch_first=True)
        self.visual = nn.Sequential(
            nn.Linear(self.image.output_dim, visual), nn.ReLU(), nn.Linear(visual, visual)
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden, p["attention_hidden"]),
            nn.ReLU(),
            nn.Linear(p["attention_hidden"], visual),
        )
        self.classifier = nn.Linear(hidden + self.text.max_length, 2)
        self.register_load_state_dict_post_hook(self._validate_weights)

    @staticmethod
    def _validate_weights(model, _incompatible):
        weight = model.text.embedding.weight
        if not torch.isfinite(weight).all() or weight[0].count_nonzero():
            raise ValueError("att-RNN checkpoint has invalid Word2Vec vectors or PAD row.")
        if model.text.ready != (model.training_texts_sha256 is not None):
            raise ValueError("att-RNN checkpoint word readiness and training identity disagree.")

    def prepare(self, train_dataset):
        texts = [row["text"] for row in train_dataset]
        self.text.prepare(texts)
        self.training_texts_sha256 = hashlib.sha256(
            json.dumps(texts, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def forward(self, batch):
        texts, paths = batch["text"], batch["image_path"]
        if not texts or len(texts) != len(paths):
            raise ValueError("att-RNN requires one text and image per news item.")
        words, _, mask = self.text(texts)
        packed = pack_padded_sequence(
            words, mask.sum(1).cpu(), batch_first=True, enforce_sorted=False
        )
        hidden, _ = self.lstm(packed)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True, total_length=self.text.max_length)
        text = hidden.sum(1) / mask.sum(1, keepdim=True).to(hidden.dtype)
        visual = self.visual(self.image(paths)).softmax(-1)
        weights = self.attention(hidden).softmax(-1)
        affinity = (weights * visual[:, None, :]).sum(-1) * mask.to(hidden.dtype)
        return self.classifier(torch.cat((text, affinity), dim=-1))

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
            "training_texts_sha256": self.training_texts_sha256,
        }

    def set_extra_state(self, state):
        digest = state.get("training_texts_sha256") if isinstance(state, dict) else None
        if (
            not isinstance(state, dict)
            or state.get("format_version") != 1
            or state.get("recipe") != self.recipe
            or digest is not None
            and (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            )
        ):
            raise ValueError("att-RNN checkpoint has an incompatible recipe or training identity.")
        self.training_texts_sha256 = digest
