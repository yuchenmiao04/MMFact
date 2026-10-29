"""CMC head-tail text inputs and common pretrained image peers."""

import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from mmfact.models_classification._encoders import _PretrainedEncoder
from mmfact.models_classification._supervised_vision import SupervisedViTEncoder


def _positive_int(value, name, minimum=1):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"CMC {name} must be an integer >= {minimum}.")


class _CMCEncoder(_PretrainedEncoder):
    def _set_frozen(self, freeze):
        if not isinstance(freeze, bool):
            raise ValueError("CMC freeze must be boolean.")
        if self.preprocessing is not None:
            self.preprocessing.reset()
        super()._set_frozen(freeze)

    def get_extra_state(self):
        return {
            "format_version": 1,
            "encoding_spec": deepcopy(self.encoding_spec),
            "freeze": self.freeze,
        }

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or set(state) != {"format_version", "encoding_spec", "freeze"}
            or state["format_version"] != 1
            or state["encoding_spec"] != self.encoding_spec
            or not isinstance(state["freeze"], bool)
        ):
            raise ValueError("CMC encoder checkpoint configuration differs.")
        self._set_frozen(state["freeze"])


class CMCTextEncoder(_CMCEncoder):
    """Common BERT: news pooler or ordered sentence means with head/tail truncation."""

    def __init__(
        self,
        model_name,
        language="zh",
        revision="main",
        max_length=None,
        max_sentences=None,
        freeze=False,
    ):
        super().__init__()
        if not isinstance(language, str) or language not in {"zh", "en"}:
            raise ValueError("CMC text language must be zh or en.")
        if not isinstance(model_name, (str, Path)) or not str(model_name).strip():
            raise ValueError("CMC text model_name must be a nonempty name or path.")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("CMC text revision must be a nonempty string.")
        if not isinstance(freeze, bool):
            raise ValueError("CMC freeze must be boolean.")
        max_length = (250 if language == "zh" else 120) if max_length is None else max_length
        max_sentences = (1 if language == "zh" else 5) if max_sentences is None else max_sentences
        _positive_int(max_length, "max_length", minimum=2)
        _positive_int(max_sentences, "max_sentences")
        if language == "zh" and max_sentences != 1:
            raise ValueError("CMC Chinese BERT uses exactly one news sequence.")

        from transformers import BertModel, BertTokenizerFast

        self.tokenizer = BertTokenizerFast.from_pretrained(model_name, revision=revision)
        self.network = BertModel.from_pretrained(model_name, revision=revision)
        if self.tokenizer.pad_token_id is None:
            raise ValueError("CMC text tokenizer requires a real padding token.")
        if max_length > self.network.config.max_position_embeddings:
            raise ValueError("CMC max_length exceeds the BERT position limit.")
        self.language, self.max_length = language, max_length
        self.max_sentences = max_sentences
        self.hidden_size = self.network.config.hidden_size
        self.output_dim = self.hidden_size * max_sentences
        vocabulary = json.dumps(self.tokenizer.get_vocab(), sort_keys=True).encode("utf-8")
        tokenization = json.loads(self.tokenizer.backend_tokenizer.to_str())
        self.encoding_spec = {
            "encoder": "cmc-head-tail-text",
            "language": language,
            "model_name": str(model_name),
            "revision": getattr(self.network.config, "_commit_hash", None) or revision,
            "config": self.network.config.to_dict(),
            "tokenizer": type(self.tokenizer).__name__,
            "vocabulary_sha256": hashlib.sha256(vocabulary).hexdigest(),
            "tokenization": tokenization,
            "special_tokens": self.tokenizer.special_tokens_map,
            "pad_id": self.tokenizer.pad_token_id,
            "max_length": max_length,
            "max_sentences": max_sentences,
            "truncation": "encoded-head-ceil-tail-floor-including-special-tokens",
            "padding": "fixed-right-real-pad-id-with-attention-mask",
            "sentence_split": "whole-news" if language == "zh" else "punctuation-newline-v1",
            "pooling": "bert-pooler" if language == "zh" else "masked-mean-including-special",
            "missing_sentences": "zero",
            "output": "ordered-flatten",
        }
        self._set_frozen(freeze)

    def _sentences(self, text):
        return [
            sentence.strip()
            for sentence in re.findall(r"[^.!?\n]+(?:[.!?]+|(?=\n|$))", text)
            if sentence.strip()
        ][: self.max_sentences]

    def _tokenize(self, texts):
        ids = torch.full(
            (len(texts), self.max_length), self.tokenizer.pad_token_id, dtype=torch.long
        )
        mask = torch.zeros_like(ids)
        for row, text in enumerate(texts):
            tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=False)
            if len(tokens) > self.max_length:
                tail = self.max_length // 2
                tokens = tokens[: self.max_length - tail] + tokens[-tail:]
            ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            mask[row, : len(tokens)] = 1
        return {"input_ids": ids.to(self.device), "attention_mask": mask.to(self.device)}

    def _encode(self, texts):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze):
            if self.language == "zh":
                return self.network(**self._tokenize(texts)).pooler_output
            sentences, positions = [], []
            for row, text in enumerate(texts):
                for slot, sentence in enumerate(self._sentences(text)):
                    sentences.append(sentence)
                    positions.append(row * self.max_sentences + slot)
            output = next(self.network.parameters()).new_zeros(
                len(texts) * self.max_sentences, self.hidden_size
            )
            if sentences:
                inputs = self._tokenize(sentences)
                hidden = self.network(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden)
                means = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                indices = torch.tensor(positions, dtype=torch.long, device=self.device)
                output = output.index_copy(0, indices, means)
            return output.reshape(len(texts), self.output_dim)


class _CMCResNetImageEncoder(_CMCEncoder):
    """Common pretrained ResNet50 avgpool, with online stage-one flip."""

    input_kind = "image"
    packages = ("torch", "torchvision", "Pillow")

    def __init__(
        self,
        architecture="resnet50",
        weights="IMAGENET1K_V1",
        image_size=None,
        freeze=False,
        flip_probability=0.5,
        center_crop=True,
        stem_maxpool=True,
    ):
        super().__init__()
        if architecture != "resnet50":
            raise ValueError("CMC common image peer requires resnet50.")
        if weights not in (None, "IMAGENET1K_V1"):
            raise ValueError("CMC weights must be IMAGENET1K_V1 or null.")
        if not isinstance(center_crop, bool) or not isinstance(stem_maxpool, bool):
            raise ValueError("CMC center_crop and stem_maxpool must be boolean.")
        if not isinstance(freeze, bool):
            raise ValueError("CMC freeze must be boolean.")
        image_size = [224, 224] if image_size is None else image_size
        if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
            raise ValueError("CMC image_size must be [height, width].")
        for value in image_size:
            _positive_int(value, "image_size")
        if (
            not isinstance(flip_probability, (float, int))
            or isinstance(flip_probability, bool)
            or not math.isfinite(flip_probability)
            or not 0 <= flip_probability <= 1
        ):
            raise ValueError("CMC flip_probability must be finite in [0, 1].")

        from torchvision import models, transforms

        selected = models.ResNet50_Weights[weights] if weights is not None else None
        network = models.resnet50(weights=selected)
        if not stem_maxpool:
            network.maxpool = nn.Identity()
        self.output_dim = network.fc.in_features
        self.network = nn.Sequential(*list(network.children())[:-1], nn.Flatten(1))
        native = (selected or models.ResNet50_Weights.IMAGENET1K_V1).transforms()
        if list(image_size) == [224, 224] and center_crop:
            self.transform = native
        else:
            operations = [
                transforms.Resize(
                    round(max(image_size) * 256 / 224) if center_crop else tuple(image_size),
                    interpolation=native.interpolation,
                )
            ]
            if center_crop:
                operations.append(transforms.CenterCrop(tuple(image_size)))
            operations.extend(
                [transforms.ToTensor(), transforms.Normalize(native.mean, native.std)]
            )
            self.transform = transforms.Compose(operations)
        self.image_size, self.flip_probability = tuple(image_size), float(flip_probability)
        self.augment = True
        self.encoding_spec = {
            "encoder": "cmc-image-peer",
            "architecture": architecture,
            "weights": weights,
            "stem_maxpool": stem_maxpool,
            "image_size": list(image_size),
            "transform": repr(self.transform),
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "output": "avgpool-flatten",
            "augmentation": "none-for-fixed-eval-features",
            "training_flip_probability": self.flip_probability,
            "small_image": "resize-all-valid-images",
        }
        self._set_frozen(freeze)

    def set_augmentation(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError("CMC augmentation must be boolean.")
        self.augment = enabled

    def _pixels(self, paths, augment=False):
        values = []
        for path in paths:
            with Image.open(path) as source:
                value = self.transform(source.convert("RGB"))
                if augment and torch.rand(()) < self.flip_probability:
                    value = value.flip(-1)
                values.append(value)
        return torch.stack(values).to(self.device)

    def _features(self, paths, augment):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze):
            return self.network(self._pixels(paths, augment))

    def _encode(self, paths):
        return self._features(paths, augment=False)

    def forward(self, paths):
        if self.training and self.augment:
            return self._features(paths, augment=True)
        return super().forward(paths)

    def get_extra_state(self):
        return {**super().get_extra_state(), "augmentation": self.augment}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or not isinstance(state.get("augmentation"), bool):
            raise ValueError("CMC image checkpoint augmentation must be boolean.")
        super().set_extra_state(
            {key: value for key, value in state.items() if key != "augmentation"}
        )
        self.set_augmentation(state["augmentation"])


class _CMCViTImageEncoder(SupervisedViTEncoder, _CMCEncoder):
    """The same mutual/fusion stage policy with the shared supervised ViT."""

    def __init__(self, *, image_size=None, flip_probability=0.5, freeze=False, **options):
        image_size = [224, 224] if image_size is None else image_size
        if list(image_size) != [224, 224]:
            raise ValueError("CMC supervised ViT uses the common 224x224 input recipe.")
        if (
            type(flip_probability) not in (int, float)
            or not math.isfinite(flip_probability)
            or not 0 <= flip_probability <= 1
        ):
            raise ValueError("CMC flip_probability must be finite in [0, 1].")
        super().__init__(freeze=freeze, **options)
        # CMC trains its own peer classifier from CLS. Keeping an unused frozen
        # ImageNet head would violate its strict all-peer stage/restore policy.
        self.network.classifier = nn.Identity()
        self.image_size = tuple(image_size)
        self.flip_probability = float(flip_probability)
        self.augment = True
        self.encoding_spec.update(
            encoder="cmc-supervised-vit-image-peer",
            augmentation="none-for-fixed-eval-features",
            training_flip_probability=self.flip_probability,
        )

    def _features(self, paths, augment):
        return self.encode_pixels(self._pixels(paths, augment))

    set_augmentation = _CMCResNetImageEncoder.set_augmentation
    _pixels = _CMCResNetImageEncoder._pixels
    _encode = _CMCResNetImageEncoder._encode

    def forward(self, paths):
        if self.training and self.augment:
            return self._features(paths, augment=True)
        return super().forward(paths)

    def get_extra_state(self):
        return {**_CMCEncoder.get_extra_state(self), "augmentation": self.augment}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or not isinstance(state.get("augmentation"), bool):
            raise ValueError("CMC image checkpoint augmentation must be boolean.")
        _CMCEncoder.set_extra_state(
            self, {key: value for key, value in state.items() if key != "augmentation"}
        )
        self.set_augmentation(state["augmentation"])


def CMCImageEncoder(*, backbone="resnet", **options):
    """Select the common visual backbone without changing CMC's training stages."""
    if backbone == "vit":
        return _CMCViTImageEncoder(backbone=backbone, **options)
    if backbone == "resnet":
        return _CMCResNetImageEncoder(**options)
    raise ValueError("CMC visual backbone must be vit or resnet.")
