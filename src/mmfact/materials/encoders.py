"""Select immutable Hub inputs while leaving trainable method encoders intact."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from mmfact.data.cache_policy import PreparedInputError
from mmfact.models_classification._encoders import WordEncoder


def hub_inputs(config):
    return config.get("preprocessing", {}).get("layout") == "hub"


@lru_cache(maxsize=16)
def _store(root, dataset):
    from mmfact.materials.hub import HubMaterials

    return HubMaterials(root, dataset)


def material_store(config):
    options = config.get("preprocessing", {})
    root = options.get("materials_root")
    if not root:
        raise PreparedInputError("Hub inputs require preprocessing.materials_root.")
    return _store(str(Path(root).expanduser().resolve()), config["dataset"]["name"])


def material_mask(encoder, texts):
    """Use stored padding masks where available, retaining the native tokenizer rule."""
    if hasattr(encoder, "matrix"):
        return encoder.token_mask(texts)
    return encoder.tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=encoder.max_length,
        return_tensors="pt",
    )["attention_mask"]


class _Tokenizer:
    """Load only tokenizer files, on demand, from the prepared local HF cache."""

    def __init__(self, name, revision=None):
        self.name, self.revision, self._value = name, revision, None

    def _load(self):
        if self._value is None:
            from transformers import AutoTokenizer

            try:
                self._value = AutoTokenizer.from_pretrained(
                    self.name, revision=self.revision, local_files_only=True, use_fast=True
                )
            except OSError as error:
                raise PreparedInputError(
                    f"Tokenizer files for {self.name!r} are missing from the local HF cache. "
                    "Download the tokenizer before running this method."
                ) from error
        return self._value

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._load(), name)


def pretrained_name(config, name):
    """Honor a downloaded local model; placeholder READMEs do not shadow HF cache."""
    if hub_inputs(config):
        local = Path(config["preprocessing"]["materials_root"]) / "pretrained" / name
        if (local / "config.json").is_file():
            return str(local)
    return name


def _native(config, native_factory, args, options):
    options = dict(options)
    original = options.get("model_name")
    if original:
        options["model_name"] = pretrained_name(config, original)
    encoder = native_factory(*args, **options)
    if original and original != options.get("model_name") and hasattr(encoder, "encoding_spec"):
        encoder.encoding_spec["model_name"] = original
        if isinstance(encoder.encoding_spec.get("config"), dict):
            encoder.encoding_spec["config"]["_name_or_path"] = original
    return encoder


def _feature(config, role, options):
    """Select catalog aliases and the independently delivered region matrices."""
    from mmfact.materials.hub import FeatureMatrix
    from mmfact.materials.vision import vision_feature

    store = material_store(config)
    method = config["model"]["name"]
    selected = vision_feature(store.entries, method, role, options)
    if selected is not None:
        return FeatureMatrix(store.root, selected)
    if method == "NLIN":
        return store.feature_for("FND-CLIP", f"clip_{role}")
    special = {
        ("LIIMR", "image_encoder"): "features/regions/roi36",
        ("C3N", "text_encoder"): "features/c3n/text_nouns",
        ("C3N", "image_encoder"): "features/c3n/image_crops",
        ("Event-Radar", "text_encoder"): "features/text/event_radar_tokens",
        ("Event-Radar", "image_encoder"): "features/regions/event_radar_objects",
        ("Event-Radar", "dct_encoder"): "inputs/frequency/block_dct_168",
    }.get((method, role))
    if special:
        try:
            metadata = json.loads((store.root / special / "metadata.json").read_text("utf-8"))
        except (OSError, ValueError) as error:
            raise PreparedInputError(f"Missing or invalid prepared matrix: {special}") from error
        kind = "text" if metadata.get("id_field") == "text_id" else "image"
        settings = {key: value for key, value in options.items() if key != "freeze"}
        if method == "Event-Radar" and kind == "text" and len(metadata["shape"]) == 3:
            settings["max_length"] = metadata["shape"][1]
        return FeatureMatrix(
            store.root,
            {
                "name": special,
                "values": f"{special}/values.npy",
                "keys": f"{special}/keys.json",
                "id_field": metadata["id_field"],
                "input_kind": kind,
                "shape": metadata["shape"],
                "dtype": metadata["dtype"],
                "settings": settings,
                **({"mask": f"{special}/{metadata['mask']}"} if metadata.get("mask") else {}),
            },
        )
    return store.feature_for(method, role)


def material_encoder(config, role, native_factory, *args, **options):
    """Use a catalog binding only for the method's immutable encoder branch."""
    if not hub_inputs(config) or options.get("freeze") is False or options.get("augment"):
        return _native(config, native_factory, args, options)
    if isinstance(native_factory, type) and issubclass(native_factory, WordEncoder):
        return HubWordEncoder(config, role, **options)
    from mmfact.materials.hub import HubEncoder

    matrix = _feature(config, role, options)
    settings = matrix.settings
    for key in ("max_length", "model_name", "revision", "normalize", "mask_padding"):
        if key in options and key in settings and options[key] != settings[key]:
            raise PreparedInputError(f"Prepared {role} disagrees with configured {key!r}.")
    tokenizer = None
    if settings.get("model_name") and matrix.input_kind == "text":
        name = settings["model_name"]
        local = Path(config["preprocessing"]["materials_root"]) / "pretrained" / name
        has_tokenizer = any(
            (local / file).is_file() for file in ("tokenizer.json", "vocab.txt", "vocab.json")
        )
        tokenizer = _Tokenizer(str(local) if has_tokenizer else name, settings.get("revision"))
    if native_factory.__name__ == "_DescriptionEncoder":
        if settings.get("max_length", args[2]) != args[2]:
            raise PreparedInputError(
                "Prepared ERIC description token length disagrees with configuration."
            )
        return _Descriptions(matrix, args[1], args[2], args[3], tokenizer)
    encoder = HubEncoder(matrix, tokenizer=tokenizer)
    if native_factory.__name__ == "_ArgumentEncoder":
        encoder.inputs_from_rows = _pair_inputs
    if native_factory.__name__ == "_EventText":
        from types import MethodType

        encoder._cls_pooling = "chinese" in str(settings.get("model_name", "")).lower()
        encoder._tokenize = lambda texts: encoder.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=encoder.max_length,
            return_tensors="pt",
        )
        encoder.event_features = MethodType(native_factory.event_features, encoder)
    for key, value in options.items():
        if key not in {"freeze", "output_dim", "max_length", "sequence_length"}:
            setattr(encoder, key, value)
    for key in ("max_sentences", "body_length", "grid_size", "max_words", "layer_groups"):
        if key in settings:
            setattr(encoder, key, settings[key])
    # Sentence matrices and special pooled outputs keep their original shape contract.
    if role == "text_encoder" and config["model"]["name"] == "SpotFakePlus":
        encoder.max_sentences = matrix.shape[1]
    return encoder


def _pair_inputs(rows):
    return [
        hashlib.sha256(
            (
                f"{hashlib.sha256(row['text'].encode('utf-8')).hexdigest()}:"
                f"{Path(row['image_path']).stem}"
            ).encode()
        ).hexdigest()
        for row in rows
    ]


class _Descriptions(nn.Module):
    """Gather fixed entity descriptions while preserving empty/padded groups."""

    def __init__(self, matrix, lookup, max_length, max_entities, tokenizer):
        super().__init__()
        from mmfact.materials.hub import HubEncoder

        self.encoder = HubEncoder(matrix, tokenizer=tokenizer)
        self.lookup, self.max_length, self.max_entities = lookup, max_length, max_entities
        self.output_dim, self.freeze = self.encoder.output_dim, True
        self.preprocessing, self.cache_features = None, False

    def inputs_from_rows(self, rows):
        return [json.dumps(self.lookup(row), ensure_ascii=False) for row in rows]

    def forward(self, inputs):
        groups = [json.loads(value) for value in inputs]
        if any(
            not isinstance(group, list)
            or len(group) > self.max_entities
            or any(not isinstance(text, str) or not text.strip() for text in group)
            for group in groups
        ):
            raise PreparedInputError("Invalid prepared ERIC entity description groups.")
        output = self.encoder._device.new_zeros(len(groups), self.max_entities, self.output_dim)
        texts = [text for group in groups for text in group]
        if texts:
            positions = [
                (row, column) for row, group in enumerate(groups) for column in range(len(group))
            ]
            rows, columns = zip(*positions)
            output[list(rows), list(columns)] = self.encoder(texts)
        return output


class HubWordEncoder(WordEncoder):
    """Restore train-fitted vectors; token IDs remain fixed, embeddings may train."""

    def __init__(self, config, role, **options):
        super().__init__(**options)
        self._config, self._role = config, role
        store = material_store(config)
        method = config["model"]["name"]
        self._tokens = store.feature_for(method, role)
        resource = store.training_for(method, role)
        weights = torch.as_tensor(resource.array("embeddings").copy())
        vocabulary = {row["token"]: row["token_id"] for row in resource.jsonl("vocabulary")}
        if (
            weights.shape != self.embedding.weight.shape
            or vocabulary.get("<pad>") != self.pad_id
            or vocabulary.get("<unk>") != self.unk_id
            or set(vocabulary.values()) != set(range(len(vocabulary)))
            or len(vocabulary) > self.max_vocab_size
            or not torch.isfinite(weights).all()
        ):
            raise PreparedInputError("Hub Word2Vec vectors/vocabulary disagree with the model.")
        with torch.no_grad():
            self.embedding.weight.copy_(weights)
        self.vocabulary, self.ready = vocabulary, True

    def prepare(self, texts):
        # State was fitted on the frozen train split and loaded at construction.
        if not self.ready:
            raise PreparedInputError("Hub Word2Vec state is not ready.")

    def forward(self, texts):
        keys = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
        ids, _ = self._tokens.read(keys, device=self.embedding.weight.device)
        ids = ids.long()
        if ids.shape != (len(texts), self.max_length):
            raise PreparedInputError("Hub token ID length disagrees with the word encoder.")
        if ((ids < 0) | (ids >= len(self.vocabulary))).any():
            raise PreparedInputError("Hub token IDs are outside the fixed vocabulary.")
        return self.embedding(ids), ids, ids.ne(self.pad_id)
