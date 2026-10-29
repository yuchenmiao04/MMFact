"""Fit text resources on training data; share fixed transforms across models."""

import hashlib
from importlib.metadata import version

import torch

from mmfact.data.cache_policy import PreparedInputError, input_mode, prepared_only
from mmfact.data.features import (
    FeatureCache,
    file_digest,
    load_resource,
    resource_path,
    state_digest,
)
from mmfact.models_classification._encoders import WordEncoder, _PretrainedEncoder


def _text_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _versions(*packages):
    return {package: version(package) for package in packages}


class _EncoderCache:
    """One input collection and preprocessing recipe, independent of the detector."""

    def __init__(self, options, inputs, keys):
        self.root = options["cache_dir"]
        self.batch_size = options["batch_size"]
        self.mode = input_mode(options.get("mode"))
        self.input_keys = dict(zip(inputs, keys))
        unique = dict(zip(keys, inputs))
        self.keys = sorted(unique)
        self.inputs = [unique[key] for key in self.keys]
        self.cache = None
        self.resource = None

    def reset(self):
        self.cache = None

    @staticmethod
    def _word_spec(encoder):
        return {
            "encoder": "word2vec",
            "tokenization": 1,
            "dtype": str(encoder.embedding.weight.dtype),
            **{
                key: getattr(encoder, key)
                for key in (
                    "dimension",
                    "max_length",
                    "max_vocab_size",
                    "language",
                    "window",
                    "epochs",
                    "seed",
                )
            },
            "versions": _versions("gensim", "jieba", "numpy"),
        }

    def prepare_words(self, encoder, texts):
        texts = list(texts)
        spec = {
            **self._word_spec(encoder),
            "training_texts": [_text_digest(text) for text in texts],
        }

        def fit():
            encoder._fit(texts)
            return {"weight": encoder.embedding.weight.detach().cpu(), **encoder.get_extra_state()}

        state = load_resource(self.root, spec, fit, mode=self.mode)
        weights, vocabulary = state["weight"], state["vocabulary"]
        if (
            tuple(weights.shape) != tuple(encoder.embedding.weight.shape)
            or not torch.isfinite(weights).all()
            or state["ready"] is not True
            or vocabulary.get("<pad>") != 0
            or vocabulary.get("<unk>") != 1
            or set(vocabulary.values()) != set(range(len(vocabulary)))
            or len(vocabulary) > encoder.max_vocab_size
        ):
            raise ValueError(
                "Invalid cached Word2Vec weights/vocabulary; delete its resource and rebuild."
            )
        with torch.no_grad():
            encoder.embedding.weight.copy_(state["weight"])
        encoder.set_extra_state(state)
        self.resource = str(resource_path(self.root, spec))

    def _build(self, spec, encode):
        if self.cache is None:
            cache = FeatureCache(self.root, spec, self.keys, mode=self.mode)

            def batches():
                for start in range(0, len(self.inputs), self.batch_size):
                    yield encode(self.inputs[start : start + self.batch_size])

            self.cache = cache.build(batches)
        return self.cache

    def word_ids(self, encoder, texts):
        if self.cache is None:
            spec = {
                **self._word_spec(encoder),
                "encoder": "word_ids",
                "vocabulary": encoder.vocabulary,
            }
            self._build(spec, encoder._token_ids)
        if any(text not in self.input_keys for text in texts):
            if prepared_only(self.mode):
                raise PreparedInputError("Prepared word IDs do not cover the requested text.")
            return encoder._token_ids(texts)
        return self.cache.read([self.input_keys[text] for text in texts]).to(
            encoder.embedding.weight.device
        )

    def features(self, encoder, inputs):
        if self.cache is None:
            spec = {
                **encoder.encoding_spec,
                "weights_sha256": state_digest(encoder.network),
                "versions": _versions(*encoder.packages),
            }
            self._build(spec, encoder._encode)
        if any(value not in self.input_keys for value in inputs):
            if prepared_only(self.mode):
                raise PreparedInputError("Prepared features do not cover the requested input.")
            return encoder._encode(inputs)
        return self.cache.read([self.input_keys[value] for value in inputs]).to(encoder.device)

    def describe(self):
        result = {}
        if self.resource is not None:
            result["word2vec"] = self.resource
        if self.cache is not None:
            result.update(
                path=str(self.cache.path), reused=self.cache.hit, shape=list(self.cache.shape)
            )
        return result


def configure_preprocessing(model, datasets, options):
    """Bind each encoder's permitted inputs without fitting or inspecting labels."""
    if options.get("layout") == "hub":
        _configure_hub(model, options)
        return
    encoders = [
        module
        for module in model.modules()
        if isinstance(module, (WordEncoder, _PretrainedEncoder))
    ]
    for encoder in encoders:
        encoder.preprocessing = None
    if not options["enabled"] or not encoders:
        return
    rows = [row for dataset in datasets.values() for row in dataset]
    image_hashes = {}
    for encoder in encoders:
        if isinstance(encoder, _PretrainedEncoder) and (
            not encoder.freeze or not encoder.cache_features
        ):
            continue
        inputs = (
            encoder.inputs_from_rows(rows)
            if isinstance(encoder, _PretrainedEncoder)
            else [row["text"] for row in rows]
        )
        variable = encoder.variable_inputs if isinstance(encoder, _PretrainedEncoder) else False
        if type(variable) is not bool:
            raise ValueError("Encoder variable_inputs must be boolean.")
        if (
            not isinstance(inputs, (list, tuple))
            or (not inputs if variable else len(inputs) != len(rows))
            or any(not isinstance(value, str) for value in inputs)
        ):
            raise ValueError(
                "Encoder inputs_from_rows must return one string per row, "
                "or a nonempty string collection with variable_inputs=True."
            )
        if isinstance(encoder, _PretrainedEncoder) and encoder.input_kind == "image":
            for path in dict.fromkeys(inputs):
                if path not in image_hashes:
                    image_hashes[path] = file_digest(path)
            keys = [image_hashes[path] for path in inputs]
        else:
            keys = [_text_digest(text) for text in inputs]
        encoder.preprocessing = _EncoderCache(options, inputs, keys)


class _HubCache:
    """Let native special encoders consume an existing catalog matrix in eval mode."""

    def __init__(self, matrix, root):
        from mmfact.materials.hub import HubEncoder

        self.reader, self.root = HubEncoder(matrix), root

    def features(self, encoder, inputs):
        return self.reader.to(encoder.device)(inputs)

    def reset(self):
        pass

    def describe(self):
        return {
            "path": str(self.reader.matrix.root / self.reader.matrix.entry["values"]),
            "reused": True,
        }


def _configure_hub(model, options):
    from mmfact.materials.encoders import HubWordEncoder, material_store
    from mmfact.materials.hub import FeatureMatrix, HubEncoder
    from mmfact.materials.vision import branch_options, vision_feature
    from mmfact.models_classification._supervised_vision import SupervisedViTEncoder

    store = material_store(model.config)
    name = model.config["model"]["name"]
    for role, encoder in model.named_modules():
        if isinstance(encoder, (HubEncoder, HubWordEncoder)):
            continue
        if not isinstance(encoder, _PretrainedEncoder):
            continue
        encoder.preprocessing = None
        if not encoder.freeze or not encoder.cache_features or getattr(encoder, "augment", False):
            continue
        if isinstance(encoder, SupervisedViTEncoder):
            branch = branch_options(model.config["model"]["parameters"], role)
            selected = vision_feature(store.entries, name, role, branch)
            if selected is not None:
                encoder.preprocessing = _HubCache(
                    FeatureMatrix(store.root, selected), options["materials_root"]
                )
            # A trained encoder may become frozen in a later stage. Its original
            # trainable/unknown role must not bind to initial or ResNet features.
            continue
        matches = [
            entry
            for entry in store.entries
            if any(
                consumer.get("method") == name and consumer.get("input") == role
                for consumer in entry.get("consumers", [])
            )
        ]
        if matches:
            encoder.preprocessing = _HubCache(
                store.feature_for(name, role), options["materials_root"]
            )


def preprocess(model, datasets, options):
    """Materialize all deterministic inputs, without an optimizer or classifier step."""
    if options.get("layout") == "hub":
        raise ValueError(
            "Hub materials are already prepared; use --dry-run to check the experiment."
        )
    if not options["enabled"]:
        raise ValueError("Preprocess mode requires preprocessing.enabled=true.")
    configure_preprocessing(model, datasets, options)
    model.prepare(datasets["train"])
    result = {}
    with torch.no_grad():
        for name, encoder in model.named_modules():
            context = getattr(encoder, "preprocessing", None)
            if context is None:
                continue
            if isinstance(encoder, WordEncoder):
                context.word_ids(encoder, context.inputs[:1])
            else:
                context.features(encoder, context.inputs[:1])
            result[name] = context.describe()
    return result
