"""LogicDM: dependency/patch graph, sparse predicates and product-t-norm clauses."""

import hashlib
import json
import math
from copy import deepcopy
from importlib.metadata import version

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder
from mmfact.models_classification._encoders import (
    BertEncoder,
    ResNetEncoder,
    RGBEncoder,
    _PretrainedEncoder,
)
from mmfact.registry import register_model
from mmfact.models_classification._supervised_vision import ViTIndependentPatches


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _sparsemax(values, mask=None):
    """Euclidean simplex projection on the final axis; empty masked rows are zero."""
    if mask is None:
        mask = torch.ones_like(values, dtype=torch.bool)
    valid = mask.any(-1, keepdim=True)
    values = values.masked_fill(~mask, -torch.inf)
    shifted = torch.where(valid, values - values.max(-1, keepdim=True).values, 0.0)
    ordered = shifted.sort(-1, descending=True).values
    ranks = torch.arange(1, values.shape[-1] + 1, device=values.device)
    cumulative = ordered.cumsum(-1)
    support = (1 + ranks * ordered > cumulative).sum(-1, keepdim=True).clamp_min(1)
    threshold = (cumulative.gather(-1, support - 1) - 1) / support
    return (shifted - threshold).clamp_min(0).masked_fill(~mask | ~valid, 0)


def _syntax_digest(resource):
    payload = {key: value for key, value in resource.items() if key not in {"bytes", "sha256"}}
    return _sha(resource["bytes"] + json.dumps(payload, sort_keys=True).encode("utf-8"))


class _SyntaxEncoder(_PretrainedEncoder):
    """Real spaCy parser, with complete fixed resources embedded in the checkpoint."""

    packages = ("spacy", "thinc", "numpy", "torch")

    def __init__(self, model_name, language, model_version, max_words):
        super().__init__()
        self.options = dict(
            model_name=str(model_name),
            language=language,
            model_version=model_version,
            max_words=max_words,
        )
        self.network = nn.Identity()
        self._set_frozen(True)
        self.parser = None
        self.resource = None
        self.encoding_spec = {"encoder": "logicdm-spacy-spans-heads-v1", **self.options}

    def _validate_parser(self, parser):
        if (
            parser.lang != self.options["language"]
            or parser.meta.get("version") != self.options["model_version"]
            or "parser" not in parser.pipe_names
        ):
            raise ValueError(
                "LogicDM needs the configured language/version and a real spaCy parser."
            )

    def _ensure_parser(self):
        if self.parser is not None:
            return
        import spacy

        parser = spacy.load(
            self.options["model_name"],
            exclude=["ner", "tagger", "lemmatizer", "attribute_ruler", "senter"],
        )
        self._validate_parser(parser)
        self.parser = parser
        payload = parser.to_bytes()
        self.resource = {
            "bytes": payload,
            "config": parser.config.to_str(),
            "meta": deepcopy(parser.meta),
            "runtime_versions": {
                package: version(package)
                for package in (
                    "spacy",
                    "thinc",
                    "numpy",
                    *(
                        ("spacy-pkuseg",)
                        if getattr(parser.tokenizer, "segmenter", None) == "pkuseg"
                        else ()
                    ),
                )
            },
        }
        self.resource["sha256"] = _syntax_digest(self.resource)
        self._update_spec()

    def _update_spec(self):
        self.encoding_spec.update(
            resource_sha256=self.resource["sha256"],
            parser_meta=deepcopy(self.resource["meta"]),
            parser_config=self.resource["config"],
            runtime_versions=deepcopy(self.resource["runtime_versions"]),
            output="nonspace-char-start-end-head; -1-padding; whole-text-parse",
        )

    def inputs_from_rows(self, rows):
        self._ensure_parser()
        return super().inputs_from_rows(rows)

    def _encode(self, texts):
        self._ensure_parser()
        results = []
        for document in self.parser.pipe(texts, batch_size=32):
            tokens = [token for token in document if not token.is_space]
            indices = {token.i: index for index, token in enumerate(tokens)}
            rows = torch.full((self.options["max_words"], 3), -1, dtype=torch.long)
            for index, token in enumerate(tokens[: len(rows)]):
                rows[index] = torch.tensor(
                    [token.idx, token.idx + len(token), indices.get(token.head.i, -1)]
                )
            results.append(rows)
        return torch.stack(results).to(self.device)

    def get_extra_state(self):
        return {"options": self.options, "resource": self.resource}

    def set_extra_state(self, state):
        if state["options"] != self.options:
            raise ValueError("LogicDM syntax configuration does not match the checkpoint.")
        resource = deepcopy(state["resource"])
        if resource is None:
            self.parser, self.resource = None, None
            return
        if (
            not isinstance(resource["bytes"], bytes)
            or _syntax_digest(resource) != resource["sha256"]
            or resource["runtime_versions"]
            != {package: version(package) for package in resource["runtime_versions"]}
        ):
            raise ValueError("LogicDM parser content/version differs from the checkpoint.")
        import spacy
        from confection import Config

        config = Config().from_str(resource["config"])
        parser = spacy.util.load_model_from_config(config, auto_fill=True)
        parser.from_bytes(resource["bytes"])
        self._validate_parser(parser)
        if parser.meta != resource["meta"]:
            raise ValueError("LogicDM parser metadata differs from its saved resource.")
        self.parser, self.resource = parser, resource
        self._update_spec()


class _PatchEncoder(ResNetEncoder):
    """Forty-nine independently encoded 32x32 patches, not pooled feature-map cells."""

    def __init__(self, weights, image_size, grid_size, patch_batch_size, architecture="resnet50"):
        super().__init__(architecture=architecture, weights=weights, freeze=True, output="avgpool")
        self.pixels = RGBEncoder(image_size=image_size, minimum_size=0, normalize=True)
        self.pixels.cache_features = False
        self.grid_size, self.patch_batch_size = grid_size, patch_batch_size
        self.encoding_spec = {
            "encoder": "logicdm-resnet-independent-patches",
            "architecture": architecture,
            "weights": weights,
            "pixels": self.pixels.encoding_spec,
            "grid_size": grid_size,
            "patch_order": "row-major",
            "patch_batch_size": patch_batch_size,
            "output": "avgpool",
        }

    def _encode(self, paths):
        pixels = self.pixels(paths)
        size = pixels.shape[-1] // self.grid_size
        results = []
        for image in pixels:
            patches = image.unfold(1, size, size).unfold(2, size, size)
            patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, 3, size, size)
            with torch.no_grad():
                features = [
                    self.network(patches[start : start + self.patch_batch_size])
                    for start in range(0, len(patches), self.patch_batch_size)
                ]
            results.append(torch.cat(features))
        return torch.stack(results)


def _patch_encoder(**options):
    if options.get("backbone") == "vit":
        return ViTIndependentPatches(resize="bilinear", **options)
    return _PatchEncoder(**options)


def _word_encoding(tokenizer, texts, syntax, max_length):
    """Tokenize parser words independently; word_ids exactly identify complete nodes."""
    groups = [
        [text[start:end] for start, end, _ in spans.tolist() if start >= 0]
        for text, spans in zip(texts, syntax.cpu())
    ]
    encoded = tokenizer(
        groups,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    mappings, heads = [], []
    for item, words in enumerate(groups):
        # A truncated last word may retain some pieces; it must not become a full node.
        counts = tokenizer(words, add_special_tokens=False, truncation=False)["input_ids"]
        word_ids = encoded.word_ids(item)
        selected, original, token_heads = [], [], []
        for index, pieces in enumerate(counts):
            positions = [position for position, word_id in enumerate(word_ids) if word_id == index]
            if positions and len(positions) == len(pieces):
                selected.append(torch.tensor(positions, dtype=torch.long))
                original.append(index)
                token_heads.append(int(syntax[item, index, 2]))
        if not selected:
            raise ValueError("LogicDM text has no complete nonspace word after tokenization.")
        remap = {index: position for position, index in enumerate(original)}
        mappings.append(selected)
        heads.append([remap.get(head, -1) for head in token_heads])
    return encoded, mappings, heads


def _align_words(tokenizer, texts, syntax, max_length):
    _, mappings, heads = _word_encoding(tokenizer, texts, syntax, max_length)
    return mappings, heads


class _WordBertEncoder(BertEncoder):
    """Frozen BERT consumes the same parser word list used by word_ids and the graph."""

    def __init__(self, syntax_encoder, **options):
        super().__init__(**options)
        # The detector registers this shared parser once; this is a non-owning reference.
        object.__setattr__(self, "syntax", syntax_encoder)
        self.encoding_spec.update(
            encoder="logicdm-word-aligned-bert-v1",
            tokenization="is_split_into_words",
            word_pooling="complete-word-subtoken-mean; partial-last-word-dropped",
        )

    def _update_spec(self):
        self.syntax._ensure_parser()
        self.encoding_spec["syntax_recipe"] = deepcopy(self.syntax.encoding_spec)

    def inputs_from_rows(self, rows):
        self._update_spec()
        return super().inputs_from_rows(rows)

    def _tokenize(self, texts):
        self._update_spec()
        encoded, _, _ = _word_encoding(
            self.tokenizer,
            texts,
            self.syntax(texts),
            self.max_length,
        )
        return encoded


def _adjacency(heads, lengths, grid_size, maximum, device, dtype):
    patches = grid_size**2
    # Binary, non-learned structure: assemble it on CPU and transfer once, rather
    # than launching a CUDA assignment for each dependency edge.
    result = torch.zeros(len(heads), maximum + patches, maximum + patches, dtype=torch.bool)
    coordinates = torch.arange(patches)
    row, column = coordinates // grid_size, coordinates % grid_size
    visual = ((row[:, None] - row).abs() <= 1) & ((column[:, None] - column).abs() <= 1)
    result[:, maximum:, maximum:] = visual
    positions = torch.arange(maximum)
    valid = positions[None] < torch.tensor(lengths)[:, None]
    result[:, positions, positions] = valid
    result[:, :maximum, maximum:] = valid[..., None]
    result[:, maximum:, :maximum] = valid[:, None]
    edges = []
    for item, (dependencies, length) in enumerate(zip(heads, lengths)):
        for index, head in enumerate(dependencies):
            if head >= 0:
                edges.append((item, index, head))
    if edges:
        item, node, head = torch.tensor(edges, dtype=torch.long).T
        result[item, node, head] = True
        result[item, head, node] = True
    return result.to(device=device, dtype=dtype)


def _pool_word_tokens(tokens, memberships):
    """The same complete-word subtoken means, without per-word CUDA calls."""
    membership = pad_sequence(memberships, batch_first=True).to(device=tokens.device, dtype=tokens.dtype)
    counts = membership.sum(-1, keepdim=True).clamp_min(1)
    return torch.bmm(membership, tokens) / counts


class _GraphConvolution(nn.Module):
    """Author graph layer; retain degree+1, bias and post-ReLU LayerNorm."""

    def __init__(self, dimension, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dimension, dimension))
        self.bias = nn.Parameter(torch.empty(dimension))
        self.norm = nn.LayerNorm(dimension, eps=eps)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.bias[:, None])

    def forward(self, features, adjacency):
        valid = adjacency.any(-1)
        values = adjacency @ (features @ self.weight) / (adjacency.sum(-1, keepdim=True) + 1)
        values = self.norm(F.relu(values + self.bias))
        return values.masked_fill(~valid[..., None], 0)


class _Objects(nn.Module):
    """Exact top-k over each full family, with selected-value gradient correction."""

    def __init__(self, dimension, top_k, chunk_size):
        super().__init__()
        self.top_k, self.chunk_size = top_k, chunk_size
        self.projections = nn.ModuleList(
            nn.Sequential(nn.Linear(dimension, dimension), nn.ReLU())
            if index < 2
            else nn.Sequential(
                nn.Linear(4 * dimension, 2 * dimension),
                nn.ReLU(),
                nn.Linear(2 * dimension, dimension),
                nn.ReLU(),
            )
            for index in range(5)
        )
        self.importance = nn.ModuleList(nn.Linear(dimension, 1) for _ in range(5))
        # ReLU scoring with a positive bias avoids an initially dead entire family.
        for score in self.importance:
            nn.init.constant_(score.bias, 1.0)

    @staticmethod
    def _pairs(left, right):
        return torch.cat((left, right, left * right, left - right), -1)

    def _select(self, left, right, family):
        count = len(left) if right is None else len(left) * len(right)
        projection, importance = self.projections[family], self.importance[family]

        def objects(indices):
            if right is None:
                values = left[indices]
            else:
                values = self._pairs(left[indices // len(right)], right[indices % len(right)])
            return projection(values)

        # Selection is discrete. Recomputing only selected values keeps exact gradients
        # without retaining a quadratic all-pair feature graph.
        best_scores = left.new_empty(0)
        best_indices = torch.empty(0, device=left.device, dtype=torch.long)
        with torch.no_grad():
            for start in range(0, count, self.chunk_size):
                indices = torch.arange(
                    start, min(start + self.chunk_size, count), device=left.device
                )
                if family in {2, 3}:
                    indices = indices[indices // len(right) != indices % len(right)]
                scores = F.relu(importance(objects(indices))).squeeze(-1)
                scores, indices = (
                    torch.cat((best_scores, scores)),
                    torch.cat((best_indices, indices)),
                )
                # Stable index order resolves equal scores independent of chunk size.
                positions = scores.argsort(descending=True, stable=True)[: self.top_k]
                best_scores, best_indices = scores[positions], indices[positions]
        selected = objects(best_indices)
        scores = F.relu(importance(selected))
        selected = selected * scores
        valid = torch.arange(self.top_k, device=left.device) < len(selected)
        return F.pad(selected, (0, 0, 0, self.top_k - len(selected))), valid

    def _batch_select(self, left, right, left_mask, right_mask, family):
        """Scan the same lexicographic candidates, batching independent samples.

        Only the discrete scan is no-grad; selected values and importance scores
        are recomputed exactly as in _select. Invalid padding never enters top-k.
        """
        batch, length, dimension = left.shape
        count = length if right is None else length * right.shape[1]
        if not count:
            raise ValueError("LogicDM requires nonempty stored text/image dimensions.")
        projection, importance = self.projections[family], self.importance[family]

        def objects(indices):
            # indices has shape [batch, candidate_count].
            if right is None:
                return projection(left.gather(1, indices[..., None].expand(-1, -1, dimension)))
            width = right.shape[1]
            first = left.gather(1, (indices // width)[..., None].expand(-1, -1, dimension))
            second = right.gather(1, (indices % width)[..., None].expand(-1, -1, dimension))
            return projection(self._pairs(first, second))

        best_scores = left.new_empty(batch, 0)
        best_indices = torch.empty(batch, 0, device=left.device, dtype=torch.long)
        with torch.no_grad():
            for start in range(0, count, self.chunk_size):
                indices = torch.arange(start, min(start + self.chunk_size, count), device=left.device)
                if right is None:
                    valid = left_mask[:, indices]
                else:
                    width = right.shape[1]
                    valid = left_mask[:, indices // width] & right_mask[:, indices % width]
                    if family in {2, 3}:
                        valid = valid & (indices // width != indices % width)[None]
                indices = indices[None].expand(batch, -1)
                scores = F.relu(importance(objects(indices))).squeeze(-1).masked_fill(~valid, -torch.inf)
                scores = torch.cat((best_scores, scores), 1)
                candidates = torch.cat((best_indices, indices), 1)
                positions = scores.argsort(dim=1, descending=True, stable=True)[:, : self.top_k]
                best_scores = scores.gather(1, positions)
                best_indices = candidates.gather(1, positions)
        # Do not use isfinite: a genuinely overflowing valid score must propagate
        # to the existing loss/gradient checks, never be silently discarded.
        valid = best_scores != -torch.inf
        if best_indices.shape[1] < self.top_k:
            missing = self.top_k - best_indices.shape[1]
            best_indices = F.pad(best_indices, (0, missing))
            valid = F.pad(valid, (0, missing), value=False)
        selected = objects(best_indices)
        selected = selected * F.relu(importance(selected))
        return selected.masked_fill(~valid[..., None], 0), valid

    def forward(self, text, image, mask):
        if text.shape[0] != image.shape[0] or mask.shape != text.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("LogicDM object inputs need aligned batches and boolean word masks.")
        patches = torch.ones(image.shape[:2], device=image.device, dtype=torch.bool)
        groups = ((text, None, mask, None), (image, None, patches, None),
                  (text, text, mask, mask), (image, image, patches, patches),
                  (text, image, mask, patches))
        selected = [self._batch_select(left, right, lm, rm, family)
                    for family, (left, right, lm, rm) in enumerate(groups)]
        return torch.stack([value for value, _ in selected], 1), torch.stack([valid for _, valid in selected], 1)


class _AuthorClause(nn.Module):
    """Author predicate/guide architecture with explicit axis and gradient corrections."""

    def __init__(
        self,
        dimension,
        top_k,
        correlations,
        rate,
        heads=5,
        guide_heads=5,
        threshold=0.4,
        dropout=0.2,
    ):
        super().__init__()
        self.atom_count = math.floor(5 * top_k * rate)
        self.threshold = threshold
        self.labels = nn.Parameter(torch.empty(2, dimension))
        self.correlations = nn.Parameter(torch.empty(5, correlations, dimension))
        self.instance_key = nn.Linear(dimension, dimension)
        self.label_key = nn.Linear(dimension, dimension)
        self.correlation_key = nn.Linear(dimension, dimension)
        self.correlation_value = nn.Linear(dimension, dimension)
        self.binding = nn.Sequential(
            nn.Linear(4 * dimension, 2 * dimension),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(2 * dimension, dimension),
            nn.ReLU(),
        )
        self.answer_instance = nn.Linear(dimension, dimension)
        self.answer_label = nn.Linear(dimension, dimension)
        self.answer_score = nn.Sequential(
            nn.Linear(4 * dimension, 2 * dimension),
            nn.ReLU(),
            nn.Linear(2 * dimension, 1),
        )
        self.text_score, self.image_score = (nn.Linear(dimension, 1) for _ in range(2))
        self.mco = nn.TransformerEncoderLayer(
            dimension,
            heads,
            2 * dimension,
            dropout,
            batch_first=True,
        )
        self.guides = nn.ModuleList(
            nn.ModuleList((nn.Linear(dimension, 1), nn.Linear(dimension, 1)))
            for _ in range(guide_heads)
        )
        self.truth = nn.Sequential(
            nn.Linear(4 * dimension, 2 * dimension),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(2 * dimension, 1),
            nn.Sigmoid(),
        )
        nn.init.xavier_uniform_(self.labels)
        for correlations in self.correlations:
            nn.init.xavier_uniform_(correlations)

    @staticmethod
    def _comparison(left, right):
        return torch.cat((left, right, left - right, left * right), -1)

    @staticmethod
    def _pool(values, score, mask):
        weights = score(values).squeeze(-1).masked_fill(~mask, -torch.inf).softmax(-1)
        return (values * weights.unsqueeze(-1)).sum(-2)

    def _predicates(self, objects):
        batch, _, top_k, dimension = objects.shape
        labels = self.label_key(self.labels)[None, None].expand(batch, top_k, -1, -1)
        predicates = []
        for family in range(5):
            instances = self.instance_key(objects[:, family])[:, :, None].expand(-1, -1, 2, -1)
            correlations = self.correlations[family]
            key = self.binding(_Objects._pairs(instances, labels))
            scores = key @ self.correlation_key(correlations).T
            predicates.append(self.correlation_value(_sparsemax(scores) @ correlations))
        return torch.cat(predicates, 1)

    def _mix(self, predicates, valid):
        batch, count, labels, dimension = predicates.shape
        values = predicates.permute(0, 2, 1, 3).reshape(batch * labels, count, dimension)
        mask = valid[:, None].expand(-1, labels, -1).reshape(batch * labels, count)
        values = values.masked_fill(~mask[..., None], 0)
        # An entirely empty private-test row must not softmax over only masked keys.
        safe_mask = mask.clone()
        safe_mask[~mask.any(-1), 0] = True
        values = self.mco(values, src_key_padding_mask=~safe_mask)
        values = values.masked_fill(~mask[..., None], 0)
        return values.reshape(batch, labels, count, dimension).permute(0, 2, 1, 3)

    def _conjunction(self, scores, truth, valid):
        selected = (scores >= self.threshold) & valid
        fallback = scores.masked_fill(~valid, -torch.inf).topk(self.atom_count, -1).indices
        fallback_mask = torch.zeros_like(valid).scatter_(-1, fallback, True) & valid
        selected = torch.where(selected.any(-1, keepdim=True), selected, fallback_mask)
        # Keep the selected score in both threshold and fallback gradient paths.
        return torch.where(selected, truth * scores, torch.ones_like(truth)).prod(-1)

    def _heads(self, objects, object_mask, text, image, text_mask, pair_projection):
        predicates = self._predicates(objects)
        objects = objects.flatten(1, 2)[:, :, None].expand(-1, -1, 2, -1)
        labels = self.labels[None, None].expand_as(objects)
        valid = object_mask.flatten(1, 2)
        mixed = self._mix(predicates, valid)
        text = self._pool(text, self.text_score, text_mask)
        image = self._pool(
            image, self.image_score, torch.ones_like(image[..., 0], dtype=torch.bool)
        )
        global_input = pair_projection(_Objects._pairs(text, image))
        answer_scores = (
            self.answer_score(
                self._comparison(self.answer_instance(objects), self.answer_label(labels))
            )
            .squeeze(-1)
            .transpose(1, 2)
        )
        valid = valid[:, None].expand(-1, 2, -1)
        answer_scores = _sparsemax(answer_scores, valid)
        # Eq5 label binding; the actual author truth MLP/dropout is retained.
        truth = (
            self.truth(self._comparison(predicates, objects * labels)).squeeze(-1).transpose(1, 2)
        )
        clauses = []
        for predicate_guide, input_guide in self.guides:
            scores = (
                predicate_guide(mixed).squeeze(-1) * input_guide(global_input)[:, None]
            ).transpose(1, 2)
            scores = _sparsemax(_sparsemax(scores, valid) * answer_scores, valid)
            clauses.append(self._conjunction(scores, truth, valid))
        return torch.stack(clauses, 1)

    def forward(self, objects, object_mask, text, image, text_mask, pair_projection):
        return (
            self._heads(objects, object_mask, text, image, text_mask, pair_projection).max(1).values
        )


class _Clause(nn.Module):
    """Paper Eqs.2–5 with fixed-length, unweighted product-t-norm clauses.

    Hard top-q is an integer selection. Its exact derivative to ranking-only
    parameters is absent, as in the stated forward equations; no straight-
    through estimator, auxiliary ranking target, or score factor is introduced.
    Object/predicate/label/truth parameters still learn through selected atoms.
    """

    def __init__(self, dimension, top_k, correlations, rate, dropout=0.2):
        super().__init__()
        self.atom_count = math.floor(5 * top_k * rate)
        self.labels = nn.Parameter(torch.empty(2, dimension))
        self.correlations = nn.Parameter(torch.empty(5, correlations, dimension))
        self.binding = nn.ModuleList(nn.Linear(2 * dimension, dimension, bias=False) for _ in range(5))
        self.text_score = nn.Linear(dimension, 1, bias=False)
        self.image_score = nn.Linear(dimension, 1, bias=False)
        self.input_guide = nn.Linear(2 * dimension, dimension, bias=False)
        self.label_guide = nn.Linear(4 * dimension, 1, bias=False)
        # Retain the author's two-layer scalar-truth parameterization, but bind
        # objects to the candidate label exactly as p = o * y in paper Eq.5.
        self.truth = nn.Sequential(
            nn.Linear(4 * dimension, 2 * dimension),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(2 * dimension, 1),
            nn.Sigmoid(),
        )
        nn.init.xavier_uniform_(self.labels)
        for values in self.correlations:
            nn.init.xavier_uniform_(values)

    _comparison = staticmethod(_AuthorClause._comparison)
    _pool = staticmethod(_AuthorClause._pool)

    def _predicates(self, objects):
        batch, _, top_k, dimension = objects.shape
        labels = self.labels[None, None].expand(batch, top_k, -1, -1)
        values = []
        for family in range(5):
            instance = objects[:, family, :, None].expand(-1, -1, 2, -1)
            key = self.binding[family](torch.cat((instance, labels), -1))
            correlations = self.correlations[family]
            values.append(_sparsemax(key @ correlations.T) @ correlations)
        return torch.cat(values, 1)

    def _scores(self, predicates, labels, text, image, text_mask, valid):
        text = self._pool(text, self.text_score, text_mask)
        image = self._pool(image, self.image_score, torch.ones_like(image[..., 0], dtype=torch.bool))
        guide = self.input_guide(torch.cat((text, image), -1))
        # Both Eq.3 attentions normalize over predicates, independently for each
        # sample and candidate label; the label guide reads B, rather than o.
        input_scores = (predicates * guide[:, None, None]).sum(-1).transpose(1, 2)
        label_scores = self.label_guide(self._comparison(predicates, labels)).squeeze(-1).transpose(1, 2)
        input_scores = _sparsemax(input_scores, valid)
        label_scores = _sparsemax(label_scores, valid)
        return _sparsemax(input_scores * label_scores, valid)

    def _select(self, scores, valid):
        if torch.any(valid.sum(-1) < self.atom_count):
            raise ValueError("Paper clause length exceeds the actual valid object count.")
        # Stable ties use the declared family/object ordering; no validation or
        # true label is consulted when selecting either candidate's atoms.
        return scores.masked_fill(~valid, -torch.inf).argsort(
            dim=-1, descending=True, stable=True
        )[..., : self.atom_count]

    def forward(self, objects, object_mask, text, image, text_mask, *, return_features=False):
        predicates = self._predicates(objects)
        objects = objects.flatten(1, 2)[:, :, None].expand(-1, -1, 2, -1)
        labels = self.labels[None, None].expand_as(objects)
        valid = object_mask.flatten(1, 2)[:, None].expand(-1, 2, -1)
        scores = self._scores(predicates, labels, text, image, text_mask, valid)
        indices = self._select(scores, valid)
        comparison = self._comparison(predicates, objects * labels)
        atom_truth = self.truth(comparison).squeeze(-1).transpose(1, 2)
        head_truth = atom_truth.gather(-1, indices).prod(-1)
        if not return_features:
            return head_truth
        # Actual pre-truth-head inputs, pooled over selected atoms separately
        # for each candidate, then concatenated. These are 8*d features, never
        # probabilities or representations conditioned on the observed label.
        comparison = comparison.transpose(1, 2)
        chosen = comparison.gather(2, indices[..., None].expand(-1, -1, -1, comparison.shape[-1]))
        return head_truth, chosen.mean(2).flatten(1)


def _disjunction(clauses):
    return 1 - (1 - torch.stack(clauses)).prod(0)


def _validate(p):
    variant = p.get("logic_variant", "author_corrected_v3")
    if variant not in {"author_corrected_v3", "paper_hard_topq_v1"}:
        raise ValueError("Unknown LogicDM logic_variant.")
    positive = [
        "dimension", "gcn_layers", "top_k", "correlations", "pair_chunk_size", "image_hidden"
    ]
    if variant == "author_corrected_v3":
        positive.extend(("predicate_heads", "guide_heads"))
    if any(
        type(p[name]) is not int or p[name] < 1
        for name in positive
    ):
        raise ValueError("LogicDM dimensions and chunk sizes must be positive integers.")
    if variant == "author_corrected_v3" and p["dimension"] % p["predicate_heads"]:
        raise ValueError("LogicDM dimension must be divisible by predicate_heads.")
    for key in ("logic_dropout", * (("clause_threshold",) if variant == "author_corrected_v3" else ())):
        value = p[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"LogicDM {key} must be finite in [0, 1).")
    image, text, syntax = p["image"], p["text"], p["syntax"]
    if (
        any(
            type(image[name]) is not int or image[name] < 1
            for name in (
                "image_size",
                "grid_size",
                "patch_batch_size",
            )
        )
        or image["image_size"] % image["grid_size"]
        or text["output"] != "tokens"
        or text["freeze"] is not True
        or text["mask_padding"] is not True
        or type(text["max_length"]) is not int
        or text["max_length"] < 3
        or syntax["language"] not in {"en", "zh"}
        or not isinstance(syntax["model_name"], str)
        or not syntax["model_name"]
        or not isinstance(syntax["model_version"], str)
        or not syntax["model_version"]
        or not isinstance(p["evaluated_layers"], list)
        or not p["evaluated_layers"]
        or len(set(p["evaluated_layers"])) != len(p["evaluated_layers"])
        or max(p["evaluated_layers"]) != p["gcn_layers"]
        or any(
            type(layer) is not int or not 0 <= layer <= p["gcn_layers"]
            for layer in p["evaluated_layers"]
        )
    ):
        raise ValueError("LogicDM needs frozen masked BERT, valid patches/parser and graph layers.")
    for key in ("image_layer_norm_eps", "gcn_layer_norm_eps"):
        if type(p[key]) not in (int, float) or not math.isfinite(p[key]) or p[key] <= 0:
            raise ValueError(f"LogicDM {key} must be positive and finite.")
    rate = p["clause_rate"]
    if type(rate) not in (float, int) or not math.isfinite(rate) or not 0 < rate < 1:
        raise ValueError("LogicDM clause_rate must be finite in (0, 1).")
    patches = image["grid_size"] ** 2
    minimum_atoms = 1 + 2 * min(p["top_k"], patches) + min(p["top_k"], patches * (patches - 1))
    if not 1 <= math.floor(5 * p["top_k"] * rate) <= minimum_atoms:
        raise ValueError("LogicDM clause length is empty or can exceed valid singleton objects.")


@register_model("LogicDM")
class LogicDM(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        _validate(p)
        self.parameters_spec = deepcopy(p)
        self.syntax_encoder = material_encoder(
            config,
            "syntax_encoder",
            _SyntaxEncoder,
            **p["syntax"],
            max_words=p["text"]["max_length"],
        )
        self.text_encoder = material_encoder(
            config, "text_encoder", _WordBertEncoder, self.syntax_encoder, **p["text"]
        )
        self.image_encoder = material_encoder(config, "image_encoder", _patch_encoder, **p["image"])
        dimension = p["dimension"]
        self.lstm = nn.LSTM(self.text_encoder.output_dim, dimension, batch_first=True)
        self.image_projection = nn.Sequential(
            nn.Linear(self.image_encoder.output_dim, p["image_hidden"]),
            nn.ReLU(),
            nn.Linear(p["image_hidden"], dimension),
            nn.ReLU(),
            nn.LayerNorm(dimension, eps=p["image_layer_norm_eps"]),
        )
        self.gcn = nn.ModuleList(
            _GraphConvolution(dimension, p["gcn_layer_norm_eps"]) for _ in range(p["gcn_layers"])
        )
        self.objects = _Objects(dimension, p["top_k"], p["pair_chunk_size"])
        self.paper_logic = p.get("logic_variant", "author_corrected_v3") == "paper_hard_topq_v1"
        if self.paper_logic:
            self.clause = _Clause(dimension, p["top_k"], p["correlations"], p["clause_rate"], p["logic_dropout"])
        else:
            self.clause = _AuthorClause(
                dimension, p["top_k"], p["correlations"], p["clause_rate"],
                p["predicate_heads"], p["guide_heads"], p["clause_threshold"], p["logic_dropout"],
            )
        self._alignment_cache = {}
        self._alignment_tokenizer = None
        self._alignment_profile = None

    def _word_structures(self, texts, syntax):
        tokenizer = self.text_encoder.tokenizer
        maximum = self.text_encoder.max_length
        profile = (maximum, tokenizer.padding_side, tokenizer.truncation_side)
        if self._alignment_tokenizer is not tokenizer or self._alignment_profile != profile:
            self._alignment_cache = {}
            self._alignment_tokenizer, self._alignment_profile = tokenizer, profile
        syntax = syntax.detach().cpu()
        keys = [(text, _sha(row.numpy().tobytes())) for text, row in zip(texts, syntax)]
        missing = {}
        for key, text, row in zip(keys, texts, syntax):
            if key not in self._alignment_cache:
                missing[key] = (text, row)
        if missing:
            records = list(missing.values())
            mappings, heads = _align_words(tokenizer, [text for text, _ in records],
                                          torch.stack([row for _, row in records]), maximum)
            for key, words, dependencies in zip(missing, mappings, heads):
                membership = torch.zeros(len(words), maximum, dtype=torch.bool)
                for position, indices in enumerate(words):
                    membership[position, indices] = True
                self._alignment_cache[key] = (membership, dependencies)
        # Only immutable tokenizer/span-derived structure is cached. No token
        # embeddings, trainable features, labels, domains or predictions enter it.
        records = [self._alignment_cache[key] for key in keys]
        return [value for value, _ in records], [dependencies for _, dependencies in records]

    def _features(self, batch):
        texts = batch["text"]
        tokens = self.text_encoder(texts)
        syntax = self.syntax_encoder(texts)
        memberships, heads = self._word_structures(texts, syntax)
        lengths = [len(words) for words in memberships]
        maximum = max(lengths)
        values = _pool_word_tokens(tokens, memberships)
        packed = pack_padded_sequence(values, lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.lstm(packed)
        text, _ = pad_packed_sequence(encoded, batch_first=True, total_length=maximum)
        mask = (
            torch.arange(maximum, device=text.device)[None]
            < torch.tensor(lengths, device=text.device)[:, None]
        )
        image = self.image_projection(self.image_encoder(batch["image_path"]))
        graph = _adjacency(
            heads,
            lengths,
            self.parameters_spec["image"]["grid_size"],
            maximum,
            text.device,
            text.dtype,
        )
        return text, image, mask, graph

    def _reason(self, batch, return_features=False):
        text, image, mask, graph = self._features(batch)
        maximum = text.shape[1]
        features = torch.cat((text, image), 1)
        clauses = []
        representations = []
        layers = self.parameters_spec["evaluated_layers"]
        for layer in range(max(layers) + 1):
            if layer:
                features = self.gcn[layer - 1](features, graph)
            if layer in layers:
                text, image = features[:, :maximum], features[:, maximum:]
                objects, valid = self.objects(text, image, mask)
                if self.paper_logic:
                    result = self.clause(objects, valid, text, image, mask, return_features=return_features)
                    if return_features:
                        result, representation = result
                        representations.append(representation)
                    clauses.append(result)
                else:
                    clauses.append(self.clause(objects, valid, text, image, mask, self.objects.projections[4]))
        result = clauses[0] if len(clauses) == 1 else _disjunction(clauses)
        return (result, torch.cat(representations, -1)) if return_features else result

    def predict(self, batch):
        return self._reason(batch)

    def predict_with_features(self, batch):
        if not self.paper_logic:
            raise ValueError("The documented selected-atom representation belongs to paper_hard_topq_v1.")
        return self._reason(batch, return_features=True)

    def calculate_loss(self, batch):
        truth, labels = self.predict(batch), batch["labels"]
        if self.paper_logic:
            return {"truth": F.cross_entropy(truth, labels)}
        real, fake = truth.unbind(-1)
        return {
            "truth": F.cross_entropy(truth, labels),
            "real_complement": F.cross_entropy(torch.stack((real, 1 - real), -1), labels),
            "fake_complement": F.cross_entropy(torch.stack((1 - fake, fake), -1), labels),
        }

    def get_extra_state(self):
        return {
            "parameters": self.parameters_spec,
            "implementation": (
                "logicdm-paper-eq2-eq5-hard-topq-single-ce-v3-batched-structures"
                if self.paper_logic else "logicdm-source-mco-five-guides-three-ce-l2-corrected-v3"
            ),
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("LogicDM parameters/implementation do not match the checkpoint.")
        # Derived structures are reproducible from fixed inputs and are never
        # checkpoint resources. Recompute after any checkpoint restoration.
        self._alignment_cache = {}
        self._alignment_tokenizer = None
        self._alignment_profile = None
