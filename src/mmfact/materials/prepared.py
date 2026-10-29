"""Resolve portable material recipes; never invoke a material producer."""

import json
from copy import deepcopy
from pathlib import Path

import yaml

from mmfact.config import (
    ConfigError,
    _deep_merge,
    _resolve_encoder_presets,
    load_config,
)
from mmfact.data.cache_policy import PreparedInputError
from mmfact.materials.vision import resolve_vision_catalog

_DATASETS = {"weibo17", "weibo21", "gossipcop"}
_SPECIAL = {"liimr", "nlin", "mmcan", "event_radar", "mmrgv", "hammer"}
_DIRECTORIES = {
    "safe": "annotations/captions",
    "aar": "annotations/arguments",
    "mmfn": "annotations/mmfn",
    "em_fend": "annotations/em_fend",
    "nslm": "retrieval/reverse_image",
    "eric_fnd": "retrieval/wikipedia",
}
_PATH_KEYS = {"root", "directory", "manifest", "checkpoint", "regions", "resources"}


def _catalog_parameters(parameters, model, catalog):
    """Use the delivered immutable recipe, keeping losses and trainable flags intact."""
    from mmfact.materials.download import _options

    catalog = resolve_vision_catalog(catalog, model, parameters)
    for entry in catalog["features"]:
        consumers = [
            consumer
            for consumer in entry.get("consumers", [])
            if consumer.get("method") == model
            or (model == "NLIN" and consumer.get("method") == "FND-CLIP")
        ]
        for consumer in consumers:
            role = consumer["input"]
            branch = _options(parameters, role)
            if not branch or branch.get("freeze") is False:
                continue
            settings = {**entry.get("settings", {}), **consumer.get("settings_overrides", {})}
            for key in (
                "model_name",
                "revision",
                "max_length",
                "mask_padding",
                "normalize",
                "max_sentences",
            ):
                if key in branch and key in settings:
                    branch[key] = deepcopy(settings[key])
            pooling = settings.get("pooling")
            if pooling in {"cls", "tokens", "flatten", "cls_layers", "mean"} and "output" in branch:
                branch["output"] = pooling
            if pooling == "pooler_output" and "output" in branch:
                branch["output"] = "pooled"
            if isinstance(settings.get("cleaning"), dict) and isinstance(
                branch.get("cleaning"), str
            ):
                branch["cleaning"] = settings["cleaning"]["language"]


def _resolved_explicit(resolved, explicit):
    """Restore requested keys from load_config's canonical values, not stale YAML.

    A saved ResNet recipe may be selected as ViT (or vice versa). Encoder preset
    expansion and backbone selection have already happened in ``load_config``;
    replaying the unexpanded YAML would reintroduce discarded backbone options.
    An explicit preset stands for its whole resolved branch.
    """
    if not isinstance(explicit, dict) or not isinstance(resolved, dict):
        return deepcopy(resolved)
    if "encoder_preset" in explicit:
        return deepcopy(resolved)
    return {
        key: _resolved_explicit(resolved[key], value)
        for key, value in explicit.items()
        if key in resolved
    }


def _hub_config(config, root, overrides):
    """Bind the final shared release; production recipes are not runtime inputs."""
    from mmfact.materials.download import requires_images

    if (overrides or {}).get("preprocessing", {}).get("mode") == "raw":
        raise ConfigError(
            "The Hub release uses prepared inputs; use a raw dataset to rebuild inputs."
        )
    # Final model presets already contain the experimental encoder settings.
    # Frozen caches are bound below; trainable branches keep their chosen backbone.
    result = deepcopy(config)
    dataset = result["dataset"]["name"]
    directory = root / "datasets" / dataset
    catalog = json.loads((directory / "feature_catalog.json").read_text("utf-8"))
    result["dataset"].update(
        root=str(root),
        splits={
            key: f"datasets/{dataset}/data/{split}/metadata.jsonl"
            for key, split in (("train", "train"), ("val", "validation"), ("test", "test"))
        },
    )
    vocabulary = json.loads((directory / "labels/vocabulary.json").read_text("utf-8"))
    model = result["model"]["name"]
    _catalog_parameters(result["model"]["parameters"], model, catalog)
    if model == "HAMMER":
        result["model"]["parameters"]["initialization"]["checkpoint"] = str(
            root / "pretrained/hammer/ALBEF_4M.pth"
        )
        result["model"]["parameters"]["auxiliary"]["root"] = str(root / "auxiliary/dgm4")
    domain_field = {
        "RDCM": "domain_3",
        "ADOSE": "domain_4",
        "MMDFND": "domain",
        "DAMMFND": "domains",
    }.get(model)
    if domain_field:
        names = vocabulary[domain_field]
        result["model"]["parameters"]["domains"].update(
            mode="metadata",
            num_domains=len(names),
            domain_names=names,
            **({"field": domain_field} if model != "MMDFND" else {}),
        )
    # Explicit experiment settings still win, using their already resolved values.
    for section in ("model", "training"):
        if section in (overrides or {}):
            _deep_merge(result[section], _resolved_explicit(config[section], overrides[section]))
    needed = requires_images(result, catalog)
    result["dataset"].update(require_images=needed, verify_images=False, hash_images=False)
    result["preprocessing"].update(
        mode="prepared",
        layout="hub",
        materials_root=str(root),
        cache_dir=str(root / ".cache/mmfact"),
    )
    if not result["preprocessing"]["enabled"]:
        raise ConfigError("Prepared inputs require preprocessing.enabled=true.")
    return result


def _slug(name):
    return name.lower().replace("-", "_")


def _changed(current, defaults):
    """Retain resolved caller changes without replacing production encoder defaults."""
    if not isinstance(current, dict) or not isinstance(defaults, dict):
        return deepcopy(current)
    result = {}
    for key, value in current.items():
        if key not in defaults or value != defaults[key]:
            result[key] = (
                _changed(value, defaults[key])
                if isinstance(value, dict) and isinstance(defaults.get(key), dict)
                else deepcopy(value)
            )
    return result


def _resolve_paths(value, root):
    if isinstance(value, list):
        return [_resolve_paths(child, root) for child in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, child in value.items():
        if (
            isinstance(child, str)
            and child
            and (key in _PATH_KEYS or key.endswith(("_path", "_manifest")))
        ):
            path = Path(child).expanduser()
            child = str(path if path.is_absolute() else (root / path).resolve())
        result[key] = _resolve_paths(child, root)
    return result


def _eric_nlin_recipe(recipe):
    """Reuse NLIN's source resource identity, not its detector or feature recipe."""
    parameters = recipe["model"].get("parameters")
    resources = parameters.get("resources") if isinstance(parameters, dict) else None
    if not isinstance(resources, dict) or not isinstance(resources.get("manifest"), str):
        raise PreparedInputError("NLIN material recipe requires a resources.manifest for ERIC-FND.")
    if not resources["manifest"].strip():
        raise PreparedInputError("NLIN material recipe requires a nonempty resources.manifest.")
    return {
        "model": {
            "parameters": {
                "resources": {
                    "mode": "nlin",
                    **{
                        key: deepcopy(resources[key])
                        for key in ("manifest", "language", "tools")
                        if key in resources
                    },
                }
            }
        }
    }


def load_material_config(config, materials_root, *, overrides=None):
    """Adopt the delivered recipe while keeping dataset/training choices intact.

    ``overrides`` is an optional already-resolved explicit caller mapping. It is
    applied last, including explicit values equal to packaged model defaults.
    For the final Hub layout, ``config`` must already include these choices via
    ``load_config``; the mapping identifies which canonical values take priority.
    Without it, differences from the packaged defaults remain authoritative.
    All relative material/resource paths are based on ``materials_root``.
    """
    root = Path(materials_root).expanduser().resolve()
    dataset, name = config["dataset"]["name"], config["model"]["name"]
    if dataset not in _DATASETS:
        raise ConfigError(f"Prepared materials do not support dataset {dataset!r}.")
    slug = _slug(name)
    organized = root / "datasets" / dataset
    if (organized / "feature_catalog.json").is_file():
        if (
            slug == "mmcan"
            and (organized / "splits/train.jsonl").is_file()
            and not (organized / "data/train/metadata.jsonl").is_file()
        ):
            # The earlier MMCAN release shares matrices but uses split/image indexes.
            result = deepcopy(config)
            result["dataset"].update(
                root=str(organized),
                splits={split: f"splits/{split}.jsonl" for split in ("train", "val", "test")},
            )
            result["preprocessing"].update(mode="prepared", materials_root=str(root))
            if not result["preprocessing"]["enabled"]:
                raise ConfigError("Prepared inputs require preprocessing.enabled=true.")
        else:
            result = _hub_config(config, root, overrides)
        if slug == "mmcan":
            result["model"]["parameters"]["shared_materials"] = True
            result["model"]["parameters"]["matching"].update(
                directory=None, translation_manifest=None
            )
        return result
    resources = deepcopy(config["model"]["parameters"].get("resources", {}))
    explicit_resources = (overrides or {}).get("model", {}).get("parameters", {}).get("resources")
    if isinstance(resources, dict) and isinstance(explicit_resources, dict):
        _deep_merge(resources, explicit_resources)
    eric_nlin = slug == "eric_fnd" and resources.get("mode", "nlin") == "nlin"
    relative = (
        "methods/nlin"
        if eric_nlin
        else _DIRECTORIES.get(
            slug, f"{'methods' if slug in _SPECIAL else 'features/recipes'}/{slug}"
        )
    )
    source_name = "NLIN" if eric_nlin else name
    directory = root / dataset / relative
    paths = [directory / "integration.yaml"]
    paths = [path for path in paths if path.is_file()]
    if not paths:
        raise PreparedInputError(f"Prepared integration recipe is missing in {directory}.")
    recipe = {}
    for path in paths:
        try:
            value = yaml.safe_load(path.read_text("utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise PreparedInputError(f"Invalid material integration recipe at {path}.") from error
        if not isinstance(value, dict) or not isinstance(value.get("model"), dict):
            raise PreparedInputError(f"Material recipe must contain model settings: {path}.")
        supplied = value["model"].get("name", source_name)
        if not isinstance(supplied, str) or _slug(supplied) != _slug(source_name):
            raise PreparedInputError(
                f"Material recipe model {supplied!r} does not match {source_name!r}."
            )
        if value.get("dataset", {}).get("name", dataset) != dataset:
            raise PreparedInputError(f"Material recipe dataset does not match {dataset!r}.")
        _deep_merge(recipe, value)
    if eric_nlin:
        recipe = _eric_nlin_recipe(recipe)
    result = deepcopy(config)
    defaults = load_config(name, dataset)
    caller = _changed(config["model"], defaults["model"])
    model = _deep_merge(deepcopy(defaults["model"]), _resolve_paths(recipe["model"], root))
    result["model"] = _deep_merge(model, caller)
    if overrides:
        # The runner supplies only values that were explicitly requested.
        explicit = deepcopy(overrides)
        if "parameters" in explicit.get("model", {}):
            explicit["model"]["parameters"] = _resolve_encoder_presets(
                explicit["model"]["parameters"], dataset
            )
        for key in ("model", "preprocessing"):
            if key in explicit:
                _deep_merge(result[key], explicit[key])
    options = result["preprocessing"]
    options.setdefault("mode", "prepared")
    options["materials_root"] = str(root)
    if "cache_dir" in (overrides or {}).get("preprocessing", {}):
        options["cache_dir"] = config["preprocessing"]["cache_dir"]
    else:
        options["cache_dir"] = str(root / "shared/cache")
    if options["mode"] == "prepared" and not options["enabled"]:
        raise ConfigError("Prepared inputs require preprocessing.enabled=true.")
    if "batch_size" in recipe.get("preprocessing", {}) and (
        options["batch_size"] == defaults["preprocessing"]["batch_size"]
        and "batch_size" not in (overrides or {}).get("preprocessing", {})
    ):
        options["batch_size"] = recipe["preprocessing"]["batch_size"]
    return result
