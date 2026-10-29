"""Layered configuration and explicitly bounded hyperparameter search.

Precedence is overall defaults, a dataset preset, an optional model preset and
its dataset-specific settings, external YAML, then dotted command-line overrides.
Unknown model names are allowed so an external config can register a model;
datasets must currently have a packaged preset. Paths are relative to the
working directory from which :func:`load_config` is called.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from mmfact import configs


class ConfigError(ValueError):
    """A configuration value or search dimension is invalid."""


_CONFIG_DIR = Path(configs.__file__).parent
_PRESET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SEARCH_TRAINING_KEYS = {
    "training.learning_rate",
    "training.weight_decay",
    "training.batch_size",
}
_TOP_KEYS = {
    "dataset",
    "model",
    "preprocessing",
    "training",
    "evaluation",
    "search",
    "output_dir",
}

# Only ordinary supervised visual branches participate in this comparison.
# MAE, paired CLIP, detection, and method-specific forensic networks are separate.
SUPERVISED_VISION_MODELS = frozenset({
    "ADOSE", "BTIC", "CAFE", "CMC", "COOLANT", "CSFND", "ConcatMLP", "EANN",
    "EM-FEND", "ERIC-FND", "FND-CLIP", "HMCAN", "ImageOnly", "LateFusion",
    "LogicDM", "MCAN", "MRML", "MSACA", "MVAE", "NSLM", "RDCM", "SpotFake",
    "SpotFakePlus", "att-RNN",
})


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"Configuration file must contain a mapping: {path}")
    return loaded


def _model_preset(name: str) -> Path | None:
    """Match display names and import names without silently choosing an alias."""
    presets = sorted(path for path in (_CONFIG_DIR / "model").glob("*.yaml") if path.is_file())
    for path in presets:
        if path.name == f"{name}.yaml":
            return path
    key = name.lower().replace("-", "_")
    matches = [path for path in presets if path.stem.lower().replace("-", "_") == key]
    if len(matches) > 1:
        choices = ", ".join(path.name for path in matches)
        raise ConfigError(f"Ambiguous model preset {name!r}: {choices}; use an exact name.")
    return matches[0] if matches else None


def _deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _resolve_encoder_presets(parameters: Any, dataset: str) -> Any:
    """Expand explicit backbone references; saved configs contain actual settings."""
    presets = None

    def resolve(value: Any) -> Any:
        nonlocal presets
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if not isinstance(value, dict):
            return value
        options = dict(value)
        if "encoder_preset" in options:
            if presets is None:
                presets = _read_yaml(_CONFIG_DIR / "encoders.yaml")
            name = options.pop("encoder_preset")
            if not isinstance(name, str) or name not in presets:
                raise ConfigError(f"Unknown encoder_preset: {name!r}")
            preset = presets[name]
            inherited = deepcopy(preset["default"])
            _deep_merge(inherited, preset.get("dataset_overrides", {}).get(dataset, {}))
            options = _deep_merge(inherited, options)
        return {key: resolve(item) for key, item in options.items()}

    return resolve(parameters)


def _resolve_supervised_vision(config: dict, legacy_parameters: dict) -> None:
    """Select the visual recipe without changing trained heads or hyperparameters.

    Model YAMLs retain the existing ResNet recipe and experimental parameters.
    The explicit selector changes only the ordinary image branches. Keeping a
    copy of the preset recipe also permits switching a saved ViT config back to
    ResNet without carrying ViT-only checkpoint or preprocessing fields across.
    """
    model = config["model"]
    choice = model.get("vision_backbone")
    if choice is None:
        return  # Methods without an ordinary supervised selector stay unchanged.
    preset = _read_yaml(_CONFIG_DIR / "encoders.yaml")["supervised_vit"]["default"]
    recipe_keys = ("architecture", "weights", "model_name", "revision",
                   "backbone", "center_crop", "stem_maxpool")

    def visit(parameters, reference):
        for name, value in parameters.items():
            if not isinstance(value, dict):
                continue
            original = reference.get(name, {}) if isinstance(reference, dict) else {}
            if name not in {"image", "contrastive_image"}:
                visit(value, original)
                continue
            if choice == "resnet":
                if value.get("backbone") == "vit":
                    for key in recipe_keys:
                        value.pop(key, None)
                        if key in original:
                            value[key] = deepcopy(original[key])
                    if model["name"] == "MSACA":
                        value["scales"] = deepcopy(original["scales"])
                continue
            value.pop("weights", None)
            value.pop("stem_maxpool", None)
            value.update(deepcopy(preset))
            if model["name"] == "MSACA":
                value["scales"] = [224, 112, 64]

    visit(model["parameters"], legacy_parameters)


def _set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    if not key or any(not part for part in parts):
        raise ConfigError(f"Expected a nonempty dotted configuration key: {key!r}")
    # Grid dimensions are themselves dotted keys, rather than nested options.
    if key.startswith("search.parameters."):
        search = config.setdefault("search", {})
        if not isinstance(search, dict):
            raise ConfigError("Cannot set search parameter: search is not a mapping")
        parameters = search.setdefault("parameters", {})
        if not isinstance(parameters, dict):
            raise ConfigError("Cannot set search parameter: search.parameters is not a mapping")
        parameters[key.removeprefix("search.parameters.")] = deepcopy(value)
        return
    current = config
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        if not isinstance(current[part], dict):
            raise ConfigError(f"Cannot set {key!r}: {part!r} is not a mapping")
        current = current[part]
    current[parts[-1]] = deepcopy(value)


def _mapping(config: Mapping[str, Any], key: str, allowed: set[str]) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a mapping")
    _known_keys(value, allowed, key)
    return value


def _known_keys(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = [str(key) for key in value if key not in allowed]
    if unknown:
        raise ConfigError(f"Unknown {path} option(s): {', '.join(sorted(unknown))}")


def _nonempty_string(value: Any, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a nonempty string")


def _integer(value: Any, path: str, minimum: int = 1) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigError(f"{path} must be an integer >= {minimum}")


def _number(value: Any, path: str, *, positive: bool) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        comparison = "> 0" if positive else ">= 0"
        raise ConfigError(f"{path} must be a finite number {comparison}")


def _validate_json_value(value: Any, path: str, active: set[int] | None = None) -> None:
    """Keep saved configs/checkpoints portable without guessing YAML conversions."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError(f"{path} must contain finite numbers; got {value!r}")
        return
    if not isinstance(value, (dict, list)):
        raise ConfigError(f"{path} must contain JSON-compatible values; got {type(value).__name__}")
    if active is None:
        active = set()
    identity = id(value)
    if identity in active:
        raise ConfigError(f"{path} contains a cyclic YAML mapping or list")
    active.add(identity)
    try:
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ConfigError(
                        f"{path} mapping keys must be strings; got {type(key).__name__}"
                    )
                _validate_json_value(item, f"{path}.{key}", active)
        else:
            for index, item in enumerate(value):
                _validate_json_value(item, f"{path}[{index}]", active)
    finally:
        active.remove(identity)


def _validate_search(search: Mapping[str, Any]) -> None:
    parameters = search.get("parameters")
    if not isinstance(parameters, dict):
        raise ConfigError("search.parameters must be a mapping of dotted keys to lists")
    for key, values in parameters.items():
        if not isinstance(key, str):
            raise ConfigError("search.parameters keys must be dotted strings")
        if "encoder_preset" in key.split("."):
            raise ConfigError(
                "search cannot vary encoder_preset; search resolved model parameters instead"
            )
        model_parameter = key.startswith("model.parameters.") and all(key.split("."))
        if not model_parameter and key not in _SEARCH_TRAINING_KEYS:
            raise ConfigError(
                f"Unsupported search parameter {key!r}; search may vary model.parameters.* "
                "or training.learning_rate, training.weight_decay, training.batch_size. "
                "Datasets, evaluation metrics, and seeds are fixed across candidates."
            )
        if not isinstance(values, list) or not values:
            raise ConfigError(f"search.parameters.{key} must be a nonempty list")
        if key == "training.batch_size":
            for value in values:
                _integer(value, f"search.parameters.{key}")
        elif key in _SEARCH_TRAINING_KEYS:
            for value in values:
                _number(
                    value,
                    f"search.parameters.{key}",
                    positive=key.endswith("learning_rate"),
                )


def _validate_optimizer(algorithm, options, prefix):
    if algorithm not in ("adam", "adamw", "sgd", "adabelief", "riemannian_adam"):
        raise ConfigError(
            f"{prefix}.optimizer must be adam, adamw, sgd, adabelief, or riemannian_adam"
        )
    path = f"{prefix}.optimizer_parameters"
    if not isinstance(options, dict):
        raise ConfigError(f"{path} must be a mapping")
    allowed = (
        {"momentum", "dampening", "nesterov"} if algorithm == "sgd" else {"betas", "eps", "amsgrad"}
    )
    if algorithm == "adabelief":
        allowed |= {"weight_decouple", "fixed_decay", "rectify", "print_change_log"}
    if algorithm == "riemannian_adam":
        allowed |= {"stabilize"}
    _known_keys(options, allowed, path)
    for key, value in options.items():
        if key == "betas":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ConfigError(f"{path}.betas must contain two values")
            for beta in value:
                _number(beta, "optimizer beta", positive=False)
                if beta >= 1:
                    raise ConfigError("optimizer beta must be less than 1")
        elif key in {"eps", "momentum", "dampening"}:
            _number(value, f"{path}.{key}", positive=key == "eps")
        elif key == "stabilize":
            if value is not None:
                _integer(value, f"{path}.{key}")
        elif not isinstance(value, bool):
            raise ConfigError(f"{path}.{key} must be boolean")


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate structural options and require JSON-compatible freeform values."""
    if not isinstance(config, dict):
        raise ConfigError("Configuration must be a mapping")
    _validate_json_value(config, "config")
    _known_keys(config, _TOP_KEYS, "top-level")
    dataset = _mapping(
        config,
        "dataset",
        {
            "name",
            "root",
            "splits",
            "label_map",
            "verify_images",
            "hash_images",
            "require_images",
        },
    )
    _nonempty_string(dataset.get("name"), "dataset.name")
    _nonempty_string(dataset.get("root"), "dataset.root")
    splits = dataset.get("splits")
    if not isinstance(splits, dict):
        raise ConfigError("dataset.splits must be a mapping")
    _known_keys(splits, {"train", "val", "test"}, "dataset.splits")
    for split in ("train", "val", "test"):
        _nonempty_string(splits.get(split), f"dataset.splits.{split}")
    label_map = dataset.get("label_map")
    if not isinstance(label_map, dict) or not label_map:
        raise ConfigError("dataset.label_map must be a nonempty mapping")
    for label, index in label_map.items():
        _nonempty_string(label, "dataset.label_map key (quote numeric YAML keys)")
        if not isinstance(index, int) or isinstance(index, bool) or index not in (0, 1):
            raise ConfigError(f"dataset.label_map.{label} must be 0 (real) or 1 (fake)")
    if set(label_map.values()) != {0, 1}:
        raise ConfigError("dataset.label_map must contain labels for both real (0) and fake (1)")
    if not isinstance(dataset.get("verify_images"), bool):
        raise ConfigError("dataset.verify_images must be a boolean")
    if not isinstance(dataset.get("hash_images"), bool):
        raise ConfigError("dataset.hash_images must be a boolean")
    if type(dataset.get("require_images", True)) is not bool:
        raise ConfigError("dataset.require_images must be a boolean")
    if not dataset.get("require_images", True) and (
        dataset["verify_images"] or dataset["hash_images"]
    ):
        raise ConfigError("Image verification/hashing requires dataset.require_images=true")

    model = _mapping(config, "model", {"name", "module", "parameters", "provenance", "vision_backbone"})
    _nonempty_string(model.get("name"), "model.name")
    if "vision_backbone" in model:
        if model["name"] not in SUPERVISED_VISION_MODELS:
            raise ConfigError("model.vision_backbone is only available for ordinary supervised visual branches")
        if not isinstance(model["vision_backbone"], str) or model["vision_backbone"] not in {"vit", "resnet"}:
            raise ConfigError("model.vision_backbone must be vit or resnet")
    if model.get("module") is not None:
        _nonempty_string(model["module"], "model.module")
    for key in ("parameters", "provenance"):
        if not isinstance(model.get(key), dict):
            raise ConfigError(f"model.{key} must be a mapping")

    preprocessing = _mapping(
        config,
        "preprocessing",
        {"enabled", "cache_dir", "batch_size", "mode", "materials_root", "layout"},
    )
    if not isinstance(preprocessing.get("enabled"), bool):
        raise ConfigError("preprocessing.enabled must be a boolean")
    _nonempty_string(preprocessing.get("cache_dir"), "preprocessing.cache_dir")
    _integer(preprocessing.get("batch_size"), "preprocessing.batch_size")
    if not isinstance(preprocessing.get("mode", "raw"), str) or preprocessing.get(
        "mode", "raw"
    ) not in {"raw", "prepared"}:
        raise ConfigError("preprocessing.mode must be raw or prepared")
    if preprocessing.get("mode") == "prepared" and not preprocessing["enabled"]:
        raise ConfigError("Prepared inputs require preprocessing.enabled=true")
    if preprocessing.get("materials_root") is not None:
        _nonempty_string(preprocessing["materials_root"], "preprocessing.materials_root")
    if preprocessing.get("layout") not in {None, "hub"}:
        raise ConfigError("preprocessing.layout must be hub when specified")

    training = _mapping(
        config,
        "training",
        {
            "seeds",
            "epochs",
            "batch_size",
            "eval_batch_size",
            "num_workers",
            "learning_rate",
            "weight_decay",
            "optimizer",
            "optimizer_parameters",
            "parameter_groups",
            "weight_decay_exclude_bias_norm",
            "patience",
            "device",
            "gradient_clip_norm",
            "gradient_accumulation_steps",
            "precision",
            "scheduler",
            "stage_overrides",
        },
    )
    seeds = training.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ConfigError("training.seeds must be a nonempty list of unique integer seeds")
    for seed in seeds:
        _integer(seed, "training.seeds entry", minimum=0)
        if seed > 2**32 - 1:
            raise ConfigError("training.seeds entries must be <= 2**32 - 1")
    if len(set(seeds)) != len(seeds):
        raise ConfigError("training.seeds must contain unique seeds")
    for key in ("epochs", "batch_size", "eval_batch_size", "patience"):
        _integer(training.get(key), f"training.{key}")
    _integer(training.get("num_workers"), "training.num_workers", minimum=0)
    _number(training.get("learning_rate"), "training.learning_rate", positive=True)
    _number(training.get("weight_decay"), "training.weight_decay", positive=False)
    groups = training.get("parameter_groups", {})
    if not isinstance(groups, dict):
        raise ConfigError("training.parameter_groups must be a mapping")
    for name, options in groups.items():
        _nonempty_string(name, "training.parameter_groups name")
        if not isinstance(options, dict):
            raise ConfigError(f"training.parameter_groups.{name} must be a mapping")
        prefix = f"training.parameter_groups.{name}"
        _known_keys(
            options,
            {"learning_rate", "weight_decay", "optimizer", "optimizer_parameters"},
            prefix,
        )
        for key in ("learning_rate", "weight_decay"):
            if key in options:
                _number(options[key], f"{prefix}.{key}", positive=key == "learning_rate")
        algorithm = options.get("optimizer", training.get("optimizer"))
        inherited = (
            training.get("optimizer_parameters", {})
            if algorithm == training.get("optimizer")
            else {}
        )
        _validate_optimizer(algorithm, options.get("optimizer_parameters", inherited), prefix)
    if not isinstance(training.get("weight_decay_exclude_bias_norm"), bool):
        raise ConfigError("training.weight_decay_exclude_bias_norm must be boolean")
    _validate_optimizer(
        training.get("optimizer"), training.get("optimizer_parameters", {}), "training"
    )
    _nonempty_string(training.get("device"), "training.device")
    if training.get("gradient_clip_norm") is not None:
        _number(training["gradient_clip_norm"], "training.gradient_clip_norm", positive=True)
    _integer(
        training.get("gradient_accumulation_steps", 1),
        "training.gradient_accumulation_steps",
    )
    if not isinstance(training.get("precision", "full"), str) or training.get(
        "precision", "full"
    ) not in {"full", "fp16", "bf16"}:
        raise ConfigError("training.precision must be full, fp16, or bf16")
    schedule = training.get("scheduler")
    if schedule is not None:
        if not isinstance(schedule, dict):
            raise ConfigError("training.scheduler must be a mapping or null")
        if schedule.get("name") == "inverse":
            _known_keys(schedule, {"name", "gamma", "power"}, "training.scheduler")
            for key in ("gamma", "power"):
                _number(schedule.get(key), f"training.scheduler.{key}", positive=True)
        elif schedule.get("name") == "step":
            _known_keys(schedule, {"name", "step_size", "gamma"}, "training.scheduler")
            _integer(schedule.get("step_size"), "training.scheduler.step_size", minimum=1)
            _number(schedule.get("gamma"), "training.scheduler.gamma", positive=True)
            if schedule["gamma"] > 1:
                raise ConfigError("training.scheduler.gamma must not exceed 1 for step decay")
        elif schedule.get("name") == "linear_warmup":
            _known_keys(schedule, {"name", "warmup_ratio"}, "training.scheduler")
            _number(
                schedule.get("warmup_ratio"),
                "training.scheduler.warmup_ratio",
                positive=False,
            )
            if schedule["warmup_ratio"] >= 1:
                raise ConfigError("training.scheduler.warmup_ratio must be less than 1")
        elif schedule.get("name") == "warmup_cosine":
            _known_keys(
                schedule,
                {"name", "warmup_steps", "warmup_start_lr", "min_lr"},
                "training.scheduler",
            )
            _integer(
                schedule.get("warmup_steps"),
                "training.scheduler.warmup_steps",
                minimum=0,
            )
            for key in ("warmup_start_lr", "min_lr"):
                _number(schedule.get(key), f"training.scheduler.{key}", positive=False)
                if schedule[key] > training["learning_rate"]:
                    raise ConfigError(f"training.scheduler.{key} must not exceed learning_rate")
        elif schedule.get("name") == "cosine":
            _known_keys(schedule, {"name", "min_lr"}, "training.scheduler")
            _number(schedule.get("min_lr"), "training.scheduler.min_lr", positive=False)
            if schedule["min_lr"] > training["learning_rate"]:
                raise ConfigError("training.scheduler.min_lr must not exceed learning_rate")
        elif schedule.get("name") == "plateau":
            _known_keys(
                schedule,
                {"name", "monitor", "factor", "patience", "min_lr"},
                "training.scheduler",
            )
            _nonempty_string(schedule.get("monitor"), "training.scheduler.monitor")
            if schedule["monitor"].startswith("validation.") and schedule["monitor"] not in {
                "validation.loss",
                "validation.accuracy",
                "validation.macro_f1",
            }:
                raise ConfigError("Unknown training.scheduler validation monitor")
            _number(schedule.get("factor"), "training.scheduler.factor", positive=True)
            if schedule["factor"] >= 1:
                raise ConfigError("training.scheduler.factor must be less than 1")
            _integer(schedule.get("patience"), "training.scheduler.patience", minimum=0)
            _number(schedule.get("min_lr"), "training.scheduler.min_lr", positive=False)
        else:
            raise ConfigError(
                "training.scheduler.name must be inverse, step, linear_warmup, "
                "warmup_cosine, cosine, or plateau"
            )

    evaluation = _mapping(config, "evaluation", {"selection_metric"})
    if evaluation.get("selection_metric") not in ("accuracy", "macro_f1"):
        raise ConfigError("evaluation.selection_metric must be accuracy or macro_f1")
    stages = training.get("stage_overrides", {})
    if not isinstance(stages, dict):
        raise ConfigError("training.stage_overrides must be a mapping")
    stage_keys = {
        "epochs",
        "patience",
        "batch_size",
        "eval_batch_size",
        "learning_rate",
        "weight_decay",
        "optimizer",
        "optimizer_parameters",
        "parameter_groups",
        "gradient_clip_norm",
        "gradient_accumulation_steps",
        "precision",
        "scheduler",
        "selection_metric",
        "min_delta",
    }
    for name, options in stages.items():
        if not isinstance(name, str) or not _PRESET_NAME.fullmatch(name):
            raise ConfigError("training.stage_overrides names must be simple identifiers")
        if not isinstance(options, dict):
            raise ConfigError(f"training.stage_overrides.{name} must be a mapping")
        _known_keys(options, stage_keys, f"training.stage_overrides.{name}")
        if options.get("selection_metric", "macro_f1") not in (
            "loss",
            "accuracy",
            "macro_f1",
            "last",
        ):
            raise ConfigError("Stage selection_metric must be loss, accuracy, macro_f1, or last")
        _number(
            options.get("min_delta", 0.0),
            f"stage_overrides.{name}.min_delta",
            positive=False,
        )
        if options.get("selection_metric") == "last" and options.get("min_delta", 0.0) != 0:
            raise ConfigError("Stage last selection does not accept a nonzero min_delta")
        candidate = deepcopy(config)
        candidate["training"].pop("stage_overrides", None)
        candidate["training"].update(
            {
                key: value
                for key, value in options.items()
                if key not in {"selection_metric", "min_delta"}
            }
        )
        validate_config(candidate)
    search = _mapping(config, "search", {"parameters"})
    _validate_search(search)
    _nonempty_string(config.get("output_dir"), "output_dir")


def load_config(
    model: str = "EANN",
    dataset: str = "weibo17",
    config_files: Sequence[str | Path] = (),
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Load, merge, validate, and resolve a complete experiment configuration.

    Each override has the form ``training.batch_size=16``; the value is parsed
    with ``yaml.safe_load`` so lists, booleans, nulls, and numbers retain their
    types. Unknown model names inherit defaults and can supply ``model.module``
    through external YAML. Relative dataset, cache, and output paths resolve
    against cwd.
    """
    for name, value in (("model", model), ("dataset", dataset)):
        if not isinstance(value, str) or not _PRESET_NAME.fullmatch(value):
            raise ConfigError(f"{name} preset must be a simple alphanumeric name: {value!r}")
    if isinstance(config_files, (str, Path)) or not isinstance(config_files, Sequence):
        raise ConfigError("config_files must be a sequence of paths")
    if isinstance(overrides, str) or not isinstance(overrides, Sequence):
        raise ConfigError("overrides must be a sequence of key=value strings")
    config = _read_yaml(_CONFIG_DIR / "overall.yaml")
    dataset_file = _CONFIG_DIR / "dataset" / f"{dataset}.yaml"
    if not dataset_file.is_file():
        raise ConfigError(f"Unknown dataset preset {dataset!r}; no file at {dataset_file}")
    _deep_merge(config, _read_yaml(dataset_file))
    model_file = _model_preset(model)
    model_config = _read_yaml(model_file) if model_file else {"model": {"name": model}}
    dataset_overrides = model_config.pop("dataset_overrides", {})
    if not isinstance(dataset_overrides, dict):
        raise ConfigError("Model preset dataset_overrides must be a mapping")
    _deep_merge(config, model_config)
    specific = dataset_overrides.get(dataset, {})
    if not isinstance(specific, dict) or set(specific) - {"model", "training"}:
        raise ConfigError("Dataset-specific model settings may only override model and training")
    _deep_merge(config, specific)
    legacy_parameters = _resolve_encoder_presets(
        deepcopy(config["model"]["parameters"]), dataset
    )
    for path in config_files:
        _deep_merge(config, _read_yaml(Path(path).expanduser()))
    for override in overrides:
        if not isinstance(override, str) or "=" not in override:
            raise ConfigError(f"Override must have the form dotted.key=value: {override!r}")
        key, raw_value = override.split("=", 1)
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML value in override {override!r}: {exc}") from exc
        _set_dotted(config, key, value)
    # Validate basic values first: recursive aliases must not enter the resolver.
    validate_config(config)
    config["model"]["parameters"] = _resolve_encoder_presets(config["model"]["parameters"], dataset)
    _resolve_supervised_vision(config, legacy_parameters)
    validate_config(config)
    config["dataset"]["root"] = str(Path(config["dataset"]["root"]).expanduser().resolve())
    config["preprocessing"]["cache_dir"] = str(
        Path(config["preprocessing"]["cache_dir"]).expanduser().resolve()
    )
    config["output_dir"] = str(Path(config["output_dir"]).expanduser().resolve())
    return config


def expand_search(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return independent grid candidates, keeping all repeated-run seeds fixed.

    Search dimension order follows the YAML mapping order. With an empty grid,
    one deep copy is returned. The original grid is retained in each candidate
    for provenance; callers execute each candidate rather than expanding it
    recursively. Seeds are repetition controls and are never a search dimension.
    """
    validate_config(config)
    parameters = config["search"]["parameters"]
    keys = list(parameters)
    combinations = product(*(parameters[key] for key in keys))
    candidates = []
    for values in combinations:
        candidate = deepcopy(dict(config))
        for key, value in zip(keys, values):
            _set_dotted(candidate, key, value)
        validate_config(candidate)
        candidates.append(candidate)
    return candidates
