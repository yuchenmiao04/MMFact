"""Plan and download the files used by one classification method, at a fixed revision."""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import yaml

from mmfact import configs
from mmfact.config import ConfigError, load_config
from mmfact.registry import list_models
from mmfact.materials.vision import branch_options, refit_training_resource, resolve_vision_catalog


def _slug(name):
    return name.lower().replace("-", "_")


def _relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ConfigError(f"Invalid material path: {value!r}.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ConfigError(f"Material path must stay inside its dataset: {value!r}.")
    return path.as_posix()


@dataclass(frozen=True)
class MaterialPlan:
    """Exact Hub paths; downloading them preserves repository-relative locations."""

    model: str
    dataset: str
    root: Path
    repo_id: str
    revision: str
    files: tuple[str, ...]
    total_bytes: int | None
    raw_images: bool
    external: tuple[dict, ...] = ()

    def to_dict(self):
        return {
            "model": self.model,
            "dataset": self.dataset,
            "root": str(self.root),
            "repo_id": self.repo_id,
            "revision": self.revision,
            "files": list(self.files),
            "total_bytes": self.total_bytes,
            "raw_images": self.raw_images,
            "external": list(self.external),
        }


def _definitions():
    path = Path(configs.__file__).parent / "materials.yaml"
    return yaml.safe_load(path.read_text("utf-8"))


def _hub():
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except ImportError as error:
        raise ConfigError("Material downloads require pip install 'mmfact[hub]'.") from error
    return HfApi, hf_hub_download, snapshot_download


def _read_catalog(root, relative, *, repo_id, revision, local_files_only, token):
    path = root / relative
    if not local_files_only or not path.is_file():
        _, fetch, _ = _hub()
        path = Path(
            fetch(
                repo_id,
                filename=relative,
                repo_type="dataset",
                revision=revision,
                local_files_only=local_files_only,
                token=token,
            )
        )
    try:
        result = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError) as error:
        raise ConfigError(f"Cannot read material catalog {relative}.") from error
    if not isinstance(result, dict):
        raise ConfigError(f"Material catalog must be a mapping: {relative}.")
    return result


def _options(parameters, role):
    """Map public catalog roles to the ordinary model configuration branches."""
    return branch_options(parameters, role)


def _entries(catalog, kind, model):
    entries = catalog.get(kind)
    if not isinstance(entries, list):
        raise ConfigError(f"Material catalog requires a {kind} list.")
    for entry in entries:
        consumers = entry.get("consumers", [])
        roles = [c.get("input", "") for c in consumers if _slug(c.get("method", "")) == model]
        if roles:
            yield entry, roles


def _keep_feature(entry, roles, parameters):
    # Fixed pixels/tokens/frequency remain useful when subsequent layers are trained.
    if entry.get("values", "").startswith("inputs/"):
        return True
    return any(
        _options(parameters, role).get("freeze") is not False
        and not _options(parameters, role).get("augment")
        for role in roles
    )


def _image_branches(parameters):
    for key, value in parameters.items():
        if not isinstance(value, dict):
            continue
        if key in {"image", "clip_image", "pattern_image", "contrastive_image"}:
            yield key, value
        else:
            yield from _image_branches(value)


def _requires_images(parameters, features, prepared_roles=()):
    for branch, options in _image_branches(parameters):
        if (
            branch in prepared_roles
            and options.get("freeze") is not False
            and not options.get("augment")
        ):
            continue
        # Pattern CNNs consume the fixed unnormalized pixels rather than originals.
        role = {
            "image": {"image", "image_encoder"},
            "clip_image": {"clip_image", "clip_image_encoder"},
            "pattern_image": {"pattern_pixels"},
            "contrastive_image": {"contrastive_image_encoder"},
        }[branch]
        matching = [
            entry for entry, roles in features if any(r.split(".", 1)[0] in role for r in roles)
        ]
        pixels = any(e["values"].startswith("inputs/pixels/") for e in matching)
        if options.get("augment") or options.get("flip_probability", 0):
            return True
        if options.get("freeze") is False and not pixels:
            return True
        if not matching:
            return True
    return False


def requires_images(config, catalog):
    """Use the same image requirement for download plans and experiment loading."""
    parameters = config["model"]["parameters"]
    definitions = _definitions()
    slug = _slug(config["model"]["name"])
    features = [
        (entry, roles)
        for entry, roles in _feature_entries(catalog, slug, definitions, parameters)
        if _keep_feature(entry, roles, parameters)
    ]
    return _requires_images(
        parameters, features, definitions.get("prepared_image_roles", {}).get(slug, ())
    )


def _feature_entries(catalog, slug, definitions, parameters=None):
    catalog = resolve_vision_catalog(catalog, slug, parameters or {})
    entries = list(_entries(catalog, "features", slug))
    for source in definitions.get("feature_sources", {}).get(slug, []):
        matches = [
            entry
            for entry in catalog["features"]
            if any(
                consumer.get("method") == source["method"]
                and consumer.get("input") == source["input"]
                for consumer in entry.get("consumers", [])
            )
        ]
        if len(matches) != 1:
            raise ConfigError(f"Shared feature source requires one catalog entry: {source}.")
        entries.append((matches[0], [source["input"]]))
    return entries


def _backbones(parameters, prefix=""):
    for key, value in parameters.items():
        if not isinstance(value, dict):
            continue
        role = f"{prefix}.{key}".strip(".")
        frozen_vit = value.get("backbone") == "vit" and value.get("freeze") is not False
        if value.get("model_name") and (
            (value.get("freeze") is not True and not frozen_vit) or value.get("augment")
        ):
            identifier = value["model_name"]
            yield {
                "name": role,
                "model_name": identifier,
                "revision": value.get("revision"),
                "url": f"https://huggingface.co/{identifier}",
            }
        if (
            value.get("architecture") == "resnet50"
            and value.get("weights") == "IMAGENET1K_V1"
            and value.get("freeze") is not True
        ):
            yield {
                "name": role,
                "directory": "pretrained/torch/hub/checkpoints",
                "url": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
            }
        yield from _backbones(value, role)


def plan_materials(
    model: str,
    dataset: str,
    root: str | Path = "materials",
    *,
    revision: str | None = None,
    repo_id: str | None = None,
    local_files_only: bool = False,
    config: dict | None = None,
    files: Mapping[str, int | None] | Sequence[str] | None = None,
    catalogs: Mapping[str, dict] | None = None,
    token: str | bool | None = None,
) -> MaterialPlan:
    """Select manifests, actual catalog consumers and their dependencies, never matrices.

    Planning fetches only repository metadata and two small catalogs. Supply ``files``
    and ``catalogs`` for a fully offline plan, or use an already downloaded root with
    ``local_files_only=True``. ``root`` is the material root, above ``datasets/``.
    """
    definitions = _definitions()
    if dataset not in definitions["datasets"]:
        raise ConfigError(f"Prepared materials do not support dataset {dataset!r}.")
    slug = _slug(model)
    if slug not in list_models():
        raise ConfigError(f"No classification material plan for model {model!r}.")
    if config is None:
        config = load_config(model, dataset)
    if _slug(config["model"]["name"]) != slug or config["dataset"]["name"] != dataset:
        raise ConfigError("Material plan and model/dataset configuration do not match.")
    parameters = config["model"]["parameters"]
    root = Path(root).expanduser().resolve()
    repo_id, revision = repo_id or definitions["repo_id"], revision or definitions["revision"]
    prefix = f"datasets/{dataset}/"
    loaded = {}
    for name, path in (
        ("features", "feature_catalog.json"),
        ("resources", "training/catalog.json"),
    ):
        loaded[name] = (
            catalogs[name]
            if catalogs is not None and name in catalogs
            else _read_catalog(
                root,
                prefix + path,
                repo_id=repo_id,
                revision=revision,
                local_files_only=local_files_only,
                token=token,
            )
        )
    feature_entries = _feature_entries(loaded["features"], slug, definitions, parameters)
    features = [
        (entry, roles)
        for entry, roles in feature_entries
        if _keep_feature(entry, roles, parameters)
    ]
    selected = {prefix + _relative(path) for path in definitions["common"]}
    patterns = [prefix + _relative(p) for p in definitions["requirements"].get(slug, [])]
    patterns.extend(
        prefix + _relative(p)
        for p in definitions.get("dataset_requirements", {}).get(slug, {}).get(dataset, [])
    )
    for entry, _ in features:
        for key in ("values", "mask", "rows"):
            if entry.get(key):
                selected.add(prefix + _relative(entry[key]))
        # Directory-format producers also carry keys/settings next to the matrix.
        parent = PurePosixPath(_relative(entry["values"])).parent
        if PurePosixPath(entry["values"]).name == "values.npy":
            patterns.append(prefix + str(parent / "*"))
    for entry, _ in _entries(loaded["resources"], "resources", slug):
        if refit_training_resource(slug, parameters, entry.get("type")):
            continue
        directory = _relative(entry["directory"])
        patterns.append(prefix + directory + "/*")
        for path in entry.get("files", {}).values():
            selected.add(prefix + _relative(path))
    raw_images = _requires_images(
        parameters, features, definitions.get("prepared_image_roles", {}).get(slug, ())
    )
    if raw_images:
        patterns.extend(
            prefix + f"data/{split}/images/*" for split in ("train", "validation", "test")
        )
    if files is None:
        if local_files_only:
            inventory = {
                path.relative_to(root).as_posix(): path.stat().st_size
                for path in (root / "datasets" / dataset).rglob("*")
                if path.is_file()
            }
        else:
            api, _, _ = _hub()
            entries = api(token=token).list_repo_tree(
                repo_id,
                path_in_repo=f"datasets/{dataset}",
                recursive=True,
                repo_type="dataset",
                revision=revision,
            )
            inventory = {entry.path: entry.size for entry in entries if hasattr(entry, "size")}
    else:
        inventory = dict(files) if isinstance(files, Mapping) else dict.fromkeys(files)
    for pattern in patterns:
        matches = [p for p in inventory if fnmatch.fnmatchcase(p, pattern)]
        if not matches:
            raise ConfigError(f"Required material is unavailable: {pattern}.")
        selected.update(matches)
    missing = selected - inventory.keys()
    if missing:
        raise ConfigError(f"Required material is unavailable: {', '.join(sorted(missing))}.")
    # Git placeholder guides belong to Git; a download must never overwrite them.
    selected = {
        path for path in selected if PurePosixPath(path).name not in {"README.md", ".gitkeep"}
    }
    for path in selected:
        _relative(path)
        if not path.startswith(prefix):
            raise ConfigError("A material plan must remain inside its selected dataset.")
    sizes = [inventory[p] for p in selected]
    total = sum(sizes) if all(isinstance(n, int) and n >= 0 for n in sizes) else None
    external = [
        *(
            [] if slug == "rdcm" and parameters.get("image", {}).get("backbone") == "vit"
            else definitions["external"].get(slug, [])
        ),
        *_backbones(parameters),
    ]
    if slug in {"dammfnd", "mmdfnd", "logicdm", "event_radar"}:
        text = parameters["text"]
        external.append(
            {
                "name": "text tokenizer",
                "model_name": text["model_name"],
                "revision": text.get("revision"),
                "tokenizer_only": True,
                "directory": f"pretrained/{text['model_name']}",
                "url": f"https://huggingface.co/{text['model_name']}",
            }
        )
    return MaterialPlan(
        config["model"]["name"],
        dataset,
        root,
        repo_id,
        revision,
        tuple(sorted(selected)),
        total,
        raw_images,
        tuple(external),
    )


def download_materials(
    plan: MaterialPlan,
    *,
    max_workers: int = 8,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> Path:
    """Download only a plan's files; external models/auxiliary datasets stay explicit."""
    if not isinstance(max_workers, int) or isinstance(max_workers, bool) or max_workers < 1:
        raise ConfigError("max_workers must be a positive integer.")
    _, _, download = _hub()
    patterns, image_directories = [], set()
    for relative in plan.files:
        if plan.raw_images and "/data/" in relative and "/images/" in relative:
            image_directories.add(relative.split("/images/", 1)[0] + "/images/*")
        else:
            patterns.append(relative)
    # Every raw-image plan includes the entire selected dataset's three image folders.
    # Compact their allowlist so Hub filtering does not compare 26k paths with 12k names.
    patterns.extend(sorted(image_directories))
    download(
        plan.repo_id,
        repo_type="dataset",
        revision=plan.revision,
        local_dir=plan.root,
        allow_patterns=patterns,
        ignore_patterns=["README.md", "*/README.md", "*/.gitkeep"],
        max_workers=max_workers,
        token=token,
        local_files_only=local_files_only,
    )
    missing = [relative for relative in plan.files if not (plan.root / relative).is_file()]
    if missing:
        raise ConfigError(f"Download is incomplete: {', '.join(missing[:5])}.")
    return plan.root
