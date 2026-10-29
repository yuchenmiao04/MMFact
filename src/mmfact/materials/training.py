"""Persist fixed training-only resources without serializing learned detectors."""

import inspect
from importlib.metadata import version

import torch

from mmfact.data.cache_policy import PreparedInputError, cache_policy, prepared_only
from mmfact.data.features import (
    file_digest,
    fingerprint,
    load_resource,
    resource_path,
    state_digest,
)
from mmfact.materials.vision import refit_training_resource


def encoder_identity(encoder):
    if hasattr(encoder, "matrix"):
        return {"spec": encoder.encoding_spec, "catalog_sha256": encoder.matrix.identity}
    return {
        "spec": encoder.encoding_spec,
        "weights_sha256": state_digest(encoder.network),
        "versions": {package: version(package) for package in encoder.packages},
    }


def hub_training_resource(config, name, *, expected=None):
    """Find an exported train-only resource without fitting or rebuilding it."""
    from mmfact.materials.encoders import hub_inputs, material_store

    if not hub_inputs(config):
        return None
    if refit_training_resource(config["model"]["name"], config["model"]["parameters"], name):
        # The released CSFND assignments were fitted on ResNet. ViT runs fit
        # new train-only assignments from the selected immutable matrices.
        return None
    store = material_store(config)
    entries = [entry for entry in store.training_catalog if entry.get("type") == name]
    if len(entries) != 1:
        raise PreparedInputError(f"Expected one exported training resource {name!r}.")
    resource = store.training(entries[0]["name"])
    settings = resource.settings
    if settings.get("fitting_split") != "train":
        raise PreparedInputError(f"Prepared training resource {name!r} must be fitted on train.")
    for key, value in (expected or {}).items():
        if settings.get(key) != value:
            raise PreparedInputError(f"Prepared training resource {name!r} disagrees on {key!r}.")
    return resource


def resource_tensor(resource, name, *, dtype=None):
    """Copy a small exported fitted resource into checkpoint-owned CPU state."""
    value = torch.from_numpy(resource.array(name).copy())
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise PreparedInputError(f"Prepared training resource {name!r} is nonfinite.")
    return value.to(dtype=dtype)


def resource_rows(resource, key=None):
    """Return an exported row index in matrix order, validating its coverage."""
    rows = resource.jsonl("rows")
    if any(type(row.get("row")) is not int for row in rows):
        raise PreparedInputError("Prepared training rows require integer row positions.")
    rows = sorted(rows, key=lambda row: row["row"])
    if [row["row"] for row in rows] != list(range(len(rows))):
        raise PreparedInputError("Prepared training rows must cover every position once.")
    if key is not None:
        keys = [row.get(key) for row in rows]
        if any(not isinstance(value, str) or not value for value in keys) or len(set(keys)) != len(
            keys
        ):
            raise PreparedInputError(f"Prepared training rows require unique {key!r} identities.")
    return rows


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_cpu(child) for child in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"Prepared resources cannot contain {type(value).__name__} objects.")


def load_prepared_resource(model, name, identity, build, *, encoders):
    """Use native resources when preprocessing is bound, otherwise build in memory.

    ``identity`` is lazy so ordinary direct model use need not expose any cache
    interface. Only the caller's explicitly selected immutable state is saved.
    """
    if hasattr(model, "prepared_resources"):
        model.prepared_resources.pop(name, None)
    options = model.config.get("preprocessing", {})
    refit = options.get("layout") == "hub" and refit_training_resource(
        model.config["model"]["name"], model.config["model"]["parameters"], name
    )
    if options.get("layout") == "hub" and not refit:
        raise PreparedInputError(
            f"Hub training resource {name!r} must be read from training/catalog.json; "
            "the training loop does not rebuild exported resources."
        )
    if not refit and (
        not options.get("enabled")
        or not encoders
        or not all(getattr(encoder, "preprocessing", None) is not None for encoder in encoders)
    ):
        if prepared_only(options.get("mode")):
            raise PreparedInputError(
                f"Prepared training resource {name!r} requires bound feature caches."
            )
        return build()
    roots = {
        str(encoder.preprocessing.root)
        for encoder in encoders
        if hasattr(encoder.preprocessing, "root")
    }
    if len(roots) > 1:
        raise ValueError("Training resource encoders must share one preprocessing cache root.")
    root = next(iter(roots)) if roots else options["cache_dir"]
    spec = {
        "resource": name,
        "schema": 1,
        "identity": identity(),
        "source_sha256": file_digest(inspect.getfile(type(model))),
        "helper_sha256": file_digest(__file__),
        "versions": {package: version(package) for package in ("torch", "numpy", "scikit-learn")},
    }
    path = resource_path(root, spec)
    reused = path.is_file()
    # Only these small, training-only resources may be rebuilt after a source
    # correction. Their encoders retain their own prepared-only cache policy.
    rebuild = refit or (prepared_only(options.get("mode")) and name in {
        "mhr_reference_graph",
        "csfnd_initial_clusters",
    })
    if rebuild:
        with cache_policy("raw"):
            value = load_resource(root, spec, lambda: _cpu(build()), mode="raw")
    else:
        value = load_resource(root, spec, lambda: _cpu(build()), mode=options.get("mode"))
    if not hasattr(model, "prepared_resources"):
        model.prepared_resources = {}
    model.prepared_resources[name] = {
        "path": str(path),
        "reused": reused,
        "identity_sha256": fingerprint(spec),
        "fitting_split": "train",
    }
    return value
