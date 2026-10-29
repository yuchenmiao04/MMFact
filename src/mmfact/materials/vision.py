"""Bind ordinary supervised visual branches to their explicitly selected backbone.

The published ViT matrices intentionally have no legacy consumers. Resolve their
roles here for planning, configuration and runtime so an old ResNet consumer can
never silently win over a requested ViT branch.
"""

from copy import deepcopy

from mmfact.config import ConfigError


def branch_options(parameters, role):
    first = role.split(".", 1)[0]
    key = {
        "text_encoder": "text",
        "image_encoder": "image",
        "instance_encoder": "image",
        "contrastive_image_encoder": "contrastive_image",
        "clip_text_encoder": "clip_text",
        "clip_image_encoder": "clip_image",
        "reference_encoder": "reference_text",
        "pattern_pixels": "pattern_image",
    }.get(first, first)
    branch = parameters.get(key)
    return branch if isinstance(branch, dict) else {}


def _slug(name):
    return name.lower().replace("-", "_")


def _visual_roles(method, parameters):
    """Ordinary branches only; paired CLIP and detector/frequency roles stay separate."""
    method = _slug(method)
    image = parameters.get("image", {})
    if image.get("backbone") == "vit":
        role = "image" if method in {"eann", "mvae", "att_rnn"} else "image_encoder"
        yield role, image
        if method == "rdcm":
            yield "instance_encoder", {**image, "output": "logits"}
    contrastive = parameters.get("contrastive_image", {})
    if contrastive.get("backbone") == "vit":
        yield "contrastive_image_encoder", contrastive


def _family(method, role, options):
    method = _slug(method)
    if options.get("backbone") != "vit" or role not in {
        "image", "image_encoder", "contrastive_image_encoder", "instance_encoder"
    }:
        return None
    if options.get("freeze") is False or options.get("augment"):
        return None
    if method not in {"btic", "logicdm", "msaca"} and options.get("image_size", 224) != 224:
        raise ConfigError("Shared frozen ViT features use 224px images.")
    if role == "instance_encoder" or options.get("output") == "logits":
        return "logits1000"
    if method == "btic":
        if options.get("grid_size") != 8 or options.get("image_size") != 640:
            raise ConfigError("Prepared BTIC ViT features require 64 independent crops from 640px.")
        return "patch8x8_independent"
    if method == "logicdm":
        if options.get("grid_size") != 7 or options.get("image_size") != 224:
            raise ConfigError("Prepared LogicDM ViT features require 49 independent crops from 224px.")
        return "patch7x7_independent"
    if method == "msaca":
        if list(options.get("scales", ())) != [224, 112, 64]:
            raise ConfigError("Prepared MSACA ViT features require ordered scales 224/112/64.")
        return "scales224_112_64"
    if options.get("output") == "spatial":
        if options.get("grid_size", 4) != 4:
            raise ConfigError("The shared frozen ViT release contains only the 4x4 spatial grid.")
        return "spatial4x4"
    if options.get("output", "avgpool") not in {"avgpool", "cls"}:
        raise ConfigError("Unsupported prepared supervised ViT output.")
    return "cls768"


def vision_feature(entries, method, role, options):
    """Return a checked, normalized matrix entry, or None for another/native branch."""
    family = _family(method, role, options)
    if family is None:
        return None
    name = f"supervised_vit_{family}"
    matches = [entry for entry in entries if entry.get("name") == name]
    if len(matches) != 1:
        raise ConfigError(
            f"Selected ViT branch {method}.{role} requires {name!r}; "
            "download materials from the ViT release instead of falling back to ResNet."
        )
    entry = deepcopy(matches[0])
    settings = entry.get("settings", {})
    suffix = {
        "cls768": [768], "logits1000": [1000], "spatial4x4": [768, 4, 4],
        "patch8x8_independent": [64, 768], "patch7x7_independent": [49, 768],
        "scales224_112_64": [3, 768],
    }[family]
    if (
        entry.get("input_kind") != "image"
        or entry.get("shape", [])[1:] != suffix
        or entry.get("dtype") != "float32"
        or entry.get("values") != f"features/image/vit_supervised/{family}.npy"
        or settings.get("output_family") != family
        or settings.get("model_id") != options.get("model_name")
        or settings.get("model_revision") != options.get("revision")
        or settings.get("frozen") is not True
        or settings.get("eval") is not True
        or settings.get("normalization_mean") != [0.5, 0.5, 0.5]
        or settings.get("normalization_std") != [0.5, 0.5, 0.5]
        or settings.get("pixel_order") != "RGB"
        or options.get("center_crop", False) is not False
    ):
        raise ConfigError(f"Prepared {name!r} does not match the selected supervised ViT recipe.")
    settings.update(
        model_name=settings["model_id"], revision=settings["model_revision"], backbone="vit"
    )
    if family == "spatial4x4":
        settings["grid_size"] = 4
    if family.startswith("patch"):
        settings["grid_size"] = 8 if family.startswith("patch8") else 7
    if family == "scales224_112_64":
        settings["scales"] = [224, 112, 64]
    return entry


def resolve_vision_catalog(catalog, method, parameters):
    """Give selected frozen ViT entries virtual consumers without editing Hub files."""
    replacements = list(_visual_roles(method, parameters))
    if not replacements:
        return catalog
    result = deepcopy(catalog)
    roles = {role for role, _ in replacements}
    for entry in result["features"]:
        entry["consumers"] = [
            consumer for consumer in entry.get("consumers", [])
            if not (_slug(consumer.get("method", "")) == _slug(method)
                    and consumer.get("input") in roles)
        ]
    for role, options in replacements:
        selected = vision_feature(catalog["features"], method, role, options)
        if selected is None:
            continue
        target = next(entry for entry in result["features"] if entry["name"] == selected["name"])
        target["settings"] = selected["settings"]
        target["consumers"].append({"method": method, "input": role})
    return result


def refit_training_resource(model, parameters, name):
    """Only CSFND's published visual clusters depend on the replaced backbone."""
    return (
        _slug(model) == "csfnd" and name == "csfnd_initial_clusters"
        and parameters.get("image", {}).get("backbone") == "vit"
    )
