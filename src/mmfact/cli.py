"""Command line entry point; run `mmfact --help` after installation."""

import argparse
import json
import sys

from mmfact.quick_start import quick_start


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="MMFact multimodal misinformation detection toolbox"
    )
    parser.add_argument("--model", "-m", default="EANN")
    parser.add_argument("--dataset", "-d", default="weibo17")
    parser.add_argument("--config", action="append", default=[], metavar="YAML")
    parser.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate data/config without a model"
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List implemented built-in adapters"
    )
    parser.add_argument("--mode", choices=["train", "evaluate", "preprocess"], default=None)
    parser.add_argument("--checkpoint", help="Framework checkpoint for evaluate mode")
    parser.add_argument("--input-mode", choices=["raw", "prepared"])
    parser.add_argument("--materials-root", help="Existing materials directory")
    parser.add_argument(
        "--vision-backbone", choices=["vit", "resnet"],
        help="Ordinary supervised visual backbone; defaults to ViT in supported models",
    )
    material_action = parser.add_mutually_exclusive_group()
    material_action.add_argument(
        "--list-materials",
        action="store_true",
        help="List the files required by a model",
    )
    material_action.add_argument(
        "--download-materials",
        action="store_true",
        help="Download only the selected method's materials",
    )
    parser.add_argument(
        "--material-revision", help="Override the fixed Hugging Face material revision"
    )
    parser.add_argument("--download-workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.vision_backbone is not None:
        args.overrides.append(f"model.vision_backbone={args.vision_backbone}")
    try:
        if args.list_materials or args.download_materials:
            from mmfact.config import load_config
            from mmfact.materials.download import download_materials, plan_materials
            from mmfact.materials.encoding import apply_encoding_profile

            config = apply_encoding_profile(
                load_config(args.model, args.dataset, args.config, args.overrides)
            )
            plan = plan_materials(
                args.model,
                args.dataset,
                args.materials_root or "materials",
                revision=args.material_revision,
                config=config,
            )
            if args.download_materials and not args.dry_run:
                download_materials(plan, max_workers=args.download_workers)
            summary = plan.to_dict()
            images = [path for path in summary["files"] if "/images/" in path]
            summary["file_count"] = len(summary["files"])
            summary["image_file_count"] = len(images)
            summary["files"] = [path for path in summary["files"] if "/images/" not in path]
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        if args.list_models:
            from mmfact.registry import list_models

            print(json.dumps({"implemented_models": list_models()}))
            return 0
        result = quick_start(
            args.model,
            args.dataset,
            args.config,
            args.overrides,
            dry_run=args.dry_run,
            mode=args.mode or "train",
            checkpoint=args.checkpoint,
            input_mode=args.input_mode,
            materials_root=args.materials_root,
        )
    except (ValueError, FileNotFoundError, ImportError, NotImplementedError) as error:
        print(f"MMFact: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
