"""Extract a runnable lightweight Waymo/trajdata cache from an index CSV.

The objective-metric scripts only need the scenes listed in
``data/waymo_data index.csv``. This script copies those scene caches and maps
from the full InterHub/trajdata cache into this project, then writes a filtered
``scenes_list.dill`` so the lightweight cache does not reference missing
scenes.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_CSV = PROJECT_ROOT / "data" / "waymo_data index.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "waymo_lightweight_cache"

DEFAULT_FULL_CACHE_MAP = {
    "waymo_0-299": "/home/zjr/\u6587\u6863/InterHub_cache/waymo_0-299",
    "waymo_300-499": "/home/zjr/\u6587\u6863/InterHub_cache/waymo_300-499",
    "waymo_500-799": "/home/zjr/\u6587\u6863/InterHub_cache/waymo_500-799",
    "waymo_800-999": "/home/zjr/\u6587\u6863/InterHub_cache/waymo_800-999",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the Waymo/trajdata scenes referenced by an index CSV into a "
            "project-local lightweight cache."
        )
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=DEFAULT_INDEX_CSV,
        help=f"Index CSV to read. Default: {DEFAULT_INDEX_CSV}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Destination lightweight cache root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "Root containing cache folders such as waymo_0-299, waymo_300-499. "
            "When set, it overrides the built-in full-cache paths."
        ),
    )
    parser.add_argument(
        "--source-map",
        action="append",
        default=[],
        metavar="FOLDER=PATH",
        help=(
            "Override one folder source cache path. Can be passed multiple "
            "times, e.g. --source-map waymo_0-299=E:\\waymo_0-299"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace destination scene/map files and directories when they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate required source files and print the copy plan only.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Copy available scenes and report missing scenes instead of failing.",
    )
    return parser.parse_args()


def parse_source_map(items: list[str]) -> dict[str, Path]:
    source_map: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --source-map value {item!r}; expected FOLDER=PATH.")
        folder, path = item.split("=", 1)
        folder = folder.strip()
        if not folder:
            raise ValueError(f"Invalid --source-map value {item!r}; folder is empty.")
        source_map[folder] = Path(path).expanduser()
    return source_map


def collect_required_scenes(index_csv: Path) -> dict[tuple[str, str], set[int]]:
    required: dict[tuple[str, str], set[int]] = defaultdict(set)
    with index_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing_columns = {"dataset", "folder", "scenario_idx"} - set(reader.fieldnames or [])
        if missing_columns:
            columns = ", ".join(sorted(missing_columns))
            raise ValueError(f"{index_csv} is missing required column(s): {columns}")

        for row_number, row in enumerate(reader, start=2):
            dataset = (row.get("dataset") or "").strip()
            folder = (row.get("folder") or "").strip()
            scenario_text = (row.get("scenario_idx") or "").strip()
            if not dataset or not folder or not scenario_text:
                raise ValueError(
                    f"{index_csv}:{row_number} has empty dataset/folder/scenario_idx."
                )
            try:
                scenario_idx = int(float(scenario_text))
            except ValueError as exc:
                raise ValueError(
                    f"{index_csv}:{row_number} has invalid scenario_idx {scenario_text!r}."
                ) from exc
            required[(folder, dataset)].add(scenario_idx)
    return required


def resolve_source_cache(folder: str, args: argparse.Namespace, overrides: dict[str, Path]) -> Path:
    if folder in overrides:
        return overrides[folder]
    if args.source_root is not None:
        return args.source_root.expanduser() / folder
    if folder not in DEFAULT_FULL_CACHE_MAP:
        raise KeyError(
            f"No source cache path configured for folder {folder!r}. "
            "Use --source-root or --source-map."
        )
    return Path(DEFAULT_FULL_CACHE_MAP[folder]).expanduser()


def map_stem(dataset: str, scenario_idx: int) -> str:
    return f"{dataset}_{scenario_idx}"


def required_map_paths(source_env_dir: Path, dataset: str, scenario_idx: int) -> list[Path]:
    stem = map_stem(dataset, scenario_idx)
    maps_dir = source_env_dir / "maps"
    return [
        maps_dir / f"{stem}.pb",
        maps_dir / f"{stem}_kdtrees.dill",
        maps_dir / f"{stem}_2.00px_m.zarr",
        maps_dir / f"{stem}_2.00px_m.dill",
    ]


def related_map_paths(source_env_dir: Path, dataset: str, scenario_idx: int) -> list[Path]:
    stem = map_stem(dataset, scenario_idx)
    maps_dir = source_env_dir / "maps"
    return sorted(
        path
        for path in maps_dir.iterdir()
        if path.name.startswith(f"{stem}.") or path.name.startswith(f"{stem}_")
    )


def copy_directory(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and overwrite:
        shutil.rmtree(dst)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def copy_path(src: Path, dst: Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        copy_directory(src, dst, overwrite)
        return
    if dst.exists() and overwrite:
        dst.unlink()
    if not dst.exists():
        shutil.copy2(src, dst)


def get_scene_data_idx(scene: object) -> int | None:
    for attr in ("data_idx", "raw_data_idx"):
        value = getattr(scene, attr, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    name = getattr(scene, "name", "")
    match = re.fullmatch(r"scene_(\d+)", str(name))
    return int(match.group(1)) if match else None


def write_filtered_scenes_list(
    source_env_dir: Path,
    dest_env_dir: Path,
    scenario_ids: set[int],
) -> int:
    try:
        import dill
    except ImportError as exc:
        raise RuntimeError(
            "The Python package 'dill' is required to filter scenes_list.dill. "
            "Install the same environment used by trajdata/InterHub before running."
        ) from exc

    source_path = source_env_dir / "scenes_list.dill"
    with source_path.open("rb") as f:
        scenes = dill.load(f)

    filtered = [scene for scene in scenes if get_scene_data_idx(scene) in scenario_ids]
    found = {get_scene_data_idx(scene) for scene in filtered}
    missing = scenario_ids - {idx for idx in found if idx is not None}
    if missing:
        preview = ", ".join(str(idx) for idx in sorted(missing)[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"{source_path} does not contain {len(missing)} required scene(s): "
            f"{preview}{suffix}"
        )

    dest_env_dir.mkdir(parents=True, exist_ok=True)
    with (dest_env_dir / "scenes_list.dill").open("wb") as f:
        dill.dump(filtered, f)
    return len(filtered)


def validate_group(source_env_dir: Path, dataset: str, scenario_ids: set[int]) -> list[str]:
    missing: list[str] = []
    if not (source_env_dir / "scenes_list.dill").is_file():
        missing.append(str(source_env_dir / "scenes_list.dill"))
    for scenario_idx in sorted(scenario_ids):
        scene_dir = source_env_dir / f"scene_{scenario_idx}"
        if not scene_dir.is_dir():
            missing.append(str(scene_dir))
        for map_path in required_map_paths(source_env_dir, dataset, scenario_idx):
            if not map_path.exists():
                missing.append(str(map_path))
    return missing


def extract_group(
    folder: str,
    dataset: str,
    scenario_ids: set[int],
    args: argparse.Namespace,
    source_overrides: dict[str, Path],
) -> dict:
    source_cache = resolve_source_cache(folder, args, source_overrides)
    source_env_dir = source_cache / dataset
    dest_env_dir = args.output_root / folder / dataset

    missing = validate_group(source_env_dir, dataset, scenario_ids)
    if missing and not args.allow_missing:
        preview = "\n  ".join(missing[:20])
        suffix = "\n  ..." if len(missing) > 20 else ""
        raise FileNotFoundError(
            f"Missing {len(missing)} required source path(s) for {folder}/{dataset}:\n"
            f"  {preview}{suffix}"
        )

    available_ids = {
        scenario_idx
        for scenario_idx in scenario_ids
        if (source_env_dir / f"scene_{scenario_idx}").is_dir()
        and all(map_path.exists() for map_path in required_map_paths(source_env_dir, dataset, scenario_idx))
    }

    if args.dry_run:
        return {
            "folder": folder,
            "dataset": dataset,
            "source": str(source_cache),
            "destination": str(args.output_root / folder),
            "requested_scenes": len(scenario_ids),
            "available_scenes": len(available_ids),
            "missing_paths": missing,
        }

    copied_map_items = 0
    for scenario_idx in sorted(available_ids):
        copy_directory(
            source_env_dir / f"scene_{scenario_idx}",
            dest_env_dir / f"scene_{scenario_idx}",
            args.overwrite,
        )
        for map_path in related_map_paths(source_env_dir, dataset, scenario_idx):
            copy_path(map_path, dest_env_dir / "maps" / map_path.name, args.overwrite)
            copied_map_items += 1

    filtered_count = write_filtered_scenes_list(source_env_dir, dest_env_dir, available_ids)
    return {
        "folder": folder,
        "dataset": dataset,
        "source": str(source_cache),
        "destination": str(args.output_root / folder),
        "requested_scenes": len(scenario_ids),
        "copied_scenes": len(available_ids),
        "copied_map_items": copied_map_items,
        "filtered_scenes_list": filtered_count,
        "missing_paths": missing,
    }


def main() -> int:
    args = parse_args()
    args.index_csv = args.index_csv.resolve()
    args.output_root = args.output_root.resolve()
    source_overrides = parse_source_map(args.source_map)

    required = collect_required_scenes(args.index_csv)
    total_scenes = sum(len(scene_ids) for scene_ids in required.values())
    print(f"Index: {args.index_csv}")
    print(f"Unique dataset/folder groups: {len(required)}")
    print(f"Unique scenes requested: {total_scenes}")
    print(f"Output root: {args.output_root}")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "index_csv": str(args.index_csv),
        "output_root": str(args.output_root),
        "dry_run": args.dry_run,
        "groups": [],
    }

    for (folder, dataset), scenario_ids in sorted(required.items()):
        print(f"\n[{folder}/{dataset}] {len(scenario_ids)} scene(s)")
        group = extract_group(folder, dataset, scenario_ids, args, source_overrides)
        manifest["groups"].append(group)
        if group.get("missing_paths"):
            print(f"  missing paths: {len(group['missing_paths'])}")
        if args.dry_run:
            print(f"  source: {group['source']}")
            print(f"  available: {group['available_scenes']}/{group['requested_scenes']}")
        else:
            print(f"  copied: {group['copied_scenes']}/{group['requested_scenes']}")
            print(f"  destination: {group['destination']}")

    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = args.output_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nWrote manifest: {manifest_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
