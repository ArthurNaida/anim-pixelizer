from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from fixer import run_fixer
from llm import generate_frames
from postprocess import run_postprocess
from preprocess import PreprocessOptions, slice_sheet
from review import review_batch


DEFAULT_CONFIG_PATH = Path("config.json")
STAGE_NAMES = ("llm", "preprocess", "fixer", "postprocess", "review")


def load_config(config_path: Path) -> dict:
    """Load JSON configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_images(target: Path) -> list[Path]:
    """Collect a single image file or sorted image files from a directory."""
    if target.is_file():
        return [target]
    if target.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        files = sorted(p for p in target.iterdir() if p.is_file() and p.suffix.lower() in exts)
        if not files:
            raise FileNotFoundError(f"No image files found in directory: {target}")
        return files
    raise FileNotFoundError(f"Path does not exist: {target}")


def resolve_output_paths(
    inputs: list[Path],
    output: Path | None,
    default_dir: Path,
    suffix: str,
) -> list[Path]:
    """Resolve output file paths for single-file or batch processing."""
    if output is not None and output.exists() and output.is_dir():
        output.mkdir(parents=True, exist_ok=True)
        return [output / f"{inp.stem}{suffix}.png" for inp in inputs]

    if len(inputs) == 1 and output is not None and output.suffix:
        output.parent.mkdir(parents=True, exist_ok=True)
        return [output]

    out_dir = output if output is not None else default_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return [out_dir / f"{inp.stem}{suffix}.png" for inp in inputs]

def _build_preprocess_options(cfg: dict) -> PreprocessOptions:
    """Build PreprocessOptions from config dictionary."""
    p_cfg = cfg.get("preprocess", {})
    return PreprocessOptions(
        pad=int(p_cfg.get("pad", 2)),
        min_size=int(p_cfg.get("min_size", 16)),
        erase_dividers=bool(p_cfg.get("erase_dividers", True)),
        divider_min_coverage=float(p_cfg.get("divider_min_coverage", 0.85)),
        divider_max_luma=float(p_cfg.get("divider_max_luma", 80.0)),
        divider_max_channel_diff=int(p_cfg.get("divider_max_channel_diff", 45)),
        divider_target_color=p_cfg.get("divider_target_color"),
        divider_color_tolerance=float(p_cfg.get("divider_color_tolerance", 35.0)),
        divider_thickness=int(p_cfg.get("divider_thickness", 1)),
        bg_fuzz_percent=float(p_cfg.get("bg_fuzz_percent", 15.0)),
        bg_color=p_cfg.get("bg_color"),
        bg_seed_border=bool(p_cfg.get("bg_seed_border", True)),
        clean_bg_pockets=bool(p_cfg.get("clean_bg_pockets", True)),
        pocket_fuzz_percent=float(p_cfg.get("pocket_fuzz_percent", 18.0)),
        pocket_max_opaque_neighbors=int(p_cfg.get("pocket_max_opaque_neighbors", 3)),
        merge_shards=bool(p_cfg.get("merge_shards", True)),
        shard_merge_ratio=float(p_cfg.get("shard_merge_ratio", 0.45)),
        max_shard_gap=float(p_cfg.get("max_shard_gap", 96.0)),
        clean_neutral_noise=bool(p_cfg.get("clean_neutral_noise", True)),
        max_channel_diff=int(p_cfg.get("max_channel_diff", 18)),
        min_gray=int(p_cfg.get("min_gray", 120)),
        max_gray=int(p_cfg.get("max_gray", 255)),
        min_component_pixels=int(p_cfg.get("min_component_pixels", 25)),
        alpha_threshold=int(p_cfg.get("alpha_threshold", 20)),
        expected_frames=p_cfg.get("expected_frames"),
        strict_frames=bool(p_cfg.get("strict_frames", True)),
        bg_mode=str(p_cfg.get("bg_mode", "transparent")),
        solid_bg_color=str(p_cfg.get("solid_bg_color", "#000000")),
    )

def execute_preprocess(inputs: list[Path], output: Path | None, cfg: dict, workdir: Path) -> list[Path]:
    """Validate, slice sprite sheets into frames, and apply background rules."""
    opts = _build_preprocess_options(cfg)
    out_dir = output if output is not None else (workdir / "preprocess")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_frames: list[Path] = []
    for sheet in inputs:
        frames = slice_sheet(sheet_path=sheet, output_dir=out_dir, options=opts)
        for fr in frames:
            print(f"[preprocess] {sheet} -> {fr}")
        all_frames.extend(frames)
    return all_frames


def execute_fixer(inputs: list[Path], output: Path | None, cfg: dict, workdir: Path) -> list[Path]:
    """Run pixel-art-fixer and optional post-command on input images."""
    fixer_cfg = cfg.get("fixer", {})
    target_size = tuple(cfg["target_size"]) if cfg.get("target_size") else None
    outputs = resolve_output_paths(inputs, output, workdir / "fixer", "_fixed")

    results: list[Path] = []
    for inp, out in zip(inputs, outputs):
        res = run_fixer(
            input_path=inp,
            output_path=out,
            executable=fixer_cfg.get("executable", "pixelfixer"),
            subcommand=fixer_cfg.get("subcommand", "process"),
            extra_args=fixer_cfg.get("extra_args", []),
            command_template=fixer_cfg.get("command"),
            post_command_template=fixer_cfg.get("post_command"),
            enable_post_command=fixer_cfg.get("enable_post_command", False),
            palette_image=fixer_cfg.get("palette_image"),
            target_size=target_size,
            resize_to_target=fixer_cfg.get("resize_to_target", False),
            fallback_nearest_resize=fixer_cfg.get("fallback_nearest_resize", True),
        )
        print(f"[fixer] {inp} -> {res}")
        results.append(res)
    return results


def execute_postprocess(inputs: list[Path], output: Path | None, cfg: dict, workdir: Path) -> list[Path]:
    """Run deterministic post-processing (alpha, noise, palette, outline) on frames."""
    pp_cfg = cfg.get("postprocess", {})
    outputs = resolve_output_paths(inputs, output, workdir / "postprocess", "_clean")

    results: list[Path] = []
    for inp, out in zip(inputs, outputs):
        res = run_postprocess(
            input_path=inp,
            output_path=out,
            palette_hex=cfg.get("palette", []),
            alpha_threshold=pp_cfg.get("alpha_threshold", 128),
            min_neighbors=pp_cfg.get("min_neighbors", 1),
            noise_window=pp_cfg.get("noise_window", 3),
            noise_max_density=pp_cfg.get("noise_max_density", 0.35),
            paint_outline=pp_cfg.get("paint_outline", True),
            outline_color_hex=pp_cfg.get("outline_color", "#141013"),
            snap_palette=pp_cfg.get("snap_palette", True),
        )
        print(f"[postprocess] {inp} -> {res}")
        results.append(res)
    return results

def _add_preprocess_override_args(parser: argparse.ArgumentParser) -> None:
    """Attach CLI override flags for preprocess stage."""
    parser.add_argument("--pad", type=int, default=None, help="Padding around extracted frame")
    parser.add_argument("--min-size", type=int, default=None, help="Minimum frame width/height")
    parser.add_argument("--fuzz", type=float, default=None, help="Background flood-fill fuzz percent")
    parser.add_argument("--bg-color", type=str, default=None, help="Explicit background hex color (#RRGGBB)")
    parser.add_argument("--no-dividers", action="store_true", help="Disable divider line detection and erasure")
    parser.add_argument("--divider-min-coverage", type=float, default=None, help="Min line coverage ratio (0..1)")
    parser.add_argument("--divider-max-luma", type=float, default=None, help="Max luma for dark divider lines")
    parser.add_argument("--divider-thickness", type=int, default=None, help="Erase thickness for divider lines")
    parser.add_argument("--no-clean-pockets", action="store_true", help="Disable isolated background pocket cleanup")
    parser.add_argument("--pocket-fuzz", type=float, default=None, help="Fuzz percent for isolated bg pockets")
    parser.add_argument("--no-merge-shards", action="store_true", help="Disable merging nearby shard boxes")
    parser.add_argument(
        "--shard-merge-ratio",
        type=float,
        default=None,
        help="Merge shards closer than ratio * average nearest-neighbor distance",
    )
    parser.add_argument("--max-shard-gap", type=float, default=None, help="Max pixel gap to merge shard boxes")
    parser.add_argument("--expected-frames", type=int, default=None, help="Expected frame count for validation")
    parser.add_argument("--no-strict-frames", action="store_true", help="Do not raise on frame count mismatch")
    parser.add_argument("--bg-mode", type=str, choices=["transparent", "solid"], default=None, help="Output bg mode")
    parser.add_argument("--solid-bg-color", type=str, default=None, help="Solid background hex color")


def _apply_preprocess_overrides(cfg: dict, args: argparse.Namespace) -> None:
    """Apply preprocess CLI overrides onto loaded config."""
    p_cfg = cfg.setdefault("preprocess", {})
    mapping = {
        "pad": "pad",
        "min_size": "min_size",
        "fuzz": "bg_fuzz_percent",
        "bg_color": "bg_color",
        "divider_min_coverage": "divider_min_coverage",
        "divider_max_luma": "divider_max_luma",
        "divider_thickness": "divider_thickness",
        "pocket_fuzz": "pocket_fuzz_percent",
        "shard_merge_ratio": "shard_merge_ratio",
        "max_shard_gap": "max_shard_gap",
        "expected_frames": "expected_frames",
        "bg_mode": "bg_mode",
        "solid_bg_color": "solid_bg_color",
    }
    for arg_name, cfg_key in mapping.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            p_cfg[cfg_key] = val

    if getattr(args, "no_dividers", False):
        p_cfg["erase_dividers"] = False
    if getattr(args, "no_clean_pockets", False):
        p_cfg["clean_bg_pockets"] = False
    if getattr(args, "no_merge_shards", False):
        p_cfg["merge_shards"] = False
    if getattr(args, "no_strict_frames", False):
        p_cfg["strict_frames"] = False


def _add_fixer_override_args(parser: argparse.ArgumentParser) -> None:
    """Attach CLI override flags for fixer stage."""
    parser.add_argument("--size", type=int, nargs=2, metavar=("W", "H"), help="Target width and height")
    parser.add_argument(
        "--fixer-arg",
        action="append",
        default=None,
        help="Extra argument passed to fixer (can be specified multiple times)",
    )
    parser.add_argument("--palette-image", type=str, default=None, help="Path to palette image for remap")
    parser.add_argument("--post-cmd", action="store_true", help="Enable fixer post_command")
    parser.add_argument("--no-post-cmd", action="store_true", help="Disable fixer post_command")
    parser.add_argument(
        "--resize-to-target",
        action="store_true",
        help="Force nearest-neighbor resize to target_size after fixer",
    )


def _apply_fixer_overrides(cfg: dict, args: argparse.Namespace) -> None:
    """Apply fixer CLI overrides onto loaded config."""
    fixer_cfg = cfg.setdefault("fixer", {})
    if getattr(args, "size", None):
        cfg["target_size"] = list(args.size)
    if getattr(args, "fixer_arg", None) is not None:
        fixer_cfg["extra_args"] = args.fixer_arg
    if getattr(args, "palette_image", None):
        fixer_cfg["palette_image"] = args.palette_image
    if getattr(args, "post_cmd", False):
        fixer_cfg["enable_post_command"] = True
    if getattr(args, "no_post_cmd", False):
        fixer_cfg["enable_post_command"] = False
    if getattr(args, "resize_to_target", False):
        fixer_cfg["resize_to_target"] = True


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser using stage names."""
    parser = argparse.ArgumentParser(
        prog="anim-pixelizer",
        description="Deterministic pixel-art animation pipeline: llm -> preprocess -> fixer -> postprocess -> review.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.json (default: ./config.json)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging for pipeline stages",
    )
    parser.add_argument(
        "-w",
        "--workdir",
        type=Path,
        default=None,
        help="Working directory for intermediate artifacts (overrides config.json)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # llm stage (stub)
    p_llm = subparsers.add_parser("llm", help="Generate raw sprite sheet/frames via LLM (stub)")
    p_llm.add_argument("-r", "--ref", type=Path, default=None, help="Reference sprite path")
    p_llm.add_argument("-p", "--prompt", type=str, required=True, help="Animation prompt")
    p_llm.add_argument("-o", "--output", type=Path, default=None, help="Output directory")
    p_llm.add_argument("-n", "--frames", type=int, default=4, help="Number of frames")

    # preprocess stage
    p_pre = subparsers.add_parser("preprocess", help="Validate, slice sprite sheets, and process background")
    p_pre.add_argument("input", type=Path, help="Input sprite sheet file or directory")
    p_pre.add_argument("-o", "--output", type=Path, default=None, help="Output directory for sliced frames")
    _add_preprocess_override_args(p_pre)

    # fixer stage
    p_fix = subparsers.add_parser("fixer", help="Run pixel-art-fixer on frames")
    p_fix.add_argument("input", type=Path, help="Input image file or directory")
    p_fix.add_argument("-o", "--output", type=Path, default=None, help="Output file or directory")
    _add_fixer_override_args(p_fix)

    # postprocess stage
    p_post = subparsers.add_parser("postprocess", help="Clean noise, snap palette, and paint contours")
    p_post.add_argument("input", type=Path, help="Input image file or directory")
    p_post.add_argument("-o", "--output", type=Path, default=None, help="Output file or directory")
    p_post.add_argument("--no-outline", action="store_true", help="Disable outline painting")
    p_post.add_argument("--no-palette", action="store_true", help="Disable palette snapping")

    # review stage
    p_rev = subparsers.add_parser("review", help="Interactively review frames")
    p_rev.add_argument("input", type=Path, help="Input image file or directory")

    # run stage chain
    p_run = subparsers.add_parser("run", help="Run pipeline stages for file or directory")
    p_run.add_argument("input", type=Path, help="Input file or directory")
    p_run.add_argument(
        "--from",
        dest="from_stage",
        type=str,
        choices=["preprocess", "fixer", "postprocess"],
        default="fixer",
        help="Starting stage (default: fixer)",
    )
    p_run.add_argument("--no-review", action="store_true", help="Skip interactive review stage")
    _add_preprocess_override_args(p_run)
    _add_fixer_override_args(p_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    workdir = args.workdir if args.workdir is not None else Path(cfg.get("workdir", "workdir"))
    _apply_preprocess_overrides(cfg, args)
    _apply_fixer_overrides(cfg, args)

    if args.command == "llm":
        out_dir = args.output or (workdir / "llm")
        generate_frames(
            reference_path=args.ref,
            prompt=args.prompt,
            output_dir=out_dir,
            frame_count=args.frames,
        )
        return 0

    if args.command == "preprocess":
        inputs = collect_images(args.input)
        execute_preprocess(inputs, args.output, cfg, workdir)
        return 0

    if args.command == "fixer":
        inputs = collect_images(args.input)
        execute_fixer(inputs, args.output, cfg, workdir)
        return 0

    if args.command == "postprocess":
        if args.no_outline:
            cfg.setdefault("postprocess", {})["paint_outline"] = False
        if args.no_palette:
            cfg.setdefault("postprocess", {})["snap_palette"] = False
        inputs = collect_images(args.input)
        execute_postprocess(inputs, args.output, cfg, workdir)
        return 0

    if args.command == "review":
        inputs = collect_images(args.input)
        rev_cfg = cfg.get("review", {})
        approved = review_batch(
            image_paths=inputs,
            rework_callback=None,
            viewer_cmd=rev_cfg.get("viewer_cmd", ["xdg-open"]),
            wait_for_close=rev_cfg.get("wait_for_close", True),
        )
        print(f"\nApproved frames: {len(approved)}/{len(inputs)}")
        return 0

    if args.command == "run":
        raw_inputs = collect_images(args.input)
        source_by_stem: dict[str, Path] = {}

        if args.from_stage == "preprocess":
            frames = execute_preprocess(raw_inputs, None, cfg, workdir)
            for fr in frames:
                source_by_stem[fr.stem] = fr
            fixed = execute_fixer(frames, None, cfg, workdir)
            for fr, fx in zip(frames, fixed):
                source_by_stem[fx.stem] = source_by_stem[fr.stem]
            cleaned = execute_postprocess(fixed, None, cfg, workdir)
            for fx, cl in zip(fixed, cleaned):
                source_by_stem[cl.stem] = source_by_stem[fx.stem]
        elif args.from_stage == "fixer":
            fixed = execute_fixer(raw_inputs, None, cfg, workdir)
            for orig, fx in zip(raw_inputs, fixed):
                source_by_stem[fx.stem] = orig
            cleaned = execute_postprocess(fixed, None, cfg, workdir)
            for fx, cl in zip(fixed, cleaned):
                source_by_stem[cl.stem] = source_by_stem[fx.stem]
        else:
            for orig in raw_inputs:
                source_by_stem[orig.stem] = orig
            cleaned = execute_postprocess(raw_inputs, None, cfg, workdir)
            for orig, cl in zip(raw_inputs, cleaned):
                source_by_stem[cl.stem] = orig

        if args.no_review:
            return 0

        def rework(step: int, current_clean_path: Path) -> Path:
            # Map legacy numeric choice (2 -> fixer, 3 -> postprocess) cleanly.
            orig = source_by_stem.get(current_clean_path.stem, current_clean_path)
            if step == 1:
                raise NotImplementedError("Stage 'llm' is not implemented yet.")
            if step == 2:
                fx = execute_fixer([orig], None, cfg, workdir)[0]
                cl = execute_postprocess([fx], None, cfg, workdir)[0]
                source_by_stem[cl.stem] = orig
                return cl
            if step == 3:
                target_in = workdir / "fixer" / f"{orig.stem}_fixed.png"
                if not target_in.exists():
                    target_in = orig
                cl = execute_postprocess([target_in], None, cfg, workdir)[0]
                source_by_stem[cl.stem] = orig
                return cl
            return current_clean_path

        rev_cfg = cfg.get("review", {})
        approved = review_batch(
            image_paths=cleaned,
            rework_callback=rework,
            viewer_cmd=rev_cfg.get("viewer_cmd", ["xdg-open"]),
            wait_for_close=rev_cfg.get("wait_for_close", True),
            available_steps=(2, 3),
        )
        print(f"\nDone. Approved frames: {len(approved)}/{len(cleaned)}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())