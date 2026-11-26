from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import numpy as np
from PIL import Image

from preprocess.clean import (
    clear_bg_noise,
    clear_isolated_bg_pockets,
    detect_and_erase_dividers,
    floodfill_background,
    remove_small_islands,
)
from preprocess.extract import extract_foreground_crops
from preprocess.validate import FrameCountMismatchError, validate_frame_count


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessOptions:
    """Options for divider erasure, background flood-fill, sprite extraction, and validation."""

    pad: int = 2
    min_size: int = 16
    # Divider line detection and erasure (executed before background removal).
    erase_dividers: bool = True
    divider_min_coverage: float = 0.85
    divider_max_luma: float = 80.0
    divider_max_channel_diff: int = 45
    divider_target_color: str | None = None
    divider_color_tolerance: float = 35.0
    divider_thickness: int = 1
    # Automatic flood-fill background removal and isolated pocket cleanup.
    bg_fuzz_percent: float = 15.0
    bg_color: str | None = None  # None = auto-detect from border/corners, or "#RRGGBB"
    bg_seed_border: bool = True
    clean_bg_pockets: bool = True
    pocket_fuzz_percent: float = 18.0
    pocket_max_opaque_neighbors: int = 3
    # Shard/fragment grouping relative to neighbor spacing.
    merge_shards: bool = True
    shard_merge_ratio: float = 0.45
    max_shard_gap: float = 96.0
    # Secondary noise/island cleanup.
    clean_neutral_noise: bool = True
    max_channel_diff: int = 18
    min_gray: int = 120
    max_gray: int = 255
    min_component_pixels: int = 25
    alpha_threshold: int = 20
    # Frame count validation and output mode.
    expected_frames: int | None = None
    strict_frames: bool = True
    bg_mode: str = "transparent"  # "transparent" | "solid"
    solid_bg_color: str = "#000000"


def process_sheet(
    sheet_path: Path,
    output_dir: Path,
    stem_prefix: str | None = None,
    options: PreprocessOptions | None = None,
) -> list[Path]:
    """Erase dividers, flood-fill background, clean noise/islands, extract and validate frames."""
    opts = options or PreprocessOptions()
    prefix = stem_prefix or sheet_path.stem

    logger.info("Processing sheet '%s' (prefix='%s', out='%s')", sheet_path, prefix, output_dir)
    with Image.open(sheet_path) as raw_img:
        rgba_arr = np.array(raw_img.convert("RGBA"), dtype=np.uint8)

    h, w, _ = rgba_arr.shape
    logger.info("Loaded '%s' (%dx%d)", sheet_path, w, h)

    # 1. Erase divider lines first so separated regions remain distinct.
    rgba_arr, h_lines, v_lines = detect_and_erase_dividers(
        rgba_arr,
        enabled=opts.erase_dividers,
        min_coverage=opts.divider_min_coverage,
        max_luma=opts.divider_max_luma,
        max_channel_diff=opts.divider_max_channel_diff,
        target_color_hex=opts.divider_target_color,
        color_tolerance=opts.divider_color_tolerance,
        erase_thickness=opts.divider_thickness,
        alpha_threshold=opts.alpha_threshold,
    )

    # 2. Auto-detect background and flood-fill with fuzz tolerance.
    rgba_arr, cleared_bg = floodfill_background(
        rgba_arr,
        fuzz_percent=opts.bg_fuzz_percent,
        bg_color_hex=opts.bg_color,
        seed_border=opts.bg_seed_border,
        alpha_threshold=opts.alpha_threshold,
    )

    # 2b. Clean enclosed/isolated background pockets unreachable from border flood-fill.
    cleared_pockets = 0
    if opts.clean_bg_pockets:
        rgba_arr, cleared_pockets = clear_isolated_bg_pockets(
            rgba_arr,
            fuzz_percent=opts.pocket_fuzz_percent,
            bg_color_hex=opts.bg_color,
            alpha_threshold=opts.alpha_threshold,
            max_opaque_neighbors=opts.pocket_max_opaque_neighbors,
        )

    # 3. Remove residual neutral noise and small islands.
    rgba_img = Image.fromarray(rgba_arr, mode="RGBA").copy()
    px = rgba_img.load()

    cleared_noise = 0
    if opts.clean_neutral_noise:
        cleared_noise = clear_bg_noise(
            px=px,
            w=w,
            h=h,
            max_channel_diff=opts.max_channel_diff,
            min_gray=opts.min_gray,
            max_gray=opts.max_gray,
            alpha_threshold=opts.alpha_threshold,
        )

    removed_islands, removed_island_px = remove_small_islands(
        px=px,
        w=w,
        h=h,
        min_pixels=opts.min_component_pixels,
        alpha_threshold=opts.alpha_threshold,
    )

    logger.info(
        "Cleanup summary for '%s': dividers=(H:%d, V:%d), cleared_bg=%d, cleared_pockets=%d, cleared_noise=%d, removed_islands=%d (%d px)",
        sheet_path.name,
        h_lines,
        v_lines,
        cleared_bg,
        cleared_pockets,
        cleared_noise,
        removed_islands,
        removed_island_px,
    )

    frames = extract_foreground_crops(
        rgba_img=rgba_img,
        output_dir=output_dir,
        stem_prefix=prefix,
        pad=opts.pad,
        min_size=opts.min_size,
        alpha_threshold=opts.alpha_threshold,
        bg_mode=opts.bg_mode,
        solid_bg_color=opts.solid_bg_color,
    )

    validate_frame_count(
        frames=frames,
        expected_frames=opts.expected_frames,
        raise_on_mismatch=opts.strict_frames,
    )
    return frames


slice_sheet = process_sheet