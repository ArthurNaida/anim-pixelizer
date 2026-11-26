from __future__ import annotations

from preprocess.clean import (
    apply_solid_background,
    clear_bg_noise,
    is_bg_noise,
    parse_hex_rgb,
    remove_small_islands,
)
from preprocess.extract import extract_foreground_crops
from preprocess.orchestrator import PreprocessOptions, process_sheet, slice_sheet
from preprocess.validate import FrameCountMismatchError, ValidationResult, validate_frame_count

__all__ = [
    "PreprocessOptions",
    "FrameCountMismatchError",
    "ValidationResult",
    "process_sheet",
    "slice_sheet",
    "validate_frame_count",
    "extract_foreground_crops",
    "clear_bg_noise",
    "remove_small_islands",
    "apply_solid_background",
    "is_bg_noise",
    "parse_hex_rgb",
    "detect_and_erase_dividers",
    "detect_background_color",
    "floodfill_background"
]