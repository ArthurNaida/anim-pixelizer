from __future__ import annotations

from collections import Counter, deque
import logging
import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)
MAX_RGB_DISTANCE = float(np.sqrt(255**2 + 255**2 + 255**2))


def parse_hex_rgb(hex_color: str) -> tuple[int, int, int]:
    """Parse a #RRGGBB hex string into an RGB tuple."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _divider_pixel_mask(
    rgba: np.ndarray,
    alpha_threshold: int = 20,
    max_luma: float = 80.0,
    max_channel_diff: int = 45,
    target_color: tuple[int, int, int] | None = None,
    color_tolerance: float = 35.0,
) -> np.ndarray:
    """
    Build a boolean mask of pixels eligible to belong to a divider line.
    Supports either a dark/low-luma interval or a target RGB color interval.
    """
    opaque = rgba[..., 3] > alpha_threshold
    rgb = rgba[..., :3].astype(np.float32)

    if target_color is not None:
        ref = np.array(target_color, dtype=np.float32)
        dist = np.linalg.norm(rgb - ref[None, None, :], axis=-1)
        return opaque & (dist <= float(color_tolerance))

    luma = (rgb[..., 0] + rgb[..., 1] + rgb[..., 2]) / 3.0
    spread = np.max(rgb, axis=-1) - np.min(rgb, axis=-1)
    return opaque & (luma <= float(max_luma)) & (spread <= float(max_channel_diff))


def _segment_matches_confidence(line_mask: np.ndarray, min_coverage: float) -> bool:
    """
    Check whether a 1D boolean line spans across the image within a confidence interval.
    Requires both overall coverage along the axis and a dominant contiguous span.
    """
    length = line_mask.shape[0]
    if length == 0:
        return False

    matched = int(np.count_nonzero(line_mask))
    coverage = matched / float(length)
    if coverage < min_coverage:
        return False

    # Compute longest contiguous run of matching pixels along the line.
    best_run = 0
    current_run = 0
    for flag in line_mask:
        if flag:
            current_run += 1
            if current_run > best_run:
                best_run = current_run
        else:
            current_run = 0

    contiguous_ratio = best_run / float(length)
    return contiguous_ratio >= max(0.5, min_coverage * 0.85)


def detect_and_erase_dividers(
    rgba: np.ndarray,
    enabled: bool = True,
    min_coverage: float = 0.85,
    max_luma: float = 80.0,
    max_channel_diff: int = 45,
    target_color_hex: str | None = None,
    color_tolerance: float = 35.0,
    erase_thickness: int = 1,
    alpha_threshold: int = 20,
) -> tuple[np.ndarray, int, int]:
    """
    Detect long straight horizontal and vertical divider lines spanning the sheet
    and erase them to transparent [0, 0, 0, 0] before background removal.
    Returns (updated_rgba, h_lines_erased, v_lines_erased).
    """
    out = rgba.copy()
    if not enabled:
        return out, 0, 0

    h, w, _ = out.shape
    if h == 0 or w == 0:
        return out, 0, 0

    target_rgb = parse_hex_rgb(target_color_hex) if target_color_hex else None
    eligible = _divider_pixel_mask(
        out,
        alpha_threshold=alpha_threshold,
        max_luma=max_luma,
        max_channel_diff=max_channel_diff,
        target_color=target_rgb,
        color_tolerance=color_tolerance,
    )

    erase_mask = np.zeros((h, w), dtype=bool)
    half_t = max(0, (erase_thickness - 1) // 2)
    extra_t = max(0, erase_thickness - 1 - half_t)

    h_lines = 0
    for y in range(h):
        if _segment_matches_confidence(eligible[y, :], min_coverage=min_coverage):
            y0 = max(0, y - half_t)
            y1 = min(h, y + extra_t + 1)
            erase_mask[y0:y1, :] = True
            h_lines += 1

    v_lines = 0
    for x in range(w):
        if _segment_matches_confidence(eligible[:, x], min_coverage=min_coverage):
            x0 = max(0, x - half_t)
            x1 = min(w, x + extra_t + 1)
            erase_mask[:, x0:x1] = True
            v_lines += 1

    erased_pixels = int(np.count_nonzero(erase_mask & (out[..., 3] > alpha_threshold)))
    out[erase_mask] = [0, 0, 0, 0]

    logger.info(
        "Divider pass: erased %d horizontal and %d vertical lines (%d px, min_coverage=%.2f)",
        h_lines,
        v_lines,
        erased_pixels,
        min_coverage,
    )
    return out, h_lines, v_lines


def detect_background_color(
    rgba: np.ndarray,
    alpha_threshold: int = 20,
) -> tuple[int, int, int] | None:
    """
    Automatically detect the dominant background RGB color from border and corner pixels.
    """
    h, w, _ = rgba.shape
    if h == 0 or w == 0:
        return None

    border_coords: list[tuple[int, int]] = []
    # Prioritize corners first, then top/bottom/left/right border pixels.
    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    border_coords.extend(corners)

    for x in range(w):
        border_coords.append((x, 0))
        border_coords.append((x, h - 1))
    for y in range(h):
        border_coords.append((0, y))
        border_coords.append((w - 1, y))

    samples: list[tuple[int, int, int]] = []
    for x, y in border_coords:
        r, g, b, a = rgba[y, x]
        if int(a) > alpha_threshold:
            samples.append((int(r), int(g), int(b)))

    if not samples:
        return None

    bg_rgb, _ = Counter(samples).most_common(1)[0]
    logger.debug("Auto-detected background RGB=%s from %d opaque border samples", bg_rgb, len(samples))
    return bg_rgb


def floodfill_background(
    rgba: np.ndarray,
    fuzz_percent: float = 15.0,
    bg_color_hex: str | None = None,
    seed_border: bool = True,
    alpha_threshold: int = 20,
) -> tuple[np.ndarray, int]:
    """
    Erase exterior background connected to image borders/corners using fuzz tolerance
    (similar to ImageMagick -fuzz N% -fill none -draw 'color ... floodfill').
    Regions already cleared to transparent (including erased dividers) remain empty.
    """
    out = rgba.copy()
    h, w, _ = out.shape
    if h == 0 or w == 0:
        return out, 0

    if bg_color_hex:
        ref_rgb = np.array(parse_hex_rgb(bg_color_hex), dtype=np.float32)
    else:
        detected = detect_background_color(out, alpha_threshold=alpha_threshold)
        if detected is None:
            return out, 0
        ref_rgb = np.array(detected, dtype=np.float32)

    threshold = (max(0.0, float(fuzz_percent)) / 100.0) * MAX_RGB_DISTANCE
    rgb = out[..., :3].astype(np.float32)
    alpha = out[..., 3]

    # Precompute mask of pixels within fuzz tolerance of the detected/configured background.
    matches_bg = np.linalg.norm(rgb - ref_rgb[None, None, :], axis=-1) <= threshold
    already_empty = alpha <= alpha_threshold

    seeds: list[tuple[int, int]] = []
    if seed_border:
        for x in range(w):
            seeds.append((x, 0))
            seeds.append((x, h - 1))
        for y in range(h):
            seeds.append((0, y))
            seeds.append((w - 1, y))
    else:
        seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]

    visited = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for sx, sy in seeds:
        if 0 <= sx < w and 0 <= sy < h and not visited[sy, sx]:
            visited[sy, sx] = True
            # Flood-fill propagates through empty/transparent space (e.g., erased dividers)
            # and matching background pixels, stopping at non-matching foreground pixels.
            if already_empty[sy, sx] or matches_bg[sy, sx]:
                queue.append((sx, sy))

    cleared = 0
    while queue:
        cx, cy = queue.popleft()
        if not already_empty[cy, cx]:
            out[cy, cx] = [0, 0, 0, 0]
            already_empty[cy, cx] = True
            cleared += 1

        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                visited[ny, nx] = True
                if already_empty[ny, nx] or matches_bg[ny, nx]:
                    queue.append((nx, ny))

    logger.info(
        "Flood-fill background pass: cleared %d pixels (ref_rgb=%s, fuzz=%.1f%%)",
        cleared,
        tuple(int(v) for v in ref_rgb),
        fuzz_percent,
    )
    return out, cleared


def is_bg_noise(
    r: int,
    g: int,
    b: int,
    a: int,
    max_channel_diff: int = 18,
    min_gray: int = 120,
    max_gray: int = 255,
    alpha_threshold: int = 20,
) -> bool:
    """Return True if a pixel is transparent or falls into neutral gray/white background noise."""
    if a <= alpha_threshold:
        return True

    spread = max(r, g, b) - min(r, g, b)
    if spread > max_channel_diff:
        return False

    luma = (r + g + b) / 3.0
    return min_gray <= luma <= max_gray


def clear_bg_noise(
    px,
    w: int,
    h: int,
    max_channel_diff: int = 18,
    min_gray: int = 120,
    max_gray: int = 255,
    alpha_threshold: int = 20,
) -> int:
    """Clear remaining neutral gray/white background noise pixels to transparent RGBA."""
    cleared = 0
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a > alpha_threshold and is_bg_noise(
                r,
                g,
                b,
                a,
                max_channel_diff=max_channel_diff,
                min_gray=min_gray,
                max_gray=max_gray,
                alpha_threshold=alpha_threshold,
            ):
                px[x, y] = (0, 0, 0, 0)
                cleared += 1

    logger.debug(
        "Cleared %d neutral noise pixels (max_channel_diff=%d, gray=[%d..%d], alpha_threshold=%d)",
        cleared,
        max_channel_diff,
        min_gray,
        max_gray,
        alpha_threshold,
    )
    return cleared


def remove_small_islands(
    px,
    w: int,
    h: int,
    min_pixels: int = 25,
    alpha_threshold: int = 20,
) -> tuple[int, int]:
    """Remove small 4-connected foreground components; return (removed_islands, removed_pixels)."""
    visited = [[False] * w for _ in range(h)]
    removed_islands = 0
    removed_pixels = 0

    for y in range(h):
        for x in range(w):
            if px[x, y][3] > alpha_threshold and not visited[y][x]:
                component: list[tuple[int, int]] = []
                stack: list[tuple[int, int]] = [(x, y)]
                visited[y][x] = True

                while stack:
                    cx, cy = stack.pop()
                    component.append((cx, cy))

                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if (
                            0 <= nx < w
                            and 0 <= ny < h
                            and not visited[ny][nx]
                            and px[nx, ny][3] > alpha_threshold
                        ):
                            visited[ny][nx] = True
                            stack.append((nx, ny))

                if len(component) < min_pixels:
                    for cx, cy in component:
                        px[cx, cy] = (0, 0, 0, 0)
                    removed_islands += 1
                    removed_pixels += len(component)

    logger.debug(
        "Removed %d small islands (%d pixels total, min_pixels=%d)",
        removed_islands,
        removed_pixels,
        min_pixels,
    )
    return removed_islands, removed_pixels


def apply_solid_background(crop: Image.Image, solid_bg_color: str) -> Image.Image:
    """Composite a transparent RGBA sprite crop onto a solid RGB background."""
    bg_rgb = parse_hex_rgb(solid_bg_color)
    rgba = np.array(crop.convert("RGBA"), dtype=np.uint8)
    canvas = np.empty_like(rgba)
    canvas[..., :3] = bg_rgb
    canvas[..., 3] = 255

    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    blended = rgba[..., :3].astype(np.float32) * alpha + canvas[..., :3].astype(np.float32) * (
        1.0 - alpha
    )
    canvas[..., :3] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    return Image.fromarray(canvas, mode="RGBA")

def clear_isolated_bg_pockets(
    rgba: np.ndarray,
    fuzz_percent: float = 18.0,
    bg_color_hex: str | None = None,
    alpha_threshold: int = 20,
    max_opaque_neighbors: int = 3,
) -> tuple[np.ndarray, int]:
    """
    Clear enclosed/isolated background pockets matching the background color
    that were unreachable from border flood-fill, while protecting dense interior sprite pixels.
    """
    out = rgba.copy()
    h, w, _ = out.shape
    if h < 3 or w < 3:
        return out, 0

    if bg_color_hex:
        ref_rgb = np.array(parse_hex_rgb(bg_color_hex), dtype=np.float32)
    else:
        detected = detect_background_color(out, alpha_threshold=alpha_threshold)
        if detected is None:
            return out, 0
        ref_rgb = np.array(detected, dtype=np.float32)

    threshold = (max(0.0, float(fuzz_percent)) / 100.0) * MAX_RGB_DISTANCE
    rgb = out[..., :3].astype(np.float32)
    opaque = (out[..., 3] > alpha_threshold).astype(np.uint8)
    matches_bg = np.linalg.norm(rgb - ref_rgb[None, None, :], axis=-1) <= threshold

    # Count 8-connected opaque neighbors for each pixel.
    padded = np.pad(opaque, 1, mode="constant", constant_values=0)
    neighbor_sum = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
        + padded[1:-1, :-2] + padded[1:-1, 2:]
        + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    )

    pocket_mask = (opaque == 1) & matches_bg & (neighbor_sum <= max_opaque_neighbors)
    cleared = int(np.count_nonzero(pocket_mask))
    out[pocket_mask] = [0, 0, 0, 0]

    logger.info(
        "Pocket cleanup pass: cleared %d isolated bg pixels (fuzz=%.1f%%, max_neighbors=%d)",
        cleared,
        fuzz_percent,
        max_opaque_neighbors,
    )
    return out, cleared