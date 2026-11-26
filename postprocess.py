from __future__ import annotations

from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a #RRGGBB hex string into an RGB tuple."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def binarize_alpha(rgba: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Convert semi-transparent pixels into binary alpha (0 or 255)."""
    out = rgba.copy()
    mask = out[..., 3] >= threshold
    out[mask, 3] = 255
    out[~mask] = [0, 0, 0, 0]
    return out


def snap_to_palette(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Snap RGB pixels to the nearest color in palette using squared Euclidean distance."""
    diff = rgb[..., None, :].astype(np.int32) - palette[None, None, :, :].astype(np.int32)
    dist_sq = np.sum(diff * diff, axis=-1)
    nearest_idx = np.argmin(dist_sq, axis=-1)
    return palette[nearest_idx].astype(np.uint8)


def remove_separated_pixels(rgba: np.ndarray, min_neighbors: int = 1) -> np.ndarray:
    """Remove opaque pixels that have fewer than min_neighbors opaque 8-connected neighbors."""
    out = rgba.copy()
    alpha = out[..., 3] > 0
    h, w = alpha.shape

    padded = np.pad(alpha.astype(np.uint8), 1, mode="constant", constant_values=0)
    neighbor_count = np.zeros((h, w), dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            neighbor_count += padded[dy : dy + h, dx : dx + w]

    isolated = alpha & (neighbor_count < min_neighbors)
    out[isolated] = [0, 0, 0, 0]
    return out


def clean_noise_clusters(
    rgba: np.ndarray,
    window_size: int = 3,
    max_density: float = 0.35,
) -> np.ndarray:
    """Replace pixels inside low-density opaque windows with the most common neighbor color."""
    out = rgba.copy()
    alpha = out[..., 3] > 0
    h, w = alpha.shape
    pad = window_size // 2

    padded_alpha = np.pad(alpha.astype(np.float32), pad, mode="constant", constant_values=0.0)
    area = float(window_size * window_size)
    density = np.zeros((h, w), dtype=np.float32)

    for dy in range(window_size):
        for dx in range(window_size):
            density += padded_alpha[dy : dy + h, dx : dx + w]
    density /= area

    noisy_mask = alpha & (density <= max_density)
    ys, xs = np.where(noisy_mask)

    # Iterate deterministically in row-major order.
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - 1), min(h, y + 2)
        x0, x1 = max(0, x - 1), min(w, x + 2)

        neighbors: list[tuple[int, int, int, int]] = []
        for abs_y in range(y0, y1):
            for abs_x in range(x0, x1):
                if abs_y == y and abs_x == x:
                    continue
                px = out[abs_y, abs_x]
                if px[3] > 0:
                    neighbors.append((int(px[0]), int(px[1]), int(px[2]), int(px[3])))

        if neighbors:
            counts = Counter(neighbors)
            best_color = max(counts.items(), key=lambda item: item[1])[0]
            out[y, x] = best_color
        else:
            out[y, x] = [0, 0, 0, 0]

    return out


def paint_contours(rgba: np.ndarray, outline_color: tuple[int, int, int]) -> np.ndarray:
    """Paint 4-connected boundary opaque pixels with outline_color."""
    out = rgba.copy()
    opaque = out[..., 3] > 0
    h, w = opaque.shape

    padded = np.pad(opaque, 1, mode="constant", constant_values=False)
    has_transparent_neighbor = (
        ~padded[0:h, 1 : w + 1]
        | ~padded[2 : h + 2, 1 : w + 1]
        | ~padded[1 : h + 1, 0:w]
        | ~padded[1 : h + 1, 2 : w + 2]
    )

    contour_mask = opaque & has_transparent_neighbor
    out[contour_mask, 0:3] = outline_color
    out[contour_mask, 3] = 255
    return out


def run_postprocess(
    input_path: Path,
    output_path: Path,
    palette_hex: list[str],
    alpha_threshold: int = 128,
    min_neighbors: int = 1,
    noise_window: int = 3,
    noise_max_density: float = 0.35,
    paint_outline: bool = True,
    outline_color_hex: str = "#141013",
    snap_palette: bool = True,
) -> Path:
    """Execute deterministic post-processing steps on a single sprite frame."""
    img = Image.open(input_path).convert("RGBA")
    rgba = np.array(img, dtype=np.uint8)

    rgba = binarize_alpha(rgba, threshold=alpha_threshold)
    rgba = remove_separated_pixels(rgba, min_neighbors=min_neighbors)
    rgba = clean_noise_clusters(
        rgba,
        window_size=noise_window,
        max_density=noise_max_density,
    )

    palette_arr: np.ndarray | None = None
    if palette_hex:
        palette_arr = np.array([hex_to_rgb(c) for c in palette_hex], dtype=np.uint8)

    if snap_palette and palette_arr is not None:
        opaque = rgba[..., 3] > 0
        if np.any(opaque):
            rgba[opaque, 0:3] = snap_to_palette(rgba[opaque, 0:3], palette_arr)

    if paint_outline:
        outline_rgb = hex_to_rgb(outline_color_hex)
        if snap_palette and palette_arr is not None:
            snapped = snap_to_palette(np.array([[outline_rgb]], dtype=np.uint8), palette_arr)[0, 0]
            outline_rgb = (int(snapped[0]), int(snapped[1]), int(snapped[2]))
        rgba = paint_contours(rgba, outline_rgb)

    rgba = remove_separated_pixels(rgba, min_neighbors=min_neighbors)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(output_path)
    return output_path