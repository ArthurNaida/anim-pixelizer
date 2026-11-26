from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
from PIL import Image

from preprocess.clean import apply_solid_background


logger = logging.getLogger(__name__)


def _box_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Compute Chebyshev/Euclidean edge-to-edge gap between two axis-aligned boxes (0 if overlapping)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    dx = 0
    if bx0 > ax1:
        dx = bx0 - ax1
    elif ax0 > bx1:
        dx = ax0 - bx1

    dy = 0
    if by0 > ay1:
        dy = by0 - ay1
    elif ay0 > by1:
        dy = ay0 - by1

    return float(np.hypot(dx, dy))


def _merge_shard_boxes(
    boxes: list[tuple[int, int, int, int]],
    shard_merge_ratio: float = 0.45,
    max_shard_gap: float = 96.0,
) -> list[tuple[int, int, int, int]]:
    """
    Merge nearby shard/fragment boxes into their parent frame box using
    nearest-neighbor distance scale so detached shards do not become extra frames.
    """
    n = len(boxes)
    if n <= 1:
        return list(boxes)

    # Estimate typical inter-frame spacing from each box's nearest neighbor distance.
    nn_distances: list[float] = []
    for i in range(n):
        best = float("inf")
        for j in range(n):
            if i == j:
                continue
            d = _box_distance(boxes[i], boxes[j])
            if d < best:
                best = d
        if np.isfinite(best):
            nn_distances.append(best)

    avg_dist = float(np.mean(nn_distances)) if nn_distances else max_shard_gap
    merge_threshold = min(float(max_shard_gap), max(8.0, avg_dist * float(shard_merge_ratio)))

    logger.debug(
        "Shard grouping: components=%d, avg_nn_dist=%.1fpx, merge_threshold=%.1fpx (ratio=%.2f)",
        n,
        avg_dist,
        merge_threshold,
        shard_merge_ratio,
    )

    # Connected components over boxes whose edge-to-edge distance <= merge_threshold.
    visited = [False] * n
    merged_boxes: list[tuple[int, int, int, int]] = []

    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        x0, y0, x1, y1 = boxes[start]
        members = 0

        while stack:
            cur = stack.pop()
            members += 1
            cx0, cy0, cx1, cy1 = boxes[cur]
            if cx0 < x0:
                x0 = cx0
            if cy0 < y0:
                y0 = cy0
            if cx1 > x1:
                x1 = cx1
            if cy1 > y1:
                y1 = cy1

            for nxt in range(n):
                if not visited[nxt] and _box_distance(boxes[cur], boxes[nxt]) <= merge_threshold:
                    visited[nxt] = True
                    stack.append(nxt)

        merged_boxes.append((x0, y0, x1, y1))
        if members > 1:
            logger.debug("Merged %d nearby components into frame box (%d,%d)-(%d,%d)", members, x0, y0, x1, y1)

    # Sort top-to-bottom, then left-to-right for deterministic frame ordering.
    merged_boxes.sort(key=lambda b: (b[1], b[0]))
    return merged_boxes


def extract_foreground_crops(
    rgba_img: Image.Image,
    output_dir: Path,
    stem_prefix: str = "sprite",
    pad: int = 2,
    min_size: int = 16,
    alpha_threshold: int = 20,
    merge_shards: bool = True,
    shard_merge_ratio: float = 0.45,
    max_shard_gap: float = 96.0,
    bg_mode: str = "transparent",
    solid_bg_color: str = "#000000",
) -> list[Path]:
    """Find foreground components, optionally merge nearby shards, and save cropped frames."""
    w, h = rgba_img.size
    px = rgba_img.load()

    def is_fg(x: int, y: int) -> bool:
        return px[x, y][3] > alpha_threshold

    visited = [[False] * w for _ in range(h)]
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_boxes: list[tuple[int, int, int, int]] = []
    for y in range(h):
        for x in range(w):
            if is_fg(x, y) and not visited[y][x]:
                stack: list[tuple[int, int]] = [(x, y)]
                visited[y][x] = True
                min_x = max_x = x
                min_y = max_y = y

                while stack:
                    cx, cy = stack.pop()
                    if cx < min_x:
                        min_x = cx
                    if cx > max_x:
                        max_x = cx
                    if cy < min_y:
                        min_y = cy
                    if cy > max_y:
                        max_y = cy

                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if 0 <= nx < w and 0 <= ny < h and not visited[ny][nx] and is_fg(nx, ny):
                            visited[ny][nx] = True
                            stack.append((nx, ny))

                raw_boxes.append((min_x, min_y, max_x, max_y))

    candidate_boxes = (
        _merge_shard_boxes(
            raw_boxes,
            shard_merge_ratio=shard_merge_ratio,
            max_shard_gap=max_shard_gap,
        )
        if merge_shards
        else raw_boxes
    )

    saved_paths: list[Path] = []
    skipped_small = 0
    idx = 0

    for min_x, min_y, max_x, max_y in candidate_boxes:
        bw = max_x - min_x + 1
        bh = max_y - min_y + 1
        if bw >= min_size and bh >= min_size:
            x0 = max(0, min_x - pad)
            y0 = max(0, min_y - pad)
            x1 = min(w, max_x + pad + 1)
            y1 = min(h, max_y + pad + 1)
            crop = rgba_img.crop((x0, y0, x1, y1))

            if bg_mode.lower() == "solid":
                crop = apply_solid_background(crop, solid_bg_color)

            out_file = output_dir / f"{stem_prefix}_{idx:02d}.png"
            crop.save(out_file)
            logger.debug("Saved frame #%d (%dx%d) -> %s", idx, bw, bh, out_file)
            saved_paths.append(out_file)
            idx += 1
        else:
            skipped_small += 1
            logger.debug("Skipped box (%d,%d)-(%d,%d) size %dx%d < min_size=%d", min_x, min_y, max_x, max_y, bw, bh, min_size)

    logger.info(
        "Extracted %d frames from %d raw components (%d grouped boxes, skipped %d < %dpx) into %s",
        len(saved_paths),
        len(raw_boxes),
        len(candidate_boxes),
        skipped_small,
        min_size,
        output_dir,
    )
    return saved_paths