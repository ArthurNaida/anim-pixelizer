from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Callable


def open_image(image_path: Path, viewer_cmd: list[str], wait_for_close: bool = True) -> None:
    """Open an image using the configured viewer command."""
    cmd = [*viewer_cmd, str(image_path)]
    if wait_for_close:
        subprocess.run(cmd, check=False)
    else:
        subprocess.Popen(cmd)


def ask_yes_no(prompt: str) -> bool:
    """Prompt the user for a yes/no response."""
    while True:
        ans = input(f"{prompt} [y/n]: ").strip().lower()
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please enter 'y' or 'n'.")


def ask_rework_step(available_steps: tuple[int, ...] = (1, 2, 3)) -> int | None:
    """Ask which pipeline step to rework from, or 'n' to skip."""
    allowed_str = ", ".join(str(s) for s in available_steps)
    while True:
        ans = input(
            f"Want to rework from step? ({allowed_str}, n to skip): "
        ).strip().lower()
        if ans in {"n", "no"}:
            return None
        if ans.isdigit() and int(ans) in available_steps:
            return int(ans)
        print(f"Please enter one of: {allowed_str}, or 'n'.")


def review_single(
    image_path: Path,
    rework_callback: Callable[[int, Path], Path] | None = None,
    viewer_cmd: list[str] | None = None,
    wait_for_close: bool = True,
    available_steps: tuple[int, ...] = (1, 2, 3),
) -> tuple[bool, Path]:
    """Review a single image and optionally trigger step rework."""
    cmd = viewer_cmd or ["xdg-open"]
    current_path = image_path

    while True:
        print(f"\n[Review] Opening: {current_path}")
        open_image(current_path, viewer_cmd=cmd, wait_for_close=wait_for_close)

        if ask_yes_no("is it okay?"):
            return True, current_path

        if rework_callback is None:
            return False, current_path

        step = ask_rework_step(available_steps=available_steps)
        if step is None:
            return False, current_path

        current_path = rework_callback(step, current_path)


def review_batch(
    image_paths: list[Path],
    rework_callback: Callable[[int, Path], Path] | None = None,
    viewer_cmd: list[str] | None = None,
    wait_for_close: bool = True,
    available_steps: tuple[int, ...] = (1, 2, 3),
) -> list[Path]:
    """Review a list of images sequentially and return approved paths."""
    approved: list[Path] = []
    for idx, img_path in enumerate(image_paths, start=1):
        print(f"\n--- Frame {idx}/{len(image_paths)} ---")
        ok, final_path = review_single(
            image_path=img_path,
            rework_callback=rework_callback,
            viewer_cmd=viewer_cmd,
            wait_for_close=wait_for_close,
            available_steps=available_steps,
        )
        if ok:
            approved.append(final_path)
    return approved