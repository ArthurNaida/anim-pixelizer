from __future__ import annotations

from pathlib import Path
import shlex
import shutil
import subprocess
from PIL import Image


def _render_args(template_args: list[str], context: dict[str, str]) -> list[str]:
    """Render format placeholders in command arguments deterministically."""
    return [arg.format(**context) for arg in template_args]


def _is_executable_available(exe: str) -> bool:
    """Return True if executable is found in PATH or exists on disk."""
    return shutil.which(exe) is not None or Path(exe).exists()


def run_fixer(
    input_path: Path,
    output_path: Path,
    executable: str = "pixelfixer",
    subcommand: str = "process",
    extra_args: list[str] | None = None,
    command_template: list[str] | None = None,
    post_command_template: list[str] | None = None,
    enable_post_command: bool = False,
    palette_image: str | None = None,
    target_size: tuple[int, int] | None = None,
    resize_to_target: bool = False,
    fallback_nearest_resize: bool = True,
) -> Path:
    """Run pixel-art-fixer and optional post-processing command on a single image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = target_size if target_size else (0, 0)

    context = {
        "input": str(input_path),
        "output": str(output_path),
        "width": str(width),
        "height": str(height),
        "palette_image": str(palette_image) if palette_image else "",
    }

    if command_template:
        cmd = _render_args(command_template, context)
    else:
        cmd = [executable]
        if subcommand:
            cmd.append(subcommand)
        cmd.extend([str(input_path), str(output_path)])
        if extra_args:
            cmd.extend(_render_args(extra_args, context))

    if not _is_executable_available(cmd[0]):
        if not fallback_nearest_resize:
            raise FileNotFoundError(
                f"Executable '{cmd[0]}' was not found. Configure 'fixer.executable' or 'fixer.command' in config.json."
            )
        img = Image.open(input_path).convert("RGBA")
        if target_size and img.size != tuple(target_size):
            img = img.resize(tuple(target_size), Image.Resampling.NEAREST)
        img.save(output_path)
        return output_path

    subprocess.run(cmd, check=True)

    if not output_path.exists():
        raise RuntimeError(
            f"Fixer command succeeded ({' '.join(shlex.quote(c) for c in cmd)}), "
            f"but output file '{output_path}' was not created."
        )

    if enable_post_command and post_command_template:
        rendered_post = _render_args(post_command_template, context)
        # Skip palette remap step if no palette image path is configured.
        if "{palette_image}" in post_command_template and not palette_image:
            filtered_post: list[str] = []
            skip_next = False
            for token in post_command_template:
                if skip_next:
                    skip_next = False
                    continue
                if token == "-remap":
                    skip_next = True
                    continue
                filtered_post.append(token.format(**context))
            rendered_post = filtered_post

        if rendered_post and _is_executable_available(rendered_post[0]):
            subprocess.run(rendered_post, check=True)

    if resize_to_target and target_size:
        img = Image.open(output_path).convert("RGBA")
        if img.size != tuple(target_size):
            img = img.resize(tuple(target_size), Image.Resampling.NEAREST)
            img.save(output_path)

    return output_path