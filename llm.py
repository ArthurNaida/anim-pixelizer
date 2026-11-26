from __future__ import annotations

from pathlib import Path


def generate_frames(
    reference_path: Path | None,
    prompt: str,
    output_dir: Path,
    frame_count: int = 4,
    model: str = "gemini-2.5-flash-image",
) -> list[Path]:
    """Step 1 stub: generate raw animation frames via LLM."""
    raise NotImplementedError(
        "Step 1 (llm.py) is not implemented yet. Use step2, step3, step4, or run."
    )