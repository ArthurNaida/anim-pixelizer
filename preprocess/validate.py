from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class FrameCountMismatchError(ValueError):
    """Raised when extracted frame count does not match expected frame count."""

    def __init__(self, actual: int, expected: int) -> None:
        self.actual = actual
        self.expected = expected
        super().__init__(f"wrong number of frames: {actual} instead of {expected}")


@dataclass(frozen=True)
class ValidationResult:
    """Result of frame validation against expected count."""

    actual_count: int
    expected_count: int | None
    is_valid: bool
    message: str


def validate_frame_count(
    frames: list[Path],
    expected_frames: int | None,
    raise_on_mismatch: bool = True,
) -> ValidationResult:
    """Validate actual extracted frame count against user-specified expected_frames."""
    actual = len(frames)
    if expected_frames is None:
        msg = f"extracted {actual} frames (no expected frame count set)"
        logger.info("Validation skipped: %s", msg)
        return ValidationResult(
            actual_count=actual,
            expected_count=None,
            is_valid=True,
            message=msg,
        )

    if actual != expected_frames:
        msg = f"wrong number of frames: {actual} instead of {expected_frames}"
        logger.warning("Validation failed: %s", msg)
        if raise_on_mismatch:
            raise FrameCountMismatchError(actual=actual, expected=expected_frames)
        return ValidationResult(
            actual_count=actual,
            expected_count=expected_frames,
            is_valid=False,
            message=msg,
        )

    msg = f"frame count valid: {actual}"
    logger.info("Validation passed: %s", msg)
    return ValidationResult(
        actual_count=actual,
        expected_count=expected_frames,
        is_valid=True,
        message=msg,
    )