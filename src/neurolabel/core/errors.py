"""Public NeuroLabel exceptions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .validation import ValidationReport


class NeuroLabelError(Exception):
    """Base class for recoverable NeuroLabel errors."""


class AtlasSpecificationError(NeuroLabelError):
    """Raised when an atlas specification is missing or malformed."""


class AtlasValidationError(NeuroLabelError):
    """Raised when an image violates the discrete-atlas contract."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        summary = "; ".join(
            f"{finding.code}: {finding.message}" for finding in report.errors
        )
        super().__init__(
            f"Atlas validation failed with {len(report.errors)} error(s): {summary}. "
            "Inspect the ValidationReport before using this atlas."
        )


class UnknownParcelError(NeuroLabelError):
    """Raised when a requested label is absent from an atlas."""

    def __init__(
        self,
        requested_id: object,
        atlas_id: str,
        available_ids: Iterable[int],
    ) -> None:
        available = tuple(sorted(set(available_ids)))
        self.requested_id = requested_id
        self.atlas_id = atlas_id
        self.available_ids = available
        display = _format_ranges(available) if available else "none"
        super().__init__(
            f"Parcel label {requested_id!r} is not present in atlas {atlas_id!r}; "
            f"available labels: {display}."
        )


def _format_ranges(values: tuple[int, ...]) -> str:
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}..{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}..{previous}")
    return ", ".join(ranges)
