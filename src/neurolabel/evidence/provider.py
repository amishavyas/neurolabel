"""Structural contract implemented by evidence providers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from .models import EvidenceRecord

if TYPE_CHECKING:
    from neurolabel.core.models import Parcel


class EvidenceProvider(Protocol):
    """A source that describes parcels without changing them."""

    @property
    def name(self) -> str:
        """Return the stable provider name."""

        ...

    def describe(self, parcel: Parcel) -> Sequence[EvidenceRecord]:
        """Return independent evidence about ``parcel``."""

        ...
