"""Stable public NeuroLabel package API."""

from .core.atlas import Atlas, load_atlas
from .core.parcel import Parcel, ParcelGeometry
from .core.provenance import Provenance
from .evidence.models import EvidenceDomain, EvidenceRecord
from .evidence.provider import EvidenceProvider

__all__ = [
    "Atlas",
    "EvidenceDomain",
    "EvidenceProvider",
    "EvidenceRecord",
    "Parcel",
    "ParcelGeometry",
    "Provenance",
    "load_atlas",
]
