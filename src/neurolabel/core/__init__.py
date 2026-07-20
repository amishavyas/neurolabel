"""Core, interpretation-free atlas API."""

from .atlas import Atlas, load_atlas
from .exceptions import (
    AtlasSpecificationError,
    AtlasValidationError,
    NeuroLabelError,
    UnknownParcelError,
)
from .extraction import extract_label_mask
from .parcel import Parcel, ParcelGeometry
from .provenance import Provenance
from .specification import AtlasSpecification, builtin_atlas_ids, load_specification
from .validation import ValidationFinding, ValidationReport, validate_nifti

__all__ = [
    "Atlas",
    "AtlasSpecification",
    "AtlasSpecificationError",
    "AtlasValidationError",
    "NeuroLabelError",
    "Parcel",
    "ParcelGeometry",
    "Provenance",
    "UnknownParcelError",
    "ValidationFinding",
    "ValidationReport",
    "builtin_atlas_ids",
    "extract_label_mask",
    "load_atlas",
    "load_specification",
    "validate_nifti",
]
