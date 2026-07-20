"""Validated atlas loading without resampling or interpretation."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from .errors import AtlasSpecificationError, AtlasValidationError, UnknownParcelError
from .extraction import _integer_label
from .parcel import Parcel
from .provenance import Provenance
from .specification import AtlasSpecification, builtin_atlas_ids, load_specification
from .validation import ValidationReport, _validate


class Atlas:
    """A validated discrete atlas preserving its source NIfTI image."""

    __slots__ = (
        "_image",
        "_labels",
        "_parcel_ids",
        "_provenance",
        "_report",
        "_source_path",
        "_specification",
    )

    def __init__(
        self,
        image: nib.spatialimages.SpatialImage,
        specification: AtlasSpecification,
        *,
        source_path: str | Path | None = None,
    ) -> None:
        """Validate and cache one atlas exactly once.

        Parameters
        ----------
        image
            Source NIfTI image, retained unchanged.
        specification
            Explicit atlas facts and label semantics.
        source_path
            Original file path for source-byte checksum verification.
        """
        resolved_source = None
        if source_path is not None:
            resolved_source = Path(source_path).expanduser().resolve()
        result = _validate(image, specification, source_path=resolved_source)
        if not result.report.is_valid or result.labels is None:
            raise AtlasValidationError(result.report)

        self._image = image
        self._specification = specification
        self._source_path = resolved_source
        self._report = result.report
        self._labels = result.labels
        self._parcel_ids = result.report.parcel_ids
        self._provenance = Provenance.from_image(image, specification, resolved_source)

    @property
    def atlas_id(self) -> str:
        """Return the stable atlas ID."""
        return self._specification.atlas_id

    @property
    def specification(self) -> AtlasSpecification:
        """Return the immutable atlas specification."""
        return self._specification

    @property
    def image(self) -> nib.spatialimages.SpatialImage:
        """Return the unchanged source image."""
        return self._image

    @property
    def affine(self) -> np.ndarray:
        """Return the source image affine."""
        return self._image.affine

    @property
    def header(self) -> nib.filebasedimages.FileBasedHeader:
        """Return the source image header."""
        return self._image.header

    @property
    def labels(self) -> np.ndarray:
        """Return cached read-only integer labels."""
        return self._labels

    @property
    def parcel_ids(self) -> tuple[int, ...]:
        """Return observed non-background parcel IDs."""
        return self._parcel_ids

    @property
    def validation_report(self) -> ValidationReport:
        """Return the report produced during the sole validation pass."""
        return self._report

    @property
    def provenance(self) -> Provenance:
        """Return immutable atlas provenance."""
        return self._provenance

    def get_parcel(self, parcel_id: int) -> Parcel:
        """Return a canonical parcel for an existing non-background ID."""
        normalized_id = _integer_label(parcel_id)
        if normalized_id not in self._parcel_ids:
            raise UnknownParcelError(normalized_id, self.atlas_id, self._parcel_ids)
        warnings = tuple(
            finding
            for finding in self._report.warnings
            if finding.context.get("parcel_id") == normalized_id
        )
        return Parcel.from_labels(
            self.atlas_id,
            normalized_id,
            self._labels,
            self._image,
            self._provenance,
            warnings,
        )

    def parcel(self, parcel_id: int) -> Parcel:
        """Compatibility alias for :meth:`get_parcel`."""
        return self.get_parcel(parcel_id)


def load_atlas(
    source: str | Path | nib.spatialimages.SpatialImage,
    specification: AtlasSpecification | str | Path | None = None,
) -> Atlas:
    """Load a built-in atlas or validate an explicitly specified image.

    Parameters
    ----------
    source
        Built-in ID, NIfTI path, or loaded NIfTI image.
    specification
        Required for path/image sources; an object or TOML path.

    Returns
    -------
    Atlas
        Validated atlas with cached integer labels and parcel IDs.
    """
    explicit_specification = _coerce_specification(specification)
    if not isinstance(source, (str, Path)):
        if explicit_specification is None:
            raise AtlasSpecificationError(
                "Loading an in-memory image requires an explicit "
                "AtlasSpecification or TOML manifest path."
            )
        return Atlas(source, explicit_specification)

    candidate = Path(source).expanduser()
    if candidate.is_file():
        if explicit_specification is None:
            raise AtlasSpecificationError(
                f"Loading image path {candidate} requires an explicit "
                "AtlasSpecification or TOML manifest path."
            )
        source_path = candidate.resolve()
        return Atlas(
            nib.load(source_path),
            explicit_specification,
            source_path=source_path,
        )

    if explicit_specification is None:
        atlas_id = str(source)
        atlas_specification = load_specification(atlas_id)
    elif str(source) == explicit_specification.atlas_id:
        atlas_specification = explicit_specification
    else:
        builtins = ", ".join(builtin_atlas_ids()) or "none"
        raise AtlasSpecificationError(
            f"Image path does not exist: {candidate}. For a built-in atlas, use "
            f"one of: {builtins}."
        )

    source_path = atlas_specification.resolve_image_path()
    if not source_path.is_file():
        raise AtlasSpecificationError(
            f"Built-in atlas {atlas_specification.atlas_id!r} resolves to missing "
            f"image {source_path}. Keep the repository parcellation in place."
        )
    return Atlas(
        nib.load(source_path),
        atlas_specification,
        source_path=source_path,
    )


def _coerce_specification(
    specification: AtlasSpecification | str | Path | None,
) -> AtlasSpecification | None:
    if specification is None or isinstance(specification, AtlasSpecification):
        return specification
    return load_specification(specification)
