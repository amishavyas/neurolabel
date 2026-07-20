"""Immutable source provenance for loaded atlases and parcels."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib

from ._version import __version__
from .specification import AtlasSpecification

UNAVAILABLE_CHECKSUM = "unavailable"


@dataclass(frozen=True, slots=True)
class Provenance:
    """Factual provenance captured without atlas interpretation."""

    source: str
    sha256: str
    atlas_id: str | None
    coordinate_space: str | None
    image_shape: tuple[int, ...]
    data_dtype: str
    affine: tuple[tuple[float, ...], ...]
    voxel_sizes: tuple[float, ...]
    package_version: str
    source_reference: str | None = None
    coordinate_space_status: str | None = None

    @classmethod
    def from_image(
        cls,
        image: nib.spatialimages.SpatialImage,
        specification: AtlasSpecification,
        source_path: str | Path | None,
    ) -> Provenance:
        """Construct provenance, hashing source file bytes when available."""
        path = Path(source_path).resolve() if source_path is not None else None
        return cls(
            source=str(path) if path is not None else "in-memory",
            sha256=(
                _file_sha256(path)
                if path is not None and path.is_file()
                else UNAVAILABLE_CHECKSUM
            ),
            atlas_id=specification.atlas_id,
            coordinate_space=specification.coordinate_space,
            image_shape=tuple(int(value) for value in image.shape),
            data_dtype=str(image.get_data_dtype()),
            affine=tuple(tuple(float(value) for value in row) for row in image.affine),
            voxel_sizes=tuple(float(value) for value in image.header.get_zooms()[:3]),
            package_version=__version__,
            source_reference=specification.source_reference,
            coordinate_space_status=specification.coordinate_space_status,
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
