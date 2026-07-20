"""Deterministic, observation-only audits of atlas image files."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from .specification import (
    AtlasSpecification,
    builtin_atlas_ids,
    load_specification,
)


@dataclass(frozen=True, slots=True)
class AtlasAudit:
    """Serializable observations about one atlas image."""

    atlas_id: str
    path: str
    checksum_sha256: str
    shape: tuple[int, ...]
    ndim: int
    stored_dtype: str
    affine: tuple[tuple[float, ...], ...]
    voxel_dimensions: tuple[float, ...]
    affine_finite: bool
    spatial_determinant: float | None
    nonfinite_count: int
    integer_like: bool
    max_integer_deviation: float | None
    unique_rounded_labels: tuple[int, ...]
    frequencies: tuple[tuple[int, int], ...]
    rounded_label_min: int | None
    rounded_label_max: int | None
    unique_label_count: int
    voxel_count: int
    border_values: tuple[int, ...]
    largest_label: int | None
    observed_parcel_count: int | None
    expected_parcel_count: int | None
    suspected_backgrounds: tuple[int, ...]
    contiguous: bool
    parcel_0_voxels: int
    parcel_0_nonempty: bool
    expected_parcels_nonempty: bool | None
    nonempty_labels: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return _json_safe(asdict(self))


def audit_atlas(
    image_or_path: nib.spatialimages.SpatialImage | str | Path,
    specification: AtlasSpecification | None = None,
) -> AtlasAudit:
    """Audit one image without assigning anatomical meaning.

    Parameters
    ----------
    image_or_path
        Loaded image or path to a NIfTI file.
    specification
        Optional specification used only for expected-count observations.

    Returns
    -------
    AtlasAudit
        Deterministic file, geometry, label, and border observations.
    """
    if isinstance(image_or_path, (str, Path)):
        source_path = Path(image_or_path).expanduser().resolve()
        image = nib.load(source_path)
        path = str(source_path)
        checksum = _sha256(source_path)
    else:
        image = image_or_path
        path = "in-memory"
        checksum = "unavailable"

    data = np.asanyarray(image.dataobj)
    numeric_real = np.issubdtype(data.dtype, np.number) and not np.issubdtype(
        data.dtype, np.complexfloating
    )
    finite = np.isfinite(data) if numeric_real else np.zeros(data.shape, dtype=bool)
    nonfinite_count = int(data.size - np.count_nonzero(finite))
    rounded_labels: np.ndarray | None = None
    max_deviation: float | None = None
    integer_like = False
    if numeric_real and bool(np.all(finite)):
        rounded = np.rint(data)
        deviation = np.abs(data - rounded)
        max_deviation = float(deviation.max(initial=0.0))
        integer_like = bool(np.all(np.isclose(data, rounded, atol=1e-6, rtol=0.0)))
        bounds = np.iinfo(np.int64)
        if not rounded.size or (
            rounded.min() >= bounds.min and rounded.max() <= bounds.max
        ):
            rounded_labels = rounded.astype(np.int64, copy=False)

    if rounded_labels is None:
        unique = counts = np.array([], dtype=np.int64)
        border_values: tuple[int, ...] = ()
        suspected_backgrounds: tuple[int, ...] = ()
    else:
        unique, counts = np.unique(rounded_labels, return_counts=True)
        border_sets = _border_sets(rounded_labels)
        border_values = tuple(int(value) for value in sorted(set().union(*border_sets)))
        suspected_backgrounds = (
            tuple(int(value) for value in sorted(set.intersection(*border_sets)))
            if border_sets
            else ()
        )

    unique_labels = tuple(int(value) for value in unique)
    frequencies = tuple(
        (int(label), int(count)) for label, count in zip(unique, counts, strict=True)
    )
    frequency_map = dict(frequencies)
    parcel_identifiers = (
        tuple(
            label
            for label in unique_labels
            if label not in specification.background_values
        )
        if specification is not None
        else unique_labels
    )
    contiguous = not parcel_identifiers or parcel_identifiers == tuple(
        range(parcel_identifiers[0], parcel_identifiers[-1] + 1)
    )
    expected_ids = (
        specification.expected_parcel_ids if specification is not None else None
    )
    expected_nonempty = (
        all(frequency_map.get(label, 0) > 0 for label in expected_ids)
        if expected_ids is not None
        else None
    )
    affine = np.asarray(image.affine)
    affine_finite = bool(affine.shape == (4, 4) and np.isfinite(affine).all())
    determinant = float(np.linalg.det(affine[:3, :3])) if affine_finite else None
    observed_parcel_count = (
        len(parcel_identifiers) if specification is not None else None
    )
    largest_label = int(unique[int(np.argmax(counts))]) if unique.size else None

    return AtlasAudit(
        atlas_id=specification.atlas_id if specification is not None else "unspecified",
        path=path,
        checksum_sha256=checksum,
        shape=tuple(int(value) for value in image.shape),
        ndim=len(image.shape),
        stored_dtype=np.dtype(image.get_data_dtype()).name,
        affine=tuple(tuple(float(value) for value in row) for row in affine),
        voxel_dimensions=tuple(float(value) for value in image.header.get_zooms()[:3]),
        affine_finite=affine_finite,
        spatial_determinant=determinant,
        nonfinite_count=nonfinite_count,
        integer_like=integer_like,
        max_integer_deviation=max_deviation,
        unique_rounded_labels=unique_labels,
        frequencies=frequencies,
        rounded_label_min=unique_labels[0] if unique_labels else None,
        rounded_label_max=unique_labels[-1] if unique_labels else None,
        unique_label_count=len(unique_labels),
        voxel_count=int(data.size),
        border_values=border_values,
        largest_label=largest_label,
        observed_parcel_count=observed_parcel_count,
        expected_parcel_count=(
            specification.expected_parcel_count if specification is not None else None
        ),
        suspected_backgrounds=suspected_backgrounds,
        contiguous=contiguous,
        parcel_0_voxels=frequency_map.get(0, 0),
        parcel_0_nonempty=frequency_map.get(0, 0) > 0,
        expected_parcels_nonempty=expected_nonempty,
        nonempty_labels=unique_labels,
    )


def audit_bundled() -> tuple[AtlasAudit, ...]:
    """Audit every bundled atlas in deterministic ID order."""
    audits: list[AtlasAudit] = []
    for atlas_id in builtin_atlas_ids():
        specification = load_specification(atlas_id)
        audits.append(audit_atlas(specification.resolve_image_path(), specification))
    return tuple(audits)


def _border_sets(labels: np.ndarray) -> tuple[set[int], ...]:
    if labels.ndim != 3 or any(size == 0 for size in labels.shape):
        return ()
    faces = (
        labels[0, :, :],
        labels[-1, :, :],
        labels[:, 0, :],
        labels[:, -1, :],
        labels[:, :, 0],
        labels[:, :, -1],
    )
    return tuple(set(int(value) for value in np.unique(face)) for face in faces)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
