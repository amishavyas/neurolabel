"""Scientifically conservative validation for discrete NIfTI atlases."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

import nibabel as nib
import numpy as np
from scipy import ndimage

from .specification import AtlasSpecification

Severity = Literal["error", "warning", "info"]
JSONValue: TypeAlias = (
    bool
    | int
    | float
    | str
    | None
    | tuple["JSONValue", ...]
    | Mapping[str, "JSONValue"]
)


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    """One stable, machine-readable atlas validation finding."""

    code: str
    severity: Severity
    message: str
    context: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frozen = {
            str(key): _freeze_json(value) for key, value in sorted(self.context.items())
        }
        object.__setattr__(
            self,
            "context",
            cast(Mapping[str, JSONValue], MappingProxyType(frozen)),
        )


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Structured validation result for one atlas image."""

    findings: tuple[ValidationFinding, ...]
    shape: tuple[int, ...]
    stored_dtype: str
    observed_labels: tuple[int, ...]
    parcel_ids: tuple[int, ...]
    spatial_determinant: float
    max_integer_deviation: float | None

    @property
    def errors(self) -> tuple[ValidationFinding, ...]:
        """Return error-severity findings."""
        return tuple(
            finding for finding in self.findings if finding.severity == "error"
        )

    @property
    def warnings(self) -> tuple[ValidationFinding, ...]:
        """Return warning-severity findings."""
        return tuple(
            finding for finding in self.findings if finding.severity == "warning"
        )

    @property
    def is_valid(self) -> bool:
        """Whether validation found no errors."""
        return not self.errors


@dataclass(frozen=True, slots=True)
class _ValidationResult:
    report: ValidationReport
    labels: np.ndarray | None


def validate_nifti(
    image: nib.spatialimages.SpatialImage,
    specification: AtlasSpecification | None = None,
    *,
    source_path: str | Path | None = None,
) -> ValidationReport:
    """Validate that an image is finite, discrete, and exactly 3D."""
    return _validate(image, specification, source_path=source_path).report


def _validate(
    image: nib.spatialimages.SpatialImage,
    specification: AtlasSpecification | None = None,
    *,
    source_path: str | Path | None = None,
) -> _ValidationResult:
    findings: list[ValidationFinding] = []
    shape = tuple(int(value) for value in image.shape)
    stored_dtype = np.dtype(image.get_data_dtype()).name
    affine = np.asarray(image.affine)
    determinant = float("nan")

    if len(shape) != 3:
        findings.append(
            _finding(
                "INVALID_DIMENSIONALITY",
                "error",
                f"Atlas image must be exactly 3D; observed shape {shape}.",
                shape=shape,
            )
        )

    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        findings.append(
            _finding(
                "INVALID_AFFINE",
                "error",
                "Affine must be a finite 4x4 matrix.",
                shape=tuple(int(value) for value in affine.shape),
            )
        )
    else:
        determinant = float(np.linalg.det(affine[:3, :3]))
        if not np.isfinite(abs(determinant)) or abs(determinant) <= 0:
            findings.append(
                _finding(
                    "SINGULAR_AFFINE",
                    "error",
                    "Affine spatial matrix must have a positive finite absolute "
                    "determinant.",
                    determinant=determinant,
                )
            )

    data = np.asanyarray(image.dataobj)
    labels, observed, max_deviation = _discrete_labels(data, findings)
    parcel_ids: tuple[int, ...] = ()
    if specification is not None:
        _validate_manifest_facts(
            image,
            stored_dtype,
            affine,
            specification,
            source_path,
            findings,
        )
        if labels is not None:
            parcel_ids = _validate_labels(labels, observed, specification, findings)

    report = ValidationReport(
        findings=tuple(findings),
        shape=shape,
        stored_dtype=stored_dtype,
        observed_labels=observed,
        parcel_ids=parcel_ids,
        spatial_determinant=determinant,
        max_integer_deviation=max_deviation,
    )
    return _ValidationResult(report, labels)


def _discrete_labels(
    data: np.ndarray, findings: list[ValidationFinding]
) -> tuple[np.ndarray | None, tuple[int, ...], float | None]:
    if not np.issubdtype(data.dtype, np.number) or np.issubdtype(
        data.dtype, np.complexfloating
    ):
        findings.append(
            _finding(
                "INVALID_LABEL_DATA",
                "error",
                f"Atlas labels must have a real numeric dtype; observed {data.dtype}.",
                dtype=str(data.dtype),
            )
        )
        return None, (), None
    if not data.size:
        findings.append(
            _finding(
                "EMPTY_PARCEL",
                "error",
                "Atlas image contains no voxels.",
            )
        )
        return None, (), None

    finite = np.isfinite(data)
    nonfinite_count = int(data.size - np.count_nonzero(finite))
    if nonfinite_count:
        findings.append(
            _finding(
                "NONFINITE_VALUES",
                "error",
                f"Atlas contains {nonfinite_count} nonfinite voxel value(s).",
                count=nonfinite_count,
            )
        )
        return None, (), None

    rounded = np.rint(data)
    deviations = np.abs(data - rounded)
    max_deviation = float(deviations.max(initial=0.0))
    integer_like = np.isclose(data, rounded, atol=1e-6, rtol=0.0)
    if not bool(np.all(integer_like)):
        count = int(integer_like.size - np.count_nonzero(integer_like))
        findings.append(
            _finding(
                "FRACTIONAL_LABELS",
                "error",
                f"Atlas contains {count} fractional voxel value(s); maximum "
                f"integer deviation is {max_deviation:.17g}. Conversion was not "
                "performed.",
                count=count,
                max_deviation=max_deviation,
            )
        )
        return None, (), max_deviation

    bounds = np.iinfo(np.int64)
    minimum = rounded.min().item()
    maximum = rounded.max().item()
    if minimum < bounds.min or maximum > bounds.max:
        findings.append(
            _finding(
                "LABEL_OUT_OF_RANGE",
                "error",
                "Integer-like labels exceed int64; conversion was not performed.",
            )
        )
        return None, (), max_deviation

    labels = rounded.astype(np.int64, copy=True)
    labels.setflags(write=False)
    observed = tuple(int(value) for value in np.unique(labels))
    return labels, observed, max_deviation


def _validate_manifest_facts(
    image: nib.spatialimages.SpatialImage,
    stored_dtype: str,
    affine: np.ndarray,
    specification: AtlasSpecification,
    source_path: str | Path | None,
    findings: list[ValidationFinding],
) -> None:
    shape = tuple(int(value) for value in image.shape)
    if (
        specification.expected_shape is not None
        and shape != specification.expected_shape
    ):
        findings.append(
            _finding(
                "SHAPE_MISMATCH",
                "error",
                f"Image shape {shape} does not match {specification.expected_shape}.",
                observed=shape,
                expected=specification.expected_shape,
            )
        )

    voxel_sizes = tuple(float(value) for value in image.header.get_zooms()[:3])
    expected_sizes = specification.expected_voxel_sizes
    if expected_sizes is not None and (
        len(voxel_sizes) != 3
        or not np.allclose(voxel_sizes, expected_sizes, atol=1e-6, rtol=0)
    ):
        findings.append(
            _finding(
                "VOXEL_SIZE_MISMATCH",
                "error",
                f"Voxel sizes {voxel_sizes} do not match {expected_sizes}.",
                observed=voxel_sizes,
                expected=expected_sizes,
            )
        )

    expected_dtype = specification.expected_stored_dtype
    if expected_dtype is not None and stored_dtype != np.dtype(expected_dtype).name:
        findings.append(
            _finding(
                "STORED_DTYPE_MISMATCH",
                "error",
                f"Stored dtype {stored_dtype!r} does not match {expected_dtype!r}.",
                observed=stored_dtype,
                expected=expected_dtype,
            )
        )

    expected_affine = specification.expected_affine
    if (
        expected_affine is not None
        and affine.shape == (4, 4)
        and not np.allclose(affine, expected_affine, atol=1e-6, rtol=0)
    ):
        findings.append(
            _finding(
                "AFFINE_MISMATCH",
                "error",
                "Image affine does not match the specification affine.",
            )
        )

    expected_sha256 = specification.expected_sha256
    if expected_sha256 is None:
        return
    if source_path is None:
        findings.append(
            _finding(
                "SOURCE_CHECKSUM_UNAVAILABLE",
                "info",
                "Source-byte checksum is unavailable for an in-memory image.",
            )
        )
        return
    path = Path(source_path)
    if not path.is_file():
        findings.append(
            _finding(
                "SOURCE_PATH_MISSING",
                "error",
                f"Source image path is not a file: {path}.",
                path=str(path),
            )
        )
        return
    checksum = _sha256(path)
    if checksum != expected_sha256:
        findings.append(
            _finding(
                "SOURCE_CHECKSUM_MISMATCH",
                "error",
                f"Source SHA256 {checksum} does not match {expected_sha256}.",
                observed=checksum,
                expected=expected_sha256,
            )
        )


def _validate_labels(
    labels: np.ndarray,
    observed: tuple[int, ...],
    specification: AtlasSpecification,
    findings: list[ValidationFinding],
) -> tuple[int, ...]:
    observed_set = set(observed)
    background = set(specification.background_values)
    expected_ids = specification.expected_parcel_ids
    expected = set(expected_ids or ())
    parcel_ids = tuple(value for value in observed if value not in background)

    overlap = background & expected
    if overlap:
        findings.append(
            _finding(
                "BACKGROUND_OVERLAP",
                "error",
                f"Expected IDs overlap explicit background values: {sorted(overlap)}.",
                labels=tuple(sorted(overlap)),
            )
        )

    if expected_ids is not None:
        for missing_label in sorted(expected - observed_set):
            findings.extend(
                (
                    _finding(
                        "MISSING_EXPECTED_LABEL",
                        "error",
                        f"Expected label {missing_label} is absent.",
                        parcel_id=missing_label,
                    ),
                    _finding(
                        "EMPTY_PARCEL",
                        "error",
                        f"Expected parcel {missing_label} contains no voxels.",
                        parcel_id=missing_label,
                    ),
                )
            )
        for unexpected_label in sorted(observed_set - expected - background):
            findings.append(
                _finding(
                    "UNEXPECTED_LABEL",
                    "error",
                    f"Label {unexpected_label} is not declared by the specification.",
                    parcel_id=unexpected_label,
                )
            )

    missing_background = tuple(sorted(background - observed_set))
    if missing_background:
        findings.append(
            _finding(
                "BACKGROUND_LABEL_ABSENT",
                "warning",
                f"Explicit background labels are absent: {missing_background}.",
                labels=missing_background,
            )
        )

    expected_count = specification.expected_parcel_count
    if expected_count is not None and len(parcel_ids) != expected_count:
        findings.append(
            _finding(
                "LABEL_COUNT_MISMATCH",
                "error",
                f"Observed {len(parcel_ids)} parcel labels; expected {expected_count}.",
                observed=len(parcel_ids),
                expected=expected_count,
            )
        )

    if parcel_ids and parcel_ids != tuple(range(parcel_ids[0], parcel_ids[-1] + 1)):
        findings.append(
            _finding(
                "NONCONTIGUOUS_IDENTIFIERS",
                "info",
                "Observed parcel identifiers are noncontiguous and remain valid.",
                parcel_ids=parcel_ids,
            )
        )

    if labels.ndim == 3:
        structure = ndimage.generate_binary_structure(3, 1)
        for parcel_id in parcel_ids:
            components = int(ndimage.label(labels == parcel_id, structure=structure)[1])
            if components > 1:
                findings.append(
                    _finding(
                        "DISCONNECTED_PARCEL",
                        "warning",
                        f"Parcel {parcel_id} has {components} face-connected "
                        "components.",
                        parcel_id=parcel_id,
                        connected_components=components,
                    )
                )
    return parcel_ids


def _finding(
    code: str,
    severity: Severity,
    message: str,
    **context: JSONValue,
) -> ValidationFinding:
    return ValidationFinding(code, severity, message, context)


def _freeze_json(value: JSONValue) -> JSONValue:
    if isinstance(value, Mapping):
        return cast(
            JSONValue,
            MappingProxyType(
                {str(key): _freeze_json(item) for key, item in sorted(value.items())}
            ),
        )
    if isinstance(value, tuple):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
