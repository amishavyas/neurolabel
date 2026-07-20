from __future__ import annotations

from collections.abc import Callable

import nibabel as nib
import numpy as np
import pytest

from neurolabel.core.atlas import load_atlas
from neurolabel.core.errors import AtlasValidationError
from neurolabel.core.specification import AtlasSpecification
from neurolabel.core.validation import validate_nifti


def error_codes(report: object) -> set[str]:
    return {finding.code for finding in report.errors}


def test_fractional_labels_are_errors_and_never_rounded(
    make_image: Callable[..., nib.Nifti1Image],
    make_spec: Callable[..., AtlasSpecification],
) -> None:
    image = make_image(np.array([[[0.0, 1.0, 1.25]]]))
    report = validate_nifti(image, make_spec(background_values=(-1,)))

    assert report.is_valid is False
    assert report.observed_labels == ()
    assert report.parcel_ids == ()
    assert report.max_integer_deviation == pytest.approx(0.25)
    assert error_codes(report) == {"FRACTIONAL_LABELS"}

    with pytest.raises(AtlasValidationError) as captured:
        load_atlas(image, make_spec(background_values=(-1,)))
    assert error_codes(captured.value.report) == {"FRACTIONAL_LABELS"}


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_voxel_values_are_rejected(
    make_image: Callable[..., nib.Nifti1Image],
    make_spec: Callable[..., AtlasSpecification],
    value: float,
) -> None:
    report = validate_nifti(
        make_image(np.array([[[0.0, value]]])),
        make_spec(),
    )

    assert error_codes(report) == {"NONFINITE_VALUES"}
    assert report.observed_labels == ()


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4, 1)])
def test_only_exactly_three_dimensions_are_accepted(
    make_image: Callable[..., nib.Nifti1Image],
    make_spec: Callable[..., AtlasSpecification],
    shape: tuple[int, ...],
) -> None:
    report = validate_nifti(make_image(np.zeros(shape, dtype=np.int8)), make_spec())

    assert "INVALID_DIMENSIONALITY" in error_codes(report)


def test_nonfinite_affine_is_rejected(
    make_image: Callable[..., nib.Nifti1Image],
    make_spec: Callable[..., AtlasSpecification],
) -> None:
    image = make_image(np.zeros((2, 2, 2), dtype=np.int8))
    affine = image.affine.copy()
    affine[0, 0] = np.nan
    image._affine = affine

    report = validate_nifti(image, make_spec())

    assert "INVALID_AFFINE" in error_codes(report)


def test_singular_affine_is_rejected(
    make_image: Callable[..., nib.Nifti1Image],
    make_spec: Callable[..., AtlasSpecification],
) -> None:
    image = make_image(np.zeros((2, 2, 2), dtype=np.int8))
    affine = image.affine.copy()
    affine[2] = 0
    image._affine = affine

    report = validate_nifti(image, make_spec())

    assert "SINGULAR_AFFINE" in error_codes(report)
    assert report.spatial_determinant == 0.0


def test_manifest_expectations_report_stable_codes(
    make_image: Callable[..., nib.Nifti1Image],
    make_spec: Callable[..., AtlasSpecification],
) -> None:
    image = make_image(np.array([[[0, 1, 3]]], dtype=np.int16))
    specification = make_spec(
        background_values=(0,),
        expected_parcel_count=2,
        expected_parcel_ids=(1, 2),
        expected_shape=(2, 2, 2),
        expected_voxel_sizes=(1.0, 1.0, 1.0),
        expected_affine=tuple(tuple(row) for row in np.eye(4)),
        expected_stored_dtype="uint8",
    )

    report = validate_nifti(image, specification)

    assert {
        "AFFINE_MISMATCH",
        "EMPTY_PARCEL",
        "MISSING_EXPECTED_LABEL",
        "SHAPE_MISMATCH",
        "STORED_DTYPE_MISMATCH",
        "UNEXPECTED_LABEL",
        "VOXEL_SIZE_MISMATCH",
    } <= error_codes(report)


def test_missing_declared_background_is_only_a_warning(
    make_image: Callable[..., nib.Nifti1Image],
    make_spec: Callable[..., AtlasSpecification],
) -> None:
    report = validate_nifti(
        make_image(np.ones((2, 2, 2), dtype=np.int8)),
        make_spec(background_values=(0,), expected_parcel_ids=(1,)),
    )

    assert report.is_valid
    assert [finding.code for finding in report.warnings] == ["BACKGROUND_LABEL_ABSENT"]


def test_expected_checksum_is_unavailable_for_in_memory_image(
    make_image: Callable[..., nib.Nifti1Image],
    make_spec: Callable[..., AtlasSpecification],
) -> None:
    report = validate_nifti(
        make_image(np.ones((1, 1, 1), dtype=np.int8)),
        make_spec(expected_sha256="0" * 64),
    )

    assert any(
        finding.code == "SOURCE_CHECKSUM_UNAVAILABLE" and finding.severity == "info"
        for finding in report.findings
    )
