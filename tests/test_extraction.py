from __future__ import annotations

from collections.abc import Callable

import nibabel as nib
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurolabel.core.errors import AtlasValidationError, UnknownParcelError
from neurolabel.core.extraction import extract_label_mask
from utils import extract_parcel_mask


def test_extracts_uint8_mask_and_preserves_geometry(
    make_image: Callable[..., nib.Nifti1Image],
) -> None:
    affine = np.array(
        [
            [-2.0, 0.0, 0.0, 90.0],
            [0.0, 3.0, 0.0, -126.0],
            [0.0, 0.0, 4.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    image = make_image(np.array([[[0, 1], [1, 2]]], dtype=np.int16), affine)
    source_zooms = image.header.get_zooms()

    mask, mask_image = extract_label_mask(image, 1)

    assert mask.dtype == np.uint8
    assert mask.flags.writeable is False
    assert_array_equal(mask, np.array([[[0, 1], [1, 0]]], dtype=np.uint8))
    assert mask_image.shape == image.shape
    assert_array_equal(mask_image.affine, affine)
    assert mask_image.header.get_data_dtype() == np.dtype(np.uint8)
    assert mask_image.header.get_zooms() == source_zooms
    assert_array_equal(np.asanyarray(mask_image.dataobj), mask)


def test_low_level_extraction_allows_label_zero(
    make_image: Callable[..., nib.Nifti1Image],
) -> None:
    image = make_image(np.array([[[0, 1], [0, 2]]], dtype=np.int8))

    mask, _ = extract_label_mask(image, 0)

    assert mask.sum() == 2


@pytest.mark.parametrize("label", [1.0, 1.25, True, np.int64(1).item() + 0.0])
def test_rejects_noninteger_requested_label(
    make_image: Callable[..., nib.Nifti1Image],
    label: object,
) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        extract_label_mask(make_image(np.ones((1, 1, 1), dtype=np.int8)), label)


def test_absent_label_reports_available_labels(
    make_image: Callable[..., nib.Nifti1Image],
) -> None:
    with pytest.raises(UnknownParcelError) as captured:
        extract_label_mask(
            make_image(np.array([[[0, 3, 11]]], dtype=np.int8)),
            7,
        )

    error = captured.value
    assert error.requested_id == 7
    assert error.available_ids == (0, 3, 11)
    assert "<in-memory-image>" in str(error)
    assert "0, 3, 11" in str(error)


def test_rejects_invalid_atlas_instead_of_rounding(
    make_image: Callable[..., nib.Nifti1Image],
) -> None:
    image = make_image(np.array([[[0.0, 1.25]]]))

    with pytest.raises(AtlasValidationError) as captured:
        extract_label_mask(image, 1)

    assert [finding.code for finding in captured.value.report.errors] == [
        "FRACTIONAL_LABELS"
    ]


def test_path_and_compatibility_wrapper_return_same_two_values(
    save_nifti: Callable[..., object],
) -> None:
    path = save_nifti(np.array([[[0, 2], [2, 3]]], dtype=np.int16))

    direct = extract_label_mask(path, 2)
    wrapped = extract_parcel_mask(path, 2)
    wrapped_by_name = extract_parcel_mask(nifti_file_name=path, parcel_num=2)

    assert len(direct) == len(wrapped) == len(wrapped_by_name) == 2
    assert_array_equal(wrapped[0], direct[0])
    assert_array_equal(wrapped_by_name[0], direct[0])
    assert_array_equal(wrapped[1].affine, direct[1].affine)
    assert_array_equal(np.asanyarray(wrapped[1].dataobj), direct[0])


def test_extraction_does_not_mutate_source_image(
    make_image: Callable[..., nib.Nifti1Image],
) -> None:
    source_data = np.array([[[0, 1], [2, 1]]], dtype=np.int16)
    image = make_image(source_data)
    before_data = np.asanyarray(image.dataobj).copy()
    before_affine = image.affine.copy()
    before_header = bytes(image.header.binaryblock)

    extract_label_mask(image, 1)

    assert_array_equal(np.asanyarray(image.dataobj), before_data)
    assert_array_equal(image.affine, before_affine)
    assert bytes(image.header.binaryblock) == before_header
