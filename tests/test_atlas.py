from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurolabel.core.atlas import Atlas, load_atlas
from neurolabel.core.errors import AtlasSpecificationError, UnknownParcelError
from neurolabel.core.specification import AtlasSpecification


def test_explicit_negative_background_preserves_zero_as_parcel(
    make_image: Callable[..., nib.Nifti1Image],
    make_spec: Callable[..., AtlasSpecification],
) -> None:
    image = make_image(
        np.array(
            [
                [[-1, 0], [1, 2]],
                [[0, 1], [2, -1]],
            ],
            dtype=np.int16,
        )
    )
    atlas = load_atlas(
        image,
        make_spec(
            atlas_id="zero-indexed",
            background_values=(-1,),
            expected_parcel_count=3,
            expected_parcel_ids=(0, 1, 2),
        ),
    )

    assert atlas.parcel_ids == (0, 1, 2)
    parcel = atlas.get_parcel(0)
    assert parcel.parcel_id == 0
    assert parcel.key == "zero-indexed:0"
    assert parcel.voxel_count == 2
    assert parcel.mask.sum() == 2
    assert parcel.mask.flags.writeable is False
    assert atlas.parcel(0) is not parcel
    assert atlas.parcel(0).geometry == parcel.geometry


def test_explicit_zero_background_excludes_zero(
    make_image: Callable[..., nib.Nifti1Image],
    make_spec: Callable[..., AtlasSpecification],
) -> None:
    atlas = Atlas(
        make_image(np.array([[[0, 1, 2]]], dtype=np.int8)),
        make_spec(background_values=(0,)),
    )

    assert atlas.parcel_ids == (1, 2)


def test_noncontiguous_identifiers_remain_supported(
    make_image: Callable[..., nib.Nifti1Image],
    make_spec: Callable[..., AtlasSpecification],
) -> None:
    atlas = load_atlas(
        make_image(np.array([[[0, 3, 11]]], dtype=np.int8)),
        make_spec(background_values=(-1,)),
    )

    assert atlas.parcel_ids == (0, 3, 11)
    assert all(atlas.get_parcel(parcel_id).voxel_count == 1 for parcel_id in (0, 3, 11))
    finding = next(
        item
        for item in atlas.validation_report.findings
        if item.code == "NONCONTIGUOUS_IDENTIFIERS"
    )
    assert finding.severity == "info"
    assert finding.context["parcel_ids"] == (0, 3, 11)


def test_unknown_parcel_names_request_atlas_and_available_ids(
    make_image: Callable[..., nib.Nifti1Image],
    make_spec: Callable[..., AtlasSpecification],
) -> None:
    atlas = load_atlas(
        make_image(np.array([[[0, 1, 2, 4, 5]]], dtype=np.int8)),
        make_spec(atlas_id="range-test", background_values=(-1,)),
    )

    with pytest.raises(UnknownParcelError) as captured:
        atlas.get_parcel(9)

    error = captured.value
    assert error.requested_id == 9
    assert error.atlas_id == "range-test"
    assert error.available_ids == (0, 1, 2, 4, 5)
    assert "9" in str(error)
    assert "range-test" in str(error)
    assert "0..2, 4..5" in str(error)


def test_loading_requires_a_specification_for_image_or_path(
    make_image: Callable[..., nib.Nifti1Image],
    save_nifti: Callable[..., Path],
) -> None:
    image = make_image(np.zeros((1, 1, 1), dtype=np.int8))
    path = save_nifti(np.zeros((1, 1, 1), dtype=np.int8))

    with pytest.raises(AtlasSpecificationError, match="in-memory"):
        load_atlas(image)
    with pytest.raises(AtlasSpecificationError, match="requires"):
        load_atlas(path)


def test_loading_and_extraction_do_not_mutate_source(
    make_image: Callable[..., nib.Nifti1Image],
    make_spec: Callable[..., AtlasSpecification],
) -> None:
    data = np.array([[[0, 1], [2, 1]]], dtype=np.int16)
    image = make_image(data)
    original_data = np.asanyarray(image.dataobj).copy()
    original_affine = image.affine.copy()
    original_header = bytes(image.header.binaryblock)

    atlas = load_atlas(image, make_spec(background_values=(0,)))
    atlas.get_parcel(1)

    assert atlas.image is image
    assert atlas.header is image.header
    assert_array_equal(atlas.affine, original_affine)
    assert_array_equal(np.asanyarray(image.dataobj), original_data)
    assert_array_equal(image.affine, original_affine)
    assert bytes(image.header.binaryblock) == original_header


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "background_values": (0,),
                "expected_parcel_count": 1,
                "expected_parcel_ids": (0,),
            },
            "BACKGROUND_OVERLAP",
        ),
        (
            {
                "background_values": (0,),
                "expected_parcel_count": 2,
                "expected_parcel_ids": (1,),
            },
            "does not match",
        ),
    ],
)
def test_invalid_specification_expectations_are_rejected(
    make_spec: Callable[..., AtlasSpecification],
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(AtlasSpecificationError, match=message):
        make_spec(**kwargs)
