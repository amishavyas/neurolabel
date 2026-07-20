from __future__ import annotations

from collections.abc import Callable

import nibabel as nib
import numpy as np
import pytest

from neurolabel.core.atlas import load_atlas
from neurolabel.core.parcel import ParcelGeometry
from neurolabel.core.specification import AtlasSpecification


def test_geometry_uses_voxel_centers_determinant_and_face_connectivity(
    make_image: Callable[..., nib.Nifti1Image],
    make_spec: Callable[..., AtlasSpecification],
) -> None:
    labels = np.zeros((4, 5, 6), dtype=np.int8)
    coordinates = ((0, 1, 1), (0, 1, 2), (2, 3, 4), (3, 3, 4))
    for coordinate in coordinates:
        labels[coordinate] = 7
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, -5.0],
            [0.0, 0.0, 4.0, 7.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    parcel = load_atlas(
        make_image(labels, affine),
        make_spec(background_values=(0,)),
    ).get_parcel(7)

    assert parcel.geometry.voxel_count == 4
    assert parcel.geometry.volume_mm3 == pytest.approx(96.0)
    assert parcel.geometry.centroid_voxel == pytest.approx((1.25, 2.0, 2.75))
    assert parcel.geometry.centroid_world == pytest.approx((12.5, 1.0, 18.0))
    assert parcel.geometry.bounding_box == ((0, 4), (1, 4), (1, 5))
    assert parcel.geometry.connected_components == 2
    assert parcel.count == parcel.voxel_count == 4
    assert parcel.volume == parcel.volume_mm3 == pytest.approx(96.0)
    assert parcel.centroid == parcel.centroid_world


def test_diagonal_touching_voxels_are_not_face_connected() -> None:
    mask = np.zeros((2, 2, 2), dtype=np.uint8)
    mask[0, 0, 0] = mask[1, 1, 1] = 1

    geometry = ParcelGeometry.from_mask(mask, np.eye(4))

    assert geometry.connected_components == 2


def test_geometry_rejects_empty_mask() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        ParcelGeometry.from_mask(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4))
