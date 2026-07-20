from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from neurolabel.core.specification import AtlasSpecification

DEFAULT_AFFINE = np.array(
    [
        [2.0, 0.0, 0.0, 10.0],
        [0.0, 3.0, 0.0, -5.0],
        [0.0, 0.0, 4.0, 7.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


@pytest.fixture
def make_image() -> Callable[..., nib.Nifti1Image]:
    def factory(
        data: ArrayLike,
        affine: ArrayLike | None = None,
    ) -> nib.Nifti1Image:
        image_affine = (
            DEFAULT_AFFINE.copy() if affine is None else np.asarray(affine, dtype=float)
        )
        return nib.Nifti1Image(
            np.asarray(data),
            image_affine,
        )

    return factory


@pytest.fixture
def make_spec() -> Callable[..., AtlasSpecification]:
    def factory(
        *,
        atlas_id: str = "synthetic",
        background_values: Iterable[int] = (-1,),
        expected_parcel_count: int | None = None,
        expected_parcel_ids: Iterable[int] | None = None,
        **kwargs: object,
    ) -> AtlasSpecification:
        return AtlasSpecification(
            atlas_id=atlas_id,
            display=f"{atlas_id} atlas",
            background_values=tuple(background_values),
            expected_parcel_count=expected_parcel_count,
            expected_parcel_ids=(
                None if expected_parcel_ids is None else tuple(expected_parcel_ids)
            ),
            **kwargs,
        )

    return factory


@pytest.fixture
def save_nifti(tmp_path: Path) -> Callable[..., Path]:
    def save(
        data: ArrayLike,
        *,
        affine: ArrayLike | None = None,
        name: str = "atlas.nii.gz",
    ) -> Path:
        path = tmp_path / name
        image_affine = (
            DEFAULT_AFFINE.copy() if affine is None else np.asarray(affine, dtype=float)
        )
        image = nib.Nifti1Image(
            np.asarray(data),
            image_affine,
        )
        nib.save(image, path)
        return path

    return save


@pytest.fixture
def simple_labels() -> NDArray[np.int16]:
    return np.array(
        [
            [[-1, 0], [1, 1]],
            [[2, 2], [0, -1]],
        ],
        dtype=np.int16,
    )
