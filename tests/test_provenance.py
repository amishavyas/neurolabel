from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import nibabel as nib
import numpy as np

from neurolabel.core._version import __version__
from neurolabel.core.atlas import load_atlas
from neurolabel.core.provenance import UNAVAILABLE_CHECKSUM
from neurolabel.core.specification import AtlasSpecification


def test_file_provenance_hashes_exact_source_bytes_deterministically(
    save_nifti: Callable[..., Path],
    make_spec: Callable[..., AtlasSpecification],
) -> None:
    affine = np.array(
        [
            [-2.0, 0.0, 0.0, 90.0],
            [0.0, 2.0, 0.0, -126.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    path = save_nifti(
        np.array([[[0, 1], [2, 1]]], dtype=np.int16),
        affine=affine,
    )
    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    specification = make_spec(
        atlas_id="on-disk",
        background_values=(0,),
        coordinate_space="unknown",
        source_reference="synthetic fixture",
        coordinate_space_status="TO_BE_VERIFIED",
    )

    first = load_atlas(path, specification).provenance
    second = load_atlas(path, specification).provenance

    assert first == second
    assert first.sha256 == expected_hash
    assert first.source == str(path.resolve())
    assert first.image_shape == (1, 2, 2)
    assert first.affine == tuple(tuple(float(value) for value in row) for row in affine)
    assert first.package_version == __version__
    assert first.atlas_id == "on-disk"
    assert first.coordinate_space == "unknown"
    assert first.coordinate_space_status == "TO_BE_VERIFIED"
    assert first.source_reference == "synthetic fixture"


def test_in_memory_provenance_marks_checksum_unavailable(
    make_image: Callable[..., nib.Nifti1Image],
    make_spec: Callable[..., AtlasSpecification],
) -> None:
    atlas = load_atlas(
        make_image(np.array([[[0, 1]]], dtype=np.uint8)),
        make_spec(background_values=(0,)),
    )

    assert atlas.provenance.source == "in-memory"
    assert atlas.provenance.sha256 == UNAVAILABLE_CHECKSUM
    assert atlas.provenance.image_shape == (1, 1, 2)
    assert atlas.provenance.data_dtype == "uint8"
