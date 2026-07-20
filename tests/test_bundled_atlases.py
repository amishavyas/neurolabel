from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from neurolabel.core.atlas import load_atlas
from neurolabel.core.audit import audit_atlas
from neurolabel.core.specification import builtin_atlas_ids, load_specification

EXPECTED_BUILTINS = (
    ("neurosynth_k50", 50),
    ("neurosynth_k100", 100),
    ("neurosynth_k200", 200),
)


def test_builtin_manifest_order_is_deterministic() -> None:
    assert builtin_atlas_ids() == tuple(atlas_id for atlas_id, _ in EXPECTED_BUILTINS)


@pytest.mark.parametrize(("atlas_id", "parcel_count"), EXPECTED_BUILTINS)
def test_bundled_atlas_contract_without_network(
    atlas_id: str,
    parcel_count: int,
) -> None:
    specification = load_specification(atlas_id)
    path = specification.resolve_image_path()
    original_bytes = path.read_bytes()

    atlas = load_atlas(atlas_id)
    labels, counts = np.unique(atlas.labels, return_counts=True)
    expected_ids = tuple(range(1, parcel_count + 1))

    assert atlas.atlas_id == atlas_id
    assert atlas.parcel_ids == expected_ids
    assert specification.expected_parcel_count == parcel_count
    assert specification.expected_parcel_ids == expected_ids
    assert specification.background_values == (0,)
    assert specification.coordinate_space == "unknown"
    assert specification.coordinate_space_status == "TO_BE_VERIFIED"
    assert atlas.validation_report.is_valid
    assert atlas.validation_report.observed_labels == (0, *expected_ids)
    assert atlas.validation_report.max_integer_deviation is not None
    assert (
        atlas.validation_report.max_integer_deviation
        <= specification.max_integer_deviation
    )
    assert np.isfinite(atlas.labels).all()
    assert np.equal(atlas.labels, np.rint(atlas.labels)).all()
    assert tuple(int(value) for value in labels) == (0, *expected_ids)
    assert np.all(counts > 0)

    for parcel_id in (1, parcel_count):
        parcel = atlas.get_parcel(parcel_id)
        assert parcel.parcel_id == parcel_id
        assert parcel.voxel_count > 0
        assert parcel.mask.dtype == np.uint8

    assert hashlib.sha256(original_bytes).hexdigest() == specification.expected_sha256
    assert atlas.provenance.sha256 == specification.expected_sha256
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize(("atlas_id", "parcel_count"), EXPECTED_BUILTINS)
def test_bundled_audit_records_observed_background_and_file_facts(
    atlas_id: str,
    parcel_count: int,
) -> None:
    specification = load_specification(atlas_id)

    first = audit_atlas(specification.resolve_image_path(), specification)
    second = audit_atlas(specification.resolve_image_path(), specification)

    assert first == second
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(
        second.to_dict(),
        sort_keys=True,
    )
    assert first.atlas_id == atlas_id
    assert first.checksum_sha256 == specification.expected_sha256
    assert first.ndim == 3
    assert first.shape == specification.expected_shape
    assert first.stored_dtype == specification.expected_stored_dtype
    assert first.affine_finite
    assert first.nonfinite_count == 0
    assert first.integer_like
    assert first.max_integer_deviation == specification.max_integer_deviation
    assert first.largest_label == 0
    assert first.observed_parcel_count == parcel_count
    assert first.expected_parcel_count == parcel_count
    assert first.expected_parcels_nonempty is True
    assert first.parcel_0_nonempty is True
    assert first.parcel_0_voxels > 0
    assert first.unique_rounded_labels == tuple(range(parcel_count + 1))
