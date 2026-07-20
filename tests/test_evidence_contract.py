from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import nibabel as nib
import numpy as np
import pytest

from neurolabel.core.atlas import load_atlas
from neurolabel.core.parcel import Parcel
from neurolabel.core.specification import AtlasSpecification
from neurolabel.evidence import EvidenceDomain, EvidenceProvider, EvidenceRecord


def record(**overrides: Any) -> EvidenceRecord:
    values: dict[str, Any] = {
        "parcel_key": "synthetic:1",
        "domain": EvidenceDomain.FUNCTION,
        "label": "working memory",
        "provider": "dummy",
        "provider_version": "1.0",
        "source": "synthetic evidence",
        "source_version": "2026-07",
        "method": "fixture",
        "inference_type": "descriptive",
        "statistic_name": "z",
        "statistic_value": 2.5,
        "p_value": 0.01,
        "q_value": 0.02,
        "support_count": 3,
        "metadata": {"nested": {"items": [True, None, 4, 1.5, "value"]}},
    }
    values.update(overrides)
    return EvidenceRecord(**values)


class DummyProvider:
    @property
    def name(self) -> str:
        return "dummy"

    def describe(self, parcel: Parcel) -> Sequence[EvidenceRecord]:
        return (
            record(parcel_key=parcel.key, label="first"),
            record(
                parcel_key=parcel.key,
                label="second",
                domain=EvidenceDomain.CONNECTIVITY,
            ),
        )


def test_valid_record_is_immutable_recursive_json_and_deterministic() -> None:
    evidence = record()

    first = evidence.to_dict()
    second = evidence.to_dict()
    assert first == second
    assert first["metadata"] == {"nested": {"items": [True, None, 4, 1.5, "value"]}}
    assert json.dumps(first) == json.dumps(second)
    assert json.loads(json.dumps(first)) == first

    source = {"z": 1, "a": [2, {"c": 3}]}
    frozen = record(metadata=source)
    source["z"] = 99
    source["a"].append(4)
    assert frozen.to_dict()["metadata"] == {"a": [2, {"c": 3}], "z": 1}


@pytest.mark.parametrize(
    "field",
    [
        "parcel_key",
        "label",
        "provider",
        "source",
        "method",
        "statistic_name",
    ],
)
@pytest.mark.parametrize("value", ["", "   "])
def test_required_strings_must_be_nonempty(field: str, value: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        record(**{field: value})


@pytest.mark.parametrize(
    "field",
    ["provider_version", "source_version", "inference_type"],
)
def test_optional_strings_allow_none_but_not_empty(field: str) -> None:
    assert getattr(record(**{field: None}), field) is None
    with pytest.raises(ValueError, match="must not be empty"):
        record(**{field: ""})


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_statistic_must_be_finite(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        record(statistic_value=value)


@pytest.mark.parametrize("field", ["p_value", "q_value"])
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_probability_bounds_are_closed_unit_interval(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        record(**{field: value})
    assert getattr(record(**{field: 0.0}), field) == 0.0
    assert getattr(record(**{field: 1.0}), field) == 1.0


def test_support_count_allows_none_and_nonnegative_integer() -> None:
    assert record(support_count=None).support_count is None
    assert record(support_count=0).support_count == 0
    with pytest.raises(ValueError, match="nonnegative"):
        record(support_count=-1)
    with pytest.raises(TypeError, match="integer"):
        record(support_count=1.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"statistic_value": np.float64(1.0)},
        {"p_value": np.float64(0.5)},
        {"support_count": np.int64(2)},
        {"metadata": {"numpy": np.array([1])}},
        {"metadata": {"nonfinite": float("nan")}},
        {"metadata": {"bad": {1: "non-string key"}}},
        {"metadata": {"tuple": (1, 2)}},
    ],
)
def test_numpy_nonfinite_and_non_json_values_are_rejected(
    overrides: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        record(**overrides)


def test_cyclic_metadata_is_rejected() -> None:
    metadata: dict[str, object] = {}
    metadata["self"] = metadata

    with pytest.raises(ValueError, match="reference cycles"):
        record(metadata=metadata)


def test_dummy_provider_accepts_canonical_parcel_and_serializes_records(
    make_image: Callable[..., nib.Nifti1Image],
    make_spec: Callable[..., AtlasSpecification],
) -> None:
    parcel = load_atlas(
        make_image(np.array([[[0, 1], [1, 0]]], dtype=np.int8)),
        make_spec(background_values=(0,)),
    ).get_parcel(1)
    provider: EvidenceProvider = DummyProvider()

    records = provider.describe(parcel)

    assert provider.name == "dummy"
    assert isinstance(records, Sequence)
    assert len(records) == 2
    assert all(item.parcel_key == parcel.key for item in records)
    assert all(json.dumps(item.to_dict()) for item in records)
