"""Provider-neutral evidence value objects."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TypeAlias, cast

JSONValue: TypeAlias = (
    bool | int | float | str | None | list["JSONValue"] | dict[str, "JSONValue"]
)


class EvidenceDomain(str, Enum):
    """A broad, policy-neutral category of scientific evidence."""

    ANATOMY = "anatomy"
    NETWORK = "network"
    FUNCTION = "function"
    TASK = "task"
    CLINICAL = "clinical"
    MICROSTRUCTURE = "microstructure"
    CONNECTIVITY = "connectivity"


def _nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return str(value)


def _optional_nonempty_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, field_name)


def _finite_float(value: object, field_name: str) -> float:
    if (
        _is_numpy_value(value)
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise TypeError(f"{field_name} must be a number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _is_numpy_value(value: object) -> bool:
    return type(value).__module__.partition(".")[0] == "numpy"


def _freeze_json(
    value: object,
    path: str,
    active_containers: set[int] | None = None,
) -> object:
    if _is_numpy_value(value):
        raise TypeError(f"{path} must not contain NumPy values")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{path} must contain only finite numbers")
        return number
    if isinstance(value, str):
        return str(value)
    if isinstance(value, (Mapping, list)):
        if active_containers is None:
            active_containers = set()
        identity = id(value)
        if identity in active_containers:
            raise ValueError(f"{path} must not contain reference cycles")
        active_containers.add(identity)
        try:
            if isinstance(value, Mapping):
                frozen: dict[str, object] = {}
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise TypeError(f"{path} mapping keys must be strings")
                    frozen[str(key)] = _freeze_json(
                        item,
                        f"{path}.{key}",
                        active_containers,
                    )
                return MappingProxyType(dict(sorted(frozen.items())))
            return tuple(
                _freeze_json(
                    item,
                    f"{path}[{index}]",
                    active_containers,
                )
                for index, item in enumerate(value)
            )
        finally:
            active_containers.remove(identity)
    raise TypeError(f"{path} must contain only JSON-compatible values")


def _thaw_json(value: object) -> JSONValue:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return cast(JSONValue, value)


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """One provider's evidence about one canonical parcel."""

    parcel_key: str
    domain: EvidenceDomain
    label: str
    provider: str
    provider_version: str | None
    source: str
    source_version: str | None
    method: str
    inference_type: str | None
    statistic_name: str
    statistic_value: float
    p_value: float | None = None
    q_value: float | None = None
    support_count: int | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "parcel_key",
            "label",
            "provider",
            "source",
            "method",
            "statistic_name",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonempty_string(getattr(self, field_name), field_name),
            )
        for field_name in ("provider_version", "source_version", "inference_type"):
            object.__setattr__(
                self,
                field_name,
                _optional_nonempty_string(getattr(self, field_name), field_name),
            )

        if not isinstance(self.domain, EvidenceDomain):
            raise TypeError("domain must be an EvidenceDomain")

        object.__setattr__(
            self,
            "statistic_value",
            _finite_float(self.statistic_value, "statistic_value"),
        )
        for field_name in ("p_value", "q_value"):
            value = getattr(self, field_name)
            if value is None:
                continue
            probability = _finite_float(value, field_name)
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")
            object.__setattr__(self, field_name, probability)

        if self.support_count is not None:
            if (
                _is_numpy_value(self.support_count)
                or isinstance(self.support_count, bool)
                or not isinstance(self.support_count, int)
            ):
                raise TypeError("support_count must be an integer")
            if self.support_count < 0:
                raise ValueError("support_count must be nonnegative")
            object.__setattr__(self, "support_count", int(self.support_count))
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        frozen_metadata = _freeze_json(self.metadata, "metadata")
        object.__setattr__(
            self,
            "metadata",
            cast(Mapping[str, JSONValue], frozen_metadata),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a deterministic dictionary containing only plain JSON values."""

        return {
            "parcel_key": self.parcel_key,
            "domain": self.domain.value,
            "label": self.label,
            "provider": self.provider,
            "provider_version": self.provider_version,
            "source": self.source,
            "source_version": self.source_version,
            "method": self.method,
            "inference_type": self.inference_type,
            "statistic_name": self.statistic_name,
            "statistic_value": self.statistic_value,
            "p_value": self.p_value,
            "q_value": self.q_value,
            "support_count": self.support_count,
            "metadata": _thaw_json(self.metadata),
        }
