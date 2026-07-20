"""Atlas specification loading and built-in manifest resolution."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import AtlasSpecificationError

try:
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    try:
        import tomli as _toml  # type: ignore[no-redef]
    except ModuleNotFoundError:  # pragma: no cover - environment dependent
        _toml = None  # type: ignore[assignment]


_MANIFEST_DIRECTORY = Path(__file__).parents[1] / "resources" / "atlases"


@dataclass(frozen=True, slots=True)
class AtlasSpecification:
    """Declarative atlas semantics plus optional audited file facts."""

    atlas_id: str
    display: str
    background_values: tuple[int, ...]
    expected_parcel_count: int | None = None
    expected_parcel_ids: tuple[int, ...] | None = None
    coordinate_space: str | None = None
    source_reference: str | None = None
    notes: tuple[str, ...] = ()
    image_path: str | None = None
    expected_sha256: str | None = None
    expected_shape: tuple[int, int, int] | None = None
    expected_voxel_sizes: tuple[float, float, float] | None = None
    expected_affine: tuple[tuple[float, float, float, float], ...] | None = None
    expected_stored_dtype: str | None = None
    max_integer_deviation: float | None = None
    coordinate_space_status: str | None = None
    manifest_path: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_specification(self)

    @property
    def display_name(self) -> str:
        """Return the human-readable atlas name."""
        return self.display

    def resolve_image_path(self) -> Path:
        """Resolve the declared repository image path.

        Raises
        ------
        AtlasSpecificationError
            If this synthetic or user specification has no image path.
        """
        if self.image_path is None:
            raise AtlasSpecificationError(
                f"Atlas {self.atlas_id!r} has no image_path. Pass an image/path "
                "directly to load_atlas or add image_path to its manifest."
            )
        path = Path(self.image_path).expanduser()
        if not path.is_absolute():
            path = _repository_root() / path
        return path.resolve()


def builtin_atlas_ids() -> tuple[str, ...]:
    """Return built-in atlas IDs in deterministic order."""
    paths = sorted(_MANIFEST_DIRECTORY.glob("*.toml"), key=_manifest_sort_key)
    return tuple(path.stem for path in paths)


def load_specification(source: str | Path) -> AtlasSpecification:
    """Load a built-in ID or TOML path into an immutable specification."""
    path = Path(source).expanduser()
    if not path.exists():
        candidate = _MANIFEST_DIRECTORY / f"{source}.toml"
        if candidate.is_file():
            path = candidate
        else:
            builtins = ", ".join(builtin_atlas_ids()) or "none"
            raise AtlasSpecificationError(
                f"No atlas specification found for {str(source)!r}. "
                f"Use a TOML path or one of the built-in IDs: {builtins}."
            )
    if not path.is_file():
        raise AtlasSpecificationError(
            f"Atlas specification path is not a file: {path}."
        )
    if _toml is None:
        raise AtlasSpecificationError(
            "Reading TOML on Python 3.10 requires the 'tomli' package. "
            "Install tomli or use Python 3.11+."
        )
    try:
        with path.open("rb") as stream:
            return _from_mapping(_toml.load(stream), path.resolve())
    except AtlasSpecificationError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise AtlasSpecificationError(
            f"Could not parse atlas specification {path}: {error}."
        ) from error


def _from_mapping(values: dict[str, Any], path: Path) -> AtlasSpecification:
    try:
        return AtlasSpecification(
            atlas_id=_required_text(values, "atlas_id"),
            display=_required_text(values, "display"),
            background_values=tuple(_sequence(values, "background_values", None, int)),
            expected_parcel_count=_optional_integer(values, "expected_parcel_count"),
            expected_parcel_ids=_expected_ids(values),
            coordinate_space=_optional_text(values, "coordinate_space"),
            source_reference=_optional_text(values, "source_reference"),
            notes=tuple(_sequence(values, "notes", None, str, default=[])),
            image_path=_optional_text(values, "image_path"),
            expected_sha256=_optional_text(values, "expected_sha256"),
            expected_shape=_optional_tuple(values, "expected_shape", 3, int),
            expected_voxel_sizes=_optional_float_tuple(
                values, "expected_voxel_sizes", 3
            ),
            expected_affine=_optional_affine(values),
            expected_stored_dtype=_optional_text(values, "expected_stored_dtype"),
            max_integer_deviation=_optional_float(values, "max_integer_deviation"),
            coordinate_space_status=_optional_text(values, "coordinate_space_status"),
            manifest_path=path,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AtlasSpecificationError(
            f"Invalid atlas specification {path}: {error}."
        ) from error


def _validate_specification(specification: AtlasSpecification) -> None:
    prefix = f"Invalid AtlasSpecification {specification.atlas_id!r}"
    if (
        not isinstance(specification.atlas_id, str)
        or not specification.atlas_id.strip()
    ):
        raise AtlasSpecificationError("atlas_id must be a non-empty string.")
    if not isinstance(specification.display, str) or not specification.display.strip():
        raise AtlasSpecificationError(f"{prefix}: display must be a non-empty string.")
    _integer_tuple(specification.background_values, "background_values", prefix)
    if len(set(specification.background_values)) != len(
        specification.background_values
    ):
        raise AtlasSpecificationError(f"{prefix}: background values must be unique.")

    expected_ids = specification.expected_parcel_ids
    if expected_ids is not None:
        _integer_tuple(expected_ids, "expected_parcel_ids", prefix)
        if len(set(expected_ids)) != len(expected_ids):
            raise AtlasSpecificationError(
                f"{prefix}: expected parcel IDs must be unique."
            )
        if set(expected_ids) & set(specification.background_values):
            raise AtlasSpecificationError(
                f"{prefix}: BACKGROUND_OVERLAP: expected IDs and explicit "
                "background values must not overlap."
            )

    count = specification.expected_parcel_count
    if count is not None and (
        isinstance(count, bool) or not isinstance(count, int) or count < 0
    ):
        raise AtlasSpecificationError(
            f"{prefix}: expected_parcel_count must be a nonnegative integer."
        )
    if count is not None and expected_ids is not None and count != len(expected_ids):
        raise AtlasSpecificationError(
            f"{prefix}: expected_parcel_count does not match expected_parcel_ids."
        )

    _optional_nonempty_text(specification.coordinate_space, "coordinate_space", prefix)
    _optional_nonempty_text(specification.source_reference, "source_reference", prefix)
    _optional_nonempty_text(
        specification.coordinate_space_status, "coordinate_space_status", prefix
    )
    if not isinstance(specification.notes, tuple) or any(
        not isinstance(note, str) or not note.strip() for note in specification.notes
    ):
        raise AtlasSpecificationError(
            f"{prefix}: notes must be a tuple of non-empty strings."
        )
    _validate_optional_file_facts(specification, prefix)


def _validate_optional_file_facts(
    specification: AtlasSpecification, prefix: str
) -> None:
    _optional_nonempty_text(specification.image_path, "image_path", prefix)
    _optional_nonempty_text(
        specification.expected_stored_dtype, "expected_stored_dtype", prefix
    )
    checksum = specification.expected_sha256
    if checksum is not None and (
        len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        raise AtlasSpecificationError(
            f"{prefix}: expected_sha256 must contain 64 lowercase hex characters."
        )
    shape = specification.expected_shape
    if shape is not None:
        _integer_tuple(shape, "expected_shape", prefix)
        if len(shape) != 3 or any(size <= 0 for size in shape):
            raise AtlasSpecificationError(
                f"{prefix}: expected_shape must contain three positive integers."
            )
    sizes = specification.expected_voxel_sizes
    if sizes is not None and (
        not isinstance(sizes, tuple)
        or len(sizes) != 3
        or any(not _positive_finite(value) for value in sizes)
    ):
        raise AtlasSpecificationError(
            f"{prefix}: expected voxel sizes must be three positive finite numbers."
        )
    affine = specification.expected_affine
    if affine is not None and (
        not isinstance(affine, tuple)
        or len(affine) != 4
        or any(
            not isinstance(row, tuple)
            or len(row) != 4
            or any(not _finite_number(value) for value in row)
            for row in affine
        )
    ):
        raise AtlasSpecificationError(
            f"{prefix}: expected_affine must be a finite 4x4 numeric tuple."
        )
    deviation = specification.max_integer_deviation
    if deviation is not None and (
        not _finite_number(deviation) or float(deviation) < 0
    ):
        raise AtlasSpecificationError(
            f"{prefix}: max_integer_deviation must be finite and nonnegative."
        )


def _repository_root() -> Path:
    return Path(__file__).parents[3]


def _manifest_sort_key(path: Path) -> tuple[str, int | str]:
    prefix, separator, suffix = path.stem.rpartition("_k")
    return (prefix, int(suffix)) if separator and suffix.isdigit() else (path.stem, "")


def _integer_tuple(values: object, name: str, prefix: str) -> None:
    if not isinstance(values, tuple) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        raise AtlasSpecificationError(f"{prefix}: {name} must be an integer tuple.")


def _optional_nonempty_text(value: object, name: str, prefix: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise AtlasSpecificationError(
            f"{prefix}: {name} must be None or a non-empty string."
        )


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _positive_finite(value: object) -> bool:
    return _finite_number(value) and float(value) > 0


def _required_text(values: dict[str, Any], key: str) -> str:
    value = values[key]
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{key} must be a non-empty string")
    return value


def _optional_text(values: dict[str, Any], key: str) -> str | None:
    return _required_text(values, key) if key in values else None


def _optional_integer(values: dict[str, Any], key: str) -> int | None:
    if key not in values:
        return None
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _optional_float(values: dict[str, Any], key: str) -> float | None:
    if key not in values:
        return None
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a number")
    return float(value)


def _expected_ids(values: dict[str, Any]) -> tuple[int, ...] | None:
    if "expected_parcel_ids" in values:
        return tuple(_sequence(values, "expected_parcel_ids", None, int))
    if "expected_id_range" not in values:
        return None
    start, stop = _sequence(values, "expected_id_range", 2, int)
    if stop < start:
        raise ValueError("expected_id_range stop must be at least its start")
    return tuple(range(start, stop + 1))


def _optional_tuple(
    values: dict[str, Any],
    key: str,
    length: int,
    item_type: type[Any],
) -> tuple[Any, ...] | None:
    return tuple(_sequence(values, key, length, item_type)) if key in values else None


def _optional_float_tuple(
    values: dict[str, Any], key: str, length: int
) -> tuple[float, ...] | None:
    return (
        tuple(float(value) for value in _sequence(values, key, length, (int, float)))
        if key in values
        else None
    )


def _optional_affine(
    values: dict[str, Any],
) -> tuple[tuple[float, float, float, float], ...] | None:
    if "expected_affine" not in values:
        return None
    rows = _sequence(values, "expected_affine", 4, list)
    return tuple(
        tuple(float(value) for value in _sized(row, "expected_affine row", 4))
        for row in rows
    )


def _sequence(
    values: dict[str, Any],
    key: str,
    length: int | None,
    item_type: type[Any] | tuple[type[Any], ...],
    *,
    default: list[Any] | None = None,
) -> list[Any]:
    sequence = _sized(values.get(key, default), key, length)
    if any(
        isinstance(item, bool) or not isinstance(item, item_type) for item in sequence
    ):
        raise TypeError(f"{key} contains a value of the wrong type")
    return sequence


def _sized(value: Any, name: str, length: int | None) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be an array")
    if length is not None and len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return value
