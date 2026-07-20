"""Semantic-free extraction of discrete label masks."""

from __future__ import annotations

import operator
from pathlib import Path

import nibabel as nib
import numpy as np

from .errors import AtlasValidationError, UnknownParcelError
from .validation import _validate


def extract_label_mask(
    image_or_path: nib.spatialimages.SpatialImage | str | Path,
    label: int,
) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Extract any existing integer label as a uint8 NIfTI mask."""
    normalized_label = _integer_label(label)
    if isinstance(image_or_path, (str, Path)):
        path = Path(image_or_path).expanduser().resolve()
        image = nib.load(path)
        atlas_id = str(path)
    else:
        image = image_or_path
        atlas_id = "<in-memory-image>"

    result = _validate(image)
    if not result.report.is_valid or result.labels is None:
        raise AtlasValidationError(result.report)
    if normalized_label not in result.report.observed_labels:
        raise UnknownParcelError(
            normalized_label, atlas_id, result.report.observed_labels
        )

    mask = np.asarray(result.labels == normalized_label, dtype=np.uint8)
    mask.setflags(write=False)
    return mask, _mask_image(image, mask)


def _mask_image(
    image: nib.spatialimages.SpatialImage, mask: np.ndarray
) -> nib.Nifti1Image:
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1.0, 0.0)
    return nib.Nifti1Image(
        np.asarray(mask, dtype=np.uint8),
        np.array(image.affine, copy=True),
        header,
    )


def _integer_label(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"Label must be an integer, not {value!r}.")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"Label must be an integer, not {value!r}.") from error


__all__ = ["extract_label_mask"]
