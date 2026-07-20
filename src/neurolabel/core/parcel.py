"""Canonical parcel identity and geometry."""

from __future__ import annotations

from dataclasses import dataclass, field

import nibabel as nib
import numpy as np
from scipy import ndimage

from .extraction import _mask_image
from .provenance import Provenance
from .validation import ValidationFinding

BoundingBox = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
Point3D = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ParcelGeometry:
    """Geometry derived directly from a parcel mask and affine."""

    voxel_count: int
    volume_mm3: float
    centroid_voxel: Point3D
    centroid_world: Point3D
    bounding_box: BoundingBox
    connected_components: int

    @classmethod
    def from_mask(cls, mask: np.ndarray, affine: np.ndarray) -> ParcelGeometry:
        """Compute mask geometry using face connectivity.

        Parameters
        ----------
        mask
            Three-dimensional nonempty parcel mask.
        affine
            Finite nonsingular 4x4 voxel-to-world affine.

        Returns
        -------
        ParcelGeometry
            Counts, centroids, half-open bounds, and connectivity.
        """
        coordinates = np.argwhere(mask)
        if not coordinates.size:
            raise ValueError("Parcel geometry requires a nonempty mask.")
        centroid_voxel_array = coordinates.mean(axis=0)
        centroid_world_array = nib.affines.apply_affine(affine, centroid_voxel_array)
        minimum = coordinates.min(axis=0)
        stop = coordinates.max(axis=0) + 1
        voxel_count = int(coordinates.shape[0])
        voxel_volume = float(abs(np.linalg.det(np.asarray(affine)[:3, :3])))
        components = int(
            ndimage.label(
                mask,
                structure=ndimage.generate_binary_structure(3, 1),
            )[1]
        )
        return cls(
            voxel_count=voxel_count,
            volume_mm3=voxel_count * voxel_volume,
            centroid_voxel=tuple(float(value) for value in centroid_voxel_array),
            centroid_world=tuple(float(value) for value in centroid_world_array),
            bounding_box=tuple(
                (int(lower), int(upper))
                for lower, upper in zip(minimum, stop, strict=True)
            ),
            connected_components=components,
        )


@dataclass(frozen=True, slots=True)
class Parcel:
    """One atlas parcel with identity, mask, geometry, and provenance."""

    atlas_id: str
    parcel_id: int
    mask_img: nib.Nifti1Image = field(repr=False, compare=False)
    geometry: ParcelGeometry
    provenance: Provenance
    _mask: np.ndarray = field(repr=False, compare=False)
    warnings: tuple[ValidationFinding, ...] = ()

    @classmethod
    def from_labels(
        cls,
        atlas_id: str,
        parcel_id: int,
        labels: np.ndarray,
        image: nib.spatialimages.SpatialImage,
        provenance: Provenance,
        warnings: tuple[ValidationFinding, ...] = (),
    ) -> Parcel:
        """Construct a parcel from validated integer labels."""
        mask = np.asarray(labels == parcel_id, dtype=np.uint8)
        mask.setflags(write=False)
        return cls(
            atlas_id=atlas_id,
            parcel_id=parcel_id,
            mask_img=_mask_image(image, mask),
            geometry=ParcelGeometry.from_mask(mask, image.affine),
            provenance=provenance,
            warnings=tuple(warnings),
            _mask=mask,
        )

    @property
    def key(self) -> str:
        """Return the stable atlas-qualified parcel key."""
        return f"{self.atlas_id}:{self.parcel_id}"

    @property
    def mask(self) -> np.ndarray:
        """Return the read-only uint8 parcel mask."""
        return self._mask

    @property
    def voxel_count(self) -> int:
        """Return the number of parcel voxels."""
        return self.geometry.voxel_count

    @property
    def volume_mm3(self) -> float:
        """Return parcel volume in cubic millimetres."""
        return self.geometry.volume_mm3

    @property
    def centroid_world(self) -> Point3D:
        """Return the world-coordinate centroid."""
        return self.geometry.centroid_world

    @property
    def count(self) -> int:
        """Compatibility alias for ``voxel_count``."""
        return self.voxel_count

    @property
    def volume(self) -> float:
        """Compatibility alias for ``volume_mm3``."""
        return self.volume_mm3

    @property
    def centroid(self) -> Point3D:
        """Compatibility alias for ``centroid_world``."""
        return self.centroid_world
