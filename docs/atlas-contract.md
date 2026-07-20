# Atlas contract

An atlas loader establishes canonical parcel identity and geometry. It
validates source data; it does not repair, resample, or scientifically
reinterpret it.

## Labels and background

- Background values come only from the atlas specification. No integer,
  including `0`, has universal background meaning.
- Label `0` is a valid parcel identifier when it is present and not declared
  as background. Conversely, an atlas may explicitly declare `0` as background.
- Every non-background voxel value must represent an integer parcel ID.
- For each finite voxel value `x`, validation computes
  `abs(x - nearest_integer(x))`. Every deviation must be at most `1e-6`, with
  no relative tolerance. The maximum observed deviation is retained in the
  validation report. Only after this check passes may values be converted to
  integers; values beyond the tolerance are invalid and are never silently
  rounded into a parcel.
- NaN, positive infinity, and negative infinity are invalid label values.
- Parcel IDs are the validated integer labels present in the source image
  after removal of the specification's declared background values. The
  resulting set must exactly equal the IDs declared by the specification.
  IDs need not be contiguous and must not be renumbered to close gaps or
  reflect iteration order.

## Spatial contract

- The source affine is part of the atlas contract. It must be finite,
  four-by-four, and have a nonsingular spatial three-by-three matrix. Against
  a manifest it is validated with absolute tolerance `1e-6` and no relative
  tolerance. Voxel coordinates are transformed into world coordinates with
  that source affine.
- Loading performs no resampling, interpolation, reorientation, or implicit
  registration.
- A parcel mask is exactly `validated_labels == parcel_id`; no voxels are
  added, dropped, smoothed, or joined.
- Geometry is derived from that exact mask: voxel count, affine-determinant
  volume, voxel and world centroids, half-open voxel bounding box, and
  connected-component count.
- Connected components use face connectivity: two voxels are neighbors only
  when they share a face (six-neighbor connectivity in three dimensions).
  Edge- or corner-only contact does not join components.

If a source needs resampling or other transformation, that operation belongs
in an explicit preprocessing step that produces a new, separately identified
atlas source.

## Provenance

Canonical parcels retain enough provenance to identify and reproduce their
origin:

- stable atlas ID and original parcel ID;
- source path, or an explicit in-memory marker;
- SHA-256 of available source bytes;
- coordinate-space claim and its verification status;
- source shape, stored dtype, affine, and voxel sizes;
- source reference; and
- NeuroLabel package version.

Provenance describes what was loaded. It must not claim transformations that
the loader did not perform or discard distinctions present in the source.
