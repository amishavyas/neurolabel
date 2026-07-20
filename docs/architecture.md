# Architecture

NeuroLabel separates geometric identity, scientific evidence, and future
interpretation. Dependencies flow in one direction:

1. An atlas specification declares the expected files and their provenance.
2. Loading and validation turn those files into validated source data.
3. A canonical `Parcel` captures one parcel's identity and geometry.
4. Independent evidence providers inspect that parcel and return
   `EvidenceRecord` values.
5. Future synthesis and reporting may consume the independent records.

Each stage may depend only on stages above it. In particular, atlas loading
does not depend on providers, and providers do not depend on synthesis or
reporting.

## Boundaries

- A canonical parcel is the shared input to providers, not a provider-owned
  workspace.
- Providers never mutate a parcel or its geometry. They return new evidence
  records.
- Providers remain independent of one another. A provider does not rank,
  merge, suppress, or reinterpret another provider's output.
- Evidence records preserve observations and provenance. They are not final
  parcel names.
- Synthesis and reporting are downstream concerns. Their future policy must
  not leak into atlas validation or the provider interface.

This direction keeps the atlas contract stable while allowing providers and
eventual synthesis strategies to evolve separately.
