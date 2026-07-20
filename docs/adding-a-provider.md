# Adding an evidence provider

An evidence provider reads one canonical `Parcel` and returns zero or more
independent `EvidenceRecord` values. It does not rename the parcel, mutate its
geometry, inspect another provider's output, or synthesize a final answer.

The minimal provider shape is:

```python
from neurolabel import EvidenceDomain, EvidenceRecord, Parcel


class ExampleProvider:
    name = "example"

    def describe(self, parcel: Parcel) -> list[EvidenceRecord]:
        return [
            EvidenceRecord(
                parcel_key=parcel.key,
                domain=EvidenceDomain.FUNCTION,
                label="example process",
                provider=self.name,
                provider_version="0.1",
                source="example source",
                source_version="1",
                method="example method",
                inference_type=None,
                statistic_name="score",
                statistic_value=0.75,
            )
        ]
```

`EvidenceProvider` is a structural protocol, so a provider does not need to
inherit from it. Records should use the provider's stable `name`.

## Implementation checklist

- Keep provider configuration and external clients outside the parcel.
- Treat the input parcel as immutable.
- Return a finite `Sequence[EvidenceRecord]`; an empty sequence means that the
  provider has no evidence for that parcel.
- Use the canonical parcel key unchanged in every returned record.
- Emit one record per attributable observation rather than pre-combining
  unlike results.
- Convert provider-specific values to ordinary JSON-compatible metadata before
  constructing a record.
- Do not rank domains, resolve disagreement, or produce a final label.

## Provenance checklist

Every record must make its result auditable:

- stable provider name and provider version;
- source identity and source version;
- method and inference type;
- statistic name, value, and any reported `p_value` or `q_value`;
- support count with a documented provider-specific meaning; and
- metadata needed to interpret or reproduce the observation.

Do not invent unavailable provenance. Version the provider or source whenever
a change can alter emitted evidence.

## Test checklist

- The provider satisfies the `EvidenceProvider` protocol shape.
- `name` is stable and agrees with each record's `provider`.
- `describe` leaves the parcel and its geometry unchanged.
- Known inputs produce deterministic records in deterministic order.
- No-result inputs return an empty sequence.
- Invalid, non-finite, or non-JSON provider data fails before publication.
- Every record points to the requested parcel key and carries complete
  provenance.
- Every `record.to_dict()` can be encoded by the standard-library `json`
  module without a custom encoder.

Tests should verify provider facts and contract invariants. They should not
encode cross-provider weighting or other scientific synthesis policy.
