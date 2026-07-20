# Evidence contract

An `EvidenceRecord` is one provider's attributable observation about one
canonical parcel. It is evidence for later interpretation, not a final parcel
label, recommendation, or consensus.

## Domains

Every record belongs to exactly one broad `EvidenceDomain`:

- `anatomy`
- `network`
- `function`
- `task`
- `clinical`
- `microstructure`
- `connectivity`

Domains organize evidence without deciding which domain is more authoritative.

## Statistics and provenance

A record names its provider, provider version, source, source version, method,
inference type, statistic, and support count. `statistic_name` defines the
meaning of `statistic_value`; the contract does not assume that larger values
are better or that unlike statistics are comparable.

`p_value` and `q_value` are optional because not every method produces them.
When present, each is finite and lies in the closed interval `[0, 1]`.
`support_count` is a nonnegative integer and never substitutes for a reported
statistic.

Provider-specific factual context belongs in `metadata`, which is recursively
JSON-compatible. Keys are strings; values are null, booleans, ordinary finite
numbers, strings, arrays, or objects. NumPy scalars and non-finite numbers must
be converted or rejected before a record is created. `to_dict()` returns only
ordinary JSON-compatible Python values and serializes the domain as its string
value.

## Independence and disagreement

Records from different providers may agree, complement one another, or
conflict. Providers preserve that disagreement by returning their own records
without ranking, suppressing, or combining other evidence. Consumers should
retain record-level provenance when storing or displaying results.

Any future synthesis may define comparison, weighting, uncertainty, and final
label policy. That policy is explicitly outside `EvidenceProvider.describe`
and the evidence record contract.
