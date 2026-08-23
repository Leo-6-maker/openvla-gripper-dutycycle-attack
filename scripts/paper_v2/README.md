# Paper V2 deterministic evidence export

This prospective namespace reads committed, sealed authorities from the main
repository and emits five presentation-neutral JSON exports, one tidy plot CSV,
one dependency-free TeX macro file, and one manifest under `exports/paper_v2/`.
It never modifies `paper/PAPER_V1_*`.

The generator must first be committed so its Git blob can be bound. Generate a
bundle from that committed tooling/source snapshot with:

```text
python scripts/paper_v2/export_paper_v2_evidence.py --write --source-ref <committed-source-head>
```

The canonical read-only reproducibility command is:

```text
python scripts/paper_v2/export_paper_v2_evidence.py --check
```

`--check` reads the source HEAD recorded in the committed manifest, rebuilds
all outputs from canonical Git blobs, and compares every byte. The manifest
records source HEAD/tree/timestamp, all authority input digests/blobs,
historical digest bases where declared, output hashes, and the generator
hash/blob.

The separate paper repository may copy these outputs with the manifest or
regenerate them from its source binding. It owns presentation only; scientific
numbers and claim boundaries remain owned by this repository.
