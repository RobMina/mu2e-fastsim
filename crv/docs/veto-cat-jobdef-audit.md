# Bounded Cat job-definition assignment audit

## Result

The registered `cnf.mu2e.NeutralsFlashCat.MDC2025ad.0.tar` contains exactly
**5,000,000 filename entries**, measured by streaming its complete
**360,000,549-byte** `jobpars.json`. This is 1,000 declared 5,000-entry blocks;
only the twelve retained-output blocks were compared with saved SAM metadata.
**All twelve match**, with 5,000 distinct names inside each checked block and no
mismatches. Comparison uses sorted-name hashes, not input-order equality.

The declared assignment model is
`filenames[index*5000:(index+1)*5000]`, with zero-based index taken from the output
filename sequencer and `tbs.sequencer_from_index: true`. The complete surrounding
JSON schema and scalar controls match the externally pinned contract. This is
recipe-to-catalog agreement, not certification of actual worker indices,
recovery mapping, execution order, or physics-file contents.

| Output index | Start, inclusive | Stop, exclusive | Result |
|---:|---:|---:|---|
| 88 | 440000 | 445000 | match |
| 147 | 735000 | 740000 | match |
| 229 | 1145000 | 1150000 | match |
| 330 | 1650000 | 1655000 | match |
| 407 | 2035000 | 2040000 | match |
| 566 | 2830000 | 2835000 | match |
| 668 | 3340000 | 3345000 | match |
| 727 | 3635000 | 3640000 | match |
| 744 | 3720000 | 3725000 | match |
| 760 | 3800000 | 3805000 | match |
| 788 | 3940000 | 3945000 | match |
| 807 | 4035000 | 4040000 | match |

The repeated saved parent-set hashes for 88/744/760, 668/788 and 407/727/807 are
also present in these declared recipe blocks. No old pairwise-overlap analysis or
payload check was rerun, and there was no global duplicate census. Neither the
filename count nor the number of distinct checked hashes is an independent
physical-library count.

## Bounds and binding

`crvfastsim.veto.cat_jobdef_audit` uses standard-library gzip/tar/JSON streaming,
64 KiB read chunks, and at most one target block's 5,000 filenames at a time.
The filename count was not guessed or fitted: the contract has `expected_items:
null`, an exact job-parameter byte pin, a six-million-item cap, and requires whole
blocks and all target indices. Strings, schema depth/keys, small arrays, members,
small inputs and output bytes are separately bounded. Unknown schema, malformed
JSON, duplicate keys, altered controls and unsafe tar members fail closed.

The archive is read directly at its exact known catalog path, without PNFS
namespace discovery or stat calls. Its compressed hash is checked before, during
and after scanning. Saved SAM text, original provenance, contract and source bytes
are hash-bound and rechecked before exclusive output creation. The report embeds
the source snapshot. Exclusive creation is not an atomic-publication or
crash-durability certificate.

The first attempt rejected a local PAX timestamp header before JSON scanning and
produced no report. Its source, tests and failure log remain preserved. The
revised scanner admits only bounded local `././@PaxHeader` records containing
mtime/atime/ctime decimal timestamps; it rejects path/link/size overrides, global
or sparse extensions, consecutive/orphan headers and nonzero padding. The real
archive has two such timestamp headers (25 and 28 bytes), plus `jobpars.json` and
the 122-byte `mu2e.fcl`. No schema, source pin, target index or comparison rule was
changed to obtain agreement.

## Validation and immutable identifiers

Targeted tests: **111 passed locally** (0.57 s) and **111 remotely** (0.89 s).
The successful single audit took **15.75 s wall**, **89,676 KiB peak RSS** and
wrote **47,218 bytes**. It exited zero. These are documentary-parser tests and a
configuration scan, not physics validation or a runtime projection.

| Object | SHA-256 |
|---|---|
| Compressed archive, 18,068,743 bytes | `c17855fe607c8233b160e1158767bee5b4f9c80241da1196201297c80b3651fb` |
| Full jobpars member | `f4b19e7d68a341a4af960f8aa09b99c07c583ed93786d9db6fd4aefc4549eb40` |
| Contract | `69f502844af7a5372b7f7a6613f250307ddba4b7044d693d607d7406317029d9` |
| Original retained provenance | `3ffe019bf7f72ee20cc14a136df9a11a5ccca157c282bf095b8ea4483f3c35e1` |
| Audit source | `e55bf9134c75a7648abf1d3ba70c165130a876b7d10a86f31a7157beac60daef` |
| Test source | `15119fa1ad64212340c57b75610ddfa4d6d03411c89c0080abfdc895fe92b0a8` |
| Successful JSON report | `934247ae30cc9ba7306d79e37b7bd1c475ad813cabed5af1628c6fd42b297d9d` |
| Successful audit log | `b2d162394e46baf6b7433b908c6118ab74110f6bfd05f22bb95c23aa3e5ebe00` |
| Remote test log | `edbbfb9946b7722d62dff58f4aebac3376f0fd7dad63034b681bb7a9f1c05049` |

For a separately authorized reproduction, the module accepts the saved
`continuation_v1` evidence directory, `--contract docs/cat-jobdef-contract-v1.json`,
`--expected-contract-sha256`, `--expected-archive-sha256`,
`--expected-provenance-sha256` with the pins above, and `--out` naming a new file
outside prior evidence and code. Reuse this successful report rather than rerun
it merely to reproduce the result. A comparison mismatch is report data and exit
2, not permission to tune offsets or silently repair parentage.

## Interpretation limits

The separate Flash 13147 worker finding (`veto-flash-parent-metadata.md`) exposes
a deeper parent-list defect: FCL input 32644 versus catalog parent 32205. Passing
the immediate Cat-to-Flash check does not cure that deeper ancestry problem.
Physical source-reuse weights, cross-species ancestry, additive exposure and
finite-library covariance remain unestablished. No CE rows, physics products,
selected tracks, previous controls, response draws or stock baseline were read or
rerun here. No exposure renormalization or uncertainty estimate follows. Historical
F is unchanged, finite-library uncertainty remains unspecified, and C4 stays stock.
