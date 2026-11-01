# Tier-1b default: the strict library with placement, templates, chord placement and hard-muon draws

The tier-1b step sampling of the **extracted** CRV configuration (`extracted_v04`) uses the asset set `strict_v4`, the
only named set: the association-certified strict library with crossing context, containment, point and topology
placement, the v3 template sidecar, chord placement of the same-bar descendant steps
([validation.md](validation.md)) and whole-muon draws of hard library muons
(`tier1b.attach_hard`). It is defined in `crvfastsim/tier1b_assets.py`.
`crvfastsim/tier1b.py` itself has no default library: the callers pick the set and pass its options down.

| Set | Library | Sidecars | `tier1b.sample` options | Run configurations |
|---|---|---|---|---|
| `strict_v4` | `tier1b_strict_v2/library.root`, sha256 `ebb015a7...7c7` (20 association-certified donors, generator-association-repair.md (branch `park/tier1b-library`)) | `topology_sidecar_strict_v2_v2.npz`, `templates_v3.npz` (library chords; near-miss census g4ref_v1 1-10 + odd g4ref_v2 21-219), each with its `.json` | `context`, `contain`, `point`, `topology`, `unvisited`, `nearmiss`, `chord`, `hard` | `extracted_v04` only |

Paths are relative to `$CRVFS_DATA`. `--tier1b-dir DIR` points at a copy of the set's files, as the endpoints use.

## Earlier option combinations

The sets that were the default earlier are no longer selectable by name. Their sampling is reproduced step by step
with `tier1b-apply`, a custom `--library` and explicit placement flags on the same library and sidecars (the custom
path reads the library and attaches the topology, then the template sidecar, in the same order as the set):

| Former name (default period) | `tier1b-apply` flags |
|---|---|
| `strict_v3` (2026-10-29 to 30): `strict_v4` without hard-muon draws | `--library L/library.root --crossing-context --contain-steps --point-placement --topology-sidecar L/topology_sidecar_strict_v2_v2.npz --template-sidecar L/templates_v3.npz --unvisited-layers --near-miss --chord-placement` |
| `strict_v2` (2026-10-28 to 29): v2 templates, no chord placement | the same with `--template-sidecar L/templates_v2.npz` and without `--chord-placement` |
| `legacy` (before 2026-10-28) | `--library tier1b_v7/lib_prod_as_20files.root` (that library was deleted and is not available) |

`L` is the set's directory (`$CRVFS_DATA/tier1b_strict_v2`). The whole-muon hard draws have no flag: they belong to
`strict_v4` only. `runchain` and `e2e-bench` take `--tier1b strict_v4` (the default) or a custom `--library` without
placement.

## Choosing a set

| Caller | Default | Custom library |
|---|---|---|
| `tier1b-apply` | `strict_v4` | `--library LIB` (placement only with the explicit flags `--crossing-context` ... `--chord-placement`) |
| `runchain`, `e2e-bench` | `strict_v4` for extracted; the named configurations have no set | `--library LIB` (no placement) |
| core `chain_point` | parameter `tier1b` 0 = `strict_v4`; named configurations are refused | - |
| core kit `chain`, `regression` (EAF, in place) | `strict_v4` | - |
| core `uva/chain_array_uva.sh`, `chain_node_uva.sh`, `aurora/chain_node_aurora.sh` | `T1B=strict_v4` | - |

`--tier1b` and `--library` cannot be combined.

**Named run configurations have no library of their own.** `run1a_v01`, `run1b_v01` and `run2_v10` need a custom
`--library`: their sectors are mapped onto the extracted library groups (`runconfig.lib_group`) and their non-muon
showers are off. `--tier1b strict_v4` with a named configuration is refused. The topology and template sidecars
describe the extracted v04 layer structure and the strict library's crossings, so they do not carry over to other
geometries.

## Provenance and hashing

- `tier1b_assets.load` reads the library and attaches the topology sidecar, then the template sidecar, in the same
  order as `tier1b-apply`. With sidecars it hashes the library once. The hash must equal the set's pinned sha256 and
  the `library_sha256` in both sidecar `.json` files. Each sidecar reader also checks its own `sidecar_sha256`.
- `runchain` prepared contexts sign the set name, options, pinned library hash and every set file path. A context
  prepared for one set cannot be reused for another. The context fingerprint covers the attached library frames.
- `e2e-bench` hashes every set file (library, both sidecars, both `.json`) with the other assets, and records the
  resolved set as `effective_tier1b`.
- `runchain` JSON (stats and `--emit`) records `tier1b` (set, paths, options, pinned hash). `runchain-merge` refuses
  shards that ran different sets. Shards of one set staged in different directories are compatible.
- Endpoints stream the whole set into the node-shared library directory (`chain_point`) or the job tmpfs (UVA,
  Aurora). Each file is md5- and sha256-checked against the pinned manifest. The node cache key of the set is
  a digest of all five file hashes.

## Evidence for chord placement (2026-10-29)

Held-out step closure on even g4ref_v2 shards 22-220 (5 M generated events, none used by the census), templates
(v2) -> chord, deck chi2/ndf vs G4: all/y 3.23 -> 1.73, T1/y 4.53 -> 1.60, T2/y 5.42 -> 2.84; z, NStep, lE and the
step yields are unchanged within their errors. Descendant steps only: all/y 7.73 -> 4.56, T1/y 13.3 -> 6.3. The
`follow` option (chain draw along the library muon) is not part of the default: it lowers NStep (4.67 -> 3.03) but
raises y and lE.

## Evidence for the strict_v2 switch (2026-10-28)

Held-out step closure on g4ref_v2 shards 21-220 (10 M generated events), topology placement -> templates, deck
chi2/ndf: bar 41.5 -> 20.3, EX y/z 6.80/6.95 -> 2.91/4.10, M y 4.60 -> 1.46. Descendant yields are within 3.2 % of G4
in every sector group. Against the previous
default (legacy library, no placement), the bar score is 189.9 -> 20.3 and EX y 431.9 -> 2.91. The residual T1/T2 y
and NStep shape comes from same-bar descendant steps, which the templates do not reach.

## Equivalence checks of the switch

Run against an archive of the code before the switch (crv `05eeb47`, core `233771b`) on the same host:

| Check | Input | Result |
|---|---|---|
| `tier1b-apply --tier1b legacy` vs old `--library tier1b_v7/...` | g4ref_v2 shard 21 transport, seed 21 | identical: 84,453 steps, 106,159 particles |
| `tier1b-apply` default vs the template-arm tables of the held-out closure | shards 21, 22 | identical: 79,164 / 101,217 and 82,373 / 103,793 rows |
| `e2e-bench --tier1b legacy` vs old `--library tier1b_v7/...` (detail fingerprint: every key, step, charge, digi, pulse, cluster and member) | g4ref_v1 shard 11; g4ref_v2 shard 21 | identical fingerprints |

Cost on one GPVM (4 cores, NumPy CPU path, g4ref_v1 shard 11, `e2e-bench --reuse-context --repeat 4`, mean of three
warm passes, legacy -> default):

| Quantity | legacy | strict_v2 |
|---|---:|---:|
| warm pass [s] | 34.96 | 37.61 (+7.6 %) |
| `tier1b_s` [s] | 1.56 | 2.08 |
| prepared-context audit [s] | 3.34 | 4.39 |
| `photons_s` [s] | 10.48 | 11.17 |
| one-time preparation [s] | 17.1 | 51.5 |
| peak RSS [GB] | 7.49 | 8.43 |

`tier1b-apply` alone: RSS 4.6 -> 6.15 GB. The extra setup is the library hash and the sidecar attachment; it is paid
once per process.

## Stock-equivalent DQM on independent shards

g4ref_v2 even shards 22-40 (500,000 generated events). No design step used these shards. Every arm ran in one art area
(Offline v13_41_01 with CRVDQM, Mu2e/DQM, `art/CrvFastSim`): A0 and A0' (G4 steps, stock response and reco, two
digitisation seeds), A0g (G4 steps on the gen stream), A1 and E2E with both sets. Merged with
`valhist.deck_rerun merge`; compared with `valhist.deck_rerun compare`. That runs the deck statistic read-only: first-bin
cut, clustered time errors, exact position variance from each arm's tables, floor A0 vs A0'.

| Comparison | legacy | strict_v2 |
|---|---:|---:|
| A1 vs A0g, flagged | 31 / 155 | 27 / 155 |
| E2E vs A0g, flagged | 19 / 133 | 19 / 133 |
| E2E vs A1, flagged | 0 / 133 | 0 / 133 |
| floor A0' vs A0 | 0 / 177 | |
| clusters A1 / A0g - 1 | -0.29 % | -0.03 % |
| digis A1 / A0g - 1 | -1.17 % | -0.62 % |
| cluster Y (`crvRecoDQM/Y_extracted`) A1 chi2/ndf | 2.27 (flag) | 0.93 |

Step closure on the same shards (`valhist.placement_closure`, deck chi2/ndf vs G4, legacy -> strict_v2): bar
41.6 -> 32.5, posx 23.2 -> 20.5, lE 6.0 -> 4.0, EX/y 25.3 -> 0.80, T1/y 4.79 -> 0.91, T2/y 7.06 -> 4.02, M/y 3.64 ->
1.63. The M step yield goes from -10.6 % to -4.8 % (+- 4.3 %). The strict_v2 arm's own reseed has bar 12.1 and T2/y 1.10,
so on 500,000 events the bar and T2/y scores are dominated by sampling. The flags that remain are occupancy and ADC
histograms (`BarId`, `SiPM`, per-channel digis per sector, ADC), the same set the deck showed before.
