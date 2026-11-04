# Tier-1b default: the strict library with step-tree roots, placement, templates, chord placement and root-coherent draws

The tier-1b step sampling of the **extracted** CRV configuration (`extracted_v04`) uses the asset set `strict_v5`
(default since 2026-11-02): the association-certified strict library rebuilt with the step-tree root of every
descendant row, with crossing context, containment, point and topology placement, the v3 template sidecar, chord
placement of the same-bar descendant steps ([validation.md](validation.md)) and **root-coherent draws**: a delta ray and
its cascade (all steps that share the muon's direct child) stay together along the fast muon within a module
(`tier1b.attach_anchors`, `sample(anchor=True)`). The previous default `strict_v4` (the same library without roots,
whole-muon draws of hard library muons instead of root-coherent draws) stays selectable and samples exactly as before
per seed. Both are defined in `crvfastsim/tier1b_assets.py`. `crvfastsim/tier1b.py` itself has no default library: the
callers pick the set and pass its options down.

| Set | Library | Sidecars | `tier1b.sample` options | Run configurations |
|---|---|---|---|---|
| `strict_v5` (default) | `tier1b_strict_v5/library.root`, sha256 `fc6e1518...5f4` (the 20 donors of `strict_v4` with step-tree roots; same bins, showers and reach) | `topology.npz`, `templates_v3.npz` (library chords; near-miss templates and census of `strict_v4`), each with its `.json` | `context`, `contain`, `point`, `topology`, `unvisited`, `nearmiss`, `chord`, `anchor` | `extracted_v04` only |
| `strict_v4` | `tier1b_strict_v2/library.root`, sha256 `ebb015a7...7c7` (20 association-certified donors, generator-association-repair.md (branch `park/tier1b-library`)) | `topology_sidecar_strict_v2_v2.npz`, `templates_v3.npz` (library chords; near-miss census g4ref_v1 1-10 + odd g4ref_v2 21-219), each with its `.json` | `context`, `contain`, `point`, `topology`, `unvisited`, `nearmiss`, `chord`, `hard` | `extracted_v04` only |

Paths are relative to `$CRVFS_DATA`. `--tier1b-dir DIR` points at a copy of the set's files, as the endpoints use.
The device sampler (`--tier1b-backend jax`) implements both sets ([device.md](device.md)).

## Evidence for root-coherent draws (2026-11-02)

Why: one step tree (a delta ray and its cascade, all steps that share the muon's direct child) deposits in several
layers along the muon, and 77 % of the Geant4 adjacent-layer covariance of muon-descendant energy in EX comes from single
trees. Attaching each step to the crossing of its bar cuts a tree at the crossings (84 % of the same-tree product ends
up on different crossings), so independent draws per crossing lose the correlation and the very hard clusters. The
root-coherent draw keeps each tree with the first crossing it touches; a draw takes the trees anchored on the drawn
crossing, and every row goes to the fast crossing at its library distance along the muon (draw cells carry the symbol
of the next crossing). A tree is cut where it leaves its module. The hard-muon draws of `strict_v4` are not used.

Held-out sample: even `g4ref_v2` shards 22-220, 5 M generated events, none used by the library; dev half 22-120 for
the choice (made on a half-size library), confirmation half 122-220 looked at for the chosen draw. Errors are the
paired per-shard statistic (arms share the generated events shard by shard; a null has mean 1).

| Quantity | Geant4 | `strict_v4` | `strict_v5` |
|---|---:|---:|---:|
| EX adjacent-layer correlation of muon-descendant energy, per muon (dev, two seeds) | 0.403 +- 0.023 | 0.244 +- 0.013 | 0.43 +- 0.07 / 0.39 +- 0.02 |
| NStep per event vs Geant4, chi2/ndf (dev, two seeds; confirmation) | - | 2.92 | 1.29 / 1.48; 1.58 |
| descendant energy vs Geant4 (dev) | - | -0.0 +- 0.7 % | +0.40 +- 0.67 % |
| cluster PE, fast steps + stock response (A1) vs Geant4 (A0g), paired chi2/ndf, 5 M | - | 3.28 | **1.09** |
| same, dev / confirmation half | - | 2.28 / 1.94 | 0.93 / 1.09 |
| cluster PE end to end (E2E) vs A0g, 5 M | - | 3.26 | 1.50 |
| clusters above 2000 PE, A1 / A0g, 5 M | - | 0.923 +- 0.014 | **0.991 +- 0.014** |
| digis, pulses, clusters A1 / A0g - 1, 5 M | - | +0.19, +0.24, +0.12 % (+- 0.14, 0.13, 0.10) | +0.33, +0.37, +0.23 % |
| tier-1b sampling, one CPU core | - | 16.3 k generated/s | 16.5 k generated/s |

Open with this default (limitations.md): NPc (coincidence clusters per event) is not moved (A1 vs A0g paired 2.02,
`strict_v4` 1.91); NDigis2 and NPulse2 (wide-range multiplicities) are worse (3.04 and 2.10, against 1.28 and 1.99);
muon-descendant steps pile up at the bar ends (as with `strict_v4`). With one tier-1b seed the confirmation half showed
digi, pulse and cluster yields +0.29 +- 0.16 % above `strict_v4`, flagging its occupancy histograms; a second seed gives
+0.05 +- 0.15 %, so that was draw noise (two seeds differ by about 0.2 % on either half).

**Library.** Rebuilt from the same 20 production donors with the step-tree root of every descendant row; its bins,
crossings, descendant rows, showers and reach tables are identical row for row to the `strict_v4` library (only the
`root` column is added). The topology sidecar is rebuilt on it; the template sidecar keeps the near-miss templates and
census of `strict_v4`. `tier1b-apply --tier1b strict_v4` reproduces the previous default's step and particle tables exactly (shards
22-40).

## Earlier option combinations

`strict_v4` (2026-10-30 to 2026-11-02) is selectable by name. The sets that were the default before it are not. Their sampling is reproduced step by step
with `tier1b-apply`, a custom `--library` and explicit placement flags on the same library and sidecars (the custom
path reads the library and attaches the topology, then the template sidecar, in the same order as the set):

| Former name (default period) | `tier1b-apply` flags |
|---|---|
| `strict_v3` (2026-10-29 to 30): `strict_v4` without hard-muon draws | `--library L/library.root --crossing-context --contain-steps --point-placement --topology-sidecar L/topology_sidecar_strict_v2_v2.npz --template-sidecar L/templates_v3.npz --unvisited-layers --near-miss --chord-placement` |
| `strict_v2` (2026-10-28 to 29): v2 templates, no chord placement | the same with `--template-sidecar L/templates_v2.npz` and without `--chord-placement` |
| `legacy` (before 2026-10-28) | `--library tier1b_v7/lib_prod_as_20files.root` (that library was deleted and is not available) |

`L` is the set's directory (`$CRVFS_DATA/tier1b_strict_v2`). The whole-muon hard draws have no flag: they belong to
`strict_v4` only, the root-coherent draws to `strict_v5` only. `runchain` and `e2e-bench` take `--tier1b strict_v5`
(the default), `--tier1b strict_v4` or a custom `--library` without placement.

## Choosing a set

| Caller | Default | Custom library |
|---|---|---|
| `tier1b-apply` | `strict_v5` (`--tier1b strict_v4`: the previous default) | `--library LIB` (placement only with the explicit flags `--crossing-context` ... `--chord-placement`) |
| `runchain`, `e2e-bench` | `strict_v5` for extracted (`--tier1b strict_v4`); the named configurations have no set | `--library LIB` (no placement) |
| core `chain_point` | parameter `tier1b` 0 = `strict_v5`, 1 = `strict_v4`; named configurations are refused | - |
| core kit `chain`, `regression` (EAF, in place) | `strict_v5` | - |
| core `uva/chain_array_uva.sh`, `chain_node_uva.sh`, `aurora/chain_node_aurora.sh` | `T1B=strict_v5` (or `strict_v4`) | - |

`--tier1b` and `--library` cannot be combined.

**Named run configurations have no library of their own.** `run1a_v01`, `run1b_v01` and `run2_v10` need a custom
`--library`: their sectors are mapped onto the extracted library groups (`runconfig.lib_group`) and their non-muon
showers are off. A named set with a named configuration is refused. The topology and template sidecars
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

Cost on one GPVM (4 cores, NumPy CPU path, g4ref_v1 shard 11, `e2e-bench --repeat 4` with a prepared context, mean of three
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
