# Tier-1b descendant placement: layer topology and step-level closure

## Decision

Keep topology placement **opt-in** (`tier1b-apply ... --crossing-context --contain-steps --point-placement
--topology-sidecar SIDE.npz`). On 10M training-independent events it reproduces the G4 energy in neighbouring bars
(adjacent-bar ratio 0.76-0.92 with CCP, 1.00-1.10 with topology) and own-bar energy (1.016-1.027 -> 1.000-1.016).
The step z distributions improve (EX/z 11.4 -> 7.0, all/z 11.7 -> 7.3), and the bar, EX/y and M/y scores improve
somewhat. **The step-level closure target is not met.** The remaining step shape residual is entirely in muon
descendants, mostly in two populations that placement cannot reach: descendants in layers their muon never crosses
(fast/G4 energy 0.12-0.51) and descendants of muons with no CRV crossing (never generated). Defaults and CCP stay
bit-identical: on held-out shard 21, crv `cbea0ae` and this code give identical `tier1b-apply` tables, and
`--topology-sidecar` gives the table the closure run produced.

## What changed (all opt-in; defaults bit-identical)

- `crvfastsim.tier1b`: `layer_slots`, `muon_visits`, `desc_topology`, `attach_topology`, `_topology_targets` and the
  `topology` option of `_desc_for_crossings` / `sample`. `tier1b-apply --topology-sidecar SIDE.npz` turns it on.
- `valhist.stephist`: refills the Offline `ValCrvStep` histograms from CrvStep tables (ROOT fixed-bin lookup, float32
  CrvStep members) with exact event-clustered second moments, the deck statistic, and yields with clustered errors.
- `valhist.descendant_topology`: the library topology sidecar (built from the library's own donors, checked against
  the library crossings) and a truth topology census of muon descendants for any step arm.
- `valhist.placement_closure`: runs placement arms in memory on retained transport tables against G4 steps of the
  same generated events, with independent-seed copies of an arm for the sampling floor; `report` sums and joins runs.
- `scripts/run_heldout_transport.sh`: stage-C transport of held-out generator shards with the retained seed rule.

## Topology placement

A library crossing's other-bar descendant (a step in a bar its own muon does not cross) is described relative to
that muon's **visits**: its crossings in time order, consecutive crossings of one layer plane forming one visit
(layer plane = library group and layer: the modules of a sector, and the side-by-side M sectors, share a plane).

- `tvis`: the visit holding the descendant's layer plane minus the anchor crossing's visit (0 same layer, -1 the
  layer crossed before the anchor, +1 after);
- `tside`: the width-rank offset of the descendant's bar beyond the bars crossed in that visit, times the sign of the
  muon's direction along the width axis (lean);
- in-bar thickness and width position (width lean-signed), and the along-bar offset from that visit's entry (signed
  by the lean along the bar).

On a fast crossing the same row goes to the receiving muon's visit `tvis` after its anchor, `tside` bars beyond the
bars that muon crosses there (with the receiving muon's lean), at the stored in-bar and along-bar offsets. The target
is therefore never a bar the receiving muon crosses in that visit. Rows whose layer plane the library muon never
visited (no topology), or whose visit or bar does not exist for the receiving muon, keep the existing placement
(bar-structure, point, nearest bar), so no sampled step is dropped and no energy is rescaled. Placement uses no
random numbers: every arm draws the same library crossings and showers as the same options without topology.

The library trees are not changed. The per-crossing bar, muon and visit are reconstructed from the library's donor
tables (`descendant_topology sidecar`), which also checks bin, context and direction of every library crossing and
that every other-bar offset is an exact bar-centre offset.

## Validation design

- Library: the strict association-certified production library (20 donors, 10M generated events), unchanged.
- Development: `g4ref_v1` shards 11-20 (500k events, already used by earlier step-layer studies), retained transport.
- Evaluation: `g4ref_v2` shards 21-220 (10M events, never used for training, donors or the denominator). Their stage-C
  transport was produced with `scripts/run_heldout_transport.sh` (seed = shard, as for shards 1-20). The current
  transport reproduces the retained shard-1 table exactly. The candidate (topology with M sectors on one layer plane)
  was fixed on the development shards before the evaluation shards were read.
- Reference: the G4 CrvStep table of the same generated events. `stephist closure` shows the refill equals the art
  ValCrvStep histograms bin for bin (A0g, A1 shard 11; reference tables shards 1, 11), so no art job is needed.
- Arms: default, CCP (context + contain + point) and topology (CCP + topology), all from the same seed. Two more
  independent seeds of CCP and of topology give the **sampling floor**: same model, other library draws.
- Statistics (`valhist.stephist`): the deck chi2/ndf with Poisson variance (first bin cut for E, SMom, SMom2), per-sector
  step y/z with the exact event-clustered variance, the exact clustered statistic for every histogram, and yields with
  clustered independent and same-event paired errors. All are conditional on the fixed library.

## Results

Shards 21-220, chi2/ndf vs G4, the same seed for every arm. The deck value has Poisson variance, except y/z, which
use the exact clustered variance as the deck does. The exact clustered value is in brackets. The floor is the range
over the three independent-seed pairs of the topology arm (deck [exact]). Target: deck <= 2.

| Histogram | default | CCP | topology | floor (topology seeds) | topology passes |
|---|---:|---:|---:|---:|---|
| E | 2.31 [1.46] | 1.71 [0.98] | 1.71 [0.98] | 0.86-1.21 [0.60-0.78] | yes |
| lE | 2.77 [1.09] | 2.90 [1.03] | 2.90 [1.03] | 1.89-2.28 [0.60-0.80] | no |
| pathLen | 1.79 [0.76] | 1.75 [0.75] | 1.75 [0.75] | 0.38-0.95 [0.15-0.18] | yes |
| bar | 189.9 [7.24] | 45.3 [1.47] | 41.5 [1.28] | 12.2-14.7 [0.38-0.58] | no |
| NStep | 14.5 | 8.66 | 8.66 | 0.89-1.10 | no |
| SMom | 1.76 [1.56] | 1.80 [1.61] | 1.80 [1.61] | 0.85-1.14 [0.77-1.02] | yes |
| all y / z | 163.3 / 15.3 | 8.69 / 11.74 | 8.29 / 7.31 | 0.64-0.71 / 0.48-0.63 | no / no |
| EX y / z | 431.9 / 14.6 | 7.94 / 11.38 | 6.80 / 6.95 | 0.36-0.71 / 0.48-0.68 | no / no |
| T1 y / z | 74.4 / 13.4 | 10.15 / 6.51 | 10.31 / 6.49 | 0.39-1.27 / 0.61-0.76 | no / no |
| T2 y / z | 113.3 / 2.05 | 10.80 / 1.20 | 10.71 / 1.00 | 0.41-0.49 / 0.34-0.54 | no / yes |
| M y / z | 10.99 / 5.26 | 5.23 / 3.09 | 4.60 / 3.15 | 0.52-1.09 / 0.83-1.06 | no / no |

The other two seeds show the same ordering: bar CCP 83.5 / 81.5 vs topology 76.4 / 75.5, and EX/z 10.8 / 10.9 vs
6.5 / 6.7. The deck Poisson statistic of bar, posx and lE cannot reach 2 even for the model against itself, because
shower and descendant steps cluster in events (floor 12-16 for bar and posx). The deck's floor of 0 for steps comes
from A0 and A0' sharing one G4 step set, so it is not a sampling floor.

Step yields vs G4 (relative difference, clustered independent SE; same-event paired SE in the text): all sectors
-0.14 +- 0.35 % (paired 0.29) for CCP and topology, -0.17 +- 0.35 % default, which is within its error.
Per sector, topology: EX +0.27 +- 0.31, T1 -1.05 +- 0.74, T2 -0.55 +- 0.55, M -2.26 +- 1.19 %. Muon-descendant
yields: T1 -3.40 +- 0.68 % (default -0.31 +- 0.70) and M -16.8 +- 1.9 % (default -16.6 +- 2.0). Placement only moves
steps, so topology and CCP have identical step counts.

Per step origin (exact clustered chi2/ndf vs G4, topology): primary-muon and non-muon steps are <= 1.6 in every y/z
and bar histogram. Muon descendants alone have EX y/z 14.2/10.1, T1 34.3/9.0, T2 20.1/1.35, M 47.6/11.0
(CCP 16.6/15.9, 33.2/9.0, 20.4/1.8, 53.8/11.7).

Truth topology census, energy fast/G4 (SE), topology (CCP):

| Sector | adjacent bar +1 | adjacent bar -1 | own bar | never-visited layer | no crossing (G4 MeV) | all other-bar |
|---|---:|---:|---:|---:|---:|---:|
| EX | 1.006 (0.018) (0.764) | 1.077 (0.021) (0.894) | 1.003 (1.016) | 0.176 (0.183) | 22,752 | 1.058 (0.902) |
| T1 | 1.016 (0.033) (0.796) | 1.006 (0.033) (0.791) | 1.000 (1.021) | 0.116 (0.182) | 3,095 | 0.882 (0.730) |
| T2 | 1.069 (0.042) (0.775) | 1.095 (0.042) (0.921) | 1.016 (1.027) | 0.511 (0.537) | 4,354 | 0.906 (0.757) |
| M | 0.996 (0.066) (0.706) | 0.851 (0.097) (0.648) | 0.992 (1.010) | 0.191 (0.202) | 2,040 | 0.555 (0.473) |

"All other-bar" is every label except own bar and no crossing (the other-bar observable of tier1b-descendant-audit.md), without SE, since the labels share
events. The never-visited layers hold 48.7k G4 steps in T1 (8.4 GeV): that is the whole T1 descendant deficit. In M
they hold 12.5k steps (6.7 GeV, 46 % of M other-bar energy). EX has an excess of steps more than 3 bars from the
muon in its top and bottom layers (1.15-1.76). They are most likely rows with no library topology, placed by the CCP fallback; this is not traced.

Cluster level (NumPy chain on both step arms, shards 21-30, `steplayers response` / `clusters`, deck diagonal score,
paired in brackets): default EX/y 3.32 (3.66), T1/y 2.26, T2/y 2.72, M/y 1.05; CCP 0.98 (1.04), 1.89, 1.23, 0.15;
topology 1.23 (1.33), 1.75 (2.04), 1.78 (1.85), 1.13 (1.16). Every z score is <= 0.53 (diagonal). Cluster
counts agree within 0.8 % (SE 0.3-1.2 %). On these 10 shards topology is better than default and within noise of
CCP (no cluster floor was run); no cluster score is flagged.

## Descendants of muons with no CRV crossing

G4 deposits from muons with no CRV crossing in the event are never generated: the sampler is conditioned on
crossings. On the 10M held-out events: EX 45.4k steps / 22.8 GeV, T1 7.6k / 3.1 GeV, T2 8.1k / 4.4 GeV, M 2.1k / 2.0 GeV.
That is about 0.4 % of all G4 steps (15.9M in 21-220) and 1.1 % of EX descendant energy. On its own it cannot explain the
y/z shape, but it is a fixed few-per-mille yield bias and a structural zero in every per-origin comparison.

Proposed treatment, not implemented: an opt-in **near-miss template** component built like the non-muon showers.
For every donor muon with descendant steps but no crossing, keep its descendant steps as one absolute-position
template, keyed by generated-muon cells: kinetic energy, plus the sector and distance bin of the straight-line
closest approach to the CRV bars, computed from the generator position and direction (measured, not truth
ancestry). The reach probability is templates / generated non-crossing muons in the cell, from the full generator
census, exactly as the shower denominator. Sample one template per fast primary muon with no fast crossing, with
that probability. The **never-visited-layer** population needs the same position information. Those deposits occur
when the crossing is near a sector or module edge (a muon through EX beside T1, or through T2 beside M), so their
rate depends on where the crossing is. Conditioning the draw on the distance of the crossing to the nearest edge of
the neighbouring layer plane, or moving these rows into the template component, is the next step. Neither needs
new G4 production: the 20 strict donors hold 66,141 no-crossing descendant steps (`desc_steps_unattached`).

## Reproduction

Paths are examples. Code is this checkout plus a matching core on `PYTHONPATH`, all outputs new paths.

```bash
# held-out transport (stage C, seed = shard)
bash scripts/run_heldout_transport.sh GEN_DIR NEW_TRANSPORT 21 220
# refill closure against art ValCrvStep histograms
python3 -m valhist.stephist closure --tables 'G4/crvsteps.{s}.root' --root 'A0G/valhist.{s}.root' \
    --gen 'G4/gen.{s}.root' --dir Validation/CrvSteps_fastCrvSteps_noName --shards 1,11 --out NEW.json
# topology sidecar from the library's own donors (checks the library crossings)
python3 -m valhist.descendant_topology sidecar --manifest LIBRARY.json --out NEW_SIDE.npz
# arms in memory against G4 (20-shard chunks keep the process near 6 GB)
python3 -m valhist.placement_closure run --library LIBRARY.root --sidecar SIDE.npz \
    --transport 'NEW_TRANSPORT/fastC.{s}.root' --reference 'G4/crvsteps.{s}.root' --shards 21-40 \
    --arm topo=context,contain,point,topology --arm topo2=context,contain,point,topology@2 \
    --arm ccp_c=context,contain,point --arm default_c= [--write topo=TABLES] --out RUN_21-40.json
python3 -m valhist.placement_closure report RUN_21-40.json RUN_41-60.json ... --layers topo,ccp_c
# the same placement as a table
python3 -m crvfastsim tier1b-apply fastC.SUBRUN.root --library LIBRARY.root --outdir DIR --seed K \
    --crossing-context --contain-steps --point-placement --topology-sidecar SIDE.npz
```

`report` joins runs over the same shards with `A.json+B.json`, and sums runs over disjoint shards. Measured on a
4-core GPVM: 7-14 s/shard transport, 10-21 s/shard for five arms, 84-97 s for the sidecar, about 1 min/shard per
arm for the NumPy response. Processes need 5-6 GB; two GPVMs killed runs when other users needed memory.

## Limitations

- The statistics are conditional on the fixed library, transport and reference. Donor and denominator uncertainty is
  not included, and the diagonal scores are not calibrated p-values. The floors come from three seeds only.
- Topology covers 89.5 % of library other-bar rows (87.6 % of their energy). The rest keep the CCP placement.
- Lean signs follow the muon's direction at the anchor crossing. For nearly vertical muons the sign is arbitrary,
  which is symmetric by construction.
- The census labels a deposit relative to the union of its ancestor's crossings in a layer plane. It is a truth
  diagnostic, not a response model.
- The cluster comparison uses 10 shards and the NumPy chain, not the stock reco.
