# Benchmarks: the three metrics of a fastsim commit

`benchmarks/fastsim_score` scores one commit of this repository on the three metrics every fast workflow reports.
Run on any commit, it gives a score that compares with every other commit's:

| Metric | How it is computed | Ranking |
|---|---|---|
| **Validity** | per stock histogram, chi2/ndf of the fast arm against stock on the same generated events, flagged when chi2/ndf > max(2, 3 x floor) and p < 1e-3 (floor = stock with another seed against stock); yields pass when \|ratio - 1\| - 3 se <= tolerance. Validity = (1 - flagged fraction) x (fraction of yields that pass) | key: mean over workflows and both dtypes |
| **Throughput multiplier** | `mu2efastsim.bench` warm rate / stock events per second per single-core grid slot at the same boundary | geometric mean; self-reported, labelled with the device |
| **SU per event** | SU per hour of the device / (3600 x warm rate) | private scoring only (`--private`), never on a leaderboard |

Both float32 and float64 are required for every workflow. The statistics are the ones the validation code here
already uses:
- the two-sample sum `(n1 - n2)^2 / (n1 + n2)` on the same events (`crvfastsim.validation.compare`);
- a scaled shape test (mode `shape`);
- the paired per-shard test when both arms carry shards (`trkfastsim.validation.chain_pair`).

The scorer uses NumPy only and imports nothing from this repository, so a Codabench worker can run it unchanged.

## Layout

```
submission/<W>/<dtype>/hists.json   fast arm: {"hists": {key: {"edges", "counts" | "shards"}}, "yields": {...}}
submission/<W>/<dtype>/bench.json   the mu2efastsim.bench report (warm_rate, device, dtype, boundary)
reference/<W>/stock/hists.json      stock arm on the same generated events
reference/<W>/floor/hists.json      stock with another seed (optional)
reference/<W>/reference.json        stock_rate_per_slot, boundary, mode, yield_tolerance, min_count, su_per_hour
```

Reference trees hold Mu2e simulation results and are **never committed**: keep them outside the repository and
point the scorer at them. `synthetic.py` writes a toy example of the whole layout.

## Commands

```bash
python -m benchmarks.fastsim_score.from_root fast_W5_f64.json shard_*.root --select '^Validation/Crv' --yield coincidences=Validation/CrvCoinc/N
python -m benchmarks.fastsim_score.score SUBMISSION REFERENCE [--private] [--out scores.json]
python -m benchmarks.fastsim_score.synthetic EXAMPLE && python -m benchmarks.fastsim_score.score EXAMPLE/submissions/good EXAMPLE/reference
python -m benchmarks.codabench.build_bundle OUT [--reference DIR --solution DIR]
```

`build_bundle` packages the scorer as a Codabench results-submission bundle (stock `codalab-legacy:py312` image).
With a real reference, it is for an unpublished competition or local scoring only.

Tests: `tests/test_benchmarks_fastsim.py`.
