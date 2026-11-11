# Reco response tables

The default (`recoresp.table.DEFAULT`, `DeviceChain(response="default")`) is `ce_loophelix_run1a_fulldts_pools.npz`.
The others are available by name (`recoresp.table.NAMED`).

`ce_loophelix_run1a.npz` (`"ceonly"`, the earlier default): conversion-electron response of stock LoopHelix reconstruction (KKDe fit, OffSpill
digitisation, Run 1A geometry), built with `python3 -m trkfastsim.recoresp.table` from Geant4 CeEndpoint events
(`dts.mu2e.CeEndpoint.MDC2025ba`, sequencers 001430_00000000..00000600, 19,630 events): the Geant4 primary's
StrawGasSteps from its tracker entry, injected into the same events and run through stock digitisation,
reconstruction and EventNtuple. Bins: step rows (`nrow_edges`) x entry tan(dip) (`tandip_edges`); field
`bz` = 1.0117 T at the tracker origin. Includes the t0 residual and the fit-efficiency turn-on in the entry time (no tracks below about 210 ns: the
digitisation window); the bin tables use entry times from 300 ns on. Validation on independent events is not part of
the table.

`ce_loophelix_run1a_pools.npz` (`"pools"` or `"ceonly_pools"`): the same tables, built from the same events, plus each bin's pool
(`table --pools`: `pool_res`, `pool_idx`, `pool_start`, `pool_count`). A response table with pools draws the seven
residual and quality quantities jointly from one reference track of the bin pool instead of the Gaussian copula and
inverse CDFs. Every array the CE-only file holds is identical in both files; the pool file also records its `reference` text.

`ce_loophelix_run1a_fulldts_pools.npz` (the default; also `"fulldts"`): the same bins, built from three times the
development events (sequencers 001430_00000000..00000600, 00000050..00000750 and 00001400..00001900 of the same
dataset, 21 files, 59,189 events), with the response measured on the full Geant4 events instead of the CE's own steps:
every StrawGasStep of the event (delta rays and other secondaries, the electrons coming back from the calorimeter)
through the same stock digitisation, reconstruction and EventNtuple, with pools. The bins still count the
conversion electron's own Geant4 rows, which is what the device transport (with its albedo) provides. The extra hits
lower the quality pass (direction and fit consistency: 85 % instead of 91 % of fitted tracks; the fit efficiency
is unchanged), and
the pools carry the full events' residual and quality shapes. A pool response inherits the statistical fluctuations
of its development sample: with three times the held-out size they add about a third to a held-out shape chi2 (with
equal sizes, about as much again).

`nrow_run1a.npz` (`trkfastsim.recoresp.nrow`, the opt-in approximation W2-W4 `tracker_rows=sampled`): per cell of the
tracker-entry helix (|p|, tan dip, r_max; edges in `p_edges`, `tandip_edges`, `rmax_edges`), a pool of the step rows
the default device chain gives (tracker transport plus the calorimeter albedo, `DeviceChain` defaults, float64, Run 1A
geometry and field). The pools come from 700,000 training electrons, 503,361 of them with a tracker entry:

- conversion electrons drawn from the MuminusStopsCat stop table with seeds 1001+;
- DIO starts, shards 0-49;
- RPC starts, shards 0-1.

The benchmark draws (seeds 1-3, DIO shards 50-249, RPC shards 2-3) are not among them. A cell with no training track
draws from its (|p|, tan dip) cell, else from every track.
