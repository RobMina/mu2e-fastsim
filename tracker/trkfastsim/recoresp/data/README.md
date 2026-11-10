# Reco response tables

`ce_loophelix_run1a.npz`: conversion-electron response of stock LoopHelix reconstruction (KKDe fit, OffSpill
digitisation, Run 1A geometry), built with `python3 -m trkfastsim.recoresp.table` from Geant4 CeEndpoint events
(`dts.mu2e.CeEndpoint.MDC2025ba`, sequencers 001430_00000000..00000600, 19,630 events): the Geant4 primary's
StrawGasSteps from its tracker entry, injected into the same events and run through stock digitisation,
reconstruction and EventNtuple. Bins: step rows (`nrow_edges`) x entry tan(dip) (`tandip_edges`); field
`bz` = 1.0117 T at the tracker origin. Includes the t0 residual and the fit-efficiency turn-on in the entry time (no tracks below about 210 ns: the
digitisation window); the bin tables use entry times from 300 ns on. Validation on independent events is not part of
the table.

`ce_loophelix_run1a_pools.npz` (opt-in): the same tables, built from the same events, plus each bin's pool
(`table --pools`: `pool_res`, `pool_idx`, `pool_start`, `pool_count`). A response table with pools draws the seven
residual and quality quantities jointly from one reference track of the bin pool instead of the Gaussian copula and
inverse CDFs. Every array the default file holds is identical in both files; the pool file also records its `reference` text.

`ce_loophelix_run1a_fulldts_pools.npz` (opt-in, `DeviceChain(response="fulldts")`): the same development events and
bins (`count` is identical), with the response measured on the full Geant4 events instead of the CE's own steps:
every StrawGasStep of the event (delta rays and other secondaries, the electrons coming back from the calorimeter)
through the same stock digitisation, reconstruction and EventNtuple, with pools. The bins still count the
conversion electron's own Geant4 rows, which is what the device transport (with its albedo) provides. The extra hits
lower the quality pass (direction and fit consistency: 85 % instead of 91 % of fitted tracks; the fit efficiency
is unchanged), and
the pools carry the full events' residual and quality shapes.
