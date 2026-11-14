# Stock EventNtuple files from the fast CE chain (`beamfastsim.centuple`, opt-in)

The Mu2e analysis server's EventNtuple analyses (`trigger_efficiency_ntuple` and its binned form,
`selection_efficiency_ntuple`, `momentum_resolution_ntuple`, `calo_cluster_resolution_ntuple`,
`variable_comparison_ntuple`, `occupancy_ntuple`, `fullsim_sensitivity`) and pyutils read `EventNtuple/ntuple`.
`centuple` writes that file from the fast conversion-electron chain with the stock release's own classes, so those
analyses run unchanged on fast output. Mu2eBO's `ce_chain` plot step is `trigger_efficiency_ntuple` with
`trigger_paths "apr_TrkDe_80m70p, cpr_TrkDe_80m70p"`.

```bash
# GPU, the fast-simulation image
python3 -m beamfastsim.centuple record  --out DIR --states states.root ... (or --stops STOPS.npz --electrons N) \
    --offline-root OFF --maps DSMap.header TSdMap.header --gdml RUN1A.gdml --library SHOWERS.npz \
    --calo-response CALO_RESPONSE_DIR --digi-cond ONSPILL_CONDITIONS.npz --edge-library EDGE_TABLE.npz --dtype float64
python3 -m beamfastsim.centuple trigger --digis DIR/centuple_digis.npz --menu replay.fcl --conditions trgdump.root \
    --tc-mva TimeCluster.weights.xml --precision float64 --out DIR   # no --response: assemble applies the trigger table
# CPU, NumPy + uproot
python3 -m beamfastsim.centuple donor    --files nts.dev.*.root --covered DIR/centuple_trigger.npz --out donor.npz
python3 -m beamfastsim.centuple assemble --record DIR/centuple_record.npz --trigger DIR/centuple_trigger.npz \
    --donor donor.npz --out DIR   # the shipped trigger table; --trigger-cal none: the trigger file's decisions
# the on-spill trigger table, once, from development events with both arms (trigger run without --response)
python3 -m beamfastsim.centuple trigcal --stock nts.dev.*.root --records DEV/centuple_record.npz \
    --triggers DEV/centuple_trigger.npz --out trigcal.npz
# CPU, the Mu2e environment of the stock release (PyROOT, libeventntuple_dict)
beam/beamfastsim/centuple/mu2e_env.sh python3 -m beamfastsim.centuple write --plan DIR/centuple_plan.npz \
    --donor-files nts.dev.*.root --out nts.fast.root --n-generated N
# or one design point (beam stops -> births -> all stages): python3 -m beamfastsim.centuple point --g4 ... --data D
```

fastsimkit task `beam_centuple` (params `geom`, `muons`, `electrons`, `seed`, `dtype`) runs `point` as one GPU job
on an endpoint with FNAL data and `/cvmfs` and returns the metrics `n_events`, `n_selected`, `n_triggered`,
`efficiency` (the ce_chain counts of the analysis server's default selection, computed on the written file), the
chain counters and the stops.

## Why a TTree written by ROOT

The stock EventNtuple is a TTree of split class branches (`trk.<leaf>`, `evtinfo/run`, `caloclusters.<leaf>`) and
of unsplit nested vectors of classes (`trksegs`: `std::vector<std::vector<mu2e::TrkSegInfo>>`), plus one
`trig_<path>` bool branch per trigger path. uproot writes neither split classes nor nested vectors of classes; its
RNTuple output (`trkfastsim.reco.ntuple`) reads as empty records through pyutils' `Importer`, which calls
`tree.arrays(["trk", "trksegs", ...])`. The writer therefore runs ROOT with the EventNtuple dictionaries of the stock
release (`mu2e_env.sh`: mounted `/cvmfs`, or `container/offline/offline_uva.sh` with `OFFLINE_CHAIN` on a node with
cvmfsexec). A TChain of the donor files and `CloneTree(0)` give an empty tree with the stock branches, classes,
split levels and trigger branches, sharing the branch buffers with the chain.

## How an event is made

Every fast event borrows one **donor**: a stock event of a development sample (the stock chain with every
digitized event kept), matched on

* class: Q (the response layer gives a quality-passing e- fit), F (a fit that fails the quality), N (no fit);
* the covered-trigger bit (any of the device-evaluated paths fired);
* Q: entry tan(dip), active hits, fit consistency, calorimeter match and E/p; F: entry tan(dip); N: leading cluster
  energy

(nearest 8, one drawn per event from `SeedSequence([seed, run, subrun, event])`). The donor entry is loaded, rotated
about the tracker axis to the fast electron's azimuth and shifted to its time, and the fast chain writes over what it
simulates. Everything else is the donor's.

| Branch | Class | How |
|---|---|---|
| `evtinfo.run/subrun/event` | exact | identity of the birth (the stock generator event when births come from its files) |
| `trig_<path>` of the 20 device paths (tpr, apr incl. `apr_TrkDe_80m70p`) | fast | `trigger.device` on the device digis (ports and the KalSeedFilter / AgnosticHelix response layer); with `--trigger-cal`, the device decisions without a response layer and the on-spill table (below) |
| `trig_cpr_*`, `calo_*`, `Triggerable*`, `mpr_*`, `cst_*`, `minBias_*`, two-track, proton | donor | not on the device (calorimeter-seeded, MC-truth and multi-helix paths) |
| stored events | fast + donor | the stock stream rule on the combined decisions: `cpr_Trk*_80*`, `apr_Trk*_80*`, `Triggerable{LH,TwoTrack,Calo}Path` |
| `trk.nactive`, `trk.fitcon`, `trk.chisq` (e- fit) | fast | response-layer draws (table `onspill`: the OnSpill production chain, including its digitization-window turn-on in entry time); chisq = chi2/ndf x ndof |
| other `trk` leaves of the e- fit | donor | nhits, ndof and the selection counts shifted by the active-hit difference |
| `trk` e+, mu-, mu+ hypotheses | donor | rotated and shifted |
| `trksegs` TT_Front / TT_Mid / TT_Back (e- fit) | fast | TT_Front: true entry position, response \|p\| and tan(dip), true azimuth, response t0; TT_Mid / TT_Back: the fast truth there with the front residuals |
| `trksegs` other surfaces (stopping target, OPA, IPA, ...) | donor | the donor's extrapolation (presence, used by the cut-set-80 target cuts) |
| `trksegpars_lh` at the TT surfaces | fast | from the reco state in the local field (rad = pT / (0.3 Bz), phi0 = phi - z / lam, t0 at z = 0); front d0, maxr, tanDip = response draws; errors (t0err) donor |
| `trkqual.result` | donor | nearest active hits, fit consistency, tan(dip) |
| `trkpid.result` | donor | own draw among e- donors with the same calorimeter match, nearest E/p |
| `trkcalohit` (e- fit) | fast | leading fast cluster above 10 MeV on the contact disk, else on the other disk (stock matches either disk), kept with the stock calorimeter-hit inclusion efficiency of its energy (0.56 at 10-20 MeV, 0.99 above 70 MeV): did, edep, ctime, csize, poca, mom, dt from the contact time |
| `trkmcsim[*][0]` | fast | the CE's birth (every fit of the same particle); ancestors donor |
| `trksegsmc` TT surfaces | fast | entry state; TT_Mid from the last forward straw step carried on a helix; TT_Back from the tracker exit |
| `caloclusters` | fast | front-face contacts: the calorimeter device chain (shower library, crystal response, digis, template fit, clusters); inner-edge contacts: one cluster with the Geant4 inner-edge table's visible energy |
| `caloclustersmc` | fast | nsim 1, etot = eprimary = the visible (Birks) energy of the sampled shower steps in the cluster's hit crystals, tprimary = the earliest of their times, folded by the microbunch like the cluster's reco time (calorimeter device chain `truth=True`); inner-edge clusters: the table's visible energy and the contact time |
| `calohits` | fast | approximate: one hit per crystal of each cluster, energy split evenly |
| `hitcount.nsd` | fast | device StrawDigis (no delta rays: low); nesel / nrsel / ntsel / nbkg scaled from the donor |
| `evtinfo.pbtime`, `evtinfomc`, `crv*`, `tcnt`, `trkhits`, `trkmats`, `trkmc`, `trkmcvd`, `calohitsmc`, ... | donor | not simulated |
| `EventNtuple/n_proc_events` | fast | the stored events (what the stock ntuple job of the stored stream processed); the births are in the provenance |
| `EventNtuple/version`, `trkqual_metadata` | exact | copied from the first donor file |
| `EventNtuple/fastsim_provenance` | | TNamed, JSON: the plan, donor files, covered and uncovered paths, stream paths, counters |

The full contract (every leaf the analyses and pyutils read, types, units, frames, sentinels) and the readers are in
`beamfastsim/centuple/contract.py`. Units and frames are the stock ones: mm, ns, MeV, MeV/c, the detector frame for
positions, times from the MC proton time.

## On-spill trigger table (`trigcal`, opt-in)

The device paths see the fast device digis; the fast track comes from the response layer, which does not. In stock
both come from the same hits, so an event with a good offline track nearly always fires the track paths; the fast
chain loses that correlation, and the device response layer is the off-spill CE calibration. `trigcal` restores
both from development events with both arms (the same births): per path, event class (Q / F / N: stock from its
fit, fast from the response layer) and birth time, the target is stock's pass fraction and the device's is its pass
fraction without a response layer. A fast event keeps a device pass with probability f = min(1, T / D) and turns a
device fail into a pass with probability g = max(0, (T - D) / (1 - D)); one uniform per event is shared by every
path, so the paths stay correlated and the device decision still orders the events. Cells with fewer than 50 events
on a side use the class's time-integrated factors.

## Known limits

* Tracker: no delta rays or conversions in the fast transport; active hits drawn from the response tables.
* Calorimeter: the inner edge from a Geant4 table; a known radius disagreement near the inner edge.
* Trigger: without `--trigger-cal` the device response layer is the off-spill CE calibration; with it, the paths
  follow stock per class and birth time, but not in variables the table does not bin (momentum, hits). Calorimeter
  and MC paths are the donor's. The trigger stage subtracts the per-event proton bunch time estimate of the record.
* Response: the tables count the electron's own Geant4 rows and draw the fit from stock reco of the same digitization
  mode; `record` uses `onspill` (ce_chain's OnSpill chain), built from 19,630 development events.
* Calorimeter cluster truth counts every sampled step in the cluster's crystals (no per-hit time window); records
  made before the per-crystal truth share the whole shower's visible energy among the clusters (a few MeV high).
* Donor quantities follow the fast event only through the matching keys: a quantity that depends on something the
  keys do not carry (e.g. the hit lists) is the donor's.
* No pileup: `fullsim_sensitivity` runs on unmixed CE only (its inputs are mixed in production).
