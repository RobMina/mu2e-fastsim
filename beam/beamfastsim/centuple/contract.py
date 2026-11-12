"""The EventNtuple contract of the target analyses, and how the fast chain covers each branch.

Readers pinned (the analysis server at commit e40d3d4, ``tools/analyses``):

* ``fullsim/eventntuple.py`` ``read_eventntuple`` (``fullsim_sensitivity`` and the cut-set-80 code ``fullsim/cuts.py``):
  ``tree.arrays(names)`` per group; a missing branch is an error;
* ``trigger_efficiency_ntuple.py`` ``read_track_arrays`` / ``read_file`` (also used by ``..._binned``,
  ``selection_efficiency_ntuple``, ``momentum_resolution_ntuple`` and ``variable_comparison_ntuple``):
  ``tree.arrays(filter_name=LEAVES + MC_LEAVES)``, the ``trig_<path>`` booleans, ``tree.num_entries``;
* ``calo_cluster_resolution_ntuple.py``: ``caloclusters.*`` and ``caloclustersmc.*`` leaves;
* ``occupancy_ntuple.py``: ``hitcount``, ``calohits``, ``caloclusters``, ``crvcoincs``, ``crvsummary``, ``tcnt``.

pyutils' ``Importer`` reads ``tree.arrays([top-level branch names])``, which on a split TTree returns the records with
their ``<branch>.<leaf>`` fields; this is why the output is a TTree and not an RNTuple.

Schema: EventNtuple of AnalysisMDC2025 v02_02_02 (backing SimJob MDC2025ax), the release of Mu2eBO's ce_chain
ntuple step, ``from_mcs-mockdata.fcl``. Units mm, ns, MeV, MeV/c; positions in the detector (tracker) frame, the Mu2e
frame shifted by (-3904, 0, 10171) mm; times from the MC proton time (the stock convention, no pbtime subtraction).
Sentinels are the struct defaults of the EventNtuple headers (e.g. ``trkpid.result`` -999 without a calorimeter
match, ``trkcalohit.did`` -1, ``TrkCaloHitInfo`` floats -1000).

Coverage classes: ``exact`` (the same value as stock by construction), ``fast`` (simulated by the fast chain,
approximate in the stated way), ``donor`` (taken from a matched stock development event, see ``donor.py``),
``not available``.
"""

READERS = {
    "fullsim_sensitivity / cuts (read_eventntuple)": [
        "run", "subrun", "event", "trig_<paths>", "crvcoincs.time", "crvcoincs.nHits", "crvcoincs.PEs",
        "crvcoincs.timeStart", "crvcoincs.timeEnd", "caloclusters.energyDep_", "trk.nactive", "trk.pdg",
        "trk.status", "trk.goodfit", "trkqual.result", "trkpid.result", "trksegs", "trksegpars_lh", "trkmcsim",
        "EventNtuple/n_proc_events"],
    "trigger_efficiency_ntuple (+binned, selection_efficiency, momentum_resolution, variable_comparison)": [
        "trk.{status,goodfit,pdg,nhits,ndof,nactive,ndouble,nplanes,nnullambig,nmat,nmatactive,nipaup,nipadown,"
        "nstup,nstdown,tsdainter,opainter,chisq,fitcon,maxgap,avggap,avgedep,seedalg,fitalg}",
        "trkqual.result", "trkpid.result", "trkcalohit.did", "trkcalohit.edep", "trkcalohit.dt", "trksegs",
        "trksegpars_lh", "event", "run", "subrun", "trksegsmc", "trkmcsim", "evtinfomc/nprotons", "trig_<paths>",
        "num_entries"],
    "calo_cluster_resolution_ntuple": [
        "caloclusters.{energyDep_,energyDepErr_,time_,timeErr_,diskID_,size_,isSplit_}",
        "caloclusters.cog_.fCoordinates.{fX,fY}",
        "caloclustersmc.{nsim,etot,eprimary,tprimary,prel._rel}", "event", "run", "subrun"],
    "occupancy_ntuple": [
        "hitcount/{nsd,nesel,nrsel,ntsel,nbkg}", "calohits.time_", "calohits.eDep_", "caloclusters.time_",
        "crvcoincs.time", "crvsummary/{nHitCounters,totalPEs}", "tcnt.ntrk"],
    "Mu2eBO ce_chain plot step": [
        "trigger_efficiency_ntuple with trigger_paths 'apr_TrkDe_80m70p, cpr_TrkDe_80m70p' and its default "
        "selection: metrics n_events, n_selected, n_triggered, efficiency"],
}

# (branch or leaf group, type and nesting, class, how) - the coverage table of the docs
COVERAGE = [
    ("evtinfo.run/subrun/event", "int per event", "exact", "the identity of the birth (the stock dts event)"),
    ("evtinfo.pbtime/pbterr/nprotons", "float/int per event", "donor", ""),
    ("evtinfomc.nprotons/pbtime", "int/float per event", "donor", "stock CE: nprotons 1"),
    ("primary[0] (SimInfo)", "vector<SimInfo>", "fast", "birth position, momentum, time of the CE; rest donor"),
    ("trig_<20 device paths>", "bool per event", "fast",
     "trigger.device on device digis (tpr, apr incl. apr_TrkDe_80m70p): ports + KalSeedFilter / AgnosticHelix "
     "response layer from off-spill CE replays; on-spill digis; opt-in (assemble --trigger-cal) the decisions "
     "without a response layer and the on-spill table per class and birth time (trigcal)"),
    ("trig_cpr_*, calo_*, Triggerable*, mpr_*, cst_*, minBias_*, apr_TwoTrk*, apr_proton", "bool per event", "donor",
     "the matched stock event's decision (class, covered-trigger bit, track and calorimeter keys): the calorimeter-"
     "seeded and MC paths are not on the device"),
    ("stored events", "tree entries", "fast+donor",
     "the stock stream rule (cpr_Trk*_80* | apr_Trk*_80* | Triggerable*) on the combined decisions"),
    ("trk.nactive, trk.fitcon, trk.chisq", "vector<TrkInfo>", "fast", "response layer draws (chisq = chi2/ndf x ndof)"),
    ("trk.nhits/ndof/nesel/nrsel/ntsel/nsel", "vector<TrkInfo>", "donor", "shifted by the active-hit difference"),
    ("trk.status/goodfit/seedalg/fitalg/pdg/ndouble/nplanes/.../avgedep", "vector<TrkInfo>", "donor",
     "the donor e- fit has the fast active hits (nearest), tan dip, fit consistency"),
    ("trk (e+, mu-, mu+ hypotheses)", "vector<TrkInfo> entries 2-4", "donor", "rotated in azimuth, shifted in time"),
    ("trksegs TT_Front/TT_Mid/TT_Back of the e- fit", "vector<vector<TrkSegInfo>>", "fast",
     "TT_Front: true entry position, response |p| and tan dip, true azimuth, response t0; TT_Mid/TT_Back: fast truth "
     "there with the front residuals"),
    ("trksegs other surfaces (ST 100-104, OPA 95, IPA, 90, 96)", "vector<vector<TrkSegInfo>>", "donor",
     "the donor's extrapolation, rotated and shifted: presence counts (st_boundary, has_st, no_opa) are the donor's"),
    ("trksegs momerr/dmom/inbounds/gap", "", "donor", ""),
    ("trksegpars_lh at TT surfaces", "vector<vector<LoopHelixInfo>>", "fast",
     "from the reco state in the local field; front d0/maxr/tanDip = response draws; errors (t0err) donor"),
    ("trkqual.result", "vector<MVAResultInfo>", "donor", "nearest active hits / fit consistency / tan dip"),
    ("trkpid.result", "vector<MVAResultInfo>", "donor", "own draw among e- donors with the same calo match, nearest E/p"),
    ("trkcalohit (e- fit)", "vector<TrkCaloHitInfo>", "fast",
     "match = leading fast cluster on the contact disk above 10 MeV: did, edep, ctime, csize, poca, mom; dt from the "
     "contact time; resolution fields reset/donor"),
    ("trkmcsim[*][0]", "vector<vector<SimInfo>>", "fast", "birth state of the CE (every track of the same particle)"),
    ("trkmcsim ancestors", "", "donor", ""),
    ("trksegsmc TT_Front/TT_Mid/TT_Back", "vector<vector<SurfaceStepInfo>>", "fast",
     "entry state; TT_Mid from the last forward straw step carried on a helix; TT_Back from the tracker exit"),
    ("trksegsmc other surfaces, trkmc, trkmcvd, trkcalohitmc", "", "donor", "rotated and shifted"),
    ("caloclusters", "vector<CaloClusterInfo>", "fast",
     "front-face contacts: calorimeter device chain clusters; inner-edge contacts: one cluster, the edepana "
     "inner-edge table's visible energy (known disagreement at the inner edge)"),
    ("caloclustersmc", "vector<CaloClusterInfoMC>", "fast",
     "nsim 1 (the CE only, no secondary sims); etot = eprimary = the visible (Birks) energy of the sampled shower "
     "steps in the cluster's hit crystals, tprimary = the earliest of their times (edge contacts: the table's "
     "visible energy and the contact time)"),
    ("calohits", "vector<CaloHitInfo>", "fast", "approximate: one hit per crystal of each cluster, energy split evenly"),
    ("calohitsmc, calomcsim", "", "donor", "not consistent with the fast clusters"),
    ("hitcount.nsd", "int per event", "fast", "device StrawDigis of the CE (no secondaries: no delta rays)"),
    ("hitcount.nesel/nrsel/ntsel/nbkg", "int per event", "donor", "scaled by nsd_fast / nsd_donor"),
    ("tcnt.ntrk", "int per event", "donor", ""),
    ("crvcoincs, crvsummary, crv*mc", "", "donor", "stock CE primary: no CRV activity"),
    ("trkhits, trkmats, trkhitsmc, trkdtdt, trkqual_bdt, lumistream, evtwt", "", "donor",
     "not read by the targets; the donor's (the e- fit's hit lists do not match its fast active-hit count)"),
    ("EventNtuple/n_proc_events", "TH1 one bin", "fast", "stored events (as the stock ntuple job of the stream); births in the provenance"),
    ("EventNtuple/version, trkqual_metadata", "", "exact", "copied from the first donor file"),
    ("EventNtuple/subrunNtuple", "", "not available", "not written"),
]

KNOWN_GAPS = [
    "tracker: no delta rays or conversions in the fast transport (nsd, hit counts low; the response tables carry the "
    "stock full-event fit)",
    "response layer: active hits drawn from tables (chi2/ndf 49-53 / 34-35 on the deck binning: a miss)",
    "calorimeter: inner-edge contacts from a Geant4 table, cluster-radius disagreement near 380-460 mm",
    "DIO tail above 100 MeV/c: held-out excess (not exercised by CE)",
    "device trigger: response layer calibrated on off-spill CE replays (opt-in on-spill table per class and birth "
    "time); calorimeter and MC paths from donors",
]


def table_markdown():
    rows = ["| Branch | Type | Class | How |", "|---|---|---|---|"]
    rows += [f"| {b} | {t} | {c} | {h} |" for b, t, c, h in COVERAGE]
    return "\n".join(rows)
