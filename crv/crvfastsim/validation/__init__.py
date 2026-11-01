"""crvfastsim.validation -- checks of the fast CRV chain against Geant4 and the stock chain.

    v0check, v1compare, v2compare      V0 bar geometry, V1 primary-muon crossings, V2 response + reco on fast steps
    v4photons, v4sipm                  V4 photon and SiPM-charge stages against the stock modules
    transport_validation               paired transport validation against a G4 reference
    compare, coverage                  DQM histogram merge/compare and the coverage audit of two arms
    pilot, paired_rates, paired_shapes, event_shapes, reco_detail
                                       pilot certification, event-paired rate and shape errors, event shapes,
                                       reconstruction detail of two complete arms

The CLI subcommands keep their names (python3 -m crvfastsim v1compare ...). The old module paths
(crvfastsim.analyses.<name>, valhist.<name>) are aliases of these modules for one release.
"""
