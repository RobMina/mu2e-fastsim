"""W1 (TS entrance -> stopped muons) and W2-W4 (stops or DIO/RPC electrons -> tracker steps, digis, observables):
the approximations of the beam chains and their switches.

Read by ``python3 -m beamfastsim.tsdevice`` (W1), ``python3 -m beamfastsim.cetracker`` (W2-W4) and
``python3 -m mu2efastsim.bench`` with their bench entry points (--approx-preset, --approx NAME=LEVEL,
--approximations FILE); the scheme is mu2efastsim.approximations. Every level drives an existing option; the default
levels set nothing. The record goes into the tsdevice NPZ ``meta``, the cetracker ``handoff.json`` and the bench report.
"""
from mu2efastsim.approximations import NONE, STOCK_ART, Approximation, Workflow

W1 = Workflow(
    "W1", "muon beam: TS entrance to stopped muons",
    (
        Approximation(
            "ts_transport", "TS stage: condensed device transport (RK4 in the field bank, ionisation with Urban "
            "fluctuations, cumulative Highland scattering, step caps), or the DS stage started from the reference's "
            "stock Geant4 DS-entrance states",
            levels=("condensed", "stock_ds_entrance"), default="condensed",
            sets={"stock_ds_entrance": {"arm": "stock-ds"}}, fidelity="stock_ds_entrance", alternative=STOCK_ART,
            stock="stock DS-entrance states of the reference (art ReadVirtualDetector) feed the DS stage",
            switch="tsdevice --arm (CLI only; the bench entry runs the condensed chain)"),
        Approximation(
            "ts_vacuum_step", "TS stage step length in vacuum: the fixed step everywhere (default), or a long step "
            "(up to 20 mm, at most 0.1 rad of turning and a 0.25 mm sagitta) where its chord crosses no material, "
            "stays inside the envelope and does not reach the DS plane",
            levels=("fixed", "long"), default="fixed", sets={"long": {"vacuum_step": 20.0}}, fast="long",
            fidelity="fixed", alternative=NONE,
            stock="Geant4 takes field-limited steps in vacuum (deltaChord 0.25 mm); the fixed 2.5 mm step is the "
                  "closer, slower model",
            switch="tsdevice --vacuum-step; bench_chain(vacuum_step=); Settings.vacuum_step_mm"),
    ),
    fixed={
        "pion_nuclear_interactions": "not modelled: pions reaching the DS are counted, not transported",
        "highland_gaussian_scattering": "Gaussian Highland, no single-scattering tails (stock: Geant4 Urban MSC)",
        "stopped_pion_and_capture_products": "stopped-pion daughters, muon capture and decay electrons not produced",
        "ds_layer_physics": "thin-plane foils with per-layer Highland and restricted loss (stock: Geant4 in the "
                            "foil geometry)",
    },
)

_TRACKER = (
    Approximation(
        "calo_albedo", "tracker returns from the calorimeter (backscatter): sampled from a Geant4-built library, or "
        "none; the NumPy tracker has no albedo",
        levels=("library", "none"), default="library", sets={"none": {"albedo": None}}, cli={"none": {"albedo": "none"}},
        fast="none", fidelity="library", alternative=NONE,
        stock="Geant4 transport through the calorimeter; no stock path in the chain (the library is built from it)",
        switch="cetracker --albedo (device tracker); DeviceChain(albedo=)"),
    Approximation(
        "straw_wall", "straw wall material: counted only on chords that cross the gas (default), or the complete "
        "outer-minus-gas wall with wall-only chords",
        levels=("gas_chords", "complete"), default="gas_chords", sets={"complete": {"complete_outer_wall": True}},
        fidelity="complete", alternative=NONE, stock="Geant4 straw walls; the complete wall is the closer model",
        switch="cetracker --complete-outer-wall; DeviceChain(complete_outer_wall=)"),
)
_TRACKER_FIXED = {
    "primary_only_steps": "no delta rays, conversions or other secondaries in the tracker besides the albedo "
                          "(DeviceChain(deltas=) takes a Geant4 delta-ray library .npz, opt-in; no level until the "
                          "library ships); stock StrawGasSteps from the production reco are the stock-art path",
    "reco_response": "no track fit: efficiency and residuals drawn from stock LoopHelix tables (default: measured on "
                     "the full Geant4 events, with the empirical joint draw; CE-only tables by name, "
                     "DeviceChain(response=\"ceonly\")); stock art: the injector (fasttrk_field.root) through "
                     "StrawGasStepsFromTable, stock digis and LoopHelix",
    "target_slab_physics": "flat slabs, constant field per layer, shell curvature as an axial field, no "
                           "secondaries (stock: Geant4 in the target geometry)",
}

W2 = Workflow("W2-W4", "stops or DIO/RPC electrons to tracker steps, digis and observables", _TRACKER,
              fixed={**_TRACKER_FIXED,
                     "rpc_internal_electron_only": "W4: internal conversion only, the positron dropped (stock: "
                                                   "external RPC with photon transport and both leptons)"},
              aliases=("W2", "W3", "W4"))

WORKFLOWS = {"W1": W1, "W2-W4": W2}
