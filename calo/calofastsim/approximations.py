"""W7 (calorimeter): the approximations of the device chain (calofastsim.device_chain, calofastsim.entrant_chain).

One has a level switch: ``digi_response`` (make_chain / make_steps_chain / make_entrant_chain ``digi_response=``).
The default shower source is the per-entrant library (calofastsim.entrant_chain, entrant_chain.W7_DEFAULT); the
front-only library (device_chain.make_chain) remains available. The device chain has no command line; the
stock-equivalent input is stock steps through device_chain.make_steps_chain (digis exact, hits bit-exact, clusters
identical on stock steps), a different entry point rather than a level. The library settings (placement "crystal" or
"geometric", key_neighbors, d_edge) choose between library approximations, none of them stock, so they are not levels
here. Read by ``python3 -m mu2efastsim.bench`` for calofastsim entry points; the record goes into the bench report.
"""
from mu2efastsim.approximations import NONE, Approximation, Workflow

STEPS = "stock CaloShowerSteps through device_chain.make_steps_chain / steps_batch (exact on stock steps)"

WORKFLOW = Workflow(
    "W7", "calorimeter",
    (
        Approximation(
            "digi_response", "crystal hits: drawn per (event, crystal) from a table of the full chain's outcome (hit, "
            "energy ratio, time offset) per bin of the main deposit's visible energy, without digis "
            "(calofastsim.param_response, default), or photoelectrons, waveforms, digis and the template fit",
            levels=("waveform", "parametric"), default="parametric", sets={"waveform": {"digi_response": "waveform"}},
            fast="parametric", fidelity="waveform", alternative=NONE,
            stock="stock CaloDigiMaker and CaloRecoDigiMaker; the waveform response is the closer model (the table is "
                  "built from it)",
            switch="device_chain.make_chain / make_steps_chain(digi_response=)"),
    ),
    fixed={
        "shower_library": "per-entrant library (calofastsim.entrant_chain): each particle's first crossing into a "
                          "disk envelope through the front face or inner edge draws one of the 8 library entries "
                          "nearest in entry radius (front) or depth (inner edge) within its (species, face, disk, "
                          "energy, cosine) cell; " + STEPS,
        "crystal_placement": "the donor pattern turned about the disk axis to the entry azimuth and moved onto the "
                             "crystal grid at the entry; the inner crystal ring is only 180-degree symmetric, so "
                             "innermost-ring deposits can land one ring out (cluster radius at 380-460 mm); " + STEPS,
        "entry_faces_front_inner": "only front and inner-edge entrants get a response (back and outer-edge "
                                   "crossings are not in the library); " + STEPS,
        "template_fit_single_pass": "Newton template fit without the second pass for chi2/ndf > 1 (stock: TMinuit "
                                    "with the refit)",
    },
)
