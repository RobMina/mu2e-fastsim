"""W5 (CRY cosmics -> CRV coincidences): the approximations of the runchain and their switches.

Read by ``python -m crvfastsim runchain`` and ``python3 -m crvfastsim.analyses.runchain_bench`` (--approx-preset,
--approx NAME=LEVEL, --approximations FILE); the scheme is mu2efastsim.approximations. Every level drives an existing
runchain option (a keyword of runchain.run and the runchain option of the same name); the default levels set
nothing. The record goes into runchain_stats.json, the --emit file, the --save-reco NPZ and the bench report.
"""
from mu2efastsim.approximations import PORT, STOCK_ART, Approximation, Workflow

STEPS_INPUT = "Geant4 CrvSteps from art through runchain --steps-input (response and reco only)"

WORKFLOW = Workflow(
    "W5", "cosmics to CRV veto",
    (
        Approximation(
            "muon_eloss", "stage-C muon energy loss: the run configuration's model (range-out with sub-stepped bars "
            "where range_out is set) or one bar loss at the entry momentum",
            levels=("run_config", "entry_momentum"), default="run_config",
            sets={"entry_momentum": {"legacy_transport": True}},
            fidelity="run_config", alternative=STOCK_ART, stock=STEPS_INPUT, switch="runchain --legacy-transport"),
        Approximation(
            "below_msc", "run configurations with a below-plane world (Run 1A, 1B, 2): straight lines below the MSC "
            "plane, or multiple scattering bar to bar (NumPy transport)",
            levels=("straight", "scattered"), default="straight", sets={"scattered": {"below_msc": True}},
            fidelity="scattered", alternative=STOCK_ART, stock=STEPS_INPUT, switch="runchain --below-msc"),
        Approximation(
            "pulse_fit", "with --jax: 12-sample, 3-round device pulse fit, flagged waveforms refit at full length; "
            "full_length runs the NumPy response, which fits every waveform at full length",
            levels=("short_window", "full_length"), default="short_window", sets={"full_length": {"jax": False}},
            fidelity="full_length", alternative=PORT,
            stock="full-length pulse fit port (reco.pulsefit.reconstruct_full; stock Minuit oracle reco.stockfit)",
            switch="runchain --jax (absent: NumPy response, full-length fit)"),
    ),
    fixed={
        "straight_line_transport": "Geant4 transport; the stock-equivalent input is " + STEPS_INPUT,
        "secondaries_library": "Geant4 secondaries; the stock-equivalent input is " + STEPS_INPUT,
    },
)
