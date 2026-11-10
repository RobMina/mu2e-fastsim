"""W6 (pileup mixing): the approximations of the device mixer.

None has a switch yet (the sampler and the donor source are under active development): each is listed with its stock
alternative, the native art mixers (mixing/fcl/four_stream_response.fcl). Read by ``python3 -m mixing.stream_benchmark``
(--approx-preset, --approx, --approximations; with nothing switchable only the record changes) and by
``python3 -m mu2efastsim.bench mixing.device_bench:run_pass``; the record goes into configuration.json and the bench
report.
"""
from mu2efastsim.approximations import Workflow

NATIVE = "native art mixing (fcl/four_stream_response.fcl)"

WORKFLOW = Workflow(
    "W6", "pileup mixing",
    (),
    fixed={
        "finite_donor_pools": "donor frames read in order and reused when the pool wraps; " + NATIVE,
        "independent_sampler_draws": "one shared intensity and four conditional Poissons from the JAX RNG, no event "
                                     "replay; stock seeded replay: fcl/four_stream_fast_inputs.fcl",
        "tracker_crv_only": "calorimeter and trigger not mixed; " + NATIVE,
    },
)
