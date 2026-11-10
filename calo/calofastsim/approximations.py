"""W7 (calorimeter): the approximations of the device chain (calofastsim.device_chain).

None has a level switch yet. The device chain has no command line; the stock-equivalent input is stock steps through
device_chain.make_steps_chain (digis exact, hits bit-exact, clusters identical on stock steps), a different entry
point rather than a level. pack_library's placement ("crystal" or "geometric") and radius_neighbors choose between
library approximations, neither of them stock, so they are not levels here. Read by ``python3 -m mu2efastsim.bench``
for calofastsim entry points; the record goes into the bench report.
"""
from mu2efastsim.approximations import Workflow

STEPS = "stock CaloShowerSteps through device_chain.make_steps_chain / steps_batch (exact on stock steps)"

WORKFLOW = Workflow(
    "W7", "calorimeter",
    (),
    fixed={
        "shower_library": "whole-shower library at the disk front, one of the K entries nearest in entry radius; "
                          + STEPS,
        "crystal_placement": "library shower moved onto the crystal grid at the query position; " + STEPS,
        "direct_front_entries_only": "only direct disk-front entries get a response; " + STEPS,
        "template_fit_single_pass": "Newton template fit without the second pass for chi2/ndf > 1 (stock: TMinuit "
                                    "with the refit)",
    },
)
