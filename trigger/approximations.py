"""W8 (device digis -> trigger decisions): the approximations of the device trigger.

None has a switch yet: each is listed with its stock alternative, the stock replay (trigger/replay.py with the fast
digis, paired by trigger/hybrid), which runs the stock trigger instead of the device one. Read by
``python3 -m trigger.device.run decide`` (--approx-preset, --approx, --approximations; with nothing switchable only
the record changes) and by ``python3 -m mu2efastsim.bench trigger.device.bench:step``; the record goes into the
decide JSON and the bench report. The device precision (--precision) is not an approximation level here.
"""
from mu2efastsim.approximations import Workflow

REPLAY = "stock trigger replay on the same digis (trigger/replay.py --straw-digi-tag, trigger/hybrid pairing)"

WORKFLOW = Workflow(
    "W8", "trigger paths",
    (),
    fixed={
        "compact_helix_fit": "compact device helix fit in place of RobustHelixFinder/AgnosticHelixFinder; " + REPLAY,
        "kalseed_response": "KalSeedFilter as cuts plus a per-path pass probability, per process (decide --response "
                            "ce, dio, rpc, none or a file; default ce): the shipped binned response, rows in time-cluster hits "
                            "and helix momentum (CE/DIO: the apr helix paths; RPC: every tpr/apr track and helix path), "
                            "scalar factors from CE for the other paths; none: probability 1; " + REPLAY,
        "exact_weighted_median": "time clusters use the exact weighted median where stock uses the P2 estimate; "
                                 + REPLAY,
        "tracker_path_subset": "20 single-helix tracker paths of 52 on the device; the replay runs all 52",
        "no_delta_finder": "DeltaFinder skipped (exact without pileup); " + REPLAY,
    },
)
