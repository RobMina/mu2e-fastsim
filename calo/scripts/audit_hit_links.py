"""Audit saved hit/reco-digi associations without repairing ambiguous identities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from check_hit_weights import flat, leaf, records
from calofastsim.response import weighted_reco_digis


def audit(path, tree_name, expected_events):
    with uproot.open(path) as source:
        tree = source[tree_name]
        if tree.num_entries != expected_events:
            raise ValueError(f"expected {expected_events} entries, got {tree.num_entries}")
        hits = records(tree, "calohits")
        reco = records(tree, "calorecodigis")
    nh = ak.to_numpy(ak.num(leaf(hits, "nSiPMs")))
    nr = ak.to_numpy(ak.num(leaf(reco, "time")))
    ho = np.r_[0, np.cumsum(nh)[:-1]]
    ro = np.r_[0, np.cumsum(nr)[:-1]]
    n_hits, n_reco = int(nh.sum()), int(nr.sum())
    forward = leaf(hits, "recoDigis")
    reverse = leaf(reco, "caloHitIdx")
    for name, links in (("forward", forward), ("reverse", reverse)):
        if not np.issubdtype(flat(links).dtype, np.integer):
            raise ValueError(f"noninteger {name} links")
    if bool(ak.any((forward < 0) | (forward >= ak.Array(nr)))):
        raise ValueError("forward link outside its source event")
    if bool(ak.any((reverse < -1) | (reverse >= ak.Array(nh)))):
        raise ValueError("reverse link outside its source event")
    fwd = flat(forward + ak.Array(ro)).astype(np.int64)
    forward_counts = flat(ak.num(forward, axis=2))
    groups = np.repeat(np.arange(n_hits), forward_counts)
    rev = flat(reverse + ak.Array(ho)).astype(np.int64)
    linked = flat(reverse) >= 0
    rev_counts = np.bincount(rev[linked], minlength=n_hits)
    expected = flat(leaf(hits, "nSiPMs"))
    pairs = np.rec.fromarrays([groups, fwd], names="hit,reco")
    _, pair_counts = np.unique(pairs, return_counts=True)
    use_count = np.bincount(fwd, minlength=n_reco)
    inconsistent = rev[fwd] != groups
    bad_hits = (forward_counts != expected) | (rev_counts != expected)
    bad_hits |= np.bincount(groups[inconsistent], minlength=n_hits) > 0
    event_for_hit = np.repeat(np.arange(expected_events), nh)
    event_for_reco = np.repeat(np.arange(expected_events), nr)
    sig_fields = ("time", "timeErr", "eDep", "eDepErr", "chi2", "ndf", "pileUp")
    sig = np.rec.fromarrays([event_for_reco] + [flat(leaf(reco, key)) for key in sig_fields])
    _, inv, sig_counts = np.unique(sig, return_inverse=True, return_counts=True)
    signature_multiplicity = sig_counts[inv]
    sipm = flat(leaf(reco, "SiPMID"))
    bad = np.flatnonzero(bad_hits)[:16]
    repeated_reco = np.flatnonzero(use_count > 1)[:16]
    report = {
        "scope": "saved associations diagnostic; no identity repair or physics certification",
        "events": expected_events, "hits": n_hits, "reco_digis": n_reco,
        "expected_members": int(expected.sum()), "forward_members": len(fwd),
        "reverse_members": int(linked.sum()),
        "forward_multiplicity_bad_hits": int(np.count_nonzero(forward_counts != expected)),
        "reverse_multiplicity_bad_hits": int(np.count_nonzero(rev_counts != expected)),
        "repeated_hit_reco_pairs": int(np.count_nonzero(pair_counts > 1)),
        "extra_repeated_hit_reco_members": int(np.maximum(pair_counts - 1, 0).sum()),
        "reverse_disagrees_with_forward_members": int(inconsistent.sum()),
        "reco_digis_used_multiple_times": int(np.count_nonzero(use_count > 1)),
        "unlinked_reco_digis": int(np.count_nonzero(~linked)),
        "unlinked_reco_with_duplicated_fit_signature": int(np.count_nonzero(~linked & (signature_multiplicity > 1))),
        "forward_wrong_crystal_members": int(np.count_nonzero(sipm[fwd] // 2 != flat(leaf(hits, "crystalId"))[groups])),
        "first_bad_hits": {"global_index": bad.tolist(), "event_entry": event_for_hit[bad].tolist(),
                           "local_hit_index": (bad - ho[event_for_hit[bad]]).tolist(),
                           "forward_members": forward_counts[bad].tolist(),
                           "reverse_members": rev_counts[bad].tolist(), "expected": expected[bad].tolist()},
        "repeated_reco_fit_signatures": {
            "global_reco_index": repeated_reco.tolist(),
            "event_entry": event_for_reco[repeated_reco].tolist(),
            "sensor_id": sipm[repeated_reco].tolist(),
            "fit_signature_multiplicity": signature_multiplicity[repeated_reco].tolist(),
            "fields": {name: flat(leaf(reco, name))[repeated_reco].tolist() for name in sig_fields},
        },
        "forward_reduction_diagnostic": {},
    }
    for quantity in ("time", "eDep"):
        values = flat(leaf(reco, quantity))[fwd]
        errors = flat(leaf(reco, quantity + "Err"))[fwd]
        order = np.lexsort((values if quantity == "time" else flat(leaf(reco, "time"))[fwd], groups))
        pred, uncertainty = weighted_reco_digis(values[order], errors[order], groups[order], n_hits)
        ref, referr = flat(leaf(hits, quantity)), flat(leaf(hits, quantity + "Err"))
        report["forward_reduction_diagnostic"][quantity] = {
            "value_float32_mismatches": int(np.count_nonzero(pred != ref)),
            "error_float32_mismatches": int(np.count_nonzero(uncertainty != referr)),
            "max_abs_value_difference": float(np.max(np.abs(pred - ref), initial=0)),
            "max_abs_error_difference": float(np.max(np.abs(uncertainty - referr), initial=0)),
        }
    report["associations_clean"] = not bool(np.any(bad_hits) or np.any(pair_counts > 1))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--tree", default="EventNtuple/ntuple")
    parser.add_argument("--expected-events", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.input, args.tree, args.expected_events)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
