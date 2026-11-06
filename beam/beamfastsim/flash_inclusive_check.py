"""Small vectorized FlashReferenceExporter pilot summary; not an S2 validation.

Run with --input inclusive.root --out summary.json --events 10. Identities stay
integer; no event, particle or step Python loops. Deposits are in MeV, times ns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

EVENT = ("run", "subrun", "event")
PRODUCT = (*EVENT, "product")
SUBPRODUCT = ("run", "subrun", "product")
FIELDS = {
    "events": (*EVENT, "ordinal", "ptr_failures"),
    "products": (*PRODUCT, "kind", "scope", "size", "label", "instance"),
    "particles": (*PRODUCT, "key", "pdg", "stage", "creation", "start_volume"),
    "steps": (*PRODUCT, "particle_state", "particle_product", "particle_key",
              "time", "total_edep", "nonionizing_edep", "px", "py", "pz"),
    "statuses": (*PRODUCT, "status", "overflow", "killed_step", "killed_field", "ntracks"),
    "volumes": (*SUBPRODUCT, "stage", "index", "name", "material"),
    "completion": ("complete", "events", "ordinal_base", "subrun_products"),
}
ROLES = {"photonSteps": (3, "g4run", "photonSteps"),
         "tracker": (3, "g4run", "tracker"), "source": (3, "BeamFilter", "Beam"),
         "particles": (1, "g4run", ""), "status": (4, "g4run", "")}


def keys(table, fields):
    """Structured unsigned keys avoid float promotion even above 2**63."""
    out = np.empty(len(table[fields[0]]), dtype=[(str(i), "u8") for i in range(len(fields))])
    for i, field in enumerate(fields):
        value = np.asarray(table[field])
        if value.dtype.kind not in "iu" or np.any(value < 0):
            raise ValueError(f"{field}: identity must be a nonnegative integer")
        out[str(i)] = value
    return out


def join(left, right):
    """Vectorized left join; missing = -1, ambiguous targets fail explicitly."""
    if not len(right):
        return np.full(len(left), -1, dtype=np.int64)
    order = np.argsort(right)
    sorted_right = right[order]
    if np.any(sorted_right[1:] == sorted_right[:-1]):
        raise ValueError("Duplicate/ambiguous join target")
    pos = np.searchsorted(sorted_right, left)
    safe = np.minimum(pos, len(right) - 1)
    return np.where((pos < len(right)) & (sorted_right[safe] == left), order[safe], -1)


def matched(index, mask):
    out = np.zeros(len(index), dtype=bool)
    valid = index >= 0
    out[valid] = mask[index[valid]]
    return out


def grouped(**columns):
    """Columnar group counts, including strings, without a loop over groups."""
    if not columns:
        return {}
    values = np.rec.fromarrays(list(columns.values()), names=list(columns))
    unique, counts = np.unique(values, return_counts=True)
    return {**{name: unique[name].tolist() for name in columns}, "count": counts.tolist()}


def deposits(steps, mask):
    total = steps["total_edep"][mask].astype(np.float64)
    nonion = steps["nonionizing_edep"][mask].astype(np.float64)
    return dict(steps=int(mask.sum()), total_mev=float(total.sum()),
                nonionizing_mev=float(nonion.sum()), ionizing_mev=float((total - nonion).sum()))


def summarize(tables, expected_events):
    """Summarize flat numpy branch arrays, retaining the entire event ledger."""
    if isinstance(expected_events, bool) or not isinstance(expected_events, (int, np.integer)) or expected_events < 0:
        raise ValueError("expected_events must be a nonnegative integer")
    e, p, a, s, st, v, c = (tables[name] for name in FIELDS)
    n = len(e["event"])
    ek = keys(e, EVENT)
    join(ek, ek)  # uniqueness, also for an otherwise empty export
    pk = keys(p, PRODUCT)
    pe = join(keys(p, EVENT), ek)
    event_product = (p["scope"] == 0) & (pe >= 0)
    joins, orphans = {}, {}
    for name, table in (("steps", s), ("particles", a), ("statuses", st)):
        joins[name] = join(keys(table, PRODUCT), pk[p["scope"] == 0])
        available = np.flatnonzero(p["scope"] == 0)
        valid = joins[name] >= 0
        joins[name][valid] = available[joins[name][valid]]
        orphans[name + "_without_product"] = int((~valid).sum())
        orphans[name + "_without_event"] = int((join(keys(table, EVENT), ek) < 0).sum())
    si, ai, sti = (joins[name] for name in ("steps", "particles", "statuses"))
    se, ae = join(keys(s, EVENT), ek), join(keys(a, EVENT), ek)
    roles, masks, counts = {}, {}, {}
    for role, (kind, label, instance) in ROLES.items():
        use = event_product & (p["kind"] == kind) & (p["label"] == label) & (p["instance"] == instance)
        join(keys(p, EVENT)[use], keys(p, EVENT)[use])  # reject multiple processes for a role
        index = si if kind == 3 else ai if kind == 1 else sti
        mask = matched(index, use)
        masks[role] = mask
        row_counts = np.bincount(index[mask], minlength=len(p["product"]))
        event_counts = np.zeros(n, dtype=np.int64)
        event_counts[pe[use]] = row_counts[use]
        counts[role] = event_counts
        roles[role] = dict(module=label, instance=instance, product_rows=int(use.sum()),
                           missing_events=int(n - use.sum()), declared_rows=int(p["size"][use].sum()),
                           rows=int(mask.sum()), size_mismatches=int(np.count_nonzero(row_counts[use] != p["size"][use])),
                           zero_events_with_product=int(np.count_nonzero(row_counts[use] == 0)))

    # Destination volume metadata identifies the new stage even in empty events.
    vp = (p["scope"] == 1) & (p["kind"] == 5) & (p["label"] == "g4run") & (p["instance"] == "")
    vi = join(keys(a, ("run", "subrun")), keys(p, ("run", "subrun"))[vp])
    stages = np.full(len(a["key"]), -1, dtype=np.int64)
    volume_product = np.zeros(len(a["key"]), dtype=np.uint64)
    sizes = p["size"][vp]
    known = vi >= 0
    known[known] &= sizes[vi[known]] > 0
    stages[known] = sizes[vi[known]].astype(np.int64) - 1
    volume_product[known] = p["product"][vp][vi[known]]
    photons = masks["particles"] & (a["pdg"] == 22)
    new = photons & known & (a["stage"] == stages)
    query = {**a, "product": volume_product, "index": a["start_volume"]}
    birth = join(keys(query, (*SUBPRODUCT, "stage", "index")), keys(v, (*SUBPRODUCT, "stage", "index")))
    birth[~known] = -1
    birth_index = birth[new]
    birth_name = np.full(len(birth_index), "<unresolved>", dtype=np.result_type(v["name"].dtype, "U12"))
    birth_name[birth_index >= 0] = v["name"][birth_index[birth_index >= 0]]
    new_counts = np.bincount(ae[new], minlength=n)
    particle_index = join(keys(s, (*EVENT, "particle_product", "particle_key")), keys(a, (*PRODUCT, "key")))
    resolved = (s["particle_state"] == 1) & (particle_index >= 0)
    source = masks["source"]
    src_resolved = source & resolved
    tracker = masks["tracker"]
    folded = np.fmod(s["time"], 1695.)
    selected = tracker & ((folded > 250.) | (folded < 10.))
    status = masks["status"]
    photon_steps = masks["photonSteps"]
    unique_steps = np.unique(keys(s, (*EVENT, "particle_product", "particle_key"))[photon_steps & resolved])
    unique_event_keys = np.empty(len(unique_steps), dtype=ek.dtype)
    for i in range(len(EVENT)):
        unique_event_keys[str(i)] = unique_steps[str(i)]
    unique_counts = np.bincount(join(unique_event_keys, ek), minlength=n)
    completion_ok = bool(len(c["complete"]) == 1 and c["complete"][0] == 1 and
                         c["events"][0] == n and c["subrun_products"][0] > 0)
    maximum = int(counts["photonSteps"].max(initial=0))
    photon_products = event_product & (p["kind"] == 3) & (p["label"] == "g4run") & (p["instance"] == "photonSteps")
    max_declared = int(p["size"][photon_products].max(initial=0))
    per_event = {**{field: e[field].tolist() for field in (*EVENT, "ordinal")},
                 **{role + "_rows": value.tolist() for role, value in counts.items()},
                 "new_photons": new_counts.tolist(), "recorded_photon_tracks": unique_counts.tolist()}
    for name, mask in (("tracker_raw", tracker), ("tracker_stock_time", selected)):
        for field, values in (("total", s["total_edep"].astype(float)),
                              ("nonionizing", s["nonionizing_edep"].astype(float)),
                              ("ionizing", s["total_edep"].astype(float) - s["nonionizing_edep"].astype(float))):
            per_event[name + "_" + field + "_mev"] = np.bincount(se[mask], weights=values[mask], minlength=n).tolist()
    integrity = dict(**orphans, event_products_without_event=int(np.count_nonzero((p["scope"] == 0) & (pe < 0))),
                     available_step_pointers_without_particle=int(np.count_nonzero((s["particle_state"] == 1) & (particle_index < 0))),
                     new_photons_without_birth_volume=int(np.count_nonzero(new & (birth < 0))),
                     g4run_photons_without_stage_metadata=int(np.count_nonzero(photons & ~known)))
    return dict(
        detector_response_valid_S2=False,
        scope="Inclusive raw pilot only; detector response is NOT valid S2 yet. No response-per-photon inference.",
        events=dict(expected=int(expected_events), observed=n, matches_expected=n == expected_events,
                    zero_photon_steps=int(np.count_nonzero(counts["photonSteps"] == 0)),
                    zero_new_photons=int(np.count_nonzero(new_counts == 0)),
                    zero_tracker_steps=int(np.count_nonzero(counts["tracker"] == 0)),
                    note="Zeros refer to observed rows; consult missing_events and join diagnostics, not assumed empty products."),
        completion=dict(ok=completion_ok, **{field: c[field].tolist() for field in FIELDS["completion"]}),
        products=roles, joins=integrity,
        photons=dict(newly_simulated=int(new.sum()), copied_earlier_stage=int(np.count_nonzero(photons & known & (a["stage"] < stages))),
                     unexpected_later_stage=int(np.count_nonzero(photons & known & (a["stage"] > stages))),
                     by_stage_creation_birthVolume=grouped(stage=a["stage"][new], creation=a["creation"][new],
                                                          birthVolumeIndex=a["start_volume"][new], birthVolume=birth_name),
                     max_new_per_event=int(new_counts.max(initial=0)),
                     max_recorded_tracks_per_event=int(unique_counts.max(initial=0)),
                     max_recorded_photon_steps_per_event=maximum, max_declared_photon_steps_per_event=max_declared,
                     recorder_threshold=1_000_000, at_or_above_1M=max(maximum, max_declared) >= 1_000_000,
                     note="photonSteps counts step records, not births; configured cap is not stored in this export."),
        tracker=dict(raw=deposits(s, tracker), stock_time_selected=deposits(s, selected),
                     note="Raw-step timing diagnostic, not the stock aggregated event selection; early board stream has no time cut.",
                     predicate="fmod(time_ns, 1695) > 250 OR fmod(time_ns, 1695) < 10 (strict)",
                     nonfinite_times=int(np.count_nonzero(tracker & ~np.isfinite(s["time"])))),
        source=dict(rows=int(source.sum()), momentum_magnitude_sum_mev_c=float(np.sqrt(
                    s["px"][source]**2 + s["py"][source]**2 + s["pz"][source]**2).sum()),
                    pdg_counts=grouped(pdg=a["pdg"][particle_index[src_resolved]]),
                    unresolved_particle_rows=int(np.count_nonzero(source & ~resolved)), deposits=deposits(s, source)),
        statuses=dict(rows=int(status.sum()), codes=grouped(status=st["status"][status]),
                      overflow=int(np.count_nonzero(st["overflow"][status])),
                      killed_step=int(st["killed_step"][status].sum()), killed_field=int(st["killed_field"][status].sum()),
                      tracks=int(st["ntracks"][status].sum())),
        pointers=dict(exporter_failures=int(e["ptr_failures"].sum()),
                      events_with_failures=int(np.count_nonzero(e["ptr_failures"])),
                      step_states=grouped(state=s["particle_state"])), per_event=per_event)


def read_tables(path):
    import awkward as ak
    import uproot

    with uproot.open(path) as root:
        return {name: {field: ak.to_numpy(arrays[field]) for field in fields}
                for name, fields in FIELDS.items()
                for arrays in [root["export/" + name].arrays(list(fields), library="ak")]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="JSON summary path")
    parser.add_argument("--events", required=True, type=int, help="Expected input event count, not a read limit")
    args = parser.parse_args(argv)
    start = perf_counter()
    report = summarize(read_tables(args.input), args.events)
    report["input"] = str(args.input)
    report["elapsed_seconds"] = perf_counter() - start
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['events']['observed']}/{args.events} events; "
          f"{report['photons']['newly_simulated']} new photons; "
          f"{report['tracker']['raw']['ionizing_mev']:.6g} MeV raw tracker ionization; "
          f"{report['elapsed_seconds']:.3f} s/file. NOT valid S2. {args.out}")
    return 0 if report["events"]["matches_expected"] and report["completion"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
