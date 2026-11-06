"""Saved-reference tracker ancestry census, not a new transport or response fit.

Read only events containing recorded tracker steps, then trace integer-qualified
parent identities in arrays. Birth volumes come from stage/SubRun metadata, not
StepPoint copy numbers. Nearest-photon birth is diagnostic, NOT causal interface
ownership. Full all-input event totals and the saved causal ownership remain
unchanged; only the forensic particle read is limited to nonempty tracker events.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .flash_inclusive_check import EVENT, PRODUCT, SUBPRODUCT, keys, join

PARTICLES = (*PRODUCT, "key", "pdg", "creation", "stage", "start_volume", "parent_state",
             "parent_product", "parent_key", "energy", "x", "y", "z", "time")
STEPS = (*PRODUCT, "index", "particle_state", "particle_product", "particle_key", "time", "total_edep", "nonionizing_edep")
PRODUCTS = (*PRODUCT, "kind", "scope", "label", "instance", "size")
VOLUMES = (*SUBPRODUCT, "stage", "index", "name", "material")


def read_tables(path, response):
    """Load flat columns, restricting ancestry to tracker-active events after I/O."""
    import awkward as ak
    import uproot

    def read(tree, fields):
        a = tree.arrays(list(fields), library="ak")
        return {k: ak.to_numpy(a[k]) for k in fields}

    with uproot.open(path) as root:
        events = read(root["export/events"], (*EVENT, "ordinal"))
        for name in (*EVENT, "ordinal"):
            if not np.array_equal(events[name], response["event_" + name]):
                raise ValueError("saved response and ROOT event ledger differ")
        s = read(root["export/steps"], STEPS)
        rows = response["tracker_step_row"]
        if rows.dtype.kind not in "iu" or np.any(rows < 0) or np.any(rows >= len(s["event"])):
            raise ValueError("saved tracker row index out of range")
        s = {k: v[rows] for k, v in s.items()}
        expected = s["total_edep"].astype(float) - s["nonionizing_edep"].astype(float)
        if not np.array_equal(expected, response["tracker_ionizing_mev"]):
            raise ValueError("saved tracker energies disagree with original ROOT rows")
        selected = np.unique(keys(s, EVENT))
        a = read(root["export/particles"], PARTICLES)
        keep = np.isin(keys(a, EVENT), selected)
        a = {k: v[keep] for k, v in a.items()}
        p = read(root["export/products"], PRODUCTS)
        keep = (p["scope"] == 1) | np.isin(keys(p, EVENT), selected)
        p = {k: v[keep] for k, v in p.items()}
        v = read(root["export/volumes"], VOLUMES)
    return dict(events=events, steps=s, particles=a, products=p, volumes=v)


def birth_metadata(a, p, v):
    """Resolve supported recorded particle products to their volume histories."""
    n = len(a["key"])
    ep = np.flatnonzero((p["scope"] == 0) & (p["kind"] == 1))
    pi = join(keys(a, PRODUCT), keys(p, PRODUCT)[ep])
    pi[pi >= 0] = ep[pi[pi >= 0]]
    volume_product = np.zeros(n, np.uint64)
    destination = np.full(n, -1, np.int64)
    known = np.zeros(n, bool)
    for particle_label, volume_label in (("g4run", "g4run"), ("BeamFilter", "compressPVBeam")):
        role = ((p["scope"] == 1) & (p["kind"] == 5) & (p["label"] == volume_label)
                & (p["instance"] == ""))
        vi = join(keys(a, ("run", "subrun")), keys(p, ("run", "subrun"))[role])
        use = (pi >= 0) & (vi >= 0)
        idx = np.flatnonzero(use)
        use[idx] &= (p["label"][pi[idx]] == particle_label) & (p["instance"][pi[idx]] == "")
        idx = np.flatnonzero(use)
        use[idx] &= p["size"][role][vi[idx]] > 0
        idx = np.flatnonzero(use)
        volume_product[idx] = p["product"][role][vi[idx]]
        destination[idx] = p["size"][role][vi[idx]].astype(np.int64) - 1
        known |= use
    query = {**a, "product": volume_product, "index": a["start_volume"]}
    bi = join(keys(query, (*SUBPRODUCT, "stage", "index")), keys(v, (*SUBPRODUCT, "stage", "index")))
    bi[~known] = -1
    names = np.full(n, "<unresolved>", dtype=np.result_type(v["name"].dtype, "U12"))
    materials = np.full(n, "<unresolved>", dtype=np.result_type(v["material"].dtype, "U12"))
    ok = bi >= 0
    names[ok], materials[ok] = v["name"][bi[ok]], v["material"][bi[ok]]
    new = np.zeros(n, bool)
    idx = np.flatnonzero(pi >= 0)
    new[idx] = ((p["label"][pi[idx]] == "g4run") & known[idx]
                & (a["stage"][idx] == destination[idx]))
    return dict(name=names, material=materials, destination=destination, new=new, known=ok,
                stage_known=known)


def _take(values, index, default):
    out = np.full(len(index), default, dtype=np.result_type(np.asarray(values).dtype, np.asarray(default).dtype))
    ok = index >= 0
    out[ok] = values[index[ok]]
    return out


def trace(tables, *, max_depth=128):
    """One bounded parent walk per array, preserving complete traces for review."""
    if isinstance(max_depth, bool) or not isinstance(max_depth, (int, np.integer)) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer")
    a, s, p, v = (tables[k] for k in ("particles", "steps", "products", "volumes"))
    ak = keys(a, (*PRODUCT, "key"))
    join(ak, ak)
    parent = join(keys(a, (*EVENT, "parent_product", "parent_key")), ak)
    leaf = join(keys(s, (*EVENT, "particle_product", "particle_key")), ak)
    meta = birth_metadata(a, p, v)
    n = len(leaf)
    nearest = np.full(n, -1, np.int64)
    nearest_new = nearest.copy()
    earliest_new = nearest.copy()
    foil_photon = np.zeros(n, bool)
    foil_particle = np.zeros(n, bool)
    metadata_unknown = np.zeros(n, bool)
    entry, root = nearest.copy(), nearest.copy()
    missing = (leaf < 0) | (s["particle_state"] != 1)
    cur = leaf.copy()
    active = ~missing
    chain_hit, chain_particle, chain_depth = [], [], []
    for depth in range(max_depth):
        rows = np.flatnonzero(active)
        if not len(rows):
            break
        particle = cur[rows]
        chain_hit.append(rows)
        chain_particle.append(particle)
        chain_depth.append(np.full(len(rows), depth, np.int64))
        isph = a["pdg"][particle] == 22
        use = isph & (nearest[rows] < 0)
        nearest[rows[use]] = particle[use]
        use = isph & meta["new"][particle] & (nearest_new[rows] < 0)
        nearest_new[rows[use]] = particle[use]
        use = isph & meta["new"][particle]
        earliest_new[rows[use]] = particle[use]
        metadata_unknown[rows] |= ~meta["stage_known"][particle] | ~meta["known"][particle]
        foil_birth = meta["new"][particle] & np.char.startswith(meta["name"][particle], "TargetFoil_")
        foil_photon[rows] |= foil_birth & isph
        foil_particle[rows] |= foil_birth
        state, par = a["parent_state"][particle], parent[particle]
        isroot = state == 0
        unknown = ~isroot & ((state != 1) | (par < 0))
        old_parent = np.zeros(len(rows), bool)
        ok = (state == 1) & (par >= 0)
        old_parent[ok] = a["stage"][par[ok]] < a["stage"][particle[ok]]
        use = meta["new"][particle] & (isroot | old_parent)
        entry[rows[use]] = particle[use]
        root[rows[isroot]] = particle[isroot]
        missing[rows[unknown]] = True
        active[rows[isroot | unknown]] = False
        cur[rows[~(isroot | unknown)]] = par[~(isroot | unknown)]
    ended = missing | active
    refs = dict(leaf=leaf, nearest_photon=nearest, nearest_new_photon=nearest_new,
                earliest_new_photon=earliest_new, stage_entry=entry, root=root)
    out = {**{k + "_particle_row": value for k, value in refs.items()}, "ancestry_unresolved": ended,
           "has_new_foil_photon_ancestor": foil_photon, "has_new_foil_born_ancestor": foil_particle,
           "ancestry_metadata_unknown": metadata_unknown, "foil_ancestry_complete": ~(ended | metadata_unknown)}
    for label, index in refs.items():
        out[label + "_present"] = index >= 0
        for name in ("pdg", "creation", "stage", "energy", "x", "y", "z", "time"):
            out[label + "_" + name] = _take(a[name], index, -1 if name in ("pdg", "creation", "stage") else np.nan)
        out[label + "_birth_volume"] = _take(meta["name"], index, "<none>")
        out[label + "_birth_material"] = _take(meta["material"], index, "<none>")
        out[label + "_birth_known"] = _take(meta["known"], index, False)
    out["leaf_is_stage_entry"] = (leaf == entry) & (leaf >= 0)
    for name, parts in (("trace_hit_row", chain_hit), ("trace_particle_row", chain_particle), ("trace_depth", chain_depth)):
        out[name] = np.concatenate(parts) if parts else np.array([], np.int64)
    return out


def group_energy(energy, **columns):
    """Columnar category census, including energy sums and hit counts."""
    labels = np.rec.fromarrays(list(columns.values()), names=list(columns))
    unique, index, count = np.unique(labels, return_inverse=True, return_counts=True)
    return {**{k: unique[k].tolist() for k in columns}, "tracker_steps": count.tolist(),
            "ionizing_mev": np.bincount(index, weights=energy, minlength=len(unique)).tolist()}


def summarize(tables, response, ancestry):
    e, s = tables["events"], tables["steps"]
    ek = keys(e, EVENT)
    join(ek, ek)
    se = join(keys(s, EVENT), ek)
    if np.any(se < 0):
        raise ValueError("tracker event absent from all-event ledger")
    status = response["tracker_ownership_status"]
    energy = response["tracker_ionizing_mev"]
    if energy.shape != (len(se),) or status.shape != energy.shape:
        raise ValueError("tracker response rows disagree with selected raw steps")
    result = dict(scope="Nearest-ancestor birth census, not causal ownership or an inclusive response model",
                  input_events=len(ek), tracker_steps=len(energy),
                  ancestry_unresolved_steps=int(ancestry["ancestry_unresolved"].sum()))
    for name in ("leaf", "nearest_new_photon", "earliest_new_photon", "nearest_photon", "stage_entry", "root"):
        result[name] = group_energy(energy, status=status, pdg=ancestry[name + "_pdg"],
                                   creation=ancestry[name + "_creation"], stage=ancestry[name + "_stage"],
                                   birth_volume=ancestry[name + "_birth_volume"],
                                   birth_material=ancestry[name + "_birth_material"],
                                   present=ancestry[name + "_present"], birth_known=ancestry[name + "_birth_known"],
                                   ancestry_unresolved=ancestry["ancestry_unresolved"],
                                   ancestry_metadata_unknown=ancestry["ancestry_metadata_unknown"])
    result["direct_stage_entry"] = group_energy(energy, status=status, direct=ancestry["leaf_is_stage_entry"])
    result["foil_ancestry"] = group_energy(energy, status=status,
        foil_photon_observed=ancestry["has_new_foil_photon_ancestor"],
        foil_born_observed=ancestry["has_new_foil_born_ancestor"],
        ancestry_complete=ancestry["foil_ancestry_complete"])
    result["origin_note"] = "Earliest/nearest mean observed ancestors; absence is established only with ancestry_complete=true. No-foil-birth ancestry does not prove no target traversal."
    result["top_residual_events"] = {}
    for label, code in (("residual", 0), ("owned", 1), ("unknown", 2)):
        total = np.bincount(se[status == code], weights=energy[status == code], minlength=len(ek))
        if not np.allclose(total, response["event_" + label + "_ionizing_mev"], rtol=1e-13, atol=1e-15):
            raise ValueError("event energy disagrees with saved response")
        top = np.argsort(-total)[:10]
        top = top[total[top] > 0]
        result["top_" + label + "_events"] = {**{k: e[k][top].tolist() for k in (*EVENT, "ordinal")},
                                                "ionizing_mev": total[top].tolist(),
                                                "sum_mev": float(total.sum()),
                                                "nonzero_events": int(np.count_nonzero(total)),
                                                "largest_fraction": float(total.max(initial=0.) / total.sum()) if total.sum() else None}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    start = perf_counter()
    with np.load(args.response, allow_pickle=False) as archive:
        response = {k: archive[k] for k in archive.files}
    tables = read_tables(args.input, response)
    ancestry = trace(tables)
    report = summarize(tables, response, ancestry)
    report["elapsed_seconds"] = perf_counter() - start
    report["input"] = str(args.input)
    report["response"] = str(args.response)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    payload = {**ancestry, **{"particle_" + k: v for k, v in tables["particles"].items()},
               **{"tracker_" + k: v for k, v in tables["steps"].items()},
               "tracker_ownership_status": response["tracker_ownership_status"],
               "tracker_ionizing_mev": response["tracker_ionizing_mev"]}
    np.savez_compressed(args.out.with_suffix(".npz"), **payload)
    print(f"{report['tracker_steps']} tracker steps; {report['ancestry_unresolved_steps']} unresolved chains; "
          f"{report['elapsed_seconds']:.3f} s/file. {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
