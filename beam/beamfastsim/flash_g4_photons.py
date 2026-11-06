"""New-stage target-foil photon births and first recorded spatial exit (mm/ns/MeV).

Run: python -m beamfastsim.flash_g4_photons --input inclusive.root --out photons.npz
     --geom frozen.txt --offline-root /path/to/release --events N

This is neither an uncollided/primary-brems selection nor ancestry, response,
whole-target escape, or an export-completeness certificate. All identified foil
births are retained, including those with no recorded exit or no steps. The event
ledger, not the photon population, supplies the denominator. Creation/parent
identities are preserved without interpreting ancestry. Step ``index`` is the
exporter's collection index, NOT a G4 track step number.

Only recorded straight pre/post segments and explicitly verified consecutive
two-segment continuations are used. Exact boundary endpoints with
non-transport end processes are ambiguous, not accepted as outward exits. An
explicit boundary_tolerance_mm can flag near-boundary discrepancies; it NEVER
moves coordinates or expands the foil. A near-short first candidate can be
resolved only by a contiguous, same-energy, straight transport continuation
whose recorded endpoint is clearly outside the same face. This uses the combined
recorded chord, not an extrapolated ray. Otherwise an ambiguous earlier candidate
takes precedence over a later certain crossing.
The saved pre-time and distance/C_MM_NS are separate: the latter is a chord
flight diagnostic, NOT an exact G4 crossing time (post-time is not exported).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .cechain import C_MM_NS
from .flash_geometry import from_offline
from .flash_exit import foil_exit, Z_MINUS, Z_PLUS, OUTER_RIM, INNER_HOLE
from .flash_inclusive_check import EVENT, PRODUCT, SUBPRODUCT, keys, join, matched, grouped

# Offline MCDataProducts/inc/ProcessCode.hh; not a whitelist of uncollided tracks.
TRANSPORT_PROCESSES = (49, 182)  # Transportation, CoupledTransportation
# Numerical consistency gates, not physical scattering/energy-loss allowances.
CONTINUITY_DIRECTION_TOLERANCE = 1e-7  # norm of unit-vector difference
CONTINUITY_ENERGY_RTOL = 1e-10  # relative to first pre-step photon energy
FIELDS = {
    "events": (*EVENT, "ordinal", "ptr_failures"),
    "products": (*PRODUCT, "kind", "scope", "size", "label", "instance"),
    "particles": (*PRODUCT, "key", "pdg", "stage", "creation", "start_volume",
                  "parent_state", "parent_product", "parent_key", "energy", "x", "y", "z"),
    "steps": (*PRODUCT, "index", "particle_state", "particle_product", "particle_key",
              "x", "y", "z", "post_x", "post_y", "post_z", "px", "py", "pz",
              "post_px", "post_py", "post_pz", "time", "length", "end_process"),
    "volumes": (*SUBPRODUCT, "stage", "index", "name"),
}


def segment_exit(pre, post, momentum, post_momentum, end_process,
                 foil_z, foil_ht, foil_rin, foil_rout, axis_x, *, boundary_tolerance_mm=0.):
    """Vectorized first outward annulus intersection of recorded segments.

    Arrays have (...,3) shape; geometry broadcasts. ``candidate`` includes strict
    exits and explicit near-boundary/endpoint-process ambiguities. ``exit`` is
    false for ambiguous candidates. Starts genuinely outside are not propagated
    through the foil: this helper is for segments beginning in the birth foil.
    At non-transport endpoints, even outward post-momentum does not establish
    process ordering relative to the interface. No uncollided flag is inferred.
    ``alignment_residual`` measures departure of the chord from pre-momentum;
    it is reported, not silently repaired. Invalid/zero segments are flagged.
    """
    tol = np.asarray(boundary_tolerance_mm, float)
    if tol.ndim or not np.isfinite(tol) or tol < 0:
        raise ValueError("boundary_tolerance_mm must be finite, scalar and nonnegative")
    pre, post, momentum, post_momentum = [np.asarray(x, float) for x in
                                         (pre, post, momentum, post_momentum)]
    if pre.ndim < 1 or pre.shape[-1] != 3 or any(x.shape != pre.shape for x in (post, momentum, post_momentum)):
        raise ValueError("pre/post positions and momenta must have identical (...,3) shapes")
    z, ht, rin, rout, ax = [np.broadcast_to(np.asarray(x, float), pre.shape[:-1])
                           for x in (foil_z, foil_ht, foil_rin, foil_rout, axis_x)]
    chord = post - pre
    length = np.linalg.norm(chord, axis=-1)
    energy = np.linalg.norm(momentum, axis=-1)
    post_energy = np.linalg.norm(post_momentum, axis=-1)
    finite = (np.all(np.isfinite(pre), axis=-1) & np.all(np.isfinite(post), axis=-1)
              & np.isfinite(energy) & (energy > 0) & np.isfinite(post_energy)
              & np.isfinite(length) & (length > 0))
    direction = np.divide(momentum, energy[..., None], out=np.full_like(momentum, np.nan),
                          where=energy[..., None] > 0)
    chord_direction = np.divide(chord, length[..., None], out=np.full_like(chord, np.nan),
                                where=length[..., None] > 0)
    ray = foil_exit(pre, chord, z, ht, rin, rout, ax)
    radius = np.hypot(pre[..., 0] - ax, pre[..., 1])
    inside = (np.abs(pre[..., 2] - z) <= ht) & (radius >= rin) & (radius <= rout)
    near_outside = (~inside & (np.abs(pre[..., 2] - z) <= ht + tol)
                    & (radius >= rin - tol) & (radius <= rout + tol) & (tol > 0))
    distance = ray["distance"]
    reaches = finite & ray["valid"] & (distance <= length)
    endpoint = reaches & (np.abs(distance - length) <= tol)
    # Post-momentum must really continue outward at a transport-only endpoint.
    radial_dot = ((ray["position"][..., 0] - ax) * post_momentum[..., 0]
                  + ray["position"][..., 1] * post_momentum[..., 1])
    outward = (((ray["face"] == Z_MINUS) & (post_momentum[..., 2] < 0))
               | ((ray["face"] == Z_PLUS) & (post_momentum[..., 2] > 0))
               | ((ray["face"] == OUTER_RIM) & (radial_dot >= 0)
                  & (np.hypot(post_momentum[..., 0], post_momentum[..., 1]) > 0))
               | ((ray["face"] == INNER_HOLE) & (radial_dot < 0)))
    endpoint_ambiguous = endpoint & (~np.isin(end_process, TRANSPORT_PROCESSES) | ~outward)
    near_short = finite & ray["valid"] & (distance > length) & (distance - length <= tol) & (tol > 0)
    boundary_ambiguous = finite & (near_outside | near_short)
    candidate = reaches | boundary_ambiguous
    ambiguous = endpoint_ambiguous | boundary_ambiguous
    return dict(candidate=candidate, exit=reaches & ~ambiguous, ambiguous=ambiguous,
                endpoint_ambiguous=endpoint_ambiguous, boundary_ambiguous=boundary_ambiguous,
                invalid=~finite, outside_start=finite & ~inside,
                distance_mm=np.where(reaches, distance, np.nan),
                position_mm=np.where(reaches[..., None], ray["position"], np.nan),
                direction=direction, pre_energy_mev=energy, face=np.where(reaches, ray["face"], 0),
                segmentflight_ns=np.where(reaches, distance / C_MM_NS, np.nan),
                chord_length_mm=length,
                alignment_residual=np.linalg.norm(chord_direction - direction, axis=-1))


def _xyz(table, prefix=""):
    return np.column_stack([table[prefix + c] for c in ("x", "y", "z")])


def extract(tables, geom, *, boundary_tolerance_mm=0.):
    """Return NPZ-ready column arrays from exporter tables and CEGeometry.

    Missing required branches (especially post fields) fail explicitly; no ray
    is invented from a pre-step momentum/length. Joins use full unsigned event,
    product and particle keys; ambiguous join targets raise. Unresolved identities
    are counted in report_json, not guessed. ``exit_candidate`` disambiguates the
    uint64-max first_stepindex sentinel. ``exit`` means first observed spatial
    exit only; consult per-photon ambiguity and diagnostics before using it.
    ``two_segment_reconstruction`` labels a combined recorded chord, anchored at
    the unchanged ``first_stepindex`` and pre-time. Its ``chord_length_mm`` and
    ``segmentflight_ns`` refer to that combined chord. Raw first-segment ambiguity
    flags and recorded segment counts are retained. The declared boundary
    tolerance also bounds the pre/post continuity gap; direction and relative
    energy consistency gates are explicitly recorded in report_json.
    """
    for name, fields in FIELDS.items():
        missing = set(fields) - tables[name].keys()
        if missing:
            raise ValueError(f"{name}: missing required fields: {sorted(missing)}")
    e, p, a, s, v = (tables[name] for name in FIELDS)
    b = geom.beam
    nfoil = len(b.foil_z)
    if not nfoil:
        raise ValueError("geometry has no foils")
    ek = keys(e, EVENT)
    join(ek, ek)
    ep = np.flatnonzero(p["scope"] == 0)
    pk = keys(p, PRODUCT)[ep]
    pi = join(keys(a, PRODUCT), pk)
    si = join(keys(s, PRODUCT), pk)
    pi[pi >= 0] = ep[pi[pi >= 0]]
    si[si >= 0] = ep[si[si >= 0]]
    pe, ae = join(keys(p, EVENT), ek), join(keys(a, EVENT), ek)
    event_product = (p["scope"] == 0) & (pe >= 0)
    particle_role = event_product & (p["kind"] == 1) & (p["label"] == "g4run") & (p["instance"] == "")
    step_role = event_product & (p["kind"] == 3) & (p["label"] == "g4run") & (p["instance"] == "photonSteps")
    source_role = event_product & (p["kind"] == 3) & (p["label"] == "BeamFilter") & (p["instance"] == "Beam")
    for role in (particle_role, step_role, source_role):
        join(keys(p, EVENT)[role], keys(p, EVENT)[role])
    volume_role = (p["scope"] == 1) & (p["kind"] == 5) & (p["label"] == "g4run") & (p["instance"] == "")
    vi = join(keys(a, ("run", "subrun")), keys(p, ("run", "subrun"))[volume_role])
    known = vi >= 0
    known[known] &= p["size"][volume_role][vi[known]] > 0
    stage = np.full(len(a["key"]), -1, np.int64)
    volume_product = np.zeros(len(stage), np.uint64)
    stage[known] = p["size"][volume_role][vi[known]].astype(np.int64) - 1
    volume_product[known] = p["product"][volume_role][vi[known]]
    photons = matched(pi, particle_role) & (a["pdg"] == 22)
    new = photons & known & (a["stage"] == stage)
    query = {**a, "product": volume_product, "index": a["start_volume"]}
    birth = join(keys(query, (*SUBPRODUCT, "stage", "index")), keys(v, (*SUBPRODUCT, "stage", "index")))
    birth[~known] = -1
    # Foil names, never copy numbers or closest-z guesses, determine emitting foil.
    names = np.char.mod("TargetFoil_%02d", np.arange(nfoil))
    order = np.argsort(names)
    where = np.searchsorted(names[order], np.asarray(v["name"], str))
    safe = np.minimum(where, nfoil - 1)
    vf = np.where((where < nfoil) & (names[order][safe] == v["name"]), order[safe], -1)
    foil = np.full(len(stage), -1, np.int64)
    foil[birth >= 0] = vf[birth[birth >= 0]]
    selected = new & (foil >= 0)
    rows = np.flatnonzero(selected)
    n = len(rows)
    particle_keys = keys(a, (*PRODUCT, "key"))
    join(particle_keys, particle_keys)
    step_particle = join(keys(s, (*EVENT, "particle_product", "particle_key")), particle_keys)
    photon_steps = matched(si, step_role)
    usable = photon_steps & (s["particle_state"] == 1) & matched(step_particle, selected)
    sr = np.flatnonzero(usable)
    join(keys(s, (*PRODUCT, "index"))[sr], keys(s, (*PRODUCT, "index"))[sr])
    owner = np.searchsorted(rows, step_particle[sr])
    sf = foil[step_particle[sr]]
    pre, post, momentum, post_momentum = (_xyz(s, prefix)[sr] for prefix in ("", "post_", "p", "post_p"))
    segments = segment_exit(pre, post, momentum, post_momentum, s["end_process"][sr], b.foil_z[sf],
                            b.foil_ht[sf], b.foil_rin[sf], b.foil_rout[sf], b.axis_x,
                            boundary_tolerance_mm=boundary_tolerance_mm)
    order = np.lexsort((s["index"][sr], owner))
    ordered = order[segments["candidate"][order]]
    first = ordered[np.r_[True, np.diff(owner[ordered]) != 0]] if len(ordered) else ordered
    owners = owner[first]
    out = {field: np.asarray(a[field])[rows] for field in
           (*PRODUCT, "key", "stage", "creation", "parent_state", "parent_product", "parent_key")}
    out.update(birth_energy_mev=a["energy"][rows], birth_position_mm=_xyz(a)[rows], foil_index=foil[rows],
               start_volume=a["start_volume"][rows], birth_volume_product=volume_product[rows],
               recorded_steps=np.bincount(owner, minlength=n), exit_candidate=np.zeros(n, bool),
               first_stepindex=np.full(n, np.iinfo(np.uint64).max, np.uint64),
               first_step_product=np.zeros(n, np.uint64), first_pre_time_ns=np.full(n, np.nan),
               first_end_process=np.full(n, -1, np.int64), first_recorded_length_mm=np.full(n, np.nan))
    out["exit_candidate"][owners] = True
    for name, source in (("first_stepindex", "index"), ("first_step_product", "product"),
                         ("first_pre_time_ns", "time"), ("first_end_process", "end_process"),
                         ("first_recorded_length_mm", "length")):
        out[name][owners] = s[source][sr[first]]
    for name in ("exit", "ambiguous", "endpoint_ambiguous", "boundary_ambiguous", "distance_mm",
                 "position_mm", "direction", "pre_energy_mev", "face", "segmentflight_ns",
                 "alignment_residual", "chord_length_mm"):
        values = segments[name]
        fill = False if values.dtype.kind == "b" else 0 if name == "face" else np.nan
        out[name] = np.full((n, *values.shape[1:]), fill, dtype=values.dtype)
        out[name][owners] = values[first]
    # Keep the saved-segment evidence even when the spatial crossing is resolved.
    out["first_segment_boundary_ambiguous"] = out["boundary_ambiguous"].copy()
    out["first_segment_endpoint_ambiguous"] = out["endpoint_ambiguous"].copy()
    out["two_segment_reconstruction"] = np.zeros(n, bool)
    out["continuation_stepindex"] = np.full(n, np.iinfo(np.uint64).max, np.uint64)
    out["continuation_gap_mm"] = np.full(n, np.nan)
    # Shift collection order within each photon without skipping recorded segments.
    same = owner[order[:-1]] == owner[order[1:]]
    following = np.full(len(sr), -1, np.int64)
    following[order[:-1][same]] = order[1:][same]
    previous = first[(segments["boundary_ambiguous"][first]
                      & ~segments["outside_start"][first]
                      & np.isin(s["end_process"][sr[first]], TRANSPORT_PROCESSES)
                      & (following[first] >= 0))]
    next_step = following[previous]
    f = sf[previous]
    combined = segment_exit(pre[previous], post[next_step], momentum[previous], post_momentum[next_step],
                            s["end_process"][sr[next_step]], b.foil_z[f], b.foil_ht[f],
                            b.foil_rin[f], b.foil_rout[f], b.axis_x,
                            boundary_tolerance_mm=boundary_tolerance_mm)
    gap = np.linalg.norm(post[previous] - pre[next_step], axis=1)
    chord = post[next_step] - pre[previous]
    direction = np.divide(chord, combined["chord_length_mm"][:, None],
                          out=np.full_like(chord, np.nan),
                          where=combined["chord_length_mm"][:, None] > 0)
    consistent = ((gap <= boundary_tolerance_mm) & ~segments["invalid"][next_step]
                  & combined["exit"]
                  & (np.abs(combined["distance_mm"] - segments["chord_length_mm"][previous])
                     <= boundary_tolerance_mm))
    # Require both chords and all saved momenta to agree, including the outer endpoint.
    for vector in (post[previous] - pre[previous], post[next_step] - pre[next_step],
                   momentum[previous], post_momentum[previous], momentum[next_step], post_momentum[next_step]):
        norm = np.linalg.norm(vector, axis=1)
        unit = np.divide(vector, norm[:, None], out=np.full_like(vector, np.nan), where=norm[:, None] > 0)
        consistent &= np.linalg.norm(unit - direction, axis=1) <= CONTINUITY_DIRECTION_TOLERANCE
    energy = combined["pre_energy_mev"]
    for vector in (post_momentum[previous], momentum[next_step], post_momentum[next_step]):
        consistent &= np.abs(np.linalg.norm(vector, axis=1) - energy) <= CONTINUITY_ENERGY_RTOL * energy
    radius = np.hypot(post[next_step, 0] - b.axis_x, post[next_step, 1])
    dz = post[next_step, 2] - b.foil_z[f]
    tol = boundary_tolerance_mm
    # Require the continuation endpoint to remain outside the crossed face.
    outside = (((combined["face"] == Z_MINUS) & (dz < -b.foil_ht[f] - tol))
               | ((combined["face"] == Z_PLUS) & (dz > b.foil_ht[f] + tol))
               | ((combined["face"] == OUTER_RIM) & (radius > b.foil_rout[f] + tol))
               | ((combined["face"] == INNER_HOLE) & (radius < b.foil_rin[f] - tol)))
    resolved = consistent & outside
    resolved_owners = owner[previous[resolved]]
    for name, values in combined.items():
        if name in out:  # saved first-step provenance and original segment counts stay unchanged
            out[name][resolved_owners] = values[resolved]
    out["two_segment_reconstruction"][resolved_owners] = True
    out["continuation_stepindex"][resolved_owners] = s["index"][sr[next_step[resolved]]]
    out["continuation_gap_mm"][resolved_owners] = gap[resolved]
    bp = out["birth_position_mm"]
    bf = out["foil_index"]
    br = np.hypot(bp[:, 0] - b.axis_x, bp[:, 1])
    out["birth_inside_geometry"] = (np.all(np.isfinite(bp), axis=1)
                                    & (np.abs(bp[:, 2] - b.foil_z[bf]) <= b.foil_ht[bf])
                                    & (br >= b.foil_rin[bf]) & (br <= b.foil_rout[bf]))
    for flag in ("invalid", "outside_start", "ambiguous"):
        out["recorded_" + flag + "_segments"] = np.bincount(owner[segments[flag]], minlength=n)
    for field in (*EVENT, "ordinal", "ptr_failures"):
        out["event_" + field] = e[field]
    out["event_births"] = np.bincount(ae[rows], minlength=len(ek))
    out["event_exits"] = np.bincount(ae[rows][out["exit"]], minlength=len(ek))
    out["input_event_count"] = np.asarray(len(ek))
    for label, role, index in (("photon_steps", step_role, si), ("particles", particle_role, pi),
                               ("input_states", source_role, si)):
        count = np.zeros(len(ek), np.uint64)
        present = np.zeros(len(ek), bool)
        count[pe[role]] = p["size"][role]
        present[pe[role]] = True
        out["event_" + label + "_declared"] = count
        out["event_" + label + "_present"] = present
        observed = np.bincount(index[matched(index, role)], minlength=len(p["product"]))
        mismatch = np.zeros(len(ek), bool)
        mismatch[pe[role]] = observed[role] != p["size"][role]
        out["event_" + label + "_size_mismatch"] = mismatch
    out["input_state_count_declared"] = np.asarray(out["event_input_states_declared"].sum())
    report = dict(scope="Birth-foil spatial exits only; not uncollided, response, whole-target escape or certification",
                  timing="pre-time and chord distance/c diagnostic; no exact crossing time",
                  boundary_tolerance_mm=float(boundary_tolerance_mm),
                  continuity_direction_tolerance=CONTINUITY_DIRECTION_TOLERANCE,
                  continuity_energy_rtol=CONTINUITY_ENERGY_RTOL,
                  two_segment_reconstructions=int(np.count_nonzero(out["two_segment_reconstruction"])),
                  reconstruction="Combined recorded chord; first-step provenance retained; no exact G4 crossing time",
                  copied_photons=int(np.count_nonzero(photons & known & (a["stage"] < stage))),
                  photons_without_stage=int(np.count_nonzero(photons & ~known)),
                  photons_after_destination_stage=int(np.count_nonzero(photons & known & (a["stage"] > stage))),
                  particles_without_event=int(np.count_nonzero(ae < 0)),
                  steps_without_event=int(np.count_nonzero(join(keys(s, EVENT), ek) < 0)),
                  new_photons_without_volume=int(np.count_nonzero(new & (birth < 0))),
                  new_photons_not_mapped_to_foil=int(np.count_nonzero(new & (foil < 0))),
                  particles_without_product=int(np.count_nonzero(pi < 0)),
                  steps_without_product=int(np.count_nonzero(si < 0)),
                  unresolved_photon_step_pointers=int(np.count_nonzero(photon_steps & ((s["particle_state"] != 1) | (step_particle < 0)))),
                  birth_geometry_disagreements=int(np.count_nonzero(~out["birth_inside_geometry"])),
                  ambiguous_first_candidates=int(np.count_nonzero(out["ambiguous"])))
    out["report_json"] = np.asarray(json.dumps(report, sort_keys=True))
    return out


def read_tables(path):
    import awkward as ak
    import uproot

    with uproot.open(path) as root:
        result = {}
        for name, fields in FIELDS.items():
            tree = root["export/" + name]
            missing = set(fields) - set(tree.keys())
            if missing:
                raise ValueError(f"{name}: missing required fields: {sorted(missing)}")
            arrays = tree.arrays(list(fields), library="ak")
            result[name] = {field: ak.to_numpy(arrays[field]) for field in fields}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--geom", required=True)
    parser.add_argument("--offline-root", required=True)
    parser.add_argument("--events", required=True, type=int, help="Expected all-input count, not a read limit")
    parser.add_argument("--boundary-tolerance-mm", type=float, default=0.)
    args = parser.parse_args(argv)
    if args.events < 0:
        parser.error("--events must be nonnegative")
    start = perf_counter()
    result = extract(read_tables(args.input), from_offline(args.geom, args.offline_root),
                     boundary_tolerance_mm=args.boundary_tolerance_mm)
    result.update(expected_input_event_count=np.asarray(args.events), input_file=np.asarray(str(args.input)),
                  geometry_file=np.asarray(args.geom), offline_root=np.asarray(args.offline_root))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as stream:
        np.savez_compressed(stream, **result)
    summary = json.loads(str(result["report_json"]))
    summary.update(input_events=int(result["input_event_count"]), foil_births=len(result["key"]),
                   spatial_exits=int(result["exit"].sum()),
                   photons_without_recorded_steps=int(np.count_nonzero(result["recorded_steps"] == 0)),
                   boundary_tolerance_mm=args.boundary_tolerance_mm,
                   boundary_ambiguous=int(result["boundary_ambiguous"].sum()),
                   endpoint_ambiguous=int(result["endpoint_ambiguous"].sum()),
                   max_alignment_residual=float(np.nanmax(result["alignment_residual"], initial=0.)),
                   birth_energy_mev=float(result["birth_energy_mev"].astype(np.float64).sum()),
                   exit_energy_mev=float(result["pre_energy_mev"][result["exit"]].sum()),
                   elapsed_seconds=perf_counter() - start)
    ambiguous = result["ambiguous"]
    summary["ambiguous_process_counts"] = grouped(end_process=result["first_end_process"][ambiguous])
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{result['input_event_count']}/{args.events} input events; {len(result['key'])} foil births; "
          f"{result['exit'].sum()} spatial exits; {result['ambiguous'].sum()} ambiguous; "
          f"{perf_counter() - start:.3f} s/file. Not response/certification. {args.out}")
    return int(result["input_event_count"] != args.events)


if __name__ == "__main__":
    raise SystemExit(main())
