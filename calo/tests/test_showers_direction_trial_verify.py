"""Opt-in SAVED-ONLY direction-trial verification: no production imports, ROOT,
RNG creation/replay, sampling, placement, reconstruction or report invocation.
Numeric payload/partition oracles retained from the independent prior verifier;
new pool and correlation oracles use explicit loops, corrcoef and literal deletion.
Receipts are externally pinned and checked again after the entire module.
"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pytest

ARMS = ("geometric", "crystal")
FIELDS = ("energy", "time", "x", "y", "z")
LOSSES = ("raw_inner", "raw_outer", "raw_axial", "interior_unmapped",
          "center_or_proposed_envelope", "center_too_far", "containment", "postmap")
ANCHORS = {
 "summary.json": "4bf2eb82fa744fef6aac2d22932cabe2e3820de8039525a0de2758fc5cf5918e",
 "execution.json": "34939428b0d373aa054b346b7b761d789bae04a5e8076a73fb20d4879df3e27b",
 "clusters.npz": "471027260b7c5619b51e8ed0f2576bf8083fe2f291ecbeb776d7f79eb8294ed5",
 "conditional.json": "53a4cfda7bb7f2e7b32c64e411c6dddbb77875e0b81003b445a6416ac11a6edf",
 "numbers.json": "8d317aecda07c0cdb074911132fdbac83eb6d89a2c4dd44ba8c3d3b4133f3339",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def archive(path):
    with np.load(path, allow_pickle=False) as source:
        return {k: source[k] for k in source.files}

def exact(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype
    np.testing.assert_array_equal(a, b)

def close(a, b):
    np.testing.assert_allclose(a, b, rtol=2e-12, atol=1e-10, equal_nan=True)

def scalar(a, b):
    if b is None:
        assert a is None
    else:
        assert a == pytest.approx(b, rel=2e-10, abs=1e-11)

def jackknife(fast, stock, rows):
    """Independent deletion oracle: actually remove each original-row block.

    Unlike production's bincount/sufficient-statistic implementation, this
    directly averages each surviving subset; unequal block sizes are retained.
    """
    f, s = np.asarray(fast, float), np.asarray(stock, float)
    group = np.asarray(rows) // 100
    if not len(f):
        return dict(fast=None, stock=None, difference=None, error=None)
    blocks = np.unique(group)
    delta = f - s
    leave = np.array([delta[group != block].mean() for block in blocks]) if len(blocks) > 1 else []
    error = np.sqrt((len(blocks) - 1) * np.mean((leave - np.mean(leave)) ** 2)) if len(leave) else None
    return dict(fast=f.mean(), stock=s.mean(), difference=delta.mean(), error=error)

def check_mean(saved, f, s, rows):
    for key, value in jackknife(f, s, rows).items():
        scalar(saved[key], value)

def check_ratio(saved, f, s, rows):
    f, s, group = np.asarray(f, float), np.asarray(s, float), np.asarray(rows) // 100
    blocks = np.unique(group)
    ratio, error = None, None
    if s.sum():
        ratio = f.sum() / s.sum()
        if len(blocks) > 1 and all(s[group != b].sum() for b in blocks):
            leave = np.array([f[group != b].sum() / s[group != b].sum() for b in blocks])
            error = np.sqrt((len(blocks) - 1) * np.mean((leave - leave.mean()) ** 2))
    scalar(saved["ratio"], ratio)
    scalar(saved["error"], error)

def masks(entries, draw):
    n = len(entries["energy"])
    result = {"all": np.ones(n, bool)}
    for key, edges in (("radius", (0, 450, 550, 1000, np.inf)),
                       ("energy", (0, 50, 80, 95, 110, np.inf)),
                       ("cosine", (0, .4, .7, 1.000001))):
        for lo, hi in zip(edges[:-1], edges[1:]):
            result[f"{key}_{lo}_{hi}"] = (entries[key] >= lo) & (entries[key] < hi)
    for disk in (0, 1):
        result[f"disk{disk}"] = entries["disk"] == disk
        high = result[f"disk{disk}"] & (entries["radius"] < 450) & (entries["energy"] >= 95)
        result[f"inner_highE_disk{disk}"] = high
        result[f"inner_highE_disk{disk}_coslow"] = high & (entries["cosine"] < .7)
        result[f"inner_highE_disk{disk}_coshigh"] = high & (entries["cosine"] >= .7)
    result.update(fallback=draw["fallback"], exact_no_fallback=~draw["fallback"], unsupported=~draw["supported"])
    return result

def fold(times, row, timing):
    """Literal source-order float32 gate arithmetic, not the production helper."""
    pb, mb, marker = [timing[k][row].astype(np.float32) for k in ("pbtime", "mbtime", "marker")]
    start = ((np.float32(450) - pb) - marker) - np.float32(10)
    end = (np.float32(1795) - pb) - marker
    period = np.where(timing["onspill"][row], mb.astype(float), timing["event_length"][row])
    hit = np.fmod(np.asarray(times, float) + pb.astype(float), period) - pb.astype(float)
    advance = hit < np.maximum(mb.astype(float), end.astype(float)) - period
    hit[advance] += period[advance]
    return hit, hit >= start, hit - start, end

def sums(steps, keep=None, n=2396):
    weights = np.asarray(steps["energy_birks"], float)
    if keep is not None:
        weights = weights * keep
    # add.at is deliberately distinct from production's bincount reduction.
    result = np.zeros(n)
    np.add.at(result, steps["entry"], weights)
    return result

def test_saved_whole_payload_membership_zero_outcomes_and_both_disks(saved):
    s = saved
    lib, draw, pools, entries, diag = [s[k] for k in ("library", "draw", "pools", "entries", "diagnostics")]
    donor, eligible = draw["donor"], pools["eligible"]
    assert eligible.shape == (2396, 8) and np.all(draw["supported"])
    assert not np.any(pools["outside"]) and np.sum(draw["fallback"]) == 1
    assert np.all((donor >= 0) & (donor < len(lib["entry_energy"])))
    assert np.all(np.any(eligible == donor[:, None], axis=1))
    for name, key in (("supported", "supported"), ("fallback", "fallback"),
                      ("radius_pool_size", "size"), ("bin_pool_size", "full_size")):
        exact(draw[name], pools[key])
    exact(pools["event_row"], entries["event_row"])
    exact(lib["entry_disk"][donor], entries["disk"])
    lengths = np.diff(lib["offsets"])[donor]
    offsets = np.r_[0, np.cumsum(lengths)]
    exact(draw["offsets"], offsets)
    index = np.repeat(np.arange(2396), lengths)
    exact(draw["entry"], index)
    source = lib["offsets"][donor[index]] + np.arange(len(index)) - offsets[index]
    for key, value in lib.items():
        if key.startswith("step_") and key not in ("step_relative_position", "step_relative_time"):
            exact(draw[key[5:]], value[source])
    exact(draw["time"], entries["time"][index] + lib["step_relative_time"][source])
    ds = np.searchsorted(lib["geometry_disk_disk"], entries["disk"])
    origin = np.column_stack([lib["geometry_disk_origin_" + a] for a in "xy"])[ds]
    target, initial = entries["position"][:, :2] - origin, lib["entry_position"][donor, :2] - origin
    angle = np.arctan2(target[:, 1], target[:, 0]) - np.arctan2(initial[:, 1], initial[:, 0])
    # Stored relative positions are in each donor's incident frame, not global.
    z = lib["entry_direction"][donor[index]].copy()
    z /= np.linalg.norm(z, axis=1)[:, None]
    reference = np.zeros_like(z)
    reference[:, 2] = 1
    reference[np.abs(z[:, 2]) > .9] = (1, 0, 0)
    x = np.cross(reference, z)
    x /= np.linalg.norm(x, axis=1)[:, None]
    y = np.cross(z, x)
    relative = lib["step_relative_position"][source]
    rel = x * relative[:, :1] + y * relative[:, 1:2] + z * relative[:, 2:3]
    rotated = rel.copy()
    c, sn = np.cos(angle[index]), np.sin(angle[index])
    rotated[:, 0], rotated[:, 1] = c * rel[:, 0] - sn * rel[:, 1], sn * rel[:, 0] + c * rel[:, 1]
    close(draw["position"], entries["position"][index] + rotated)
    assert np.any(draw["disk"] != entries["disk"][index]), "cross-disk steps must not be removed"
    totals = sums(dict(entry=np.repeat(np.arange(len(lib["entry_energy"])), np.diff(lib["offsets"])),
                       energy_birks=lib["step_energy_birks"]), n=len(lib["entry_energy"]))
    outcomes = dict(energy=totals, empty=np.diff(lib["offsets"]) == 0, weak=totals <= 10)
    assert len(totals) == 24120
    assert outcomes["empty"].sum() == 259 and outcomes["weak"].sum() == 835
    assert np.any(lengths == 0)
    close(sums(draw), totals[donor])
    for name, value in outcomes.items():
        close(pools["outcome_" + name], value)
        values = value[np.maximum(eligible, 0)].astype(float)
        valid = eligible >= 0
        mean = np.sum(values * valid, axis=1) / valid.sum(axis=1)
        variance = np.sum((values - mean[:, None]) ** 2 * valid, axis=1) / valid.sum(axis=1)
        for key, expected in (("expected_", mean), ("variance_", variance), ("actual_", value[donor])):
            close(diag["pool_" + key + name], expected)
        close(pools["prob_" + name], mean)
    for key in ("event_row", "supported", "fallback", "donor"):
        exact(diag["pool_" + key], entries[key] if key == "event_row" else draw[key])
    for label, payload in (("draw", draw), *s["placed"].items()):
        h = hashlib.sha256()
        for key, value in sorted(payload.items()):
            a = np.ascontiguousarray(value)
            h.update(json.dumps([key, a.dtype.descr, a.shape], sort_keys=True).encode())
            h.update(a.tobytes())
        expected = s["preparation"]["fingerprints"]
        assert h.hexdigest() == (expected["draw"] if label == "draw" else expected["placements"][label])
    for arm, steps in s["placed"].items():
        for name in draw:
            if name not in ("crystal", "position"):
                exact(steps[name], draw[name])
        assert steps["mapped"].dtype == bool
        exact(steps["mapped"], steps["crystal"] >= 0)
    for feature in ("radius", "energy", "cosine"):
        expected = lib["entry_" + feature][donor] - entries[feature]
        close(diag["css_" + feature + "_delta"], expected)
        close(diag["residual_" + feature + "_delta"], expected)
    u = lib["entry_direction"][donor].copy()
    x, y = u[:, 0].copy(), u[:, 1].copy()
    u[:, 0], u[:, 1] = np.cos(angle) * x - np.sin(angle) * y, np.sin(angle) * x + np.cos(angle) * y
    degrees = np.rad2deg(np.arccos(np.clip(np.sum(u * entries["direction"], axis=1), -1, 1)))
    close(diag["css_angle_deg"], degrees)
    close(diag["residual_angle_deg"], degrees)

def test_saved_all_query_spatial_timing_energy_partitions_and_diagnostics(saved):
    s = saved
    d, entries, lib, draw = [s[k] for k in ("diagnostics", "entries", "library", "draw")]
    rows = entries["event_row"]
    timing = {k[7:]: v for k, v in s["candidate"].items() if k.startswith("timing_")}
    hit, accepted, margin, _ = fold(entries["time"], rows, timing)
    for name, value in (("entry_folded_time", hit), ("entry_time_accepted", accepted), ("entry_gate_margin", margin)):
        close(d["residual_" + name], value)
    exact(d["css_event_row"], rows)
    exact(d["residual_event_row"], rows)
    reuse = np.bincount(draw["donor"], minlength=len(lib["entry_energy"]))
    exact(d["residual_donor_reuse"], reuse[draw["donor"]])
    collections = dict(original=s["original"], donor=draw, **s["placed"])
    for name, steps in collections.items():
        index = steps["entry"]
        mapped = steps["mapped"] if name in ARMS else np.ones(len(index), bool)
        hit, accepted, _, end = fold(steps["time"], rows[index], timing)
        total, kept, lost = sums(steps), sums(steps, mapped), sums(steps, ~mapped)
        timed, rejected = sums(steps, mapped & accepted), sums(steps, mapped & ~accepted)
        if name in ARMS:
            close(total, d["css_donor"])
            for key, expected in (("mapped", kept), ("lost", lost)):
                close(d[f"css_{name}_{key}"], expected)
                close(d[f"residual_{name}_{key}"], expected)
            close(d[f"residual_{name}_time_rejected"], rejected)
            close(total - lost - rejected - timed, np.zeros(2396))
            close(d["residual_donor"] - d[f"residual_{name}_lost"] -
                  d[f"residual_{name}_time_rejected"] - d[f"residual_{name}_time_accepted"], np.zeros(2396))
            # Independent radial/axial/guard precedence using saved positions.
            reason = np.full(len(index), 3 if name == "geometric" else 7, int)
            if name == "crystal":
                guard = steps["crystal_loss_reason"]
                assert np.all(np.isin(guard, [0, 1, 2, 3]))
                for code, value in ((1, 4), (2, 5), (3, 6)):
                    reason[guard == code] = value
            for disk in (0, 1):
                take = draw["disk"] == disk
                ds = np.flatnonzero(lib["geometry_disk_disk"] == disk)[0]
                pool = lib["geometry_crystal_disk"] == disk
                origin = np.array([lib["geometry_disk_origin_" + a][ds] for a in "xyz"])
                rotation = np.array([[lib[f"geometry_disk_axis_{a}_{b}"][ds] for a in "xyz"] for b in "xyz"])
                local = (draw["position"][take] - origin) @ rotation
                centers = np.column_stack([lib["geometry_crystal_position_" + a][pool] for a in "xyz"])
                z = ((centers - origin) @ rotation)[:, 2]
                radius = np.linalg.norm(local[:, :2], axis=1)
                codes = reason[take]
                axial = (local[:, 2] < z.min() - 1e-6) | (local[:, 2] > np.max(z + lib["geometry_crystal_size_z"][pool]) + 1e-6)
                codes[axial] = 2
                codes[radius > lib["geometry_disk_size_y"][ds]] = 1
                codes[radius < lib["geometry_disk_size_x"][ds]] = 0
                reason[take] = codes
            reason[mapped] = -1
            components = []
            for code, label in enumerate(LOSSES):
                expected = sums(steps, reason == code)
                close(d[f"css_{name}_loss_{label}"], expected)
                components.append(expected)
            close(np.sum(components, axis=0), lost)
        else:
            close(d["css_" + name], total)
            close(d["residual_" + name], total)
            exact(d[f"residual_{name}_steps"], np.bincount(index, minlength=2396))
        close(d[f"residual_{name}_time_accepted"], timed)
        delay = steps["time"] - entries["time"][index]
        for key, condition in (("end_exposed", hit >= end), ("delay_ge20", delay >= 20), ("delay_negative", delay < 0)):
            close(d[f"residual_{name}_{key}"], sums(steps, mapped & accepted & condition))
        # Occupancy counts include both disks and only positive Birks energy.
        occupancy_keep = (steps["energy_birks"] > 0) & (mapped & accepted if name in ARMS else True)
        pair, inverse = np.unique(np.column_stack((index[occupancy_keep], steps["crystal"][occupancy_keep])), axis=0, return_inverse=True)
        energy = np.zeros(len(pair))
        np.add.at(energy, inverse, steps["energy_birks"][occupancy_keep].astype(float))
        counts = np.bincount(pair[:, 0], minlength=2396)
        maximum, occupancy_energy = np.zeros(2396), np.zeros(2396)
        np.maximum.at(maximum, pair[:, 0], energy)
        np.add.at(occupancy_energy, pair[:, 0], energy)
        close(d[f"residual_{name}_crystals"], counts)
        close(d[f"residual_{name}_max_fraction"], np.divide(maximum, occupancy_energy, out=np.zeros(2396), where=occupancy_energy > 0))
        # All 2396 CSS centroids, including NaNs on zero-energy queries.
        position = np.zeros((2396, 3))
        np.add.at(position, index, (steps["energy_birks"].astype(float) * mapped)[:, None] * steps["position"])
        centroid = np.divide(position, kept[:, None], out=np.full_like(position, np.nan), where=kept[:, None] > 0)
        origin = np.array([lib["geometry_disk_origin_" + a][0] for a in "xy"])
        xy = entries["position"][:, :2] - origin
        unit = xy / np.linalg.norm(xy, axis=1)[:, None]
        delta = centroid[:, :2] - entries["position"][:, :2]
        for key, expected in (("energy", kept), ("radial", np.sum(delta * unit, axis=1)),
                              ("tangent", delta[:, 1] * unit[:, 0] - delta[:, 0] * unit[:, 1])):
            close(d[f"css_{name}_centroid_{key}"], expected)
    for arm in ("stockref", "control", *ARMS):
        exact(d[f"residual_{arm}_selected"], s["candidate"][arm + "_energy"][rows] > 10)

@pytest.fixture(scope="module")
def saved():
    root = os.environ.get("CALO_DIRECTION_TRIAL_VERIFY_BASE")
    if not root:
        pytest.skip("set CALO_DIRECTION_TRIAL_VERIFY_BASE for frozen saved-only verification")
    base = Path(root).resolve()
    if base.name == "production_direction_trial":
        base = base.parent
    trial = base / "production_direction_trial"
    source = Path(__file__).resolve().parents[1] / "calofastsim"
    pins = {}
    def read(path):
        return json.loads(Path(path).read_text())
    def pin(path, digest):
        path = Path(path)
        assert path not in pins or pins[path] == digest
        assert sha(path) == digest, str(path)
        pins[path] = digest
    def locate(path):
        path = Path(path)
        for marker, target in (("CAL2_20261105", base), ("calofastsim", source)):
            if marker in path.parts:
                return target.joinpath(*path.parts[path.parts.index(marker) + 1:])
        return path
    for name, digest in ANCHORS.items():
        pin(trial / name, digest)
    summary, execution, preparation = [read(trial / (k + ".json")) for k in ("summary", "execution", "preparation")]
    assert summary["completed"] and (summary["sampler_calls"], summary["reconstruction_calls"]) == (1, 2)
    assert (execution["draws"], execution["calls"]) == (1, 2)
    assert [x["arm"] for x in execution["runs"]] == list(ARMS)
    assert set(p.name for p in trial.iterdir()) == set(summary["output_sha256"]) | {"summary.json"}
    for receipt in (summary, execution):
        for name, digest in receipt["output_sha256"].items():
            pin(trial / name, digest)
    assert preparation["input_sha256"] == summary["input_sha256"]
    # ROOT files are opaque hash inputs only, never opened by any ROOT reader.
    for name, digest in summary["input_sha256"].items():
        pin(locate(name), digest)
    for name, digest in preparation["calo_source_sha256"].items():
        pin(source / name, digest)
    for subdir in ("production_candidate", "production_candidate_support", "production_direction_support", "production_candidate_trial"):
        receipt = read(base / subdir / "summary.json")
        for name, digest in receipt["output_sha256"].items():
            pin(base / subdir / name, digest)
    oldprep = read(base / "production_candidate_trial/preparation.json")
    assert preparation["pins"] == summary["pins"]
    assert preparation["seed"] == oldprep["seed"] == 20261105
    assert preparation["reco_seed"] == oldprep["reco_seed"] == 20261108
    for key in ("runtime", "timing", "draw_rng_before", "draw_rng_after"):
        assert preparation[key] == oldprep[key], key
    assert preparation["draw_rng_before"] != preparation["draw_rng_after"]
    assert preparation["runtime"]["backend"] == "cpu" and preparation["runtime"]["x64"]
    assert preparation["runtime"]["fit_config"]["iterations"] == 15
    assert preparation["policy"] == summary["policy"] == dict(neighbors=8, angle_degrees=10., radius_mm=25.,
        both_direction_signs=True, zero_sign="nonnegative", insufficient="original_K8", orientation="disk")
    for arm, run in zip(ARMS, execution["runs"]):
        assert read(trial / (arm + "_started.json")) == dict(arm=arm, seed=20261108, retry_authorized=False)
        assert read(trial / (arm + "_completed.json")) == run
    c = archive(trial / "clusters.npz")
    s = dict(base=base, trial=trial, summary=summary, preparation=preparation, candidate=c,
        entries={k[6:]: v for k,v in c.items() if k.startswith("entry_")},
        library=archive(base / "production_candidate/library.npz"),
        previous=archive(base / "production_candidate_trial/clusters.npz"),
        previous_diagnostics=archive(base / "production_candidate_trial/diagnostics.npz"),
        previous_draw=archive(base / "production_candidate_trial/draw.npz"),
        baseline_pools=archive(base / "production_candidate_support/candidate_pools.npz"),
        support=archive(base / "production_direction_support/arrays.npz"),
        placed={a: archive(trial / (a + "_placement.npz")) for a in ARMS},
        conditional=read(trial / "conditional.json"), numbers=read(trial / "numbers.json"))
    for name in ("draw", "pools", "original", "diagnostics", "conditional_arrays"):
        s[name] = archive(trial / (name + ".npz"))
    yield s
    for path, digest in pins.items():
        assert sha(path) == digest, "modified during verification: " + str(path)


def local(entry, lib):
    origin = np.array([lib["geometry_disk_origin_" + k][0] for k in "xy"])
    xy = entry["position"][:, :2] - origin
    unit = xy / np.linalg.norm(xy, axis=1)[:, None]
    u = entry["direction"]
    return np.column_stack(((u[:, :2] * unit).sum(axis=1),
                           u[:, 1] * unit[:, 0] - u[:, 0] * unit[:, 1], u[:, 2]))


def test_saved_identity_and_capped_pools(saved):
    s = saved
    c, old, d, p, baseline, support = [s[k] for k in ("candidate", "previous", "draw", "pools", "baseline_pools", "support")]
    rows = c["query_event_rows"]
    assert len(rows) == 2396 and np.all(np.diff(rows) > 0)
    assert len(c["full_ids"]) == len(np.unique(c["full_ids"])) == 2813
    for name in c:
        if not name.startswith(("geometric_", "crystal_", "sampling_")):
            exact(c[name], old[name])
    exact(c["event_row"], np.arange(2813))
    exact(support["full_ids"], c["full_ids"])
    exact(support["event_row"], rows)
    for name in ("event_row", "full_ids", "query_event_rows"):
        exact(s["diagnostics"][name], c[name])
    for name in ("donor", "supported", "fallback", "offsets", "radius_pool_size", "bin_pool_size"):
        exact(c["sampling_" + name], d[name])
    for arm in ARMS:
        for name, value in archive(s["trial"] / (arm + ".npz")).items():
            exact(c[arm + "_" + name], value)
        assert np.all(c[arm + "_energy"][~np.isin(np.arange(2813), rows)] == 0)
    # Prior trial did not save original.npz; compare its authenticated original
    # diagnostics, never reopen source ROOT to manufacture a reference.
    for name, value in s["previous_diagnostics"].items():
        if name.startswith(("css_original", "residual_original", "residual_entry_")):
            exact(value, s["diagnostics"][name])
    for name in ("query_cell", "chosen_cell", "library_cell", "occupied", "counts", "full_size", "exact_size", "supported", "fallback", "outside", "substituted"):
        exact(p[name], baseline[name])
    lib = s["library"]
    q = local(s["entries"], lib)
    donor = local({k[6:]:v for k,v in lib.items() if k.startswith("entry_")}, lib)
    close(q, support["query_direction"])
    close(donor, support["donor_direction"])
    for i in range(len(rows)):
        candidates = np.flatnonzero(p["library_cell"] == p["chosen_cell"][i])
        dr = np.abs(lib["entry_radius"][candidates] - s["entries"]["radius"][i])
        angles = np.degrees(np.arccos(np.clip(np.sum(donor[candidates] * q[i], axis=1), -1, 1)))
        good = (dr <= 25) & (angles <= 10) & np.all((donor[candidates, :2] >= 0) == (q[i, :2] >= 0), axis=1)
        assert p["direction_count"][i] == good.sum() == support["compatible_within25_count"][i]
        enabled = good.sum() >= 8
        assert p["direction_used"][i] == enabled
        expected = np.array(sorted(sorted(candidates[good], key=lambda j: (abs(lib["entry_radius"][j] - s["entries"]["radius"][i]), j))[:8])) if enabled else baseline["eligible"][i]
        exact(p["eligible"][i], expected)
    assert p["direction_used"].sum() == 2284 and p["direction_fallback"].sum() == 112
    exact(p["direction_fallback"], p["supported"] & ~p["direction_used"])
    for name in ("direction_count", "direction_used", "direction_fallback"):
        exact(p[name], d[name])
    fallback = p["direction_fallback"]
    exact(d["donor"][fallback], s["previous_draw"]["donor"][fallback])
    # Identical uniform schedule implies the same saved slot, without RNG replay.
    slots = np.argmax(p["eligible"] == d["donor"][:, None], axis=1)
    oldslots = np.argmax(baseline["eligible"] == s["previous_draw"]["donor"][:, None], axis=1)
    exact(slots, oldslots)


def observables(saved, diagnostic):
    """Derive the report inputs from leading clusters, not conditional arrays."""
    e = {k[6:]: v for k,v in saved.items() if k.startswith("entry_")}
    rows = e["event_row"]
    xy = e["position"][:, :2] - np.array([saved["geometry_disk_origin_" + k][0] for k in "xy"])
    unit = xy / np.linalg.norm(xy, axis=1)[:, None]
    p = np.sqrt(e["energy"] ** 2 - .51099895 ** 2)
    cell = np.zeros(len(rows), np.int64)
    for key, edges in (("energy", (0,50,80,95,110)), ("cosine", (0,.4,.7,1.000001)), ("radius", (0,450,550,750,1000))):
        cell = cell * (len(edges)-1) + np.searchsorted(edges, e[key], side="right") - 1
    cell += e["disk"].astype(np.int64) * 48
    result = dict(event_row=rows, momentum=p, cell_id=cell,
        direction_radial=np.sum(e["direction"][:, :2] * unit, axis=1),
        direction_tangent=e["direction"][:, 1] * unit[:, 0] - e["direction"][:, 0] * unit[:, 1])
    for arm in ("stockref", "control", *ARMS):
        energy = saved[arm + "_energy"][rows]
        delta = np.column_stack([saved[arm + "_" + k][rows] for k in "xy"]) - xy
        values = dict(energy=energy, truth_ep=energy/p,
            radial=np.sum(delta * unit, axis=1), tangent=delta[:, 1] * unit[:, 0] - delta[:, 0] * unit[:, 1],
            time=saved[arm + "_time"][rows] - diagnostic["residual_entry_folded_time"])
        result.update({arm + "_" + k:v for k,v in values.items()})
    for arm in ARMS:
        o, donor, oa = [diagnostic["residual_" + k] for k in ("original", "donor", "original_time_accepted")]
        terms = dict(donor=donor-o, spatial=-diagnostic["residual_" + arm + "_lost"],
            timing=-diagnostic["residual_" + arm + "_time_rejected"] + o-oa,
            downstream=result[arm + "_energy"] - diagnostic["residual_" + arm + "_time_accepted"] - (result["control_energy"]-oa),
            control=result["control_energy"]-result["stockref_energy"])
        close(sum(terms.values()), result[arm + "_energy"] - result["stockref_energy"])
        close(sum(v/p for v in terms.values()), result[arm + "_truth_ep"] - result["stockref_truth_ep"])
        result.update({arm + "_term_" + k:v for k,v in terms.items()})
    return result


def regions(s, a):
    e, d = s["entries"], s["diagnostics"]
    r = masks(e, s["draw"])
    for coord in ("radial", "tangent"):
        value = a["direction_" + coord]
        r[coord + "_direction_negative"] = value < 0
        r[coord + "_direction_nonnegative"] = value >= 0
    for lo, hi in ((0,10), (10,30), (30,np.inf)):
        v = d["residual_angle_deg"]
        r[f"angle_{lo}_{hi}"] = (v >= lo) & (v < hi)
    for disk in (0,1):
        for lo,hi in ((0,450), (450,550), (550,1000)):
            r[f"disk{disk}_radius_{lo}_{hi}"] = (e["disk"] == disk) & (e["radius"] >= lo) & (e["radius"] < hi)
    r["highEinnerdisk1"] = (e["disk"] == 1) & (e["radius"] < 450) & (e["energy"] >= 95)
    r["problem109"] = r["highEinnerdisk1"] & (e["cosine"] >= .7)
    r.update({f"cell_{i:02d}": a["cell_id"] == i for i in range(96)})
    r.update({k:s["draw"][k] for k in ("direction_used", "direction_fallback")})
    return r


def statistics(y, x):
    if not len(y):
        return dict(n=0, mean=None, std=None, covariance=None, correlation=None)
    corr = float(np.corrcoef(y, x)[0, 1]) if len(y)>1 and np.std(y)>0 and np.std(x)>0 else None
    return dict(n=len(y), mean=np.mean(y), std=np.std(y),
                covariance=np.mean((y-y.mean())*(x-x.mean())), correlation=corr)


def check_moments(record, f, o, x, rows, fm, om):
    """Literal removal of ORIGINAL blocks, including unselected-only blocks."""
    full = [statistics(y[m], x[m]) for y,m in ((f,fm),(o,om))]
    group = rows // 100
    blocks = np.unique(group)
    deleted = [[statistics(y[m & (group != b)], x[m & (group != b)])
                for y,m in ((f,fm),(o,om))] for b in blocks]
    for key in ("n", "mean", "std", "covariance", "correlation"):
        scalar(record["fast"][key], full[0][key])
        scalar(record["stock"][key], full[1][key])
        if key == "n":
            continue
        difference = None if any(v[key] is None for v in full) else full[0][key]-full[1][key]
        scalar(record["difference"][key], difference)
        error = None
        if difference is not None and len(blocks)>1 and all(v[key] is not None for pair in deleted for v in pair):
            leave = np.array([pair[0][key]-pair[1][key] for pair in deleted])
            error = np.sqrt((len(blocks)-1) * np.mean((leave-leave.mean())**2))
        scalar(record["error"][key], error)


EDGES = dict(energy=np.arange(0.,111.,5.), truth_ep=np.linspace(0,1.2,25),
             radial=np.arange(-1400.,1401.,40.), tangent=np.arange(-1400.,1401.,40.), time=np.arange(-20.,21.))


def histogram(value, edges):
    # np.histogram includes the rightmost edge; append explicit flow bins.
    return np.r_[np.sum(value < edges[0]), np.histogram(value, edges)[0], np.sum(value > edges[-1])]


def check_histogram(record, f, o, fm, om, rows, edges):
    ft, ot = histogram(f[fm], edges), histogram(o[om], edges)
    np.testing.assert_array_equal(record["fast"], ft)
    np.testing.assert_array_equal(record["stock"], ot)
    close(record["edges"], edges)
    assert record["fast_selected"] == fm.sum() and record["stock_selected"] == om.sum()
    diff = ft/ft.sum()-ot/ot.sum() if ft.sum() and ot.sum() else np.full(len(ft), np.nan)
    close(np.asarray(record["fraction_difference"],float), diff)
    blocks = np.unique(rows//100)
    leaves = []
    for b in blocks:
        keep = rows//100 != b
        fh, oh = histogram(f[fm & keep],edges), histogram(o[om & keep],edges)
        if not fh.sum() or not oh.sum():
            break
        leaves.append(fh/fh.sum()-oh/oh.sum())
    err = np.full(len(ft),np.nan)
    if len(blocks)>1 and len(leaves)==len(blocks):
        leaves = np.array(leaves)
        err = np.sqrt((len(blocks)-1)*np.mean((leaves-leaves.mean(axis=0))**2,axis=0))
    close(np.asarray(record["fraction_error"],float), err)


def test_saved_primary_correlations_paired_deltas_and_histograms(saved):
    s = saved
    a = observables(s["candidate"], s["diagnostics"])
    old = observables(s["previous"], s["previous_diagnostics"])
    ca = s["conditional_arrays"]
    for prefix, values in (("",a),("production_",old)):
        for name,value in values.items():
            close(ca[prefix+name],value)
    rr = regions(s,a)
    report = s["conditional"]["direct_production"]
    assert set(rr) == set(report["regions"])
    assert rr["problem109"].sum() == 109
    for arm in ARMS:
        for name in ("energy", "truth_ep"):
            h = report["cell_histograms"][arm][name]
            f = np.array([histogram(a[arm+"_"+name][a["cell_id"]==i],EDGES[name]) for i in range(96)])
            o = np.array([histogram(old[arm+"_"+name][a["cell_id"]==i],EDGES[name]) for i in range(96)])
            for k,v in (("fast",f),("stock",o),("difference",f-o)):
                np.testing.assert_array_equal(h[k],v)
    for label, mask in rr.items():
        rows = a["event_row"][mask]
        rec = report["regions"][label]
        assert rec["entries"] == mask.sum() and rec["source_blocks"] == len(np.unique(rows//100))
        assert s["numbers"]["direction_strata_counts"][label] == mask.sum()
        for arm in ARMS:
            item = rec["arms"][arm]
            f = {k:a[arm+"_"+k][mask] for k in EDGES}
            o = {k:old[arm+"_"+k][mask] for k in EDGES}
            stock = {k:a["stockref_"+k][mask] for k in EDGES}
            fm, om, sm = f["energy"]>10, o["energy"]>10, stock["energy"]>10
            common = fm & om & sm
            np.testing.assert_array_equal(ca[arm+"_common_three_way_selected"][mask],common)
            for key,val in dict(selected=fm.sum(),production_selected=om.sum(),stock_selected=sm.sum(),
                gains=(fm & ~om).sum(),misses=(om & ~fm).sum(),joint_two_way_selected=(fm & om).sum(),
                common_three_way_selected=common.sum()).items():
                assert item[key] == val, (label,arm,key)
            check_mean(item["selection"],fm,om,rows)
            check_ratio(item["ratio"],fm,om,rows)
            for name in ("energy", "truth_ep"):
                check_mean(item[name],f[name],o[name],rows)
            check_mean(item["ep_ge1"],f["truth_ep"]>=1,o["truth_ep"]>=1,rows)
            for name in ("radial", "tangent", "time"):
                x = (s["diagnostics"]["residual_entry_folded_time"] if name=="time" else a["direction_"+name])[mask]
                check_moments(item["spatial"][name]["own"],f[name],o[name],x,rows,fm,om)
                check_moments(item["spatial"][name]["joint_two_way"],f[name],o[name],x,rows,fm & om,fm & om)
                for contrast,y,z in (("new_vs_stock",f[name],stock[name]),("production_vs_stock",o[name],stock[name]),("new_vs_production",f[name],o[name])):
                    check_moments(item["common_three_way"][name][contrast],y,z,x,rows,common,common)
            for name in ("energy", "truth_ep"):
                check_moments(item["energy_momentum" if name=="energy" else "ep_momentum"],f[name],o[name],a["momentum"][mask],rows,np.ones(len(rows),bool),np.ones(len(rows),bool))
            for term in ("donor", "spatial", "timing", "downstream", "control"):
                ft,ot = a[arm+"_term_"+term][mask],old[arm+"_term_"+term][mask]
                check_mean(item["energy_terms"][term],ft,ot,rows)
                check_mean(item["ep_terms"][term],ft/a["momentum"][mask],ot/a["momentum"][mask],rows)
            for name, record in item["css_energy_partition"].items():
                check_mean(record, s["diagnostics"]["residual_"+name][mask],
                           s["previous_diagnostics"]["residual_"+name][mask], rows)
            for name, record in item.get("shapes",{}).items():
                selected = name in ("radial","tangent","time")
                check_histogram(record,f[name],o[name],fm if selected else np.ones(len(rows),bool),om if selected else np.ones(len(rows),bool),rows,EDGES[name])
