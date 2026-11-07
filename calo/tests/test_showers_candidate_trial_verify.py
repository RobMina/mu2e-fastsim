"""Independent, saved-only verification of the completed one-draw/two-call trial.

Opt in with CALO_CANDIDATE_TRIAL_VERIFY_BASE pointing at the artifact root or
its production_candidate_trial child. No replay, RNG, placement, preparation,
reconstruction, production report, or original candidate verifier is invoked.
External receipt paths are relocated when possible, otherwise read verbatim.
"""
import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest


ANCHORS = {
    "summary.json": "dee6f99faa4882a3ac98ded6a9a7d160aa011e1d739aab78235c443adc4eaf5a",
    "execution.json": "80591d54bcde92d7927809ec8290174809089510a00bff3564bff438e0479e52",
    "draw.npz": "abfec4dd20c7789eee986caa21fdaa5f6254e19ec2545fd1de914ca61659db50",
    "clusters.npz": "d86bc45f81a54cd8b53968950c43bfd13f472b07116a456cef5d76977e6edf94",
    "numbers.json": "8b0ff049e4f263fea15d671c000d11504286bf98a868fa4bb3b9a83c7675f8c5",
}
ARMS = ("geometric", "crystal")
FIELDS = ("energy", "time", "x", "y", "z")
LOSSES = ("raw_inner", "raw_outer", "raw_axial", "interior_unmapped",
          "center_or_proposed_envelope", "center_too_far", "containment", "postmap")


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


def test_deletion_jackknife_unequal_blocks_zeros_and_gate_edges():
    rows = np.array([0, 1, 100, 200, 201, 202])
    f = np.array([1., 3., 9., 2., 4., 8.])
    leaves = np.array([23 / 4, 18 / 5, 13 / 3])
    got = jackknife(f, np.zeros(6), rows)
    scalar(got["error"], np.sqrt(2 / 3 * np.sum((leaves - leaves.mean()) ** 2)))
    assert jackknife([], [], np.array([], int))["fast"] is None
    assert jackknife([0], [0], np.array([103]))["error"] is None
    t = dict(pbtime=np.full(4, -200., np.float32), mbtime=np.full(4, 1695., np.float32),
             marker=np.full(4, 200., np.float32), event_length=np.full(4, 1700.), onspill=np.ones(4, bool))
    hit, keep, margin, _ = fold(np.array([439., 440., 1795., 1800.]), np.arange(4), t)
    exact(keep, np.array([False, True, True, True]))
    close(margin, hit - 440)


@pytest.fixture(scope="module")
def saved():
    root = os.environ.get("CALO_CANDIDATE_TRIAL_VERIFY_BASE")
    if not root:
        pytest.skip("set CALO_CANDIDATE_TRIAL_VERIFY_BASE for saved-only real-artifact checks")
    base = Path(root).resolve()
    if base.name == "production_candidate_trial":
        base = base.parent
    trial = base / "production_candidate_trial"
    source = Path(__file__).resolve().parents[1] / "calofastsim"
    pins = {}
    def pin(path, value):
        path = Path(path)
        assert path not in pins or pins[path] == value
        assert sha(path) == value, str(path)
        pins[path] = value
        return path
    def read(path):
        return json.loads(Path(path).read_text())
    def locate(path):
        path = Path(path)
        if "calofastsim" in path.parts:
            return source.joinpath(*path.parts[path.parts.index("calofastsim") + 1:])
        # Receipt root name is fixed, but the local opt-in root may be renamed.
        marker = next((part for part in path.parts if part.startswith("CAL2_") and part[5:].isdigit()), None)
        if marker is not None:
            return base.joinpath(*path.parts[path.parts.index(marker) + 1:])
        return path  # external condition files, paths supplied by pinned receipt
    for name, value in ANCHORS.items():
        pin(trial / name, value)
    summary, execution, preparation = [read(trial / (k + ".json")) for k in ("summary", "execution", "preparation")]
    assert summary["completed"] and summary["sampler_calls"] == 1 and summary["reconstruction_calls"] == 2
    assert (execution["draws"], execution["calls"]) == (1, 2)
    assert [r["arm"] for r in execution["runs"]] == list(ARMS)
    assert {p.name for p in trial.glob("*_started.json")} == {a + "_started.json" for a in ARMS}
    assert {p.name for p in trial.glob("*_completed.json")} == {a + "_completed.json" for a in ARMS}
    assert {p.name for p in trial.iterdir() if p.is_file()} == set(summary["output_sha256"]) | {"summary.json"}
    for receipt in (summary, execution):
        for name, value in receipt["output_sha256"].items():
            assert Path(name).name == name
            pin(trial / name, value)
    assert preparation["input_sha256"] == summary["input_sha256"]
    for path, value in summary["input_sha256"].items():
        pin(locate(path), value)
    for path, value in preparation["calo_source_sha256"].items():
        pin(source / path, value)
    # Read constants without importing the executable driver.
    tree = ast.parse((source / "showers/candidate_trial.py").read_text())
    driver_pins = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "PINS" for t in node.targets))
    assert summary["pins"] == preparation["pins"] == driver_pins
    refs = {"candidate": base / "production_candidate/summary.json",
            "support": base / "production_candidate_support/summary.json",
            "k8_manifest": base / "primary_file0_radius8/preparation.json",
            "k8_numbers": base / "primary_file0_radius8/numbers.json",
            "k8_clusters": base / "primary_file0_radius8/clusters.npz"}
    for name, path in refs.items():
        pin(path, driver_pins[name])
    for name in ("candidate", "support"):
        receipt = read(refs[name])
        for filename, value in receipt["output_sha256"].items():
            assert Path(filename).name == filename
            pin(refs[name].parent / filename, value)
    assert (preparation["neighbors"], preparation["seed"], preparation["reco_seed"]) == (8, 20261105, 20261108)
    support = read(refs["support"])
    assert support["sampling_performed"] is False and support["reconstruction_performed"] is False
    assert support["candidate_query_overlap"]["shared_parent_events"] == 0
    assert support["candidate_query_overlap"]["shared_output_events"] == 0
    prior = read(refs["k8_manifest"])
    assert read(refs["k8_numbers"])["manifest"] == prior
    assert preparation["runtime"] == prior["runtime"]
    runtime = preparation["runtime"]
    assert runtime["backend"] == "cpu" and runtime["x64"] and runtime["fit_config"]["iterations"] == 15
    assert preparation["draw_rng_before"] != preparation["draw_rng_after"]  # never replay these states
    for arm, run in zip(ARMS, execution["runs"]):
        assert read(trial / (arm + "_started.json")) == dict(arm=arm, seed=20261108, retry_authorized=False)
        assert read(trial / (arm + "_completed.json")) == run
    # Executable sampling imports are guarded. No other production modules are
    # imported except the numeric export reader and its identity-only dependency.
    from calofastsim.showers import library, export
    from unittest.mock import patch
    from contextlib import ExitStack
    def forbidden(*args, **kwargs):
        raise AssertionError("saved verifier cannot build/sample/RNG/reconstruct/prepare")
    with ExitStack() as stack:
        for name in ("build", "sample"):
            stack.enter_context(patch.object(library, name, forbidden))
        for name in ("default_rng", "RandomState", "Generator", "PCG64", "seed", "random", "choice"):
            stack.enter_context(patch.object(np.random, name, forbidden))
        packed = archive(base / "production_candidate/library.npz")
        pools = archive(base / "production_candidate_support/candidate_pools.npz")
        tables = next(locate(p) for p in summary["input_sha256"] if p.endswith("calo_library_tables.root"))
        entries, original, geometry, census, total = export.read(tables, input_mode="primary")
        import uproot
        with uproot.open(tables) as stream:
            raw_ids = stream["dump/events"].arrays(["run", "subrun", "event"], library="np")
        ids = np.rec.fromarrays([raw_ids[k].astype(np.int64) for k in ("run", "subrun", "event")], names="run,subrun,event")
        result = dict(base=base, trial=trial, library=packed, pools=pools, entries=entries, original=original,
                      geometry=geometry, ids=ids, summary=summary, execution=execution, preparation=preparation,
                      numbers=read(trial / "numbers.json"), diagnostics=archive(trial / "diagnostics.npz"),
                      draw=archive(trial / "draw.npz"), candidate=archive(trial / "clusters.npz"),
                      legacy=archive(base / "primary_file0_comparison/clusters.npz"),
                      k8=archive(base / "primary_file0_radius8/clusters.npz"),
                      placed={arm: archive(trial / (arm + "_placement.npz")) for arm in ARMS})
        assert len(total) == len(ids) == 2813 and census["front_entry_events"] == 2396
        yield result
    for path, value in pins.items():
        assert sha(path) == value, "artifact changed during verification: " + str(path)


def test_saved_full_identity_reference_reuse_and_complete_query_rows(saved):
    s = saved
    rows = s["entries"]["event_row"]
    assert len(rows) == 2396 and np.all(np.diff(rows) > 0)
    assert len(np.unique(s["ids"])) == 2813
    front = np.isin(np.arange(2813), rows)
    for label in ("legacy", "k8", "candidate"):
        data = s[label]
        exact(data["event_row"], np.arange(2813))
        exact(data["full_ids"], s["ids"])
        exact(data["query_event_rows"], rows)
        for prefix, values in (("entry_", s["entries"]), ("geometry_", s["geometry"])):
            for name, value in values.items():
                exact(data[prefix + name], value)
        for arm in ("stockref", "control", *ARMS):
            energy = data[arm + "_energy"]
            assert energy.shape == (2813,) and np.all(np.isfinite(energy)) and np.all(energy >= 0)
            for field in FIELDS[1:]:
                assert data[arm + "_" + field].shape == (2813,)
                assert np.all(np.isfinite(data[arm + "_" + field][energy > 0]))
            if arm in ARMS:
                assert np.all(energy[~front] == 0)
        phase = -(s["ids"]["event"] % 5) * 5.
        expected_timing = dict(pbtime=(-200 + phase).astype(np.float32), mbtime=np.full(2813, 1695., np.float32),
            event_length=np.where(phase < -20, 1675., 1700.), onspill=np.ones(2813, bool),
            marker=np.full(2813, 200., np.float32), crystal_length=np.full(2813, 200., np.float32))
        for name, value in expected_timing.items():
            exact(data["timing_" + name], value)
    for name, value in s["legacy"].items():
        variable = name.startswith(("geometric_", "crystal_", "sampling_"))
        exact(s["k8"][("baseline_" if variable else "") + name], value)
        if not variable:
            exact(s["candidate"][name], value)
    for arm in ARMS:
        for field, value in archive(s["trial"] / (arm + ".npz")).items():
            exact(s["candidate"][arm + "_" + field], value)
    for name in ("donor", "supported", "fallback", "offsets", "radius_pool_size", "bin_pool_size"):
        exact(s["candidate"]["sampling_" + name], s["draw"][name])
    for name, value in dict(event_row=np.arange(2813), full_ids=s["ids"], query_event_rows=rows, front=front).items():
        exact(s["diagnostics"][name], value)


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


def check_cluster_report(report, f, s, rows):
    selected_f, selected_s = f["energy"] > 10, s["energy"] > 10
    both = selected_f & selected_s
    for key, value in dict(events=len(rows), fast_selected=selected_f.sum(), stock_selected=selected_s.sum(),
                           both_selected=both.sum(), fast_only=(selected_f & ~selected_s).sum(),
                           stock_only=(selected_s & ~selected_f).sum()).items():
        assert report[key] == value
    check_mean(report["acceptance_difference"], selected_f, selected_s, rows)
    check_ratio(report["acceptance_ratio"], selected_f, selected_s, rows)
    check_mean(report["energy_mean"], f["energy"], s["energy"], rows)
    if both.any():
        check_mean(report["paired_time_ns"], f["time"][both], s["time"][both], rows[both])
        scalar(report["paired_position_distance_mm"], np.hypot(f["x"][both] - s["x"][both], f["y"][both] - s["y"][both]).mean())
    else:
        assert "paired_time_ns" not in report
    hist = report["energy_shape"]
    edges = np.arange(0., 111., 5.)
    close(hist["edges"], edges)
    close(hist["fast"], np.histogram(f["energy"], edges)[0])
    close(hist["stock"], np.histogram(s["energy"], edges)[0])
    blocks = np.unique(rows // 100)
    if len(blocks) > 1:
        delta = np.array([np.histogram(f["energy"][rows // 100 == b], edges)[0] -
                          np.histogram(s["energy"][rows // 100 == b], edges)[0] for b in blocks])
        variance = len(blocks) * np.sum((delta - delta.mean(axis=0)) ** 2, axis=0) / (len(blocks) - 1)
        close(hist["variance"], variance)
        good = variance > 0
        difference = delta.sum(axis=0)
        assert hist["ndf"] == good.sum()
        scalar(hist["chi2"], np.sum(difference[good] ** 2 / variance[good]))
        for actual, numerator, var in zip(hist["pull"], difference, variance):
            scalar(actual, numerator / np.sqrt(var) if var else None)


def test_saved_all_fixed_strata_selection_gains_misses_means_errors(saved):
    s = saved
    n, d, rows = s["numbers"], s["diagnostics"], s["entries"]["event_row"]
    strata = masks(s["entries"], s["draw"])
    assert n["queries"] == 2396 and n["full_stored_rows"] == 2813
    assert n["strata_counts"] == {k: int(v.sum()) for k, v in strata.items()}
    assert strata["inner_highE_disk1"].sum() == 190
    assert strata["inner_highE_disk1_coshigh"].sum() == 109
    clusters = {arm: {field: s["legacy"][arm + "_" + field] for field in FIELDS} for arm in ("stockref", "control")}
    for label in ("legacy", "k8", "candidate"):
        clusters.update({label + "_" + arm: {field: s[label][arm + "_" + field] for field in FIELDS} for arm in ARMS})
    for name, cluster in clusters.items():
        exact(d[name + "_selected"], cluster["energy"] > 10)
        for label, mask in strata.items():
            rr = rows[mask]
            check_cluster_report(n["comparisons"][name][label], {k: v[rr] for k, v in cluster.items()},
                                 {k: v[rr] for k, v in clusters["stockref"].items()}, rr)
    for reference in ("legacy", "k8"):
        for arm in ARMS:
            for label, mask in strata.items():
                rr = rows[mask]
                check_cluster_report(n["change"][reference][arm][label],
                    {k: v[rr] for k, v in clusters["candidate_" + arm].items()},
                    {k: v[rr] for k, v in clusters[reference + "_" + arm].items()}, rr)
    expected = dict(candidate_geometric=1714, candidate_crystal=1729, stockref=1727, control=1722,
                    k8_geometric=1643, k8_crystal=1655)
    for arm, count in expected.items():
        assert n["comparisons"][arm]["all"]["fast_selected"] == count
    for arm, count in dict(candidate_geometric=86, candidate_crystal=88, k8_geometric=61, k8_crystal=61, stockref=89).items():
        assert n["comparisons"][arm]["inner_highE_disk1_coshigh"]["fast_selected"] == count
    common = np.all(np.stack([v["energy"][rows] > 10 for v in clusters.values()]), axis=0)
    assert common.sum() == n["eight_arm_common_selected"] == 1469
    exact(d["eight_arm_common_selected"], common)
    exact(d["eight_arm_common_event_rows"], rows[common])
    rr = rows[common]
    distances = {k: np.hypot(v["x"][rr] - clusters["stockref"]["x"][rr], v["y"][rr] - clusters["stockref"]["y"][rr]) for k, v in clusters.items()}
    for key, value in distances.items():
        check_mean(n["eight_arm_distance_mm"][key], value, np.zeros(len(rr)), rr)
    for reference in ("legacy", "k8"):
        for arm in ARMS:
            check_mean(n["eight_arm_distance_change_mm"][reference][arm], distances["candidate_" + arm], distances[reference + "_" + arm], rr)
    front = np.isin(np.arange(2813), rows)
    account = n["full_stored_accounting"]
    assert (account["rows"], account["front"], account["nonfront"]) == (2813, 2396, 417)
    assert not account["inclusive_sampled_acceptance_model"]
    check_cluster_report(account["original_control_all_rows"], clusters["control"], clusters["stockref"], np.arange(2813))
    for name, cluster in clusters.items():
        for label, mask in (("all", np.ones(2813, bool)), ("front", front), ("nonfront", ~front)):
            record = account["arms"][name][label]
            assert record["rows"] == mask.sum()
            assert record["selected"] == np.sum(cluster["energy"][mask] > 10)
            assert record["zero"] == np.sum(cluster["energy"][mask] == 0)
            scalar(record["energy_sum"], cluster["energy"][mask].sum())
    original_energy = sums(s["original"])
    originals = dict(energy=original_energy, empty=np.bincount(s["original"]["entry"], minlength=2396) == 0, weak=original_energy <= 10)
    for label, mask in strata.items():
        record = n["draw"][label]
        assert record["entries"] == record["supported"] == mask.sum()
        assert record["unsupported"] == 0
        assert record["fallback"] == np.sum(mask & s["draw"]["fallback"])
        for name, original in originals.items():
            actual, expected, variance = [d["pool_" + k + name][mask] for k in ("actual_", "expected_", "variance_")]
            out = record["outcomes"][name]
            for key, value in (("actual_total", actual.sum()), ("expected_total", expected.sum()),
                               ("original_total", original[mask].sum()), ("conditional_draw_sd_total", np.sqrt(variance.sum())),
                               ("conditional_draw_sd_mean", np.sqrt(variance.sum()) / mask.sum() if mask.any() else None)):
                scalar(out[key], value)
            for key, f, st in (("actual_minus_expected", actual, expected), ("actual_minus_original", actual, original[mask]),
                               ("expected_minus_original", expected, original[mask])):
                check_mean(out[key], f, st, rows[mask])


def test_saved_css_selection_partitions_and_report_uncertainties(saved):
    s = saved
    n, d, entries = s["numbers"], s["diagnostics"], s["entries"]
    rows = entries["event_row"]
    strata = masks(entries, s["draw"])
    source, donor = d["css_original"], d["css_donor"]
    stock, control = [s["candidate"][arm + "_energy"][rows] for arm in ("stockref", "control")]
    for label, record in n["css"]["strata"].items():
        mask = strata[label]
        rr, zero = rows[mask], np.zeros(mask.sum())
        assert record["entries"] == mask.sum()
        for arm in ARMS:
            item = record["arms"][arm]
            mapped, lost = [d[f"css_{arm}_{k}"] for k in ("mapped", "lost")]
            fast = s["candidate"][arm + "_energy"][rows]
            check_mean(item["donor_minus_original"], donor[mask], source[mask], rr)
            check_mean(item["placement_loss"], lost[mask], zero, rr)
            check_mean(item["mapped_minus_original"], mapped[mask], source[mask], rr)
            check_mean(item["saved_reco_delta"], fast[mask], stock[mask], rr)
            check_ratio(item["lost_fraction_of_donor"], lost[mask], donor[mask], rr)
            for loss in LOSSES:
                check_mean(item["losses"][loss], d[f"css_{arm}_loss_{loss}"][mask], zero, rr)
            selection = item["selection"]
            sg, dg, mg = source > 10, donor > 10, mapped > 10
            expected_css = dict(original=sg, donor=dg, mapped=mg, donor_to_mapped_down=dg & ~mg,
                                source_to_donor_down=sg & ~dg, source_to_donor_up=~sg & dg)
            assert selection["css_gate"] == {k: int(np.sum(v & mask)) for k, v in expected_css.items()}
            miss, gain, cmiss = (stock > 10) & (fast <= 10), (stock <= 10) & (fast > 10), (control > 10) & (fast <= 10)
            reco = selection["reco"]
            expected_reco = dict(stock_selected=stock > 10, fast_selected=fast > 10, misses=miss, gains=gain,
                control_misses=cmiss, misses_with_css_placement_gate_down=miss & dg & ~mg)
            for key, value in expected_reco.items():
                assert reco[key] == np.sum(value & mask)
            assert reco["net_deficit"] == np.sum(miss & mask) - np.sum(gain & mask)
            categories = dict(no_donor_energy=donor == 0, unsupported=~s["draw"]["supported"],
                              positive_donor_no_loss=(donor > 0) & (lost <= 1e-12),
                              positive_donor_with_loss=(donor > 0) & (lost > 1e-12))
            for key, selected in (("miss_partition", miss), ("control_miss_partition", cmiss)):
                assert reco[key] == {k: int(np.sum(mask & selected & v)) for k, v in categories.items()}
    # Timing-associated miss/gain partitions are descriptive, not causal losses.
    regions = dict(all=np.ones(2396, bool), inner=entries["radius"] < 450,
                   disk0=entries["disk"] == 0, disk1=entries["disk"] == 1,
                   inner_highE=(entries["radius"] < 450) & (entries["energy"] >= 95))
    for disk in (0, 1):
        regions[f"inner_highE_disk{disk}"] = regions["inner_highE"] & (entries["disk"] == disk)
    residual_masks = dict(strata)
    for name, value, edges in (("angle", d["residual_angle_deg"], (0, 10, 30, np.inf)),
        ("margin", d["residual_entry_gate_margin"], (-np.inf, 0, 50, np.inf)),
        ("occupancy", d["residual_donor_crystals"], (0, 1, 2, 5, np.inf)),
        ("pool", s["draw"]["bin_pool_size"], (0, 1, 9, 33, np.inf)),
        ("reuse", d["residual_donor_reuse"], (0, 1, 2, 5, np.inf))):
        for lo, hi in zip(edges[:-1], edges[1:]):
            residual_masks[f"{name}_{lo}_{hi}"] = (value >= lo) & (value < hi)
    residual_masks.update(no_fallback=~s["draw"]["fallback"], donor_empty=d["residual_donor_steps"] == 0,
        donor_zero_nonempty=(d["residual_donor_steps"] > 0) & (donor == 0), donor_positive_le10=(donor > 0) & (donor <= 10),
        donor_gt10=donor > 10, radius_pool_lt8=s["draw"]["radius_pool_size"] < 8, radius_pool_8=s["draw"]["radius_pool_size"] == 8)
    for region, records in n["residual"]["regions"].items():
        for label, record in records.items():
            mask = regions[region] & residual_masks[label]
            rr, zero = rows[mask], np.zeros(mask.sum())
            assert record["entries"] == mask.sum()
            assert record["source_blocks"] == len(np.unique(rr // 100))
            for arm in ("control", *ARMS):
                item = record["arms"][arm]
                fast = s["candidate"][arm + "_energy"][rows]
                selected, selected_stock = fast > 10, stock > 10
                miss, gain = selected_stock & ~selected, ~selected_stock & selected
                for key, value in dict(selected=selected, misses=miss, gains=gain, control_misses=(control > 10) & ~selected,
                                       jointly_selected=selected & selected_stock).items():
                    assert item[key] == np.sum(mask & value)
                check_mean(item["selection_difference"], selected[mask], selected_stock[mask], rr)
                check_ratio(item["ratio"], selected[mask], selected_stock[mask], rr)
                check_mean(item["energy"], fast[mask], stock[mask], rr)
                if arm not in ARMS:
                    continue
                mapped, timed, rejected = [d[f"residual_{arm}_{k}"] for k in ("mapped", "time_accepted", "time_rejected")]
                categories = dict(unsupported=~s["draw"]["supported"], zero_donor=donor == 0,
                    all_spatial_loss=(donor > 0) & (mapped == 0), all_time_rejected=(mapped > 0) & (timed == 0),
                    partial_time_rejected=(timed > 0) & (rejected > 1e-12), positive_no_time_rejection=(timed > 0) & (rejected <= 1e-12))
                exact(np.sum(list(categories.values()), axis=0), np.ones(2396, np.int64))
                for key, selection in (("miss_partition", miss), ("gain_partition", gain)):
                    assert item[key] == {k: int(np.sum(mask & selection & v)) for k, v in categories.items()}
                for key, feature in (("mapped", "mapped"), ("spatial_loss", "lost"), ("time_rejected", "time_rejected"),
                    ("time_accepted", "time_accepted"), ("end_exposed", "end_exposed"), ("delay_ge20", "delay_ge20"),
                    ("mean_occupied_crystals", "crystals")):
                    check_mean(item["css"][key], d[f"residual_{arm}_{feature}"][mask], zero, rr)
                check_mean(item["css"]["accepted_minus_original"], timed[mask], d["residual_original_time_accepted"][mask], rr)
                for key, value in dict(accepted_gt10=timed > 10, timing_downcross=(mapped > 10) & (timed <= 10),
                    timing_downcross_misses=miss & (mapped > 10) & (timed <= 10)).items():
                    assert item["css"][key] == np.sum(value & mask)
