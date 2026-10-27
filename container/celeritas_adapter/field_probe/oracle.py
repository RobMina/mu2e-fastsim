"""Independent mm/T/MeV-c field-only oracle: prepare NEW_OUTPUT; check OUTPUT [--request NAME]."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


# These are diagnostic synthetic-backend budgets, not Mu2e accuracy requirements or fitted tolerances.
# Celeritas 0.6.3 uses covfie host float/software linear and CUDA texture linear interpolation.
# CUDA texture fractional weights have eight fractional bits; use a conservative full 1/256 cell.
# The float allowance is 32 eps32 times field/coordinate-transform scale, including CGS conversion.
# Integration budgets assume at least fourth-order fixed integration, with a conservative 0.1 coefficient.
# Adaptive budgets allow accumulated local relative error: 20 epsilon times path/rotation scales.
# Neither acceptance nor decreasing step sizes establishes physical Mu2e accuracy or strict monotonicity.
# Reference DOP853 solves at two tolerances must agree within 2e-6 mm and 2e-9 relative momentum.
# Momentum conservation is separately required to 5e-4; path completeness to 1e-6 mm + 5e-9 L.
# Endpoint field envelopes use the affine Lipschitz majorant, not an observed-error calibration.
KAPPA = 0.299792458
TAG = 0x6F31
REQUEST_FORMAT = "m2fs-field-probe-v1"
RESULT_FORMAT = "m2fs-field-probe-result-v1"
EXPECTED_FORMAT = "m2fs-field-probe-oracle-v1"
REF_POSITION_MM = 2e-6
REF_MOMENTUM_REL = 2e-9
MOMENTUM_DRIFT_MAX = 5e-4
FIXED_CAPS_MM = (4.0, 1.0, 0.25)
ADAPTIVE_EPSILONS = (1e-3, 1e-5, 1e-7)
FAMILIES = ("zero", "constant", "affine")
RESOLUTIONS = ("coarse", "refined")
GRADIENT = np.array([[1e-5, 2e-5, -3e-5], [4e-5, -2.5e-5, 1.5e-5], [-2e-5, 3e-5, 1.5e-5]])
TILTED_FIELD = np.array([0.23, -0.31, 0.87])
_HERE = Path(__file__).resolve().parent
_CONTRACT_PATH = _HERE / "field_contract.py" if (_HERE / "field_contract.py").is_file() else _HERE.parent / "field_contract.py"
_SPEC = importlib.util.spec_from_file_location("probe_field_contract", _CONTRACT_PATH)
_CONTRACT_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONTRACT_MODULE)
CONTRACT = _CONTRACT_MODULE.CONTRACT


def require(condition, message):
    """Fail closed with an actionable diagnostic."""
    if not condition:
        raise ValueError(message)


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(data):
    return hashlib.sha256(canonical(data).encode()).hexdigest()


def keys(data, required, label):
    require(isinstance(data, dict) and set(data) == set(required.split()), f"{label}: wrong/missing keys")


def number(value, label):
    require(type(value) in (float, int) and math.isfinite(value), f"{label}: expected finite number")
    return float(value)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"{label}: invalid integer")
    return value


def array(value, shape, label):
    raw = np.asarray(value)
    require(raw.dtype.kind in "fiu" and raw.shape == shape, f"{label}: wrong numeric shape")
    result = raw.astype(float)
    require(np.isfinite(result).all(), f"{label}: nonfinite values")
    return result


def model(family):
    require(family in FAMILIES, "Unknown synthetic field family")
    b0 = np.zeros(3) if family == "zero" else TILTED_FIELD.copy()
    gradient = GRADIENT.copy() if family == "affine" else np.zeros((3, 3))
    require(abs(np.trace(gradient)) < 1e-18, "Field must be divergence free")
    return b0, gradient


def field_at(points_mm, b0, gradient):
    """Analytic B_j = b0_j + sum_i x_i G_ij in tesla; G is tesla/mm."""
    p = np.asarray(points_mm, dtype=float)
    require(p.ndim == 2 and p.shape[1] == 3 and np.isfinite(p).all(), "Expected finite N x 3 coordinates")
    return p @ array(gradient, (3, 3), "gradient") + array(b0, (3,), "b0")


def initial_arrays(position_mm, momentum_mev_c, charge_e, length_mm):
    x = np.asarray(position_mm, dtype=float)
    require(x.ndim == 2 and x.shape[1] == 3 and len(x) > 0 and np.isfinite(x).all(), "Invalid initial positions")
    p = array(momentum_mev_c, x.shape, "momenta")
    q = np.broadcast_to(np.asarray(charge_e, dtype=float), (len(x),)).copy()
    length = np.broadcast_to(np.asarray(length_mm, dtype=float), (len(x),)).copy()
    require(np.isfinite(q).all() and np.isin(q, [-1, 0, 1]).all(), "Invalid charges")
    require(np.isfinite(length).all() and np.all(length > 0), "Invalid lengths")
    require(np.all(np.linalg.norm(p, axis=1) > 0), "Zero initial momentum")
    return x, p, q, length


def constant_helix(position_mm, momentum_mev_c, charge_e, length_mm, field_tesla):
    """Vectorized exact helix: dx/ds=u, dp/ds=0.299792458 q (u cross B), s in mm."""
    x, p, q, length = initial_arrays(position_mm, momentum_mev_c, charge_e, length_mm)
    b = array(field_tesla, (3,), "constant field")
    pmag = np.linalg.norm(p, axis=1)
    u = p / pmag[:, None]
    bmag = np.linalg.norm(b)
    axis = b / bmag if bmag else np.zeros(3)
    parallel = (u @ axis)[:, None] * axis
    perpendicular = u - parallel
    transverse = np.cross(perpendicular, axis)
    theta = KAPPA * q * bmag * length / pmag
    sinc = np.sinc(theta / np.pi)
    cosc = 0.5 * theta * np.sinc(theta / (2 * np.pi)) ** 2
    displacement = length[:, None] * (parallel + sinc[:, None] * perpendicular + cosc[:, None] * transverse)
    direction = parallel + np.cos(theta)[:, None] * perpendicular + np.sin(theta)[:, None] * transverse
    return x + displacement, pmag[:, None] * direction


def affine_ode(position_mm, momentum_mev_c, charge_e, length_mm, b0, gradient, tight=True):
    """Independent DOP853 solve, batched across tracks with common normalized path t in [0,1]."""
    x, p, q, length = initial_arrays(position_mm, momentum_mev_c, charge_e, length_mm)
    b0 = array(b0, (3,), "b0")
    gradient = array(gradient, (3, 3), "gradient")
    n = len(x)
    initial = np.concatenate((x, p), axis=1)

    def rhs(t, flattened):
        state = flattened.reshape(n, 6)
        direction = state[:, 3:] / np.linalg.norm(state[:, 3:], axis=1)[:, None]
        force = KAPPA * q[:, None] * np.cross(direction, state[:, :3] @ gradient + b0)
        return (length[:, None] * np.concatenate((direction, force), axis=1)).ravel()

    # A conservative rotation cap avoids dilution of the adaptive RMS norm across independent tracks.
    field_bound = np.linalg.norm(np.abs(b0) + (np.abs(x) + length[:, None]) @ np.abs(gradient), axis=1)
    rotation_bound = np.max(KAPPA * np.abs(q) * field_bound * length / np.linalg.norm(p, axis=1))
    max_step = min(1.0, 0.15 / rotation_bound) if rotation_bound else 1.0
    atol = np.tile([2e-13] * 3 + [2e-15] * 3 if tight else [2e-11] * 3 + [2e-13] * 3, n)
    solution = solve_ivp(rhs, (0.0, 1.0), initial.ravel(), method="DOP853", rtol=3e-14 if tight else 2e-12, atol=atol, max_step=max_step)
    require(solution.success and solution.t[-1] == 1.0, "Independent DOP853 reference did not finish")
    final = solution.y[:, -1].reshape(n, 6)
    require(np.isfinite(final).all(), "Nonfinite reference endpoint")
    return final[:, :3], final[:, 3:]


def grid_geometry(data):
    lo = np.array([a["min_mm"] for a in data["axes"]], dtype=float)
    hi = np.array([a["max_mm"] for a in data["axes"]], dtype=float)
    counts = np.array([a["num"] for a in data["axes"]])
    return lo, hi, counts, (hi - lo) / (counts - 1)


def make_map(family, resolution):
    """Public asymmetric xyz-flat maps; both resolutions represent exactly the same analytic field."""
    require(resolution in RESOLUTIONS, "Unknown resolution")
    counts = (10, 15, 13) if resolution == "coarse" else (19, 29, 25)
    axes = [dict(min_mm=lo, max_mm=hi, num=n) for lo, hi, n in zip((-1600, -1900, -2200), (2000, 2300, 2600), counts)]
    points = np.stack(np.meshgrid(*(np.linspace(a["min_mm"], a["max_mm"], a["num"]) for a in axes), indexing="ij"), axis=-1)
    b0, gradient = model(family)
    result = dict(CONTRACT, axes=axes, field=field_at(points.reshape(-1, 3), b0, gradient).ravel().tolist())
    _CONTRACT_MODULE.validate(result)
    return result


def sample_points(data):
    """1016 queries: nodes, centers, random off-grid, texture thresholds, edges, faces, outside guards."""
    lo, hi, n, h = grid_geometry(data)
    rng = np.random.default_rng(20261026)
    pieces, labels = [], []

    def add(points, category):
        points = np.asarray(points).reshape(-1, 3)
        pieces.append(points)
        labels.extend([category] * len(points))

    add(lo + rng.integers(1, n - 1, size=(160, 3)) * h, "interior_node")
    add(lo + (rng.integers(1, n - 2, size=(160, 3)) + 0.5) * h, "cell_center")
    add(rng.uniform(lo + 1.01 * h, hi - 1.01 * h, size=(384, 3)), "off_grid")
    for axis in range(3):
        for k in (0, 1, 31, 63, 127, 191, 253, 254):
            base = rng.integers(2, n - 2).astype(float) + 0.37
            for epsilon in (-1e-5, 0, 1e-5):
                u = base.copy()
                u[axis] = math.floor(u[axis]) + (k + 0.5) / 256 + epsilon
                add(lo + h * u, "texture_threshold")
                u[axis] = math.floor(base[axis]) + epsilon
                add(lo + h * u, "cell_edge")
    for axis in range(3):
        for side in (-1, 1):
            face = lo[axis] + h[axis] if side < 0 else hi[axis] - h[axis]
            for category, offsets in (("inset_face", [0.0] * 10 + [1e-3] * 10), ("outside_inset", [-0.02] * 8)):
                points = rng.uniform(lo + 1.1 * h, hi - 1.1 * h, size=(len(offsets), 3))
                points[:, axis] = face - side * np.asarray(offsets)
                add(points, f"{category}_{axis}_{side:+d}")
    return np.concatenate(pieces), labels


def make_tracks(data):
    """90 field-sensitive sweep tracks plus six deliberate guard crossings; no physical boundaries."""
    tracks, metadata = [], []
    directions = np.array([[0.81, 0.42, -0.39], [-0.31, 0.88, 0.36], [0.45, -0.28, 0.85]])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    index = 0
    for momentum in (5.0, 25.0, 105.0):
        for charge in (-1, 1, 0):
            for length in ((180.0, 600.0) if charge else (600.0,)):
                direction = directions[index % 3]
                position = [float(45 * ((index % 3) - 1)), float(37 * ((index % 5) - 2)), float(31 * ((index % 4) - 1))]
                group = f"p{momentum:g}_q{charge:+d}_L{length:g}"
                variants = [("fixed", cap, 1e-7) for cap in FIXED_CAPS_MM] + [("adaptive", 16.0, eps) for eps in ADAPTIVE_EPSILONS]
                for method, cap, eps in variants:
                    track_id = f"{group}_{method}_{cap:g}_{eps:g}"
                    tracks.append(dict(id=track_id, position_mm=position, momentum_mev_c=(momentum * direction).tolist(), charge_e=charge, length_mm=length, max_step_mm=cap, method=method, epsilon_rel_max=eps, delta_chord_mm=0.05))
                    metadata.append(dict(id=track_id, group=group, expected_status=0, sweep=method))
                index += 1
    lo, hi, n, h = grid_geometry(data)
    for axis in range(3):
        for side in (-1, 1):
            position = (lo + hi) / 2
            position[axis] = lo[axis] + h[axis] + 0.5 if side < 0 else hi[axis] - h[axis] - 0.5
            momentum = np.zeros(3)
            momentum[axis] = side * 105.0
            track_id = f"guard_cross_{axis}_{side:+d}"
            tracks.append(dict(id=track_id, position_mm=position.tolist(), momentum_mev_c=momentum.tolist(), charge_e=side, length_mm=40.0, max_step_mm=4.0, method="fixed", epsilon_rel_max=1e-7, delta_chord_mm=0.05))
            metadata.append(dict(id=track_id, group=track_id, expected_status=1, sweep="guard"))
    require(len(tracks) <= 100, "Fixture exceeded track budget")
    return tracks, metadata


def field_error_bounds(data, b0, gradient):
    """Per-component absolute T bounds; full 1/256 times sum of per-axis affine cell variation."""
    lo, hi, n, h = grid_geometry(data)
    variation = h @ np.abs(gradient)
    scale = np.abs(b0) + np.maximum(np.abs(lo), np.abs(hi)) @ np.abs(gradient) + variation
    host = 32 * np.finfo(np.float32).eps * np.maximum(scale, 1e-6)
    return host, host + variation / 256


def reference_endpoints(tracks, b0, gradient):
    """Solve unique physical initial states once, independently of step/tolerance requests."""
    unique, mapping = [], []
    lookup = {}
    for track in tracks:
        identity = canonical([track[k] for k in ("position_mm", "momentum_mev_c", "charge_e", "length_mm")])
        if identity not in lookup:
            lookup[identity] = len(unique)
            unique.append(track)
        mapping.append(lookup[identity])
    x = np.array([t["position_mm"] for t in unique])
    p = np.array([t["momentum_mev_c"] for t in unique])
    q = np.array([t["charge_e"] for t in unique])
    length = np.array([t["length_mm"] for t in unique])
    loose_x, loose_p = affine_ode(x, p, q, length, b0, gradient, tight=False)
    tight_x, tight_p = affine_ode(x, p, q, length, b0, gradient, tight=True)
    position_disagreement = np.linalg.norm(tight_x - loose_x, axis=1)
    momentum_disagreement = np.linalg.norm(tight_p - loose_p, axis=1) / np.linalg.norm(p, axis=1)
    require(np.max(position_disagreement) <= REF_POSITION_MM, "Reference solve position precision not established")
    require(np.max(momentum_disagreement) <= REF_MOMENTUM_REL, "Reference solve momentum precision not established")
    if not np.any(gradient):
        exact_x, exact_p = constant_helix(x, p, q, length, b0)
        require(np.max(np.linalg.norm(tight_x - exact_x, axis=1)) <= REF_POSITION_MM, "DOP853 disagrees with analytic helix")
        require(np.max(np.linalg.norm(tight_p - exact_p, axis=1) / np.linalg.norm(p, axis=1)) <= REF_MOMENTUM_REL, "DOP853 helix sign/momentum disagreement")
        tight_x, tight_p = exact_x, exact_p
    precision = dict(position_disagreement_mm=float(np.max(position_disagreement)), momentum_relative_disagreement=float(np.max(momentum_disagreement)), rtol_loose=2e-12, rtol_tight=3e-14, solver="SciPy DOP853", truth="analytic helix" if not np.any(gradient) else "analytic affine Lorentz ODE")
    return tight_x[mapping], tight_p[mapping], precision


def build_case(family, resolution):
    """Return (map, exact request, expected sidecar) in memory; no subprocess or filesystem writes."""
    data = make_map(family, resolution)
    b0, gradient = model(family)
    points, categories = sample_points(data)
    tracks, metadata = make_tracks(data)
    stem = f"{family}-{resolution}"
    request = dict(format=REQUEST_FORMAT, map_file=f"{stem}.map.json", sample_points_mm=points.tolist(), tracks=tracks)
    lo, hi, n, h = grid_geometry(data)
    margin = np.minimum(points - (lo + h), (hi - h) - points).min(axis=1)
    valid = margin >= 0
    selected = [t for t, m in zip(tracks, metadata) if m["expected_status"] == 0]
    endpoints, momenta, precision = reference_endpoints(selected, b0, gradient)
    endpoint_iter = iter(zip(endpoints, momenta))
    for track, meta in zip(tracks, metadata):
        if meta["expected_status"] == 0:
            # The arclength ball proves containment for every intermediate oracle trajectory point.
            start_margin = np.min(np.minimum(np.array(track["position_mm"]) - lo - h, hi - h - np.array(track["position_mm"])))
            require(start_margin > track["length_mm"] + track["max_step_mm"], "Accepted fixture lacks a proven guard margin")
            end, mom = next(endpoint_iter)
            meta.update(end_position_mm=end.tolist(), end_momentum_mev_c=mom.tolist())
    host_bound, device_bound = field_error_bounds(data, b0, gradient)
    expected = dict(format=EXPECTED_FORMAT, family=family, resolution=resolution, request_sha256=digest(request), map_sha256=digest(data), units=dict(length="mm", momentum="MeV/c", field="tesla", charge="e", gradient="tesla/mm"), physical_factor=KAPPA, model=dict(b0_tesla=b0.tolist(), gradient_tesla_per_mm=gradient.tolist()), sample_categories=categories, sample_valid=valid.tolist(), sample_margin_mm=margin.tolist(), sample_field_tesla=field_at(points, b0, gradient).tolist(), field_bound_tesla=dict(host=host_bound.tolist(), device=device_bound.tolist()), tracks=metadata, reference_precision=precision, scope="Synthetic diagnostic backend acceptance only; not final Mu2e validation")
    return data, request, expected


def validate_request(request):
    keys(request, "format map_file sample_points_mm tracks", "request")
    require(request["format"] == REQUEST_FORMAT, "Wrong request format")
    name = request["map_file"]
    require(isinstance(name, str) and bool(name) and not Path(name).is_absolute() and ".." not in Path(name).parts, "Map must be a relative contained path")
    points = request["sample_points_mm"]
    require(isinstance(points, list) and len(points) > 0, "Empty/malformed samples")
    array(points, (len(points), 3), "query points")
    tracks = request["tracks"]
    require(isinstance(tracks, list) and 0 < len(tracks) <= 100, "Empty/oversized tracks")
    ids = set()
    for t in tracks:
        keys(t, "id position_mm momentum_mev_c charge_e length_mm max_step_mm method epsilon_rel_max delta_chord_mm", "track request")
        require(isinstance(t["id"], str) and t["id"] and t["id"] not in ids, "Duplicate/invalid track ID")
        ids.add(t["id"])
        array(t["position_mm"], (3,), "initial position")
        require(np.linalg.norm(array(t["momentum_mev_c"], (3,), "initial momentum")) > 0, "Zero momentum")
        require(type(t["charge_e"]) in (int, float) and t["charge_e"] in (-1, 0, 1), "Invalid charge")
        for key in ("length_mm", "max_step_mm", "epsilon_rel_max", "delta_chord_mm"):
            require(number(t[key], key) > 0, f"Nonpositive {key}")
        require(t["epsilon_rel_max"] < 1 and t["method"] in ("fixed", "adaptive"), "Invalid integration method/tolerance")


def field_endpoint_envelope(track, component_bound, gradient):
    """Comparison majorant X'' <= a deltaB + a ||G|| X; magnetic rotation is skew-symmetric."""
    a = KAPPA * abs(track["charge_e"]) / np.linalg.norm(track["momentum_mev_c"])
    length = track["length_mm"]
    db = np.linalg.norm(component_bound)
    rate = math.sqrt(a * np.linalg.norm(gradient, ord=2))
    if rate == 0:
        return 0.5 * a * db * length ** 2, a * db * length
    z = rate * length
    return a * db * 2 * math.sinh(z / 2) ** 2 / rate ** 2, a * db * math.sinh(z) / rate


def integration_budget(track, bmax):
    """A-priori fourth-order/local-error screening, not a promised bound on a transport integrator."""
    length = track["length_mm"]
    curvature = KAPPA * abs(track["charge_e"]) * bmax / np.linalg.norm(track["momentum_mev_c"])
    if not curvature:
        return 2e-4, 2e-7
    if track["method"] == "fixed":
        scale = 0.1 * (curvature * track["max_step_mm"]) ** 4
    else:
        scale = 20 * track["epsilon_rel_max"]
    return 2e-4 + length * scale, 2e-7 + max(1.0, length * curvature) * scale


def fixed_step_count(length_mm, cap_mm):
    """Replay only the probe's double/CGS path bookkeeping, not its field or ODE integration."""
    length, cap = float(length_mm) * 0.1, float(cap_mm) * 0.1
    require(math.isfinite(length) and math.isfinite(cap) and length > 0 and cap > 0,
            "Invalid fixed-step length/cap")
    path, count = 0.0, 0
    while path < length:
        path += min(length - path, cap)
        count += 1
        require(count <= 200000, "Fixed-step bookkeeping exceeded work budget")
        if length - path <= 32 * np.finfo(np.float64).eps * length:
            break
    return count


def _validate_result(request, result, expected):
    validate_request(request)
    require(expected.get("format") == EXPECTED_FORMAT and expected.get("request_sha256") == digest(request), "Wrong oracle sidecar/request identity")
    keys(result, "format input celeritas_version device samples tracks", "result")
    require(result["format"] == RESULT_FORMAT, "Wrong result format")
    require(canonical(result["input"]) == canonical(request), "Result input is not the exact request")
    require(result["celeritas_version"] == "0.6.3", "Wrong Celeritas version")
    device = result["device"]
    keys(device, "name capability synchronized", "device")
    require(isinstance(device["name"], str) and "NVIDIA" in device["name"] and "A100" in device["name"], "Missing A100 CUDA device/no CPU fallback permitted")
    require(type(device["capability"]) is int and device["capability"] == 80 and device["synchronized"] is True, "Wrong capability or unsynchronized device")
    for section in ("samples", "tracks"):
        keys(result[section], "host device", section)
        count = len(request["sample_points_mm"] if section == "samples" else request["tracks"])
        for backend in ("host", "device"):
            rows = result[section][backend]
            require(isinstance(rows, list) and len(rows) == count, f"{section}/{backend}: missing/extra records")
    ns = len(request["sample_points_mm"])
    truth_field = array(expected["sample_field_tesla"], (ns, 3), "oracle samples")
    require(np.allclose(truth_field, field_at(request["sample_points_mm"], *model(expected["family"])), rtol=0, atol=1e-15), "Oracle field values disagree with independent analytic model")
    require(isinstance(expected["sample_valid"], list) and len(expected["sample_valid"]) == ns and all(type(v) is bool for v in expected["sample_valid"]), "Malformed oracle validity")
    validity = np.array(expected["sample_valid"], dtype=bool)
    require(np.any(validity) and np.any(~validity), "Fixture lacks valid/rejected query controls")
    margins = array(expected["sample_margin_mm"], (ns,), "oracle margins")
    require(len(expected["tracks"]) == len(request["tracks"]), "Oracle track count mismatch")
    b0, gradient = model(expected["family"])
    data = make_map(expected["family"], expected["resolution"])
    require(digest(data) == expected["map_sha256"], "Oracle map identity mismatch")
    lo, hi, n, h = grid_geometry(data)
    bounds = dict(zip(("host", "device"), field_error_bounds(data, b0, gradient)))
    field_scale = np.abs(b0) + np.maximum(abs(lo), abs(hi)) @ abs(gradient)
    bmax = np.linalg.norm(field_scale)
    margin_allowance = 2e-6 + 32 * np.finfo(np.float32).eps * max(np.max(abs(lo)), np.max(abs(hi)))
    precision = expected["reference_precision"]
    require(0 <= number(precision["position_disagreement_mm"], "reference position precision") <= REF_POSITION_MM and 0 <= number(precision["momentum_relative_disagreement"], "reference momentum precision") <= REF_MOMENTUM_REL, "Reference precision not established")
    summary = dict(scope=expected["scope"], device=device, reference_precision=precision, fields={}, trajectories={}, convergence={})
    sampled = {}
    endpoint_data = {}
    for backend in ("host", "device"):
        observed = np.zeros((ns, 3))
        for i, row in enumerate(result["samples"][backend]):
            label = f"samples/{backend}/{i}"
            keys(row, "field_tesla valid field_evaluations guard_violations numerical_failures min_margin_mm completed_tag", label)
            require(integer(row["numerical_failures"], label) == 0, f"{label}: nonfinite field evaluation")
            require(type(row["completed_tag"]) is int and row["completed_tag"] == TAG, f"{label}: missing completion tag")
            require(type(row["valid"]) is bool and row["valid"] == bool(validity[i]), f"{label}: wrong guard validity")
            evaluations = integer(row["field_evaluations"], label)
            violations = integer(row["guard_violations"], label)
            margin = number(row["min_margin_mm"], label)
            require(abs(margin - margins[i]) <= margin_allowance, f"{label}: inconsistent inset margin")
            if validity[i]:
                require(evaluations == 1 and violations == 0, f"{label}: accepted query accounting")
                observed[i] = array(row["field_tesla"], (3,), label)
            else:
                require(evaluations == 0 and violations == 1 and margin < 0, f"{label}: rejected query accessed field or lacks guard violation")
                require(row["field_tesla"] == [None, None, None], f"{label}: rejected query must return null field")
        errors = abs(observed[validity] - truth_field[validity])
        require(np.all(errors <= bounds[backend]), f"{backend}: field mismatch exceeds a-priori interpolation/float bound")
        sampled[backend] = observed
        summary["fields"][backend] = dict(valid_queries=int(validity.sum()), rejected_queries=int((~validity).sum()), max_abs_error_tesla=errors.max(axis=0).tolist(), rms_error_tesla=np.sqrt(np.mean(errors ** 2, axis=0)).tolist(), p99_abs_error_tesla=np.quantile(errors, 0.99, axis=0).tolist(), acceptance_bound_tesla=bounds[backend].tolist())
        metrics = []
        for track, meta, row in zip(request["tracks"], expected["tracks"], result["tracks"][backend]):
            label = f"tracks/{backend}/{track['id']}"
            require(meta["id"] == track["id"] and meta["expected_status"] in (0, 1), "Oracle track identity/status mismatch")
            keys(row, "end_position_mm end_momentum_mev_c path_mm steps field_evaluations guard_violations numerical_failures min_margin_mm momentum_relative_drift max_step_mm_used status completed_tag", label)
            require(integer(row["numerical_failures"], label) == 0, f"{label}: numerical failure is not a boundary rejection")
            require(type(row["completed_tag"]) is int and row["completed_tag"] == TAG, f"{label}: missing completion tag")
            status = integer(row["status"], label)
            require(status == meta["expected_status"], f"{label}: unexpected integration status {status}")
            steps = integer(row["steps"], label)
            evaluations = integer(row["field_evaluations"], label)
            violations = integer(row["guard_violations"], label)
            path = number(row["path_mm"], label)
            margin = number(row["min_margin_mm"], label)
            drift = number(row["momentum_relative_drift"], label)
            used = number(row["max_step_mm_used"], label)
            path_allowance = 1e-6 + 5e-9 * track["length_mm"]
            require(0 <= path <= track["length_mm"] + path_allowance and drift >= 0 and 0 <= used <= track["max_step_mm"] * (1 + 1e-7) + 1e-8, f"{label}: invalid path, drift or step cap")
            if status == 1:
                require(violations == 1 and margin < 0 and steps == 1 and evaluations == 1
                        and path == 0 and used == 0 and drift == 0, f"{label}: wrong first-trial crossing accounting")
                retained_x = array(row["end_position_mm"], (3,), label)
                retained_p = array(row["end_momentum_mev_c"], (3,), label)
                require(np.linalg.norm(retained_x - track["position_mm"]) <= 1e-9
                        and np.array_equal(retained_p, track["momentum_mev_c"]), f"{label}: rejected trial altered retained state")
                metrics.append(dict(id=track["id"], expected_rejection=True, guard_violations=violations, field_evaluations=evaluations))
                continue
            require(abs(path - track["length_mm"]) <= path_allowance and steps > 0 and used > 0, f"{label}: incomplete path/progress")
            require(path <= steps * used + path_allowance, f"{label}: impossible path/step accounting")
            if track["method"] == "fixed":
                expected_steps = fixed_step_count(track["length_mm"], track["max_step_mm"])
                require(steps == expected_steps and evaluations == 7 * steps, f"{label}: fixed-step/evaluation accounting")
            require(violations == 0 and margin >= -margin_allowance and evaluations > 0, f"{label}: accepted trajectory missing valid field access")
            end = array(row["end_position_mm"], (3,), label)
            momentum = array(row["end_momentum_mev_c"], (3,), label)
            p0 = np.linalg.norm(track["momentum_mev_c"])
            pn = np.linalg.norm(momentum)
            require(pn > 0, f"{label}: zero endpoint momentum")
            actual_drift = abs(pn / p0 - 1)
            require(actual_drift <= MOMENTUM_DRIFT_MAX and actual_drift <= drift + 5e-8 and drift <= MOMENTUM_DRIFT_MAX, f"{label}: momentum conservation/accounting failure")
            truth_x = array(meta["end_position_mm"], (3,), "oracle endpoint")
            truth_p = array(meta["end_momentum_mev_c"], (3,), "oracle momentum")
            position_error = float(np.linalg.norm(end - truth_x))
            direction_error = float(np.linalg.norm(momentum / pn - truth_p / np.linalg.norm(truth_p)))
            field_x, field_u = field_endpoint_envelope(track, bounds[backend], gradient)
            integration_x, integration_u = integration_budget(track, bmax)
            require(position_error <= field_x + integration_x + REF_POSITION_MM, f"{label}: position error exceeds separate field + integration budgets")
            require(direction_error <= field_u + integration_u + REF_MOMENTUM_REL, f"{label}: direction error exceeds separate field + integration budgets")
            metrics.append(dict(id=track["id"], group=meta["group"], method=track["method"], max_step_mm=track["max_step_mm"], epsilon_rel_max=track["epsilon_rel_max"], steps=steps, field_evaluations=evaluations, max_step_mm_used=used, position_error_mm=position_error, direction_error=direction_error, angle_error_rad=2 * math.asin(min(1.0, direction_error / 2)), momentum_relative_drift=actual_drift, field_envelope_position_mm=field_x, field_envelope_direction=field_u, integration_budget_position_mm=integration_x, integration_budget_direction=integration_u, position_excess_over_field_envelope_mm=max(0.0, position_error - field_x)))
            endpoint_data[(backend, track["id"])] = (end, momentum / pn)
        summary["trajectories"][backend] = metrics
        groups = {}
        for row in metrics:
            if row.get("expected_rejection"):
                continue
            groups.setdefault(f"{row['group']}/{row['method']}", []).append(row)
        convergence = []
        for name, rows in groups.items():
            parameter = "max_step_mm" if rows[0]["method"] == "fixed" else "epsilon_rel_max"
            rows.sort(key=lambda r: r[parameter], reverse=True)
            errors = [r["position_error_mm"] for r in rows]
            convergence.append(dict(group=name, parameter=parameter, values=[r[parameter] for r in rows], position_errors_mm=errors, direction_errors=[r["direction_error"] for r in rows], steps=[r["steps"] for r in rows], observed_nonincreasing=all(b <= a for a, b in zip(errors, errors[1:])), finest_to_coarsest_ratio=errors[-1] / errors[0] if errors[0] else None, possible_field_floor=bool(errors[-1] <= rows[-1]["field_envelope_position_mm"] + 2e-4), interpretation="Report sensitivity and interpolation floor; strict monotonicity is not required, especially for quantized CUDA textures"))
        summary["convergence"][backend] = convergence
    host_device = abs(sampled["host"][validity] - sampled["device"][validity])
    require(np.all(host_device <= bounds["host"] + bounds["device"]), "Host-device fields exceed combined bounds")
    summary["fields"]["host_vs_device"] = dict(max_abs_difference_tesla=host_device.max(axis=0).tolist(), combined_bound_tesla=(bounds["host"] + bounds["device"]).tolist())
    summary["host_vs_device_tracks"] = [dict(id=t["id"], position_difference_mm=float(np.linalg.norm(endpoint_data[("host", t["id"])][0] - endpoint_data[("device", t["id"])][0])), direction_difference=float(np.linalg.norm(endpoint_data[("host", t["id"])][1] - endpoint_data[("device", t["id"])][1]))) for t, m in zip(request["tracks"], expected["tracks"]) if m["expected_status"] == 0]
    return summary


def check_result(request, result, expected):
    """Check an in-memory result against its independent sidecar; raises ValueError on any failure."""
    try:
        canonical(result)
        return _validate_result(request, result, expected)
    except (KeyError, TypeError, IndexError, OverflowError, AttributeError) as error:
        raise ValueError(f"Malformed probe result/sidecar: {error}") from error


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant {value} in {path}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key {key} in {path}")
            result[key] = value
        return result

    try:
        return json.loads(Path(path).read_text(), parse_constant=reject_constant, object_pairs_hook=unique_object)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON {path}: {error}") from error


def write_json(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def prepare(directory):
    """Exclusively create a new directory with six maps/requests/sidecars and an immutable fixture index."""
    directory = Path(directory)
    require(not directory.exists(), "prepare requires NEW_OUTPUT; refusing to overwrite existing evidence")
    cases = [(family, resolution, build_case(family, resolution)) for family in FAMILIES for resolution in RESOLUTIONS]
    directory.mkdir(parents=True, exist_ok=False)
    entries = []
    for family, resolution, (data, request, expected) in cases:
        stem = f"{family}-{resolution}"
        entry = dict(name=stem, map=f"{stem}.map.json", request=f"{stem}.request.json", expected=f"{stem}.expected.json", result=f"{stem}.result.json")
        write_json(directory / entry["map"], data)
        write_json(directory / entry["request"], request)
        write_json(directory / entry["expected"], expected)
        entries.append(entry)
    index = dict(format="m2fs-field-probe-fixtures-v1", cases=entries, execution="Not run; build/run C++ probe separately", scope="Public synthetic field-only fixtures, not Mu2e validation")
    write_json(directory / "fixtures.json", index)
    return index


def check_directory(directory, request_name=None):
    """Replay saved pairs without executing C++; optionally select a case name or request basename."""
    directory = Path(directory)
    index = read_json(directory / "fixtures.json")
    require(isinstance(index, dict) and index.get("format") == "m2fs-field-probe-fixtures-v1", "Wrong fixture index")
    expected_names = {f"{f}-{r}" for f in FAMILIES for r in RESOLUTIONS}
    entries = index.get("cases")
    require(isinstance(entries, list) and len(entries) == len(expected_names), "Empty/incomplete fixture suite")
    require({e.get("name") for e in entries if isinstance(e, dict)} == expected_names, "Duplicate/missing fixture identities")
    reports = {}
    for entry in entries:
        stem = entry["name"]
        require(entry == dict(name=stem, map=f"{stem}.map.json", request=f"{stem}.request.json", expected=f"{stem}.expected.json", result=f"{stem}.result.json"), "Noncanonical fixture paths")
        if request_name is not None and request_name not in (stem, entry["request"]):
            continue
        data, request, expected = (read_json(directory / entry[k]) for k in ("map", "request", "expected"))
        family, resolution = stem.split("-")
        fresh_map, fresh_request, fresh_expected = build_case(family, resolution)
        require(digest(data) == digest(fresh_map) and digest(request) == digest(fresh_request), "Fixture was modified after preparation")
        # Recompute truth at replay; permit only the declared reference precision for SciPy version differences.
        saved_tracks = expected.get("tracks", [])
        require(len(saved_tracks) == len(fresh_expected["tracks"]), "Modified oracle tracks")
        for saved, fresh in zip(saved_tracks, fresh_expected["tracks"]):
            for key, tolerance in (("end_position_mm", REF_POSITION_MM), ("end_momentum_mev_c", REF_MOMENTUM_REL)):
                if key in fresh:
                    scale = 1.0 if key == "end_position_mm" else np.linalg.norm(fresh[key])
                    require(np.linalg.norm(array(saved[key], (3,), "saved reference") - fresh[key]) <= tolerance * scale, "Modified/inaccurate saved reference")
                    saved[key] = fresh[key]
        expected["reference_precision"] = fresh_expected["reference_precision"]
        require(canonical(expected) == canonical(fresh_expected), "Modified oracle model/budgets/metadata")
        reports[stem] = check_result(request, read_json(directory / entry["result"]), fresh_expected)
    require(bool(reports), "No selected request/result pairs")
    resolution_comparison = []
    for family in FAMILIES:
        coarse, fine = reports.get(f"{family}-coarse"), reports.get(f"{family}-refined")
        if coarse and fine:
            trajectories = []
            for backend in ("host", "device"):
                fine_tracks = {t["id"]: t for t in fine["trajectories"][backend] if not t.get("expected_rejection")}
                for t in coarse["trajectories"][backend]:
                    if t.get("expected_rejection"):
                        continue
                    refined = fine_tracks[t["id"]]
                    trajectories.append(dict(backend=backend, id=t["id"], coarse_position_error_mm=t["position_error_mm"], refined_position_error_mm=refined["position_error_mm"], coarse_direction_error=t["direction_error"], refined_direction_error=refined["direction_error"], coarse_field_envelope_position_mm=t["field_envelope_position_mm"], refined_field_envelope_position_mm=refined["field_envelope_position_mm"]))
            resolution_comparison.append(dict(family=family, device_coarse_max_abs_error_tesla=coarse["fields"]["device"]["max_abs_error_tesla"], device_refined_max_abs_error_tesla=fine["fields"]["device"]["max_abs_error_tesla"], trajectories=trajectories, interpretation="Refinement halves affine cell variation and its texture bound; observed errors need not be strictly monotone"))
    return dict(format="m2fs-field-probe-check-v1", passed=True, cases=reports, resolution_comparison=resolution_comparison, scope="Synthetic diagnostic acceptance; no final Mu2e tolerance claim")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--request", help="Check one fixture stem or request basename; default checks all six")
    args = parser.parse_args(argv)
    try:
        require(args.action == "check" or args.request is None, "--request is only meaningful with check")
        result = prepare(args.directory) if args.action == "prepare" else check_directory(args.directory, args.request)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError) as error:
        parser.exit(1, f"field-probe oracle: {error}\n")


if __name__ == "__main__":
    main()
