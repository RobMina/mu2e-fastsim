"""Full-frame identity adapter around unchanged diagnostic keyed response kernels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import frame_registry as F, keyed_optical_smoke as M
from . import replay_sipm_control as S, replay_waveform_control as W
from . import saved_photon_downstream as D

CONTRACT = "frozen-registry-keyed-response-v1"


def bind(registry, expected_digest):
    if F.digest(registry) != expected_digest:
        raise ValueError("frozen frame registry digest changed")


def conditions(registry, selected, calibration, expected_digest, nbars=1280):
    """All status-live detector channels on each admitted frame, even without optical photons."""
    bind(registry, expected_digest)
    if selected.duplicated(list(F.FRAME_KEYS)).any() or not len(selected):
        raise ValueError("nonempty unique selected frame census required")
    F.encode(selected, registry)
    cols = ["input_sha256", "chan", "pedestal", "pulsearea", "pulseheight", "timeoffset", "status", "roc", "feb"]
    c = calibration[cols].copy()
    if c.duplicated(["input_sha256", "chan"]).any():
        raise ValueError("duplicate calibration identity")
    if type(nbars) is not int or not 0 < nbars * 4 < W.MULT:
        raise ValueError("invalid geometry channel count")
    for digest, sub in c.groupby("input_sha256", sort=False):
        if not np.array_equal(np.sort(sub.chan), np.arange(nbars * 4)):
            raise ValueError("incomplete calibration channel domain")
    for key in ("chan", "status", "roc", "feb"):
        if c[key].dtype.kind not in "iu" or np.any(c[key] < 0):
            raise ValueError("invalid integer calibration")
    if not np.isfinite(c[cols[1:]].to_numpy()).all() or (c[["pulsearea", "pulseheight"]] <= 0).any().any():
        raise ValueError("invalid calibration")
    if not set(selected.input_sha256) <= set(c.input_sha256):
        raise ValueError("missing input calibration")
    out = selected[[*F.FRAME_KEYS, "pbtime", "intensity"]].merge(c, on="input_sha256", how="left", validate="many_to_many")
    out = out[(out.status & 5) == 0].reset_index(drop=True)
    out["w"] = -out.pbtime.to_numpy().astype(np.float32).astype(float)
    return out


def admit(q, cond, registry, expected_digest, optical_seed):
    bind(registry, expected_digest)
    if not (q.draw_seed == optical_seed).all():
        raise ValueError("mixed or unexpected optical seed")
    np.testing.assert_array_equal(q.photon_id, M.roots(q))
    encoded_q = F.encode(q, registry)
    encoded_c = F.encode(cond, registry)
    if encoded_c.duplicated(["event", "chan"]).any():
        raise ValueError("duplicate frame/channel conditions")
    if any(q[name].dtype != bool for name in ["live_channel", *(f"{a}_kept" for a in M.P.ARMS)]):
        raise ValueError("invalid optical masks")
    present = pd.MultiIndex.from_frame(encoded_q[["event", "chan"]]).isin(pd.MultiIndex.from_frame(encoded_c[["event", "chan"]]))
    if not np.array_equal(q.live_channel, present):
        raise ValueError("optical live mask disagrees with authoritative conditions")
    if any((q[f"{a}_kept"] & ~q.live_channel).any() for a in M.P.ARMS):
        raise ValueError("dead photon admitted by timing arm")
    return encoded_q, encoded_c


def compare_partition(full, subset, registry, expected_digest):
    """Exact structural stage comparisons; fit differences remain measured, never tolerance-promoted."""
    bind(registry, expected_digest)
    result = {}
    selected = subset["phases"].frame_id.unique()
    for name, (keys, fields) in D.STAGES.items():
        a, b = full[name], subset[name]
        left = F.encode(a[a.frame_id.isin(selected)].drop(columns="frame_id"), registry)
        right = F.encode(b.drop(columns="frame_id"), registry)
        result[name] = D.compare_rows(left, right, keys, fields)[0]
    return result


def run_arm(q, cond, registry, expected_digest, optical_seed, response_seed, arm, pmap, sipm_cfg, waveform_cfg):
    """Execute one whole-frame arm; never rebuild registry or cut the frame's history."""
    eq, ec = admit(q, cond, registry, expected_digest, optical_seed)
    seed = F.helper_seed(registry, response_seed)
    dark = S.thermal_candidates(ec, sipm_cfg, seed, start=100., end=1500.)
    products, counts = D.run_arm(eq, ec, pmap, sipm_cfg, waveform_cfg, seed, arm, dark)
    restored = {name: F.restore(frame, registry) for name, frame in products.items()}
    return restored, {**counts, "contract": CONTRACT, "registry_sha256": expected_digest,
                      "response_seed": int(response_seed), "helper_seed": seed, "optical_seed": int(optical_seed),
                      "thermal_domain": "all supplied status-live detector channels", "channels": len(ec),
                      "pulse_fit_partition_exact_certified": False}
