"""Whole-frame diagnostic response with immutable registry keys and explicit allocation budgets."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import frame_registry as F, qualified_response as Q, keyed_optical_smoke as M
from . import bounded_sipm as S, bounded_waveform as W, diagnostic_fit as P
from . import saved_photon_downstream as D

CONTRACT = "bounded-qualified-diagnostic-response-v1"
STAGES = ("frames", "conditions", "candidates", "attempts", "charges", "digis", "pulses", "samples",
          "clusters", "phases", "waveforms", "fit_samples")
KEYS = {
    "frames": [], "conditions": ["chan"], "candidates": ["chan", "root"],
    "attempts": ["chan", "root", "dark", "generation", "px", "py"],
    "charges": ["chan", "root", "dark", "generation", "px", "py"],
    "digis": ["chan", "starttdc", "sample"], "pulses": ["chan", "starttdc", "seq"],
    "samples": ["chan", "cluster_starttdc", "tdc"], "clusters": ["chan", "starttdc"],
    "phases": ["chan"], "waveforms": ["chan", "starttdc"], "fit_samples": ["chan", "waveform_starttdc", "tdc"],
}


@dataclass(frozen=True)
class Limits:
    frames: int = 4
    live_channels: int = 20480
    photons: int = 300000
    sipm: S.Limits = field(default_factory=S.Limits)
    waveform: W.Limits = field(default_factory=W.Limits)
    fit: P.Limits = field(default_factory=P.Limits)

    def validate(self):
        if any(type(v) is not int or not 0 < v <= 10**7 for v in (self.frames, self.live_channels, self.photons)):
            raise ValueError("invalid response limits")
        self.sipm.validate()
        self.waveform.validate()
        self.fit.validate()


def selected_frames(registry, selected, digest):
    Q.bind(registry, digest)
    if list(selected) != list(F.COLUMNS) or not len(selected) or selected.frame_id.duplicated().any():
        raise ValueError("requires nonempty exact selected registry rows")
    if selected.frame_id.dtype.kind not in "iu" or np.any(selected.frame_id < 0) or np.any(selected.frame_id >= len(registry)):
        raise ValueError("selected frame id outside registry")
    expected = registry.iloc[selected.frame_id.to_numpy(dtype=np.int64)].reset_index(drop=True)
    pd.testing.assert_frame_equal(selected.reset_index(drop=True), expected, check_exact=True)
    return expected.sort_values("frame_id").reset_index(drop=True)


def run_arm(q, registry, selected, calibration, *, registry_digest, optical_seed, response_seed,
            arm, pmap, sipm_cfg, waveform_cfg, nbars=1280, limits=Limits()):
    """Run one arm on complete frames, including all quiet status-live channels."""
    limits.validate()
    selected = selected_frames(registry, selected, registry_digest)
    if type(nbars) is not int or not 0 < nbars < 8192:
        raise ValueError("invalid detector bar count")
    if len(selected) > limits.frames or len(selected) * nbars * 4 > limits.live_channels or len(q) > limits.photons:
        raise ValueError("response frame/channel/photon cap exceeded")
    if arm not in M.P.ARMS:
        raise ValueError("unknown timing arm")
    M.K.Draws("0" * 64, optical_seed)
    seed = F.helper_seed(registry, response_seed)
    domain = pd.MultiIndex.from_frame(selected[list(F.FRAME_KEYS)])
    if not pd.MultiIndex.from_frame(q[list(F.FRAME_KEYS)]).isin(domain).all():
        raise ValueError("photon outside selected frame census")
    cond = Q.conditions(registry, selected.drop(columns="frame_id"), calibration, registry_digest, nbars)
    eq, ec = Q.admit(q, cond, registry, registry_digest, optical_seed)
    for name in ("time_historical_relative", "time_raw_wrapped_relative"):
        if not np.isfinite(eq[name]).all():
            raise ValueError("nonfinite saved optical time")
    time_key = "time_historical_relative" if arm in ("historical", "rejection_only") else "time_raw_wrapped_relative"
    optical = eq.loc[eq[f"{arm}_kept"], ["event", "chan", "photon_id", time_key]].rename(columns={time_key: "time"})
    light = S.optical_candidates(optical, pmap, sipm_cfg, seed, limits=limits.sipm)
    dark = S.thermal_candidates(ec, sipm_cfg, seed, start=100., end=1500., limits=limits.sipm)
    if len(light) + len(dark) > limits.sipm.candidates:
        raise RuntimeError("combined candidate cap exceeded before concatenation; batch incomplete")
    candidates = pd.concat([light, dark], ignore_index=True)
    charges, attempts = S.simulate_candidates(candidates, sipm_cfg, seed, end=1500., limits=limits.sipm)
    (digis, samples, clusters, phases), wave_receipt = W.synthesize(charges, ec, seed, waveform_cfg, limits=limits.waveform)
    pulses, waveforms, fit_samples = P.reconstruct(digis, ec, limits=limits.fit)
    pulses = pulses.merge(waveforms[["wf", "starttdc"]], on="wf", how="left", validate="many_to_one")
    if pulses.starttdc.isna().any():
        raise ValueError("pulse missing waveform identity")
    products = dict(conditions=ec, candidates=candidates, attempts=attempts, charges=charges, digis=D.encode_digis(digis),
                    pulses=pulses, samples=samples, clusters=clusters, phases=phases, waveforms=waveforms, fit_samples=fit_samples)
    products = {name: F.restore(table, registry) for name, table in products.items()}
    frames = selected.copy()
    for name, table in products.items():
        if name == "samples":
            table = table[table.valid]
        counts = table.groupby("frame_id").size()
        frames[name] = frames.frame_id.map(counts).fillna(0).astype(np.int64)
    products["frames"] = frames
    info = {"contract": CONTRACT, "registry_sha256": registry_digest, "optical_seed": int(optical_seed),
            "response_seed": int(response_seed), "helper_seed": seed, "arm": arm,
            "frames": len(selected), "live_channels": len(ec), "candidate_photons": len(q),
            "photons": len(optical), "dark_candidates": len(dark), "attempts": len(attempts), "charges": len(charges),
            "digis": len(digis["chan"]), "pulses": len(pulses), "limits": asdict(limits), "waveform_admission": wave_receipt,
            "fit_contract": P.CONTRACT, "thermal_domain": "all status-live detector channels, including quiet frames",
            "execution_admitted": False, "pulse_fit_partition_exact_certified": False}
    return products, info


REQUIRED = {
    "frames": {"pbtime", "intensity", *STAGES[1:]},
    "conditions": {"chan", "pbtime", "intensity", "pedestal", "pulsearea", "pulseheight", "timeoffset", "status", "roc", "feb", "w"},
    "candidates": set(S.S._CANDIDATE_COLUMNS) - {"event"},
    "attempts": set(S.S._ATTEMPT_COLUMNS) - {"event"},
    "charges": set(S.S._ATTEMPT_COLUMNS) - {"event"},
    "digis": {"chan", "starttdc", "sample", "adc"},
    "pulses": {"wf", "seq", "pe", "peph", "time", "height", "beta", "chi2", "letime", "failed", "zerondf", "ped", "pedfromdb",
               "fit_A", "fit_mu", "fit_b", "chan", "time_relative", "time_reco", "channel_ignored", "starttdc"},
    "samples": {"cluster", "chan", "tdc", "time_relative", "valid", "cluster_prekeep", "voltage_noiseless", "noise", "voltage",
                "above_zs", "zs_margin_v", "precast_adc"},
    "clusters": {"cluster", "chan", "phase", "first_charge", "last_charge", "charges", "starttdc", "length", "prekeep"},
    "phases": {"chan", "pbtime", "intensity", "pedestal", "pulsearea", "pulseheight", "timeoffset", "status", "roc", "feb", "w", "label", "phase"},
    "waveforms": {"wf", "chan", "starttdc", "n", "pedestal_used", "max_adc_minus_pedestal", "peak_margin_adc", "near_threshold"},
    "fit_samples": {"wf", "chan", "tdc", "adc"},
}


def validate_products(products):
    """Check declared coverage, mandatory fields and physical sample support before comparing."""
    if set(products) != set(STAGES):
        raise ValueError("missing response stage")
    frames = products["frames"]
    if frames.frame_id.duplicated().any() or frames.duplicated(list(F.FRAME_KEYS)).any():
        raise ValueError("duplicate response frame")
    framekeys = [*F.FRAME_KEYS, "frame_id"]
    domain = pd.MultiIndex.from_frame(frames[framekeys])
    for name, table in products.items():
        if not (REQUIRED[name] | set(framekeys)) <= set(table):
            raise ValueError("missing mandatory stage fields: " + name)
        if not pd.MultiIndex.from_frame(table[framekeys]).isin(domain).all():
            raise ValueError("stage identity outside explicit frame census")
        if name != "frames":
            physical = table[table.valid] if name == "samples" else table
            counts = physical.groupby("frame_id").size()
            np.testing.assert_array_equal(frames[name], frames.frame_id.map(counts).fillna(0).astype(np.int64))
    for sample_name, group_name, key, size in (("samples", "clusters", "cluster", "length"), ("fit_samples", "waveforms", "wf", "n")):
        samples, groups = products[sample_name], products[group_name]
        if groups[key].duplicated().any():
            raise ValueError("duplicate local group identity")
        mapping = groups[[key, "starttdc", size, *framekeys, "chan"]]
        joined = samples.merge(mapping, on=[key, *framekeys, "chan"], how="left", validate="many_to_one")
        if joined.starttdc.isna().any() or samples.duplicated([key, "tdc"]).any():
            raise ValueError("invalid sample/group identity")
        valid = (joined.tdc >= joined.starttdc) & (joined.tdc < joined.starttdc + joined[size])
        if sample_name == "samples":
            if samples.valid.dtype != bool or not np.array_equal(samples.valid, valid):
                raise ValueError("physical sample mask disagrees with cluster length")
        elif not valid.all():
            raise ValueError("fit sample outside waveform")
        counts = joined[valid].groupby(key).size()
        np.testing.assert_array_equal(groups[size], groups[key].map(counts).fillna(0).astype(np.int64))


def canonical(products, name):
    """Physical rows only; remove local ranks and nonphysical cross-cluster matrix padding."""
    if set(products) != set(STAGES):
        raise ValueError("missing response stage")
    table = products[name].copy()
    if name == "samples":
        table = table[table.valid].copy()
        identity = products["clusters"][["cluster", "starttdc"]].rename(columns={"starttdc": "cluster_starttdc"})
        table = table.merge(identity, on="cluster", how="left", validate="many_to_one")
    if name == "fit_samples":
        identity = products["waveforms"][["wf", "starttdc"]].rename(columns={"starttdc": "waveform_starttdc"})
        table = table.merge(identity, on="wf", how="left", validate="many_to_one")
    table = table.drop(columns=[c for c in ("cluster", "wf") if c in table])
    keys = [*F.FRAME_KEYS, "frame_id", *KEYS[name]]
    if table.duplicated(keys).any():
        raise ValueError("duplicate physical stage key: " + name)
    for column in table:
        if column != "input_sha256" and table[column].dtype.kind in "iufb" and not np.isfinite(table[column]).all():
            raise ValueError("nonfinite stage value: " + name + "." + column)
    return table.sort_values(keys).reset_index(drop=True)


def compare_partition(full, parts):
    """Require disjoint declared frame domains, then compare every physical stage field exactly."""
    if not parts:
        raise ValueError("empty response partition")
    validate_products(full)
    for part in parts:
        validate_products(part)
    frames = pd.concat([p["frames"] for p in parts], ignore_index=True)
    if frames.frame_id.duplicated().any() or not frames.frame_id.isin(full["frames"].frame_id).all():
        raise ValueError("duplicate or foreign partition frame")
    result = {}
    for name in STAGES:
        actual = pd.concat([canonical(p, name) for p in parts], ignore_index=True)
        expected = canonical(full, name)
        expected = expected[expected.frame_id.isin(frames.frame_id)]
        keys = [*F.FRAME_KEYS, "frame_id", *KEYS[name]]
        actual, expected = (t.sort_values(keys).reset_index(drop=True) for t in (actual, expected))
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        result[name] = {"rows": len(actual), "exact": True}
    return {"frames": len(frames), "frame_ids": sorted(int(x) for x in frames.frame_id), "stages": result,
            "all_physical_fields_exact": True, "nonphysical_sample_padding_compared": False}
