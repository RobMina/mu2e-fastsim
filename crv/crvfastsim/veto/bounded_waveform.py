"""Pre-allocation guards around the frozen diagnostic waveform synthesis helper."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from ..reco import waveform as WF
from . import replay_waveform_control as W

CONTRACT = "diagnostic-waveform-cell-admission-v1"
MAX_CHARGE_RATIO = 1_000_000.
NOISE_SIGMA_BOUND = 38.


@dataclass(frozen=True)
class Limits:
    charges: int = 200_000
    reach_cells: int = 2_000_000
    waveform_cells: int = 2_000_000
    max_length: int = 1024
    template_cells: int = 100_000

    def validate(self):
        if any(type(v) is not int or not 0 < v <= 10**8 for v in vars(self).values()):
            raise ValueError("invalid waveform resource limits")


def _real(value):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        number = float(value)
        return np.isfinite(number) and (number != 0 or value == 0)
    except (OverflowError, TypeError, ValueError):
        return False


def _columns(frame, names, *, identities=False):
    for name in names:
        if name not in frame:
            raise ValueError(f"missing waveform column {name}")
        values = frame[name].to_numpy()
        kinds = "iu" if identities else "iuf"
        if values.dtype.kind not in kinds or values.dtype.itemsize > 8:
            raise ValueError(f"invalid numeric dtype for {name}")
        if (values.dtype.kind == "f" or name == "time") and values.dtype.itemsize != 8:
            raise ValueError(f"waveform time requires 64 bits and floating columns require float64: {name}")
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite waveform column {name}")


def _amplitude(q, cid, cfg, limits):
    charge = q.charge.to_numpy(dtype=np.float64)
    reference = float(cfg.single_pe_reference_charge)
    if float(charge.max(initial=0)) / MAX_CHARGE_RATIO > reference:
        raise ValueError("charge/reference ratio limit exceeded")
    ratio = charge / reference
    if np.any(ratio > MAX_CHARGE_RATIO):
        raise ValueError("charge/reference ratio limit exceeded")
    template = np.asarray(WF.single_pe_template(cfg.single_pe_file, cfg.single_pe_precision,
                                               cfg.single_pe_stretch, cfg.single_pe_max_time))
    if template.ndim != 1 or not 0 < template.size <= limits.template_cells:
        raise ValueError("template cell limit exceeded")
    if template.dtype.kind != "f" or template.dtype.itemsize != 8 or not np.isfinite(template).all():
        raise ValueError("invalid waveform template")
    peak = float(np.abs(template).max())
    total = float(np.bincount(cid, weights=ratio).max(initial=0))
    # The keyed Box-Muller radius is below 38 even when the uniform is zero.
    safe_float = np.finfo(float).max / 4
    if peak > safe_float / max(1., total, len(q)) or cfg.noise > safe_float / NOISE_SIGMA_BOUND:
        raise ValueError("waveform amplitude outside safe numeric range")
    voltage_bound = peak * total + float(cfg.noise) * NOISE_SIGMA_BOUND
    # Reserve one count for rounding before the frozen helper's int16 narrowing.
    adc_budget = (np.iinfo(np.int16).max - int(cfg.pedestal) - 1.) / float(cfg.adc_conversion)
    if voltage_bound > adc_budget:
        raise ValueError("waveform amplitude outside safe ADC range")


def admit(charges, channels, seed, cfg=WF.WaveformConfig(), *, limits=Limits()):
    """Conservative cell and int16 bounds; numeric columns use integers or float64."""
    limits.validate()
    if len(charges) > min(limits.charges, 2_000_000):
        raise ValueError("charge count limit exceeded")
    for name in ("single_pe_max_time", "single_pe_precision", "single_pe_stretch", "single_pe_reference_charge", "adc_conversion"):
        value = getattr(cfg, name)
        if not _real(value) or value <= 0:
            raise ValueError(f"invalid waveform configuration: {name}")
    for name in ("noise", "min_voltage"):
        value = getattr(cfg, name)
        if not _real(value) or value < 0:
            raise ValueError(f"invalid waveform configuration: {name}")
    for name in ("pedestal", "min_adc", "max_adc"):
        value = getattr(cfg, name)
        if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or not 0 <= value <= 4095:
            raise ValueError(f"invalid ADC configuration: {name}")
    if not cfg.min_adc <= cfg.pedestal <= cfg.max_adc:
        raise ValueError("invalid ADC configuration ordering")
    max_length = min(limits.max_length, 1024)
    if type(cfg.samples_zs) is not int or not 1 <= cfg.samples_zs <= max_length:
        raise ValueError("invalid zero-suppression length")
    if float(cfg.single_pe_max_time) / float(cfg.single_pe_precision) > limits.template_cells:
        raise ValueError("template cell limit exceeded")
    reach = int(np.ceil(float(cfg.single_pe_max_time) / 12.5)) + 2
    if reach > min(limits.reach_cells, max_length) or len(charges) * reach > limits.reach_cells:
        raise ValueError("charge-by-reach cell limit exceeded")
    if (reach + 2) * 12.5 / float(cfg.single_pe_precision) > 2**52:
        raise ValueError("template index outside safe waveform domain")
    _columns(charges, ("event", "chan"), identities=True)
    _columns(charges, ("time", "charge"))
    _columns(channels, ("event", "chan", "roc", "feb"), identities=True)
    phases = W.phase_table(channels, seed).set_index("label")
    q = charges.assign(label=W.labelled(charges)).sort_values(["label", "time"], kind="stable")
    if (q.charge < 0).any() or not q.label.isin(phases.index).all():
        raise ValueError("invalid charges/conditions")
    if ((q.time < -(2**40)) | (q.time > 2**40)).any():
        raise ValueError("charge time outside safe waveform domain")
    label, t = q.label.to_numpy(), q.time.to_numpy()
    cid = WF.charge_clusters(label, t, cfg)
    clusters = int(cid[-1]) + 1 if len(cid) else 0
    first = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]]) if clusters else np.zeros(0, np.int64)
    last = np.r_[first[1:] - 1, len(cid) - 1] if clusters else np.zeros(0, np.int64)
    phase = phases.loc[label[first], "phase"].to_numpy()
    start = np.ceil((t[first] - 12.5 - phase) / 12.5) * 12.5 + phase
    estimate = np.floor((t[last] - t[first] + cfg.single_pe_max_time) / 12.5)
    jmax = np.maximum(np.ceil((t[last] - start) / 12.5), 0)
    upper = np.maximum(estimate, jmax + reach)
    length = int(upper.max(initial=0)) + cfg.samples_zs if clusters else 0
    if length > max_length or clusters * length > limits.waveform_cells:
        raise ValueError("cluster-by-sample cell limit exceeded")
    _amplitude(q, cid, cfg, limits)
    return {"contract": CONTRACT, "charges": len(q), "clusters": clusters, "reach": reach,
            "charge_reach_cells": len(q) * reach, "waveform_length_upper": length,
            "waveform_cells_upper": clusters * length, "limits": vars(limits).copy(),
            "peak_memory_certified": False}


def synthesize(charges, channels, seed, cfg=WF.WaveformConfig(), window=(200., 1500.), *, limits=Limits()):
    try:
        bounds = tuple(window)
    except TypeError as exc:
        raise ValueError("invalid waveform window") from exc
    if len(bounds) != 2 or not all(_real(value) for value in bounds) or bounds[0] > bounds[1]:
        raise ValueError("invalid waveform window")
    receipt = admit(charges, channels, seed, cfg, limits=limits)
    products = W.synthesize(charges, channels, seed, cfg, bounds)
    return products, receipt
