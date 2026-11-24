"""Design muon sample generated on the device: CRY showers (mu2efastsim.cry_device, a CosmicCRY preset) carried through
the overburden by the device stage-C transport (entry_jax), then the CE-like proxy (sample.crv_entry_states).

The CRY source runs in chunks of accepted events; each chunk's selected primaries cross to the host once (the existing
CRY boundary), the preselected muons go back to the device for the world transport, and only the proxy muons are kept.
entry_compare compares two such samples (or a device sample with the host one) per generated CRY event.
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import fom as F
from . import sample as S


def build(preset, gdml, counters, events, seed, *, chunk=200_000, proxy=F.PROXY, require_gpu=True,
          selected_capacity=8192, transport_batch=4096, backend="jax"):
    """dict of the sample arrays (as sample.crv_entry_states, concatenated over chunks) plus generation provenance."""
    from mu2efastsim.cosmic_cry import PRESETS
    from mu2efastsim.cry_device import DeviceCosmicSource
    from mu2efastsim import precision as dprec
    if events < 1 or chunk < 1:
        raise ValueError("events and chunk must be positive")
    config = PRESETS[preset](gdml)
    t0 = time.perf_counter()
    ew = S.EntryWorld(gdml, counters)
    if backend == "jax":
        ew.device(transport_batch)
    setup_s = time.perf_counter() - t0
    parts, live, gen_s, tr_s, receipt = [], 0.0, 0.0, 0.0, {}
    with DeviceCosmicSource(config, seed=seed, require_gpu=require_gpu, selected_capacity=selected_capacity) as src:
        for i, start in enumerate(range(0, events, chunk)):
            n = min(chunk, events - start)
            a = time.perf_counter()
            gen, subruns, receipt = src.generate(n, subrun=i + 1)
            b = time.perf_counter()
            rng = np.random.default_rng(np.random.SeedSequence([seed, i, 0x454E54]))
            parts.append(S.crv_entry_states(gen, gdml, counters, rng, proxy, backend=backend, prepared=ew))
            tr_s += time.perf_counter() - b
            gen_s += b - a
            live += float(subruns.livetime.sum())
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("origin", "direction", "p", "p_generated", "event_key")}
    for k in ("n_generated_muons", "n_generated_events", "n_preselected", "n_ranged_out", "n_selected_generated_lines"):
        out[k] = int(sum(p[k] for p in parts))
    out["y_plane"] = parts[0]["y_plane"]
    out["live_seconds"] = live
    keep = ("generator", "sampling", "device", "particle_capacity", "selected_capacity", "device_batches",
            "device_setup_s", "device_kernel_wall_s", "device_copy_wall_s", "rng", "cosmic_config")
    out["provenance"] = dict(preset=preset, seed=seed, events=events, chunk=chunk, transport=backend,
                             transport_batch=transport_batch,
                             concrete_substituted=ew.device().substituted if backend == "jax" else None, precision=dprec.provenance(),
                             setup_s=setup_s, generation_s=gen_s, transport_s=tr_s,
                             cry={k: receipt[k] for k in keep if k in receipt})
    return out


def save(path, m, proxy, states="crv_entry"):
    extra = {k: v for k, v in m.items() if k.startswith("n_") or k in ("y_plane", "live_seconds")}
    np.savez(path, origin=m["origin"], direction=m["direction"], p=m["p"], p_generated=m["p_generated"],
             event_key=m["event_key"], proxy=json.dumps(proxy), states=states,
             provenance=json.dumps(m.get("provenance", {}), default=str), **extra)


BINS = {"p": np.geomspace(100.0, 1e6, 41), "p_generated": np.geomspace(100.0, 1e6, 41),
        "coszenith": np.linspace(0.0, 1.0, 41), "azimuth": np.linspace(-np.pi, np.pi, 37),
        "x": np.linspace(-6000.0, -1800.0, 43), "z": np.linspace(2000.0, 15200.0, 45)}


def _columns(z):
    o, d = z["origin"], z["direction"]
    return {"p": z["p"], "p_generated": z["p_generated"] if "p_generated" in z else None, "coszenith": -d[:, 1],
            "azimuth": np.arctan2(d[:, 2], d[:, 0]), "x": o[:, 0], "z": o[:, 2]}


def entry_compare(paths, n_events=None):
    """Per-generated-event rate and spectra of CE-proxy entry samples against the first (the reference).

    n_events: generated CRY events of each sample where the npz does not record n_generated_events. Spectra: counts per
    generated event in BINS (under/overflow included), Pearson chi2 of the difference with Poisson errors on both
    arms (bins with >= 10 entries in both), and the two-sample KS p-value of the unbinned shapes."""
    from scipy.stats import chi2 as chi2_dist, ks_2samp
    samples = []
    for i, path in enumerate(paths):
        z = dict(np.load(path, allow_pickle=False))
        ne = int(z["n_generated_events"]) if "n_generated_events" in z else (n_events[i] if n_events else None)
        if ne is None:
            raise ValueError(f"{path}: no n_generated_events; pass the generated event count")
        samples.append((path, z, ne))
    ref_path, ref, ref_n = samples[0]
    rc = _columns(ref)
    out = {"reference": ref_path, "samples": []}
    for path, z, ne in samples:
        c = _columns(z)
        n = len(z["p"])
        rate, rate_ref = n / ne, len(ref["p"]) / ref_n
        ratio = rate / rate_ref
        rec = {"path": path, "selected": n, "generated_events": ne, "rate_per_event": rate,
               "rate_ratio": ratio, "rate_ratio_err": ratio * np.sqrt(1 / max(n, 1) + 1 / max(len(ref["p"]), 1)),
               "spectra": {}}
        for k, edges in BINS.items():
            if c[k] is None or rc[k] is None:
                continue
            e = np.r_[-np.inf, edges, np.inf]
            h, hr = np.histogram(c[k], e)[0], np.histogram(rc[k], e)[0]
            a, b = h / ne, hr / ref_n
            var = h / ne ** 2 + hr / ref_n ** 2
            use = (h >= 10) & (hr >= 10)
            chi2 = float(np.sum((a - b)[use] ** 2 / var[use]))
            ndf = int(use.sum())
            rec["spectra"][k] = {"edges": edges.tolist(), "per_event": a.tolist(), "reference_per_event": b.tolist(),
                                 "chi2": chi2, "ndf": ndf, "chi2_p": float(chi2_dist.sf(chi2, ndf)) if ndf else None,
                                 "ks_p": float(ks_2samp(c[k], rc[k]).pvalue) if n and len(rc[k]) else None,
                                 "mean": float(np.mean(c[k])) if n else None,
                                 "reference_mean": float(np.mean(rc[k])) if len(rc[k]) else None}
        out["samples"].append(rec)
    return out


def subset(path, chunks, out):
    """The first `chunks` CRY chunks of a device sample (subruns 1..chunks) as its own sample: the same events as a
    shorter run with the same seed and chunk size."""
    z = dict(np.load(path, allow_pickle=False))
    prov = json.loads(str(z["provenance"]))
    chunk, events = int(prov["chunk"]), int(prov["events"])
    if chunks < 1 or chunks * chunk > events:
        raise ValueError(f"{path} holds {events} events in chunks of {chunk}: cannot take {chunks} chunks")
    keep = z["event_key"][:, 1] <= chunks
    m = {k: z[k][keep] for k in ("origin", "direction", "p", "p_generated", "event_key")}
    m["n_generated_events"] = chunks * chunk
    m["y_plane"] = float(z["y_plane"])
    m["provenance"] = dict(prov, subset_of=str(path), chunks=chunks, events=chunks * chunk)
    save(out, m, json.loads(str(z["proxy"])))
    return m


def summary_lines(cmp):
    """Compact text of an entry_compare result: per sample the rate ratio and each spectrum's chi2/ndf and KS p."""
    lines = []
    for s in cmp["samples"]:
        lines.append(f"{s['path']}: {s['selected']} of {s['generated_events']} events, rate ratio "
                     f"{s['rate_ratio']:.4f} +- {s['rate_ratio_err']:.4f}")
        for k, v in s["spectra"].items():
            lines.append(f"  {k:12s} chi2/ndf {v['chi2']:.1f}/{v['ndf']} (p {v['chi2_p']})  KS p {v['ks_p']}  "
                         f"mean {v['mean']} vs {v['reference_mean']}")
    return lines
