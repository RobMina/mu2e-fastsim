"""Pure bounded admission of saved stock-ADC refits; no finder or upstream execution."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import boundary_pulse_inventory as B

I, F, P = B.I, B.F, B.P
require = B.require
IDS, KEYS = B.IDS, I.KEYS
CALIBRATION = ["pedestal", "pulsearea", "pulseheight", "timeoffset", "status", "roc", "feb"]
STOCK = [*IDS, "idx", "bar", "sipm", "seq", "time", "pe", "flags"]
COMPARISON = [*IDS, "idx", "bar", "sipm", "seq", "time", "pe_stock", "flags", "chan", "wf",
              "pe_fast", "time_reco", "failed", "zerondf", "_merge", "delta_time", "delta_pe"]


def _exact(actual, expected, label):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),
            "saved storage differs: " + label)


def _bijection(left, right, keys, label):
    require(len(left) == len(right) and not left.duplicated(keys).any()
            and not right.duplicated(keys).any(), "incomplete or duplicate " + label)
    loc = pd.MultiIndex.from_frame(right[keys]).get_indexer(pd.MultiIndex.from_frame(left[keys]))
    require(np.all(loc >= 0), "unmatched " + label)
    return right.iloc[loc].reset_index(drop=True)


def _full(t, selected):
    require(not {"input_sha256", "frame_id"} & set(t), "already qualified stock/comparison")
    out = t.merge(selected[KEYS], on=IDS, how="left", validate="many_to_one", indicator="_qualified", sort=False)
    require(out._qualified.eq("both").all(), "stock/comparison outside full selected census")
    return out.drop(columns="_qualified")


def _calibration(conditions, selected, registry, calibration, nchannels):
    P._schema(calibration, ["input_sha256", "chan", *CALIBRATION], "authoritative calibration")
    require(set(calibration.input_sha256) <= set(registry.input_sha256)
            and set(selected.input_sha256) <= set(calibration.input_sha256), "calibration input domain differs")
    require(not calibration.duplicated(["input_sha256", "chan"]).any(), "duplicate calibration identity")
    for table, label in ((calibration, "calibration"), (conditions, "conditions")):
        for name, hi in (("chan", nchannels), ("status", 2**16), ("roc", 2**31), ("feb", 2**31)):
            P._integer(table[name], label + " " + name, hi)
        for name in CALIBRATION:
            P._real(table[name], label + " " + name)
        require((table[["pulsearea", "pulseheight"]] > 0).all().all(), "invalid " + label + " pulse calibration")
    require(calibration.groupby("input_sha256").size().eq(nchannels).all(), "incomplete calibration channel domain")
    c = I.qualify(conditions, selected)
    joined = c.merge(calibration[["input_sha256", "chan", *CALIBRATION]], on=["input_sha256", "chan"],
                     how="left", validate="many_to_one", suffixes=("", "_full"), indicator=True)
    require(joined._merge.eq("both").all()
            and all(np.array_equal(joined[n], joined[n + "_full"]) for n in CALIBRATION),
            "saved conditions differ from authoritative calibration")
    shift = -c.frame_id.map(selected.set_index("frame_id").pbtime).to_numpy().astype(np.float32).astype(float)
    P._real(c.w, "conditions w")
    require(np.array_equal(c.w, shift), "saved conditions bunch shift differs")


def prepare(p, w, conditions, events, stock, comparison, registry, calibration, geom, input_sha):
    """Return selected, pulses, waveforms, bridge and diagnostics without changing any input."""
    require(0 < len(events) <= 4 and len(registry) <= 9000 and len(calibration) <= 15360
            and len(conditions) <= 20480 and all(len(t) <= 2048 for t in (p, w, stock, comparison)),
            "saved refit admission row cap exceeded")
    omitted_channel_fields = not {"bar", "sipm", "chan"} & set(comparison)
    if omitted_channel_fields:
        P._schema(comparison, [n for n in COMPARISON if n not in ("bar", "sipm", "chan")], "comparison")
        P._schema(stock, STOCK, "stock")
        comparison = comparison.merge(stock[[*IDS, "idx", "bar", "sipm"]], on=[*IDS, "idx"],
                                      how="left", validate="one_to_one", indicator="_channel_join", sort=False)
        require(comparison._channel_join.eq("both").all(), "comparison stock channel identity missing")
        comparison = comparison.drop(columns="_channel_join")
        comparison["chan"] = comparison.bar.astype(np.int64) * 4 + comparison.sipm.astype(np.int64)
    for t, names, label in ((p, [], "pulses"), (w, [], "waveforms"),
                             (events, ["event", "pbtime"], "boundary events"),
                             (conditions, ["event", "chan", "w", *CALIBRATION], "conditions"),
                             (stock, STOCK, "stock"), (comparison, COMPARISON, "comparison")):
        P._schema(t, names, label)
    require(not {"input_sha256", "frame_id", "refit_idx", "stock_idx", "idx", "starttdc"} & set(p),
            "legacy pulse contains invented identity/starttdc")
    require(not {"run", "subrun", "input_sha256", "frame_id"} & set(events), "requires event-only boundary summary")
    for table in (stock, comparison):
        require(not {"input_sha256", "frame_id", "refit_idx", "stock_idx", "starttdc", "_qualified"} & set(table),
                "legacy stock/comparison contains invented bridge identity")
    P._integer(events.event, "event", 2**32)
    B.M.S.I._digest(input_sha)
    sector, _, _ = P._geometry(geom)
    nchannels = len(sector) * 4
    require(nchannels <= 5120, "geometry channel cap exceeded")
    for table, label in ((stock, "stock"), (comparison, "comparison")):
        for name in IDS:
            P._integer(table[name], label + " " + name, 2**32)
        for name, hi in (("idx", 2**31), ("bar", len(sector)), ("sipm", 4), ("seq", 2**31), ("flags", 2**31)):
            P._integer(table[name], label + " " + name, hi)
    require(comparison._merge.eq("both").all(), "comparison contains unmatched stock/refit rows")
    require(stock.pe.dtype == np.dtype("float32") and stock.time.dtype == np.dtype("float64"),
            "stock floating storage differs")
    for name in ("time", "pe"):
        P._real(stock[name], "stock " + name)
    require((stock.pe >= 0).all(), "negative stock PE")
    ordered = stock.sort_values([*IDS, "idx"])
    require(np.array_equal(ordered.idx, ordered.groupby(IDS, sort=False).cumcount()),
            "stock indices are not contiguous complete-event indices")
    require(not stock.duplicated([*IDS, "idx"]).any(), "duplicate stock collection identity")
    census = B.event_census(events, stock)
    boundary = B.inspect(p, w, conditions, census, input_sha)
    F.validate(registry)
    digest = F.digest(registry)
    wanted = census.assign(input_sha256=input_sha)
    loc = pd.MultiIndex.from_frame(registry[list(F.FRAME_KEYS)]).get_indexer(
        pd.MultiIndex.from_frame(wanted[list(F.FRAME_KEYS)]))
    require(np.all(loc >= 0), "boundary full identity outside frozen registry")
    admitted = registry.iloc[loc].reset_index(drop=True)
    with np.errstate(over="ignore"):
        pb = admitted.pbtime.to_numpy().astype(np.float32)
        saved_pb = census.pbtime.to_numpy().astype(np.float32)
    require(np.isfinite(pb).all(), "registry bunch time exceeds float32")
    _exact(saved_pb, pb, "boundary float32 pbtime")
    if "pbtime_raw" in census:
        _exact(census.pbtime_raw, admitted.pbtime, "boundary original pbtime_raw")
    selected = registry.iloc[np.sort(loc)].copy().reset_index(drop=True)
    _calibration(conditions, selected, registry, calibration, nchannels)
    linked = p.merge(w[["wf", "event", "chan", "starttdc"]], on=["wf", "event", "chan"],
                     how="left", validate="many_to_one", sort=False)
    pulses, waveforms = I.qualify(linked, selected), I.qualify(w, selected)
    require(not pulses.channel_ignored.any(), "saved refit candidate is not all-active")
    inventory = I.inspect(pulses, waveforms, selected, registry, calibration, geom)
    pulses["refit_idx"] = pulses.groupby("frame_id", sort=False).cumcount().astype(np.int64)
    fullstock, comp = _full(stock, selected), _full(comparison, selected)
    copied = _bijection(comp, fullstock, [*KEYS, "idx"], "stock/comparison collection")
    for name in stock.columns:
        target = "pe_stock" if name == "pe" else name
        require(target in comp, "comparison missing copied stock field: " + name)
        _exact(comp[target], copied[name], "stock " + name)
    P._integer(comp.chan, "comparison chan", nchannels)
    P._integer(comp.wf, "comparison wf", 2**31)
    require(np.array_equal(comp.chan, comp.bar.to_numpy(dtype=np.int64) * 4 + comp.sipm),
            "comparison stock channel differs")
    refit = _bijection(comp, pulses, [*KEYS, "wf", "seq"], "comparison/active refit waveform sequence")
    require(np.array_equal(comp.chan, refit.chan), "comparison/refit channel differs")
    for actual, expected in (("pe_fast", "pe"), ("time_reco", "time_reco"),
                             ("failed", "failed"), ("zerondf", "zerondf")):
        _exact(comp[actual], refit[expected], "refit " + actual)
    for name, expected in (("delta_time", comp.time_reco - comp.time),
                            ("delta_pe", comp.pe_fast - comp.pe_stock)):
        require(comp[name].dtype == np.dtype("float64") and np.isfinite(comp[name]).all(), "invalid " + name)
        _exact(comp[name], expected, name)
    bridge = comp.rename(columns={"idx": "stock_idx"}).copy()
    bridge["refit_idx"], bridge["starttdc"] = refit.refit_idx.to_numpy(), refit.starttdc.to_numpy()
    diagnostics = dict(inventory, boundary=boundary, registry_sha256=digest, stock_rows=len(stock),
                       comparison_rows=len(comparison), complete_stock_refit_bijection=True,
                       authoritative_saved_conditions_exact=True, original_time_bits_exact=True,
                       comparison_channel_from_stock_index=omitted_channel_fields,
                       stock_refit_differences=dict(time_changed=int(np.count_nonzero(comp.delta_time)),
                           pe_changed=int(np.count_nonzero(comp.delta_pe)),
                           max_abs_time_ns=float(comp.delta_time.abs().max()) if len(comp) else 0.,
                           max_abs_pe=float(comp.delta_pe.abs().max()) if len(comp) else 0.,
                           failed_flag_changed=int(np.count_nonzero(((comp["flags"].to_numpy() & 1) != 0) != comp.failed)),
                           zerondf_flag_changed=int(np.count_nonzero(((comp["flags"].to_numpy() & 8) != 0) != comp.zerondf))),
                       all_active=True, regenerated_finder=False, regenerated_fit=False)
    return dict(selected=selected, pulses=pulses, waveforms=waveforms, bridge=bridge, diagnostics=diagnostics)
