"""Bounded saved-pulse bridge to the unchanged current reference finder.

The authoritative identity is (input_sha256, run, subrun, event, frame_id).
Collection indices are assigned in (frame_id, chan, starttdc, seq) order. This
structural diagnostic bridge is not the stock Ptr collection order, a truth
match, or a stock-equivalence assertion. Stored wf labels are local metadata,
not physical pulse identity. No response, fitting, or optical work is executed.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import awkward as ak
import numpy as np
import pandas as pd

from . import bounded_response as B, frame_registry as F, replay_recover as R

CONTRACT = "bounded-saved-pulse-coincidence-v1"
FRAME_KEYS = [*F.FRAME_KEYS, "frame_id"]
PULSE_KEYS = ["chan", "starttdc", "seq"]
PULSE_FIELDS = [*PULSE_KEYS, "wf", "time_relative", "time_reco", "pe", "channel_ignored"]
SCALARS = [name for name in R.CLUSTER_TYPES if name not in R.IDS]


@dataclass(frozen=True)
class Limits:
    """Lowerable admission ceilings; exceeding one fails without truncation."""

    max_frames: int = 4
    max_pulses_per_frame: int = 512
    max_total_pulses: int = 2048
    max_combinations: int = 2_000_000

    def __post_init__(self):
        for name, ceiling in (("max_frames", 4), ("max_pulses_per_frame", 512),
                              ("max_total_pulses", 2048), ("max_combinations", 2_000_000)):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= ceiling:
                raise ValueError(f"{name} must be a positive native int <= {ceiling}")


def _schema(table, columns, label):
    if (not isinstance(table, pd.DataFrame) or not table.columns.is_unique
            or not set(columns) <= set(table)):
        raise ValueError(f"missing or duplicate {label} columns")


def _integer(values, name, hi=2**63):
    values = np.asarray(values)
    if (values.ndim != 1 or values.dtype.kind not in "iu"
            or np.any(values < 0) or np.any(values >= hi)):
        raise ValueError(f"invalid integer {name}")
    return values.astype(np.int64)


def _real(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError(f"invalid finite numeric {name}")
    return values.astype(np.float64)


def _geometry(G):
    sector = _integer(G.sector, "geometry sector", len(G.sector_type))
    layer = _integer(G.layer, "geometry layer", 4)
    sector_type = _integer(G.sector_type, "geometry sector type", 2**31)
    if not len(sector) or len(layer) != len(sector):
        raise ValueError("invalid geometry bar domain")
    return sector, layer, sector_type


def _status(calibration, digest, nchannels):
    _schema(calibration, ["input_sha256", "chan", "status"], "calibration")
    c = calibration.loc[calibration.input_sha256.eq(digest), ["chan", "status"]].copy()
    c["chan"] = _integer(c.chan, "calibration chan", nchannels)
    c["status"] = _integer(c.status, "calibration status", 2**16)
    c = c.sort_values("chan")
    if len(c) != nchannels or not np.array_equal(c.chan, np.arange(nchannels)):
        raise ValueError("incomplete or duplicate calibration channel domain")
    return c.status.to_numpy()


def _restore(table, frames):
    out = table.merge(frames[FRAME_KEYS], on="frame_id", how="left", validate="many_to_one", indicator=True)
    if not out._merge.eq("both").all():
        raise ValueError("native frame outside selected census")
    return out.drop(columns="_merge")


def combination_work(pulses, G, cfg):
    """Upper-bound eligible layer tuples before filtering/clustering, using fixed detector groups."""
    sector, layer, sector_type = _geometry(G)
    if len(pulses) > Limits().max_total_pulses:
        raise ValueError("total pulse cap exceeded before combination count")
    counts = pulses.groupby("frame_id").size()
    if len(counts) > Limits().max_frames or (counts > Limits().max_pulses_per_frame).any():
        raise ValueError("frame/pulse cap exceeded before combination count")
    bar = _integer(pulses.chan, "combination channel", len(sector) * 4) // 4
    table = pd.DataFrame({"frame_id": pulses.frame_id.to_numpy(), "side": pulses.chan.to_numpy() % 2,
                          "type": sector_type[sector[bar]], "layer": layer[bar]})
    admitted = np.array([name in cfg.cuts for name in G.sector_names], dtype=bool)
    table = table.loc[admitted[sector[bar]]]
    if not len(table):
        return 0
    grouped = table.groupby(["type", "frame_id", "side", "layer"]).size().unstack("layer", fill_value=0)
    grouped = grouped.reindex(columns=np.arange(4), fill_value=0)
    total = 0
    for kind in np.unique(sector_type):
        settings = [cfg.cuts[G.sector_names[s]].coincidence_layers for s in np.flatnonzero(sector_type == kind)
                    if G.sector_names[s] in cfg.cuts]
        if not settings or kind not in grouped.index.get_level_values("type"):
            continue
        if any(type(n) is not int or n not in (2, 3, 4) for n in settings):
            raise ValueError("unsupported coincidence layer requirement")
        multiplicities = grouped.loc[kind].to_numpy(dtype=np.int64)
        for size in range(min(settings), max(settings) + 1):
            for layers in combinations(range(4), size):
                total += int(np.prod(multiplicities[:, layers], axis=1).sum())
    return total


def run(pulses: pd.DataFrame, registry, selected, calibration, G, *, registry_digest, limits=Limits()):
    """Recover bounded whole frames, including quiet frames, from saved pulses only.

    Returns frames with pulses/clusters/members counts, the canonical per-frame
    pulse idx map, all stock-layout scalar cluster properties (idx is per frame),
    and native links with cluster_idx, member_order and pulse_idx. The temporary
    replay event label is frame_id, never a replacement for full input identity.
    """
    if not isinstance(limits, Limits):
        raise ValueError("requires coincidence Limits")
    limits.__post_init__()
    _schema(selected, F.COLUMNS, "selected registry")
    if len(selected) > limits.max_frames:
        raise ValueError("frame cap exceeded")
    frames = B.selected_frames(registry, selected, registry_digest)
    if frames.input_sha256.nunique() != 1:
        raise ValueError("single input per coincidence call required")
    _schema(pulses, [*FRAME_KEYS, *PULSE_FIELDS], "saved pulses")
    if len(pulses) > limits.max_total_pulses:
        raise ValueError("total pulse cap exceeded")
    sector, layer, sector_type = _geometry(G)
    nchannels = len(sector) * 4
    status = _status(calibration, frames.input_sha256.iloc[0], nchannels)
    p = pulses[[*FRAME_KEYS, *PULSE_FIELDS]].copy()
    for name in ("run", "subrun", "event"):
        p[name] = _integer(p[name], name, 2**32)
    p["frame_id"] = _integer(p.frame_id, "frame_id", len(registry))
    for name, hi in (("chan", nchannels), ("starttdc", 2**31), ("seq", 2**31), ("wf", 2**63)):
        p[name] = _integer(p[name], name, hi)
    for name in ("time_relative", "time_reco", "pe"):
        p[name] = _real(p[name], name)
    if (p.pe < 0).any():
        raise ValueError("negative pulse PE")
    if p.channel_ignored.dtype != np.dtype(bool):
        raise ValueError("channel_ignored must be boolean")
    p = p.merge(frames[[*FRAME_KEYS, "pbtime"]], on=FRAME_KEYS, how="left", validate="many_to_one", indicator=True)
    if not p._merge.eq("both").all():
        raise ValueError("pulse full identity outside selected frame census")
    if p.duplicated([*FRAME_KEYS, *PULSE_KEYS]).any():
        raise ValueError("duplicate physical pulse key")
    with np.errstate(over="ignore"):
        pbtime = frames.pbtime.to_numpy().astype(np.float32).astype(np.float64)
        pulse_pbtime = p.pbtime.to_numpy().astype(np.float32).astype(np.float64)
    if not np.isfinite(pbtime).all():
        raise ValueError("bunch time not finite as float32")
    if not np.array_equal(p.time_reco, p.time_relative.to_numpy() - pulse_pbtime):
        raise ValueError("time_reco must exactly equal time_relative - float32(pbtime)")
    pulse_status = status[p.chan.to_numpy()]
    if not np.array_equal(p.channel_ignored, (pulse_status & 2) != 0):
        raise ValueError("channel_ignored disagrees with status bit 2")
    if np.any(pulse_status & 5):
        raise ValueError("pulse on non-live response channel")
    counts = frames.frame_id.map(p.groupby("frame_id").size()).fillna(0).to_numpy(dtype=np.int64)
    if np.any(counts > limits.max_pulses_per_frame):
        raise ValueError("per-frame pulse cap exceeded")
    if combination_work(p, G, R.finder_config()) > limits.max_combinations:
        raise ValueError("conservative finder combination cap exceeded")
    p = p.sort_values(["frame_id", *PULSE_KEYS]).reset_index(drop=True)
    p["idx"] = p.groupby("frame_id", sort=False).cumcount().astype(np.int64)
    p = _restore(p[["frame_id", "idx", *PULSE_FIELDS]], frames)
    p = p[[*FRAME_KEYS, "idx", *PULSE_FIELDS]]
    events = {"run": np.zeros(len(frames), np.int64), "subrun": np.zeros(len(frames), np.int64),
              "event": frames.frame_id.to_numpy(), "pbtime": frames.pbtime.to_numpy()}
    active = p.loc[~p.channel_ignored]
    helper_pulses = {"ev": active.frame_id.to_numpy(), "chan": active.chan.to_numpy(), "seq": active.seq.to_numpy(),
                     "time": active.time_reco.to_numpy(), "pe": active.pe.to_numpy()}
    batch = R.prepare_pulses(helper_pulses, events, nchannels)
    batch.pulses["idx"] = active.idx.to_numpy(copy=True)
    batch.finder_time = active.time_relative.to_numpy(copy=True)
    native = R.recover(batch, G, status, R.finder_config(), pe_float32=False)
    clusters, members = _native_tables(native, frames, p, sector, layer, sector_type)
    for name, table in (("pulses", p), ("clusters", clusters), ("members", members)):
        frames[name] = frames.frame_id.map(table.groupby("frame_id").size()).fillna(0).astype(np.int64)
    return {"frames": frames, "pulses": p, "clusters": clusters, "members": members}


def _native_tables(native, frames, pulses, sector, layer, sector_type):
    columns = {name: ak.to_numpy(native[name], allow_missing=False) for name in R.CLUSTER_TYPES}
    c = pd.DataFrame(columns)
    if np.any(c.run != 0) or np.any(c.subrun != 0):
        raise ValueError("unexpected native temporary replay identity")
    for name, dtype in R.CLUSTER_TYPES.items():
        if dtype in ("int32", "uint32"):
            _integer(c[name], "native " + name, 2**32 if dtype == "uint32" else 2**31)
        elif dtype == "bool":
            if c[name].dtype != np.dtype(bool):
                raise ValueError("invalid native boolean")
        else:
            _real(c[name], "native " + name)
    c = _restore(c.drop(columns=["run", "subrun"]).rename(columns={"event": "frame_id"}), frames)
    if c.duplicated([*FRAME_KEYS, "idx"]).any():
        raise ValueError("duplicate native cluster identity")
    ordered = c.sort_values(["frame_id", "idx"])
    if not np.array_equal(ordered.idx, ordered.groupby("frame_id", sort=False).cumcount()):
        raise ValueError("noncanonical native cluster indices")
    counts = ak.to_numpy(ak.num(native.pulses, axis=1), allow_missing=False)
    links = _integer(ak.to_numpy(ak.flatten(native.pulses, axis=1), allow_missing=False), "native pulse link")
    if np.any(counts == 0):
        raise ValueError("empty native coincidence")
    m = pd.DataFrame({"frame_id": np.repeat(c.frame_id.to_numpy(), counts),
                      "cluster_idx": np.repeat(c.idx.to_numpy(), counts),
                      "member_order": np.arange(len(links)) - np.repeat(counts.cumsum() - counts, counts),
                      "pulse_idx": links})
    if m.duplicated(["frame_id", "pulse_idx"]).any():
        raise ValueError("duplicate native pulse membership")
    mapping = pulses.loc[~pulses.channel_ignored, [*FRAME_KEYS, "idx", *PULSE_KEYS]].rename(columns={"idx": "pulse_idx"})
    m = m.merge(mapping, on=["frame_id", "pulse_idx"], how="left", validate="many_to_one", indicator=True)
    if not m._merge.eq("both").all():
        raise ValueError("native pulse link outside canonical collection bounds")
    m = m.drop(columns="_merge")
    bars = m.chan.to_numpy(dtype=np.int64) // 4
    m["layer"], m["sector"] = layer[bars], sector[bars]
    expected_type = np.repeat(c.sectortype.to_numpy(), counts)
    if not np.array_equal(sector_type[sector[bars]], expected_type):
        raise ValueError("native member sector type disagrees with cluster")
    layer_counts = ak.to_numpy(ak.num(native.layers, axis=1), allow_missing=False)
    native_layers = _integer(ak.to_numpy(ak.flatten(native.layers, axis=1), allow_missing=False), "native layer", 4)
    if not np.array_equal(c.nLayers, layer_counts):
        raise ValueError("native nLayers disagrees with layer links")
    layer_keys = ["frame_id", "cluster_idx", "layer"]
    actual_layers = pd.DataFrame({"frame_id": np.repeat(c.frame_id.to_numpy(), layer_counts),
                                 "cluster_idx": np.repeat(c.idx.to_numpy(), layer_counts), "layer": native_layers})
    expected_layers = m[layer_keys].drop_duplicates()
    if actual_layers.duplicated(layer_keys).any():
        raise ValueError("duplicate native layer")
    actual, expected = (t.sort_values(layer_keys).reset_index(drop=True) for t in (actual_layers, expected_layers))
    if not np.array_equal(actual.to_numpy(), expected.to_numpy()):
        raise ValueError("native layers disagree with pulse geometry")
    c = c[[*FRAME_KEYS, *SCALARS]].sort_values(["frame_id", "idx"]).reset_index(drop=True)
    m = m[[*FRAME_KEYS, "cluster_idx", "member_order", "pulse_idx", *PULSE_KEYS, "layer", "sector"]]
    return c, m.sort_values(["frame_id", "cluster_idx", "member_order"]).reset_index(drop=True)


def _compare_rows(left, right, keys, fields):
    for table in (left, right):
        _schema(table, [*keys, *fields], "comparison")
        if table.duplicated(keys).any():
            raise ValueError("duplicate comparison identity")
    joined = left[[*keys, *fields]].merge(right[[*keys, *fields]], on=keys, how="outer",
                                         suffixes=("_left", "_right"), indicator=True, validate="one_to_one")
    common = joined.loc[joined._merge.eq("both")]
    different = {name: int(np.count_nonzero(common[name + "_left"].to_numpy() != common[name + "_right"].to_numpy()))
                 for name in fields}
    left_only, right_only = (int(joined._merge.eq(which).sum()) for which in ("left_only", "right_only"))
    return {"left_rows": len(left), "right_rows": len(right), "left_only": left_only, "right_only": right_only,
            "field_differences": different, "exact": not (left_only or right_only or any(different.values()))}


def compare(left, right):
    """Report exact structural differences; cross-arm equality is not a requirement.

    Pulse physical keys, index order, PE and both times are compared before native
    ordered links and every scalar cluster property. Local wf labels are excluded.
    There are no numerical tolerances and no inferred nearest-pulse matches.
    """
    specs = {
        "frames": (FRAME_KEYS, ["pbtime", "intensity", "pulses", "clusters", "members"]),
        "pulses": ([*FRAME_KEYS, *PULSE_KEYS], ["idx", "pe", "time_relative", "time_reco", "channel_ignored"]),
        "members": ([*FRAME_KEYS, "cluster_idx", "member_order"], ["pulse_idx", *PULSE_KEYS, "layer", "sector"]),
        "clusters": ([*FRAME_KEYS, "idx"], [name for name in SCALARS if name != "idx"]),
    }
    if set(left) != set(specs) or set(right) != set(specs):
        raise ValueError("requires all coincidence product tables")
    stages = {name: _compare_rows(left[name], right[name], *spec) for name, spec in specs.items()}
    return {"stages": stages, "exact": all(stage["exact"] for stage in stages.values()),
            "pulse_identity_exact": stages["pulses"]["exact"], "native_links_exact": stages["members"]["exact"],
            "cluster_properties_exact": stages["clusters"]["exact"], "numerical_tolerances": False,
            "stock_equivalence_asserted": False}
