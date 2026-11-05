"""Extract represented cosmic primary start states from selected-track MC genealogy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from .paired import KEY


def extract(selection, origin):
    """Deduplicate generator-associated muons by event and SimParticle ID, not kinematics."""
    sims = selection.trkmcsim
    flat = ak.flatten(ak.flatten(sims, axis=2), axis=1)
    event_row = ak.broadcast_arrays(ak.local_index(selection, axis=0), sims.id)[0]
    row = ak.to_numpy(ak.flatten(event_row, axis=None))
    good = ak.to_numpy(flat.valid & (flat.gen == 38) & (abs(flat.pdg) == 13))
    selected = flat[good]
    row = row[good]
    data = {k: ak.to_numpy(selection[k])[row] for k in KEY}
    data.update(key=ak.to_numpy(selected.id), pdg=ak.to_numpy(selected.pdg),
                t=ak.to_numpy(selected.time), primary_relation=ak.to_numpy(selected.prirel._rel))
    for i, axis in enumerate(("X", "Y", "Z")):
        data[axis.lower()] = ak.to_numpy(selected.pos.fCoordinates["f" + axis]) + origin[i]
        data["p" + axis.lower()] = ak.to_numpy(selected.mom.fCoordinates["f" + axis])
    table = pd.DataFrame(data)
    state_cols = ["pdg", "t", "x", "y", "z", "px", "py", "pz"]
    if (table.groupby(list(KEY) + ["key"])[state_cols].nunique() > 1).any().any():
        raise ValueError("one event/SimParticle identity carries inconsistent start states")
    table = table.drop_duplicates(list(KEY) + ["key"]).reset_index(drop=True)
    counts = table.groupby(list(KEY)).size()
    n_events = len(counts)
    relations, nrel = np.unique(table.primary_relation, return_counts=True)
    summary = dict(events=len(selection), represented_events=n_events, events_without_represented_muon=len(selection)-n_events,
                   primary_muons=len(table), duplicate_genealogy_rows=int(good.sum())-len(table),
                   primary_relations={str(int(k)): int(v) for k, v in zip(relations, nrel)},
                   coordinate_origin_mu2e_mm=list(origin),
                   scope="represented cosmic generator muons from selected-track genealogy; not an all-primary census")
    return table, summary


def bunch_times(source, selection):
    """Read true/reconstructed bunch times by exact event identity from an EventNtuple."""
    with uproot.open(source) as root:
        tree = root["EventNtuple/ntuple"]
        fields = {k: "evtinfo/" + k for k in KEY}
        fields.update(pbtime="evtinfo/pbtime", pbtime_mc="evtinfomc/pbtime")
        values = tree.arrays(list(fields.values()), library="np")
    times = pd.DataFrame({k: values[v] for k, v in fields.items()})
    wanted = pd.DataFrame({k: ak.to_numpy(selection[k]) for k in KEY})
    result = wanted.merge(times, on=list(KEY), how="left", validate="one_to_one", sort=False)
    if result[["pbtime", "pbtime_mc"]].isna().any().any():
        raise ValueError("source ntuple does not contain every selected event's bunch times")
    return result


def recover_files(paths, cached, mca, origin, out):
    """Recover stock-selected primaries one source file at a time, with the original analysis mask."""
    from .cosmic_ana import _import_mca, select_file
    from .paired import align
    Analyse, CutManager = _import_mca(mca)
    analyse = Analyse(on_spill=True, cut_config_path=str(Path(mca) / "config/common/cuts.yaml"),
                      cutset_name="dev", verbosity=0)
    selections, primary_parts, reports = [], [], []
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for shard, path in enumerate(paths):
        selected = select_file(path, analyse, CutManager, include_lineage=True)
        event = pd.DataFrame({k: ak.to_numpy(selected[k]) for k in KEY})
        cache_keys = pd.DataFrame({k: ak.to_numpy(cached[k]) for k in KEY})
        cache_keys["row"] = np.arange(len(cached))
        rows = event.merge(cache_keys, on=list(KEY), validate="one_to_one", how="left")
        if rows.row.isna().any():
            raise ValueError("original analysis selected events outside the cached population")
        reference = cached[rows.row.to_numpy(np.int64)]
        for key in ("p_mid", "t_mid"):
            if not ak.all(ak.num(selected[key]) == ak.num(reference[key])) or not ak.all(selected[key] == reference[key]):
                raise ValueError("original selection does not reproduce cached tracks")
        times = bunch_times(path, selected)
        for key in ("pbtime", "pbtime_mc"):
            selected = ak.with_field(selected, times[key].to_numpy(), key)
        selected = ak.with_field(selected, np.full(len(selected), shard, np.int32), "shard")
        primaries, summary = extract(selected, origin)
        primaries["shard"] = shard
        summary.update(source=path, shard=shard)
        ak.to_parquet(selected, out / f"selection_{shard:03d}.parquet")
        primaries.to_parquet(out / f"primaries_{shard:03d}.parquet", index=False)
        print(json.dumps(summary), flush=True)
        selections.append(selected)
        primary_parts.append(primaries)
        reports.append(summary)
    combined = align(cached, ak.concatenate(selections))
    ak.to_parquet(combined, out / "selection.parquet")
    pd.concat(primary_parts, ignore_index=True).to_parquet(out / "primaries.parquet", index=False)
    (out / "primary_summary.json").write_text(json.dumps(reports, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage")
    parser.add_argument("--ntuple")
    parser.add_argument("--files", help="source ntuple paths, one per line, for full-cache recovery")
    parser.add_argument("--cached")
    parser.add_argument("--mca")
    parser.add_argument("--detector-origin", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    if args.files:
        if not args.cached or not args.mca:
            parser.error("--files requires --cached and --mca")
        paths = [p.strip() for p in Path(args.files).read_text().splitlines() if p.strip()]
        recover_files(paths, ak.from_parquet(args.cached), args.mca, args.detector_origin, args.outdir)
        return
    if not args.lineage or not args.ntuple:
        parser.error("give --lineage and --ntuple, or --files with --cached and --mca")
    selection = ak.from_parquet(args.lineage)
    primaries, summary = extract(selection, args.detector_origin)
    times = bunch_times(args.ntuple, selection)
    summary["bunch_times"] = {k: dict(min=float(times[k].min()), max=float(times[k].max())) for k in ("pbtime", "pbtime_mc")}
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    primaries.to_parquet(out / "primaries.parquet", index=False)
    for key in ("pbtime", "pbtime_mc"):
        selection = ak.with_field(selection, times[key].to_numpy(), key)
    selection = ak.with_field(selection, np.zeros(len(selection), np.int32), "shard")
    ak.to_parquet(selection, out / "selection.parquet")
    (out / "primary_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
