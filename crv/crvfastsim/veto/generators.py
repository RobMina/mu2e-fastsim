"""Prepare retained generator inputs using event identity and an independent event census."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

from .paired import KEY

STATE = ("x", "y", "z", "px", "py", "pz", "e", "t")
IDENTITY = (*KEY, "gen_product_id", "idx")


def integer_columns(table, fields, label, *, signed=False):
    """Require integer storage before joins; never round source identities through floats."""
    for field in fields:
        if field not in table:
            raise ValueError(f"{label}: missing {field}")
        values = table[field].to_numpy()
        if values.dtype.kind not in "iu":
            raise ValueError(f"{label}: {field} requires integer storage")
        if values.dtype.kind == "u" and np.any(values > np.uint64(np.iinfo(np.int64).max)):
            raise ValueError(f"{label}: {field} exceeds signed 64-bit range")
        if not signed and np.any(values < 0):
            raise ValueError(f"{label}: negative {field}")


def validate_generators(gen):
    """Validate source keys and finite stored states without changing particle values."""
    integer_columns(gen, IDENTITY, "generators")
    integer_columns(gen, ("pdg", "genid"), "generators", signed=True)
    if gen.duplicated(list(IDENTITY)).any():
        raise ValueError("duplicate retained generator identity")
    if (gen.gen_product_id == 0).any():
        raise ValueError("invalid retained generator product/index")
    if not set(STATE) <= set(gen):
        raise ValueError("missing generator state columns")
    if not np.isfinite(gen[list(STATE)].to_numpy()).all() or (gen.e < 0).any():
        raise ValueError("nonfinite state or negative generator total energy")
    p = np.hypot(np.hypot(gen.px.astype(float), gen.py.astype(float)), gen.pz.astype(float))
    precision = max((np.finfo(gen[k].dtype).eps for k in ("px", "py", "pz", "e")
                     if gen[k].dtype.kind == "f"), default=0.)
    if (gen.e < p - max(1e-9, 4 * precision) * np.maximum(p, 1.)).any():
        raise ValueError("generator total energy is less than momentum magnitude")


def select_split(selection, split, split_at):
    """Reject empty event populations while preserving events with empty generator collections."""
    if split not in ("all", "dev", "heldout"):
        raise ValueError("unknown selection split: " + split)
    if split != "all":
        dev = selection.shard < split_at
        selection = selection[dev] if split == "dev" else selection[~dev]
    if len(selection) == 0:
        raise ValueError(f"no selected events remain after split={split!r} at shard {split_at}")
    return selection


def prepare(selection, generators, census):
    """Keep every retained species and source identity; never derive inputs from veto outcomes."""
    if len(selection) == 0:
        raise ValueError("retained inputs require a nonempty event selection")
    selected = pd.DataFrame({k: np.asarray(selection[k]) for k in (*KEY, "shard")})
    integer_columns(selected, (*KEY, "shard"), "selection")
    integer_columns(census, (*KEY, "n_gen"), "census")
    integer_columns(generators, KEY, "generators")
    if selected.duplicated(list(KEY)).any() or census.duplicated(list(KEY)).any():
        raise ValueError("duplicate selected or upstream event identity")
    expected = selected.merge(census[[*KEY, "n_gen"]], on=list(KEY), how="left", validate="one_to_one")
    if expected.n_gen.isna().any():
        raise ValueError("generator census does not cover every selected event")
    gen = generators.drop(columns="shard", errors="ignore").merge(
        selected, on=list(KEY), how="inner", validate="many_to_one", sort=False)
    validate_generators(gen)
    if "gen_product_id" in census:
        integer_columns(census, ("gen_product_id",), "census")
        if census.gen_product_id.eq(0).any():
            raise ValueError("census: null generator ProductID")
        products = gen[[*KEY, "gen_product_id"]].merge(
            census[[*KEY, "gen_product_id"]], on=list(KEY), validate="many_to_one", suffixes=("", "_census"))
        if not products.gen_product_id.eq(products.gen_product_id_census).all():
            raise ValueError("generator ProductID differs from event census")
    counts = gen.groupby(list(KEY)).size().rename("observed")
    expected = expected.merge(counts, on=list(KEY), how="left", validate="one_to_one")
    if not np.array_equal(expected.n_gen.to_numpy(), expected.observed.fillna(0).to_numpy()):
        raise ValueError("retained generator rows disagree with the selected event census")
    gen = gen.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    report = dict(input_kind="retained", selected_events=len(selected), primary_particles=len(gen),
        primary_muons=int((gen.pdg.abs() == 13).sum()), nonmuon_primaries=int((gen.pdg.abs() != 13).sum()),
        empty_generator_events=int(expected.n_gen.eq(0).sum()),
        species={str(int(k)): int(v) for k, v in gen.pdg.value_counts().items()},
        scope="complete selected retained collection; not a complete original generated stream",
        coordinates="upstream Mu2e frame and global time; no fitted offsets")
    return gen, report


def shower_support(gen, lib):
    """Report missing species/energy support separately from a sampled zero response; no RNG calls."""
    from .. import tier1b
    rows = gen.loc[gen.pdg.abs() != 13, ["pdg", "e"]].copy()
    rows["apdg"] = rows.pdg.abs()
    rows["known_mass"] = rows.apdg.isin(tier1b.MASS)
    rows["ke"] = rows.e - rows.apdg.map(tier1b.MASS).fillna(0.)
    rows["kebin"] = tier1b._ke_bin(rows.ke.to_numpy())
    rows = rows.merge(lib["libgen"][["apdg", "kebin", "ngen", "nshower"]],
                      on=["apdg", "kebin"], how="left", validate="many_to_one")
    rows["status"] = np.select(
        [~rows.known_mass, rows.ngen.isna(), rows.ngen.eq(0), rows.nshower.eq(0)],
        ["unknown_mass", "missing_cell", "zero_denominator", "zero_shower_cell"], default="supported")
    cells = rows.groupby(["pdg", "kebin", "status"], sort=True).size().rename("inputs").reset_index()
    outside = (rows.ke < tier1b.KE_EDGES[0]) | (rows.ke >= tier1b.KE_EDGES[-1])
    return dict(nonmuon_inputs=len(rows), status_counts=rows.status.value_counts().to_dict(),
                cells=cells.to_dict(orient="records"),
                incomplete_response_support=bool((rows.status.ne("supported") | outside).any()),
                outside_energy_grid=int(outside.sum()),
                limitation="Support counts describe the existing sampler, not position-conditioned response or veto closure.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "generators", "events", "outdir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--split", choices=("all", "dev", "heldout"), default="dev")
    parser.add_argument("--split-at", type=int, default=22)
    args = parser.parse_args(argv)
    selection = select_split(ak.from_parquet(args.selection), args.split, args.split_at)
    census = pd.read_parquet(args.events)
    gen, summary = prepare(selection, pd.read_parquet(args.generators), census)
    summary.update(prepared_split=args.split, split_at=args.split_at)
    events = pd.DataFrame({k: np.asarray(selection[k]) for k in KEY}).merge(
        census, on=list(KEY), validate="one_to_one")
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(selection, out / "selection.parquet")
    gen.to_parquet(out / "primaries.parquet", index=False)
    events.to_parquet(out / "generator_events.parquet", index=False)
    (out / "inputs.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
