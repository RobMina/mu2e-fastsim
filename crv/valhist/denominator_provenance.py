"""Read-only, identity-bound donor/generated energy and denominator-support provenance."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import uproot

from crvfastsim import tables, tier1b
from crvfastsim.analyses.census import primary_of
from valhist import denominator_variants as D
from valhist import library_variants as V

KEY = list(V.KEYS)
BIN = D.BIN


def records(frame):
    """Preserve integer identities and finite double precision; JSON null denotes unavailable."""
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def read(path, tree, columns=None):
    """Read the highest cycle, retaining explicit cycle and branch-type provenance."""
    with uproot.open(path) as root:
        name = "dump/" + tree
        cycles = sorted(int(k.rsplit(";", 1)[1]) for k in root.keys(recursive=True, cycle=True)
                        if k.rsplit(";", 1)[0] == name)
        if not cycles:
            raise ValueError("missing " + name + " in " + str(path))
        obj = root[name + ";" + str(cycles[-1])]
        frame = obj.arrays(columns, library="pd")
        info = dict(tree=name, cycles=cycles, selected_cycle=cycles[-1], rows=int(obj.num_entries),
                    types=obj.typenames())
    for col in set(KEY + ["idx", "key", "parent", "genidx", "sp", "pdg"]).intersection(frame.columns):
        a = frame[col].to_numpy()
        if (a.dtype.kind not in "iu" or np.any(a > np.iinfo(np.int64).max)
                or np.any(a == np.iinfo(np.int64).min)):
            raise ValueError("invalid integer identity: " + col)
    return frame, info


def energy(frame, pdg="pdg"):
    """Native total-E subtraction plus explicitly diagnostic on-shell/mass alternatives."""
    a = frame.copy()
    a["apdg"] = a[pdg].abs().astype(np.int64)
    mass = a.apdg.map(tier1b.MASS)
    a["mass_known"] = mass.notna()
    a["native_mass"] = mass.fillna(0.0)
    a["ke"] = a.e - a.native_mass
    p = np.sqrt(np.square(a[["px", "py", "pz"]].to_numpy(dtype=float)).sum(axis=1))
    on_shell = np.hypot(p, a.native_mass.to_numpy())
    a["p"] = p
    a["energy_minus_on_shell"] = a.e - on_shell
    a["invariant_mass_squared"] = (a.e - p) * (a.e + p)
    m2 = a.invariant_mass_squared.to_numpy()
    a["inferred_mass"] = np.sqrt(np.where(m2 >= 0, m2, np.nan))
    denom = on_shell + a.native_mass.to_numpy()
    a["diagnostic_ke_from_p"] = np.divide(p * p, denom, out=np.zeros_like(p), where=denom != 0)
    a["kebin"] = tier1b._ke_bin(a.ke.to_numpy())
    a["unclipped_kebin"] = np.searchsorted(tier1b.KE_EDGES, a.ke.to_numpy(), side="right") - 1
    a["diagnostic_p_kebin"] = tier1b._ke_bin(a.diagnostic_ke_from_p.to_numpy())
    a["diagnostic_bin_disagreement"] = a.kebin != a.diagnostic_p_kebin
    a["negative_ke"] = a.ke < 0
    a["underflow"] = a.ke < tier1b.KE_EDGES[0]
    a["overflow"] = a.ke >= tier1b.KE_EDGES[-1]
    a["nonfinite"] = ~np.isfinite(a[["e", "px", "py", "pz", "ke"]].to_numpy()).all(axis=1)
    return a


def gen_summary(gen):
    """Accepted donor gen rows are provenance only, never an exposure denominator."""
    frame = energy(gen.loc[gen.pdg.abs() != tables.PRIMARY_MUON_PDG].copy())
    return dict(rows=len(gen), persisted_events=len(gen[KEY].drop_duplicates()),
                nonmu_rows=len(frame), bin_summary=summarize(frame),
                genid_counts=records(gen.groupby(["genid", "pdg"]).size().rename("rows").reset_index()))


def summarize(frame):
    """Small species/bin groups only; all row operations are vectorized."""
    result = []
    for (apdg, kebin), group in frame.groupby(BIN, sort=True):
        item = dict(apdg=int(apdg), kebin=int(kebin), rows=len(group))
        for col in ["e", "ke", "p", "inferred_mass", "energy_minus_on_shell", "diagnostic_ke_from_p"]:
            finite = group[col].to_numpy()[np.isfinite(group[col].to_numpy())]
            item[col] = dict(min=float(finite.min()), max=float(finite.max())) if len(finite) else None
        for col in ["negative_ke", "underflow", "overflow", "nonfinite", "diagnostic_bin_disagreement"]:
            item[col] = int(group[col].sum())
        item["unknown_mass"] = int((~group.mass_known).sum())
        result.append(item)
    return result


def donor_showers(path):
    """Reconstruct only the builder's identity/energy projection, without rebuilding physics."""
    parts, pi = read(path, "particles")
    gen, gi = read(path, "gen")
    steps, si = read(path, "steps", KEY + ["sp"])
    if parts.duplicated(KEY + ["key"]).any() or gen.duplicated(KEY + ["idx"]).any():
        raise ValueError("ambiguous donor particle/gen identity")
    ancestry = primary_of(parts)
    s = steps.merge(ancestry[KEY + ["key", "prim_key", "prim_pdg"]].rename(columns={"key": "sp"}),
                    on=KEY + ["sp"], how="left", validate="many_to_one")
    valid = s.prim_key.notna() & (s.prim_key >= 0) & (s.prim_pdg.abs() != tables.PRIMARY_MUON_PDG)
    prim = parts.loc[(parts.isprim == 1) & (parts.pdg.abs() != tables.PRIMARY_MUON_PDG)].copy()
    gcols = {c: "gen_" + c for c in gen.columns if c not in KEY + ["idx"]}
    joined = prim.merge(gen.rename(columns={**gcols, "idx": "genidx"}), on=KEY + ["genidx"],
                        how="left", validate="many_to_one", indicator="gen_join")
    joined = joined.rename(columns={"key": "prim_key"})
    nm = s.loc[valid, KEY + ["prim_key"]].merge(joined, on=KEY + ["prim_key"], validate="many_to_one")
    sh = nm.drop_duplicates(KEY + ["prim_key"]).reset_index(drop=True)
    sh["shid"] = np.arange(len(sh), dtype=np.int64)
    sh["gen_join_missing"] = sh.gen_join != "both"
    sh["pdg_disagreement"] = sh.pdg != sh.gen_pdg
    sh = energy(sh.assign(e=sh.gen_e, px=sh.gen_px, py=sh.gen_py, pz=sh.gen_pz))
    sh["gen_pdg_native_ke"] = sh.gen_e - sh.gen_pdg.abs().map(tier1b.MASS).fillna(0.0)
    sh["gen_pdg_native_kebin"] = tier1b._ke_bin(sh.gen_pdg_native_ke.to_numpy())
    sh["start_ke_native_mass"] = sh.se - sh.native_mass
    sh["start_minus_gen_e"] = sh.se - sh.gen_e
    sh["start_minus_gen_p"] = np.sqrt(np.square(sh[["spx", "spy", "spz"]].to_numpy()).sum(axis=1)) - sh.p
    sh["start_minus_gen_t"] = sh.st - sh.gen_t
    sh["start_minus_gen_position_mm"] = np.sqrt(sum((sh["s" + c] - sh["gen_" + c]) ** 2 for c in "xyz"))
    sh = sh.drop(columns="gen_join")
    return sh, dict(particles=pi, gen=gi, steps=si, accepted_gen_population=gen_summary(gen),
                    step_rows_without_known_ancestry=int((s.prim_key.isna() | (s.prim_key < 0)).sum()),
                    nonmu_step_rows_without_primary=int(valid.sum() - len(nm)))


def support(showers, blocks, exposure, generated, existing):
    """Separate absent raw support, scaled-to-zero support, and persisted denominator errors."""
    ns = showers.groupby(BIN).size().rename("showers")
    raw = pd.concat(blocks, ignore_index=True).groupby(BIN).size().rename("raw_generator_rows")
    result = pd.concat([ns, raw], axis=1).fillna(0).astype(np.int64)
    result["scaled_generator_rows"] = np.rint(result.raw_generator_rows * (float(generated) / exposure)).astype(np.int64)
    if existing.duplicated(BIN).any():
        raise ValueError("duplicate libgen bins")
    old = existing.set_index(BIN)
    result = result.join(old.rename(columns={"ngen": "stored_ngen", "nshower": "stored_nshower"}), how="outer")
    result["missing_raw_support"] = (result.showers > 0) & (result.raw_generator_rows == 0)
    result["rounded_to_zero"] = (result.showers > 0) & (result.raw_generator_rows > 0) & (result.scaled_generator_rows == 0)
    result["nominal_saturated"] = (result.scaled_generator_rows > 0) & (result.showers > result.scaled_generator_rows)
    expected = result.raw_generator_rows > 0
    mismatch = ((expected != result.stored_ngen.notna())
                | (expected & (result.scaled_generator_rows != result.stored_ngen))
                | (expected & (result.showers != result.stored_nshower)))
    if mismatch.any():
        raise ValueError("persisted libgen does not reproduce raw/scaled generator formula")
    return result.reset_index()


def provenance_summary(donors, cells):
    """Multiplicity-weighted diagnostics without changing any native shower or support bin."""
    rows = pd.concat([pd.DataFrame(d['showers']).assign(file_occurrences=d['multiplicity'])
                      for d in donors], ignore_index=True)
    missing = cells.loc[cells.missing_raw_support, BIN]
    rows = rows.merge(missing.assign(missing_cell=True), on=BIN, how='left', validate='many_to_one')
    rows['missing_cell'] = rows.missing_cell.eq(True)
    categories = dict(all=np.ones(len(rows), bool), pdg_mismatch=rows.pdg_disagreement,
                      missing_support=rows.missing_cell, negative_ke=rows.negative_ke,
                      missing_low=rows.missing_cell & (rows.kebin == 0),
                      missing_high=rows.missing_cell & (rows.kebin > 0))
    def extrema(series, operation):
        values = series.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        return float(operation(values)) if len(values) else None
    summary = {}
    for label, selected in categories.items():
        r = rows.loc[selected]
        summary[label] = dict(unique_showers=len(r), occurrence_weighted_showers=int(r.file_occurrences.sum()),
                              pdg_mismatch_unique=int(r.pdg_disagreement.sum()),
                              pdg_mismatch_weighted=int(r.loc[r.pdg_disagreement, 'file_occurrences'].sum()),
                              missing_join_unique=int(r.gen_join_missing.sum()),
                              native_ke_min=extrema(r.ke, np.min),
                              native_ke_max=extrema(r.ke, np.max),
                              diagnostic_start_ke_min=extrema(r.start_ke_native_mass, np.min),
                              diagnostic_start_ke_max=extrema(r.start_ke_native_mass, np.max))
    summary['interpretation'] = 'Identity mismatches invalidate a pure finite-support interpretation; no affected row is dropped or repaired.'
    return summary


def run(library_json, out, evidence=()):
    """Accept only the original sealed donor product and unchanged bound physics checkout."""
    out = V._new_output(out)
    if Path(__file__).resolve().parents[1] in out.parents:
        raise ValueError("outputs must be outside the audit driver code tree")
    metadata, plan, base = D._base(library_json)
    code = D._code_identity()
    driver = V._record(__file__)
    evidence_records = [V._record(p) for p in evidence]
    selection = metadata["provenance"]["selection"]
    donors, frames = [], {}
    for index in sorted(set(selection["donor_indices"])):
        start = time.monotonic()
        record = plan["donors"][index]
        frame, trees = donor_showers(record["path"])
        frames[index] = frame
        donors.append(dict(source_index=index, source=record, trees=trees,
                           multiplicity=selection["file_multiplicity"][index],
                           bin_summary=summarize(frame), showers=records(frame)))
        print("donor %s: %.3f s" % (record["path"], time.monotonic() - start), flush=True)
    occurrences, offset = [], 0
    for occurrence, index in enumerate(selection["donor_indices"]):
        frame = frames[index]
        occurrences.append(frame[["shid", "apdg", "kebin", "ke"]].assign(shid=frame.shid + offset))
        offset += len(frame)
    reconstructed = pd.concat(occurrences, ignore_index=True)
    stored, shower_tree = read(base["root"]["path"], "libshower")
    pd.testing.assert_frame_equal(reconstructed, stored[reconstructed.columns].reset_index(drop=True),
                                  check_dtype=False, check_exact=True)
    generators, blocks, exposure = [], [], 0
    for record in plan["gen"]:
        start = time.monotonic()
        g, tree = read(record["path"], "gen")
        if g.duplicated(KEY + ["idx"]).any():
            raise ValueError("ambiguous external generator identity")
        events = len(g[KEY].drop_duplicates())
        if events != record["events"]["unique_events"]:
            raise ValueError("generator event inventory changed")
        exposure += events
        g = energy(g.loc[g.pdg.abs() != tables.PRIMARY_MUON_PDG].copy())
        if g.nonfinite.any():
            raise ValueError("nonfinite external generator energy/momentum")
        blocks.append(g[BIN])
        extremes = pd.concat([g.sort_values("ke").groupby(BIN).head(1),
                              g.sort_values("ke").groupby(BIN).tail(1)]).drop_duplicates(KEY + ["idx"])
        anomalous = g.loc[g.negative_ke | g.diagnostic_bin_disagreement | (~g.mass_known)
                          | ((g.apdg.isin([2112, 2212])) & (g.kebin == 0))]
        generators.append(dict(source=record, tree=tree, complete_persisted_events=events,
                               bin_summary=summarize(g), bin_extrema_rows=records(extremes),
                               diagnostic_rows=records(anomalous)))
        print("generator %s: %.3f s" % (record["path"], time.monotonic() - start), flush=True)
    lg, gen_tree = read(base["root"]["path"], "libgen")
    cells = support(stored, blocks, exposure, metadata["generated_scaled_to"], lg)
    if sum(len(b) for b in blocks) != metadata["generated_nonmu"]:
        raise ValueError("generated_nonmu metadata mismatch")
    result = dict(schema="valhist.denominator_provenance.v1", base=base, selection=selection,
                  generated_exposure=metadata["generated_scaled_to"], external_persisted_events=exposure,
                  native_mass={str(k): v for k, v in tier1b.MASS.items()}, ke_edges=tier1b.KE_EDGES.tolist(),
                  code_sha256=code, driver=driver, reviewed_evidence=evidence_records,
                  versions=dict(numpy=np.__version__, pandas=pd.__version__, uproot=uproot.__version__),
                  shower_tree=shower_tree, gen_tree=gen_tree, exact_shower_projection_closure=True,
                  exact_libgen_closure=True, support=records(cells), donors=donors, generators=generators,
                  provenance_summary=provenance_summary(donors, cells),
                  limitations=[
                      "Truth provenance diagnostic only; no model, replacement library or support repair.",
                      "Native bins use right search and outer-edge clipping; raw KE and unclipped bins retained.",
                      "Momentum/on-shell KE is diagnostic only, never substituted into shower or denominator counts.",
                      "Every unique selected donor shower is retained; selection order/multiplicity reconstruct global shid.",
                      "External summaries cover every nonmuon row; raw rows retain extrema and flagged/low-KE cases.",
                      "GenParticle ProductID is not persisted: matching idx alone cannot certify collection identity.",
                      "Start quantities include float32 serialization; different transport stage can change start kinematics.",
                      "Absent raw bank support does not establish physical impossibility or common generator configuration.",
                      "Evidence files are review snapshots, not proof of the historical producer binary/configuration.",
                      "Donor gen is an accepted-event table, not complete generated exposure; use sealed asserted exposure.",
                      "No held-out or new-source independence certificate, bootstrap, downstream errors or physics execution."])
    if D._base(library_json) != (metadata, plan, base):
        raise ValueError("base changed during provenance audit")
    if D._code_identity() != code or V._record(__file__) != driver:
        raise ValueError("audit/source drift")
    for r in evidence_records:
        D._check_record(r)
    payload = V._json_bytes(result)
    with out.open("xb") as stream:
        stream.write(payload)
    print("missing raw cells:", int(cells.missing_raw_support.sum()),
          "rounded-to-zero cells:", int(cells.rounded_to_zero.sum()), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library_json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--evidence", nargs="*", default=[], help="Producer/config review snapshots, not historical certification")
    run(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
