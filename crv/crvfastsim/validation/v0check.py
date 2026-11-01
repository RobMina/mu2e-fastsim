"""v0check -- validation level V0 (docs/plan.md section 5): the fast-sim CRV bar geometry against Geant4's, per bar.

Reference: the GDML export of the Geant4 world Offline builds (geometry/fcl/dump_geometry.fcl), in double precision,
plus the constructCRV.cc placement printout (crs.verbosityLevel 5) for the bar-index order. Candidate:
geometry.from_offline_config, the CosmicRayShieldMaker re-implementation on the Offline counters file. The GDML is in
Geant4 world coordinates, so one translation (the Mu2e origin in the world) is fitted and must be common to every bar.
Threshold: every bar centre and half length within 1 um, identical (sector, module, layer, counter) at every index.

    v0check mu2e_extracted_v04.gdml --g4log dump.log [--counters FILE] [--table bars.csv] --outdir DIR
"""
import numpy as np

from .. import geometry as geo
from mu2efastsim import plots

TOL_MM = 1e-3
PRINT_TOL_MM = 0.06  # the printout has one decimal
DEFAULT_COUNTERS = "Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"


def check(cand, ref, nrot, printed):
    ids_c = np.stack([cand.sector, cand.module, cand.layer, cand.counter], 1)
    ids_r = np.stack([ref.sector, ref.module, ref.layer, ref.counter], 1)
    ids_p = np.array(list(printed.keys()))
    res = {"n_bars_candidate": cand.nbars, "n_bars_gdml": ref.nbars, "n_bars_printout": len(printed),
           "n_rotated_bars_gdml": nrot,
           "index_order_matches_printout": bool(ids_p.shape == ids_c.shape and (ids_p == ids_c).all()),
           "ids_match_gdml": bool(ids_r.shape == ids_c.shape and (ids_r == ids_c).all())}
    ok = (cand.nbars == ref.nbars == len(printed) and nrot == 0
          and res["index_order_matches_printout"] and res["ids_match_gdml"])
    if ok:
        diff = ref.center - cand.center
        origin = np.median(diff, 0)
        resid = diff - origin
        pres = np.array([printed[tuple(i)] for i in ids_c]) - cand.center
        worst = int(np.abs(resid).max(1).argmax())
        res.update({
            "mu2e_origin_in_g4_world_mm": origin,
            "max_abs_center_residual_mm": float(np.abs(resid).max()),
            "max_abs_halflen_diff_mm": float(np.abs(ref.halflen - cand.halflen).max()),
            "max_abs_printout_residual_mm": float(np.abs(pres).max()),
            "local_to_world_match": bool((ref.local_to_world == cand.local_to_world).all()),
            "worst_bar": {"index": worst, "id": ids_c[worst], "residual_mm": resid[worst]},
            "per_sector": {name: {"n": int((cand.sector == s).sum()),
                                  "max_abs_center_residual_mm": float(np.abs(resid[cand.sector == s]).max())}
                           for s, name in enumerate(cand.sector_names)},
        })
        ok = (res["max_abs_center_residual_mm"] <= TOL_MM and res["max_abs_halflen_diff_mm"] <= TOL_MM
              and res["max_abs_printout_residual_mm"] <= PRINT_TOL_MM and res["local_to_world_match"])
    res["threshold_mm"] = TOL_MM
    res["pass"] = bool(ok)
    return res


def write_table(cand, path):
    with open(path, "w") as f:
        f.write("bar,sector,module,layer,counter,x,y,z,hx,hy,hz\n")
        for i in range(cand.nbars):
            f.write(",".join(map(str, [i, cand.sector_names[cand.sector[i]], cand.module[i], cand.layer[i],
                                       cand.counter[i], *np.round(cand.center[i], 6),
                                       *np.round(cand.halflen[i], 6)])) + "\n")
    print(f"wrote {path}")


def run(inputs, outdir, g4log=None, counters=None, table=None, **kw):
    if not g4log:
        raise SystemExit("v0check needs --g4log (mu2e log with the crs.verbosityLevel 5 bar printout)")
    cand = geo.from_offline_config(geo.find_counters(counters or DEFAULT_COUNTERS))
    ref, nrot = geo.from_gdml(inputs[0], cand.sector_names)
    res = check(cand, ref, nrot, geo.parse_g4_printout(g4log))
    plots.write_json(outdir, "v0_check.json", res)
    if table:
        write_table(cand, table)
    print(f"V0 {'PASS' if res['pass'] else 'FAIL'}: {res['n_bars_candidate']} bars, max centre residual "
          f"{res.get('max_abs_center_residual_mm')} mm (threshold {TOL_MM} mm)")
    return res
