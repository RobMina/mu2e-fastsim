"""Score one fastsim commit on the three metrics: validity, throughput multiplier and SU per event.

The scorer recomputes everything from binned histograms and bench.json reports, so a score never rests on a
self-reported statistic (throughput is self-reported by construction, and the device label says where it ran).

Layout (one directory per workflow; the reference side is never in this repository):

    submission/<W>/<dtype>/hists.json     the fast arm, dtype = float32 and float64 (both required)
    submission/<W>/<dtype>/bench.json     mu2efastsim.bench report of the same workflow and dtype
    reference/<W>/stock/hists.json        the stock arm on the same generated events
    reference/<W>/floor/hists.json        stock with another seed (optional: without it the threshold is 2)
    reference/<W>/reference.json          {"stock_rate_per_slot": events/s, "boundary": "...", "mode": ...,
                                           "yield_tolerance": 0.01, "min_count": 0, "su_per_hour": {device: SU}}

hists.json: {"hists": {key: H}, "yields": {name: Y}}, where H is {"edges": [...], "counts": [...]} or
{"edges": [...], "shards": {shard_id: [...]}} and Y is a count or {"shards": {shard_id: count}}.

Statistics, per histogram (the same ones the repository's validation code uses):
- mode "same_events" (default; crvfastsim.validation.compare): both arms on the same generated events, no scaling,
  chi2 = sum (n1 - n2)^2 / (n1 + n2) over bins with n1 + n2 > 0, ndf = number of those bins;
- mode "shape": the candidate scaled to the reference total s = N1/N2, chi2 = sum (n1 - s n2)^2 / (n1 + s^2 n2),
  ndf = bins - 1;
- shards in both arms (trkfastsim.validation.chain_pair): d_sb = c_B,sb - c_A,sb over the S shared shards,
  chi2 = sum_b D_b^2 / var_b, var_b = S/(S-1) sum_s (d_sb - mean_b)^2, over bins with var_b > 0 and at least
  min_count entries in the reference (a correct arm then has <chi2/ndf> = (S-1)/(S-3), which the floor absorbs).
Flag rule (DQMVAL): chi2/ndf > max(2, 3 x floor) and p < 1e-3, where the floor is the same statistic for the floor
arm against stock (threshold 2 without a floor). A yield passes when |ratio - 1| - 3 se <= yield_tolerance.

Scores per workflow and dtype: validity = (1 - flagged fraction) x (fraction of yields that pass); throughput
multiplier = bench warm_rate / stock_rate_per_slot; SU/event = su_per_hour[device] / (3600 warm_rate), only with
--private (SU numbers are never published).

    python -m benchmarks.fastsim_score.score SUBMISSION_DIR REFERENCE_DIR [--private] [--out scores.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

DTYPES = ("float32", "float64")
P_FLAG = 1e-3


class ScoreError(ValueError):
    pass


# ---------------------------------------------------------------- chi2 survival function (no scipy needed)

def chi2_sf(x: float, k: int) -> float:
    """P(chi2_k > x): the regularized upper incomplete gamma Q(k/2, x/2) (series below a+1, continued fraction above)."""
    if k <= 0 or not math.isfinite(x):
        return float("nan")
    if x <= 0:
        return 1.0
    a, z = 0.5 * k, 0.5 * x
    lg = math.lgamma(a)
    if z < a + 1:
        term = s = 1.0 / a
        n = a
        for _ in range(10000):
            n += 1
            term *= z / n
            s += term
            if abs(term) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(-z + a * math.log(z) - lg))
    b, c, d = z + 1 - a, 1e300, 1.0 / (z + 1 - a)
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return min(1.0, math.exp(-z + a * math.log(z) - lg) * h)


# ---------------------------------------------------------------- statistics

def _arr(h: dict, key: str) -> np.ndarray:
    return np.asarray(h[key], dtype=float)


def _check_edges(ref: dict, cand: dict, name: str) -> None:
    e1, e2 = _arr(ref, "edges"), _arr(cand, "edges")
    if e1.shape != e2.shape or not np.allclose(e1, e2, rtol=1e-9, atol=0):
        raise ScoreError(f"{name}: binning differs between the arms")


def _bins(h: dict, name: str) -> np.ndarray:
    if "counts" in h:
        v = _arr(h, "counts")
    elif "shards" in h:
        v = np.sum([np.asarray(s, float) for s in h["shards"].values()], axis=0)
    else:
        raise ScoreError(f"{name}: needs counts or shards")
    if v.shape != (len(h["edges"]) - 1,):
        raise ScoreError(f"{name}: {v.size} bins for {len(h['edges']) - 1} edges")
    if not np.all(np.isfinite(v)) or np.any(v < 0):
        raise ScoreError(f"{name}: counts must be finite and >= 0")
    return v


def statistic(ref: dict, cand: dict, mode: str = "same_events", min_count: float = 0, name: str = "") -> dict:
    """(chi2, ndf, chi2/ndf, p, kind) of the candidate against the reference histogram."""
    _check_edges(ref, cand, name)
    n1, n2 = _bins(ref, name), _bins(cand, name)
    shared = sorted(set(ref.get("shards", {})) & set(cand.get("shards", {})))
    if len(shared) >= 3:
        a = np.array([ref["shards"][s] for s in shared], float)
        b = np.array([cand["shards"][s] for s in shared], float)
        d = b - a
        S = len(shared)
        var = S / (S - 1) * ((d - d.mean(axis=0)) ** 2).sum(axis=0)
        use = (var > 0) & (a.sum(axis=0) >= max(min_count, 1e-300))
        chi2, ndf, kind = float((d.sum(axis=0)[use] ** 2 / var[use]).sum()), int(use.sum()), f"paired_{S}_shards"
    elif mode == "shape":
        N1, N2 = n1.sum(), n2.sum()
        if N1 <= 0 or N2 <= 0:
            return {"chi2": None, "ndf": 0, "chi2ndf": None, "p": None, "kind": "empty"}
        s = N1 / N2
        den = n1 + s * s * n2
        use = (den > 0) & (n1 >= min_count)
        chi2, ndf, kind = float(((n1 - s * n2)[use] ** 2 / den[use]).sum()), max(int(use.sum()) - 1, 0), "shape"
    elif mode == "same_events":
        den = n1 + n2
        use = (den > 0) & (n1 >= min_count)
        chi2, ndf, kind = float(((n1 - n2)[use] ** 2 / den[use]).sum()), int(use.sum()), "same_events"
    else:
        raise ScoreError(f"unknown mode {mode!r}")
    if ndf == 0:
        return {"chi2": None, "ndf": 0, "chi2ndf": None, "p": None, "kind": kind}
    return {"chi2": chi2, "ndf": ndf, "chi2ndf": chi2 / ndf, "p": chi2_sf(chi2, ndf), "kind": kind}


def _count(y, name: str) -> tuple[float, dict | None]:
    if isinstance(y, dict):
        sh = {k: float(v) for k, v in y["shards"].items()}
        return sum(sh.values()), sh
    if not isinstance(y, (int, float)) or y < 0:
        raise ScoreError(f"yield {name}: not a count")
    return float(y), None


def yield_ratio(ref, cand, tol: float, name: str = "") -> dict:
    r, rs = _count(ref, name)
    c, cs = _count(cand, name)
    if r <= 0:
        raise ScoreError(f"yield {name}: reference count is 0")
    ratio = c / r
    shared = sorted(set(rs or {}) & set(cs or {}))
    if len(shared) >= 2:
        d = np.array([cs[s] - rs[s] for s in shared])
        S = len(shared)
        se = math.sqrt(S / (S - 1) * ((d - d.mean()) ** 2).sum()) / r
    else:
        se = ratio * math.sqrt(1 / r + 1 / max(c, 1))
    return {"ratio": ratio, "se": se, "pass": abs(ratio - 1) - 3 * se <= tol}


# ---------------------------------------------------------------- one workflow, one dtype

def load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        raise ScoreError(f"missing {p.as_posix()}") from None
    except ValueError as e:
        raise ScoreError(f"{p.as_posix()}: not JSON ({e})") from None


def score_arm(cand: dict, stock: dict, floor: dict | None, refcfg: dict) -> dict:
    mode, minc = refcfg.get("mode", "same_events"), float(refcfg.get("min_count", 0))
    tol = float(refcfg.get("yield_tolerance", 0.01))
    rows, flagged = [], 0
    for key, h_ref in sorted(stock.get("hists", {}).items()):
        if key not in cand.get("hists", {}):
            raise ScoreError(f"histogram {key} missing from the submission")
        st = statistic(h_ref, cand["hists"][key], mode, minc, key)
        fl = statistic(h_ref, floor["hists"][key], mode, minc, key) if floor and key in floor.get("hists", {}) else None
        f0 = fl["chi2ndf"] if fl and fl["chi2ndf"] is not None else None
        thr = max(2.0, 3.0 * f0) if f0 is not None else 2.0
        flag = st["chi2ndf"] is not None and st["chi2ndf"] > thr and st["p"] < P_FLAG
        flagged += flag
        rows.append({"hist": key, **st, "floor": f0, "threshold": thr, "flag": bool(flag)})
    yields = []
    for name, y_ref in sorted(stock.get("yields", {}).items()):
        if name not in cand.get("yields", {}):
            raise ScoreError(f"yield {name} missing from the submission")
        yields.append({"yield": name, **yield_ratio(y_ref, cand["yields"][name], tol, name)})
    n = len(rows)
    if n == 0:
        raise ScoreError("the reference has no histograms")
    ypass = sum(y["pass"] for y in yields) / len(yields) if yields else 1.0
    return {"n_hists": n, "n_flagged": flagged, "validity": (1 - flagged / n) * ypass, "yield_pass": ypass,
            "hists": rows, "yields": yields}


def score(sub: Path, ref: Path, private: bool = False) -> dict:
    workflows = sorted(p.name for p in ref.iterdir() if (p / "reference.json").is_file())
    if not workflows:
        raise ScoreError(f"no <W>/reference.json under {ref.as_posix()}")
    out: dict = {"workflows": {}}
    vals, mults = [], []
    for w in workflows:
        cfg = load_json(ref / w / "reference.json")
        stock = load_json(ref / w / "stock" / "hists.json")
        floor = load_json(ref / w / "floor" / "hists.json") if (ref / w / "floor" / "hists.json").is_file() else None
        out["workflows"][w] = {}
        for dt in DTYPES:
            d = sub / w / dt
            if not d.is_dir():
                raise ScoreError(f"{w}: no {dt} arm (both dtypes are required)")
            arm = score_arm(load_json(d / "hists.json"), stock, floor, cfg)
            bench = load_json(d / "bench.json")
            if bench.get("dtype") not in (None, dt):
                raise ScoreError(f"{w}/{dt}: bench.json says dtype {bench.get('dtype')}")
            rate = float(bench.get("warm_rate") or 0)
            if rate <= 0:
                raise ScoreError(f"{w}/{dt}: bench.json has no positive warm_rate")
            mult = rate / float(cfg["stock_rate_per_slot"])
            res = {k: v for k, v in arm.items()}
            res.update(warm_rate=rate, device=bench.get("device"), inputs=bench.get("inputs"),
                       throughput_multiplier=mult, throughput_verified=False,
                       boundary_match=(bench.get("boundary") in (None, cfg.get("boundary"))))
            if private:
                su = (cfg.get("su_per_hour") or {}).get(bench.get("device"))
                res["su_per_event"] = None if su is None else float(su) / (3600.0 * rate)
            out["workflows"][w][dt] = res
            vals.append(arm["validity"])
            mults.append(mult)
    out["validity_score"] = float(np.mean(vals))
    out["min_validity"] = float(np.min(vals))
    out["throughput_geomean"] = float(np.exp(np.mean(np.log(mults))))
    out["n_workflows"] = len(workflows)
    return out


def leaderboard_row(res: dict) -> dict:
    """The flat keys a Codabench leaderboard shows: overall scores plus validity and multiplier per workflow/dtype."""
    row = {k: res[k] for k in ("validity_score", "min_validity", "throughput_geomean")}
    for w, by in res["workflows"].items():
        for dt, r in by.items():
            tag = f"{w.lower()}_{'f32' if dt == 'float32' else 'f64'}"
            row[f"{tag}_validity"] = r["validity"]
            row[f"{tag}_multiplier"] = r["throughput_multiplier"]
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("submission", type=Path)
    ap.add_argument("reference", type=Path)
    ap.add_argument("--private", action="store_true", help="add SU per event (never for a public leaderboard)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    try:
        res = score(a.submission, a.reference, a.private)
    except ScoreError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    text = json.dumps(res, indent=1)
    a.out.write_text(text + "\n") if a.out else print(text)
    for w, by in res["workflows"].items():
        for dt, r in by.items():
            print(f"{w} {dt}: flagged {r['n_flagged']}/{r['n_hists']}, yields {r['yield_pass']:.2f}, "
                  f"validity {r['validity']:.3f}, x{r['throughput_multiplier']:.1f} on {r['device']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
