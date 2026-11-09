"""pulsematch -- channel-by-channel PE of four V2 arms on the same events: how much the secondaries add, G4 vs fast.

Arms (CrvRecoTableDumper files of the same shards; candidate arms may run on the gen stream, they share event keys):
  --ref     G4, all steps              --ref-mu   G4, primary-muon steps only (g4muon)
  --fast    fast tier 1a + 1b          --fast-mu  fast tier 1a only
Per event and SiPM channel: total reco PE (all pulses). Channels are joined across the four arms (absent = 0 PE).
The secondaries' contribution is ref - ref-mu (G4) and fast - fast-mu (fast). Per sector group, in bins of the
muon-only PE of the same arm: channels, mean added PE, fraction with added PE > 1, and the channels that have PE only
with secondaries (muon-only arm 0). A per-sector spectrum gap that sits in the secondaries' part shows up here, and
the muon-only part is compared too (ref-mu vs fast-mu: the V2 closure of tier 1a).

    pulsematch F.root ... --fast F.root ... --fast-mu FM.root ... --ref R.root ... --ref-mu RM.root ... --outdir DIR
"""
import awkward as ak
import numpy as np

from mu2efastsim import plots

from .. import geometry
from ..reco import reftables
from .census import SECTOR_GROUP

GROUPS = ("EX", "T1", "T2", "M")
MU_BINS = np.array([0.0, 0.5, 5.0, 20.0, 50.0, 100.0, 1e9])
DEFAULT_COUNTERS = "Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"


def channel_pe(paths):
    """(keys [N,2] = (event key, channel), total PE) over all pulses, summed per channel."""
    p = reftables.read(paths, "pulses", ["run", "subrun", "event", "bar", "sipm", "pe"])
    k = np.stack([reftables.event_key(p), ak.to_numpy(p.bar).astype(np.int64) * 4 + ak.to_numpy(p.sipm)], 1)
    u, inv = np.unique(k, axis=0, return_inverse=True)
    return u, np.bincount(inv.ravel(), weights=ak.to_numpy(p.pe).astype(float))


def join4(arms):
    """Union of channel keys of the arms and each arm's PE on it (0 where absent)."""
    keys = np.unique(np.concatenate([a[0] for a in arms.values()]), axis=0)
    out = {}
    for n, (k, v) in arms.items():
        pos = np.searchsorted(keys.view([("e", np.int64), ("c", np.int64)]).ravel(),
                              k.view([("e", np.int64), ("c", np.int64)]).ravel())
        x = np.zeros(len(keys))
        x[pos] = v
        out[n] = x
    return keys, out


def _binstats(mu, tot, sel):
    b = np.searchsorted(MU_BINS, mu, "right") - 1
    add = tot - mu
    res = {}
    for i in range(len(MU_BINS) - 1):
        m = sel & (b == i)
        res[f"{MU_BINS[i]:g}-{MU_BINS[i + 1]:g}"] = {
            "channels": int(m.sum()), "added_mean": float(add[m].mean()) if m.any() else None,
            "frac_added_gt1": float((add[m] > 1).mean()) if m.any() else None,
            "tot_mean": float(tot[m].mean()) if m.any() else None}
    return res


def _same_crossing(pe, sel, lo=5.0):
    """Channels lit by the muon in both muon-only arms (>= lo PE): median ratios fast/G4 of muon-only and of total PE
    (channels with a pulse in the full G4 arm; a muon pulse can be absent there when secondaries merge it into another
    pulse window), per muon-only G4 PE bin. Biased by binning on the G4 value; the added-PE table is the clean one."""
    both = sel & (pe["ref_mu"] >= lo) & (pe["fast_mu"] >= lo)
    b = np.searchsorted(MU_BINS, pe["ref_mu"], "right") - 1
    res = {}
    for i in range(len(MU_BINS) - 1):
        m = both & (b == i)
        if m.sum() < 20:
            continue
        res[f"{MU_BINS[i]:g}-{MU_BINS[i + 1]:g}"] = {
            "channels": int(m.sum()), "mu_ratio_median": float(np.median(pe["fast_mu"][m] / pe["ref_mu"][m])),
            "tot_ratio_median": float(np.median(pe["fast"][m & (pe["ref"] > 0)] / pe["ref"][m & (pe["ref"] > 0)]))
            if (m & (pe["ref"] > 0)).any() else None,
            "tot_minus_mu_g4_mean": float((pe["ref"][m] - pe["ref_mu"][m]).mean()),
            "tot_minus_mu_fast_mean": float((pe["fast"][m] - pe["fast_mu"][m]).mean())}
    return res


def run(inputs, outdir, fast=None, fast_mu=None, ref=None, ref_mu=None, counters=None, **kw):
    if not (fast and fast_mu and ref and ref_mu):
        raise SystemExit("pulsematch needs --fast, --fast-mu, --ref and --ref-mu")
    geom = geometry.from_offline_config(geometry.find_counters(counters or DEFAULT_COUNTERS))
    arms = {"fast": channel_pe(fast), "fast_mu": channel_pe(fast_mu), "ref": channel_pe(ref), "ref_mu": channel_pe(ref_mu)}
    # events every arm saw: the muon-only arms ran on G4-accepted events; restrict to those
    seen = np.intersect1d(np.unique(arms["ref_mu"][0][:, 0]), np.unique(arms["fast_mu"][0][:, 0]))
    keys, pe = join4(arms)
    ok = np.isin(keys[:, 0], seen)
    grp = SECTOR_GROUP[geom.sector[keys[:, 1] // 4]]
    out = {}
    for g in GROUPS:
        sel = ok & (grp == g)
        out[g] = {"g4": _binstats(pe["ref_mu"], pe["ref"], sel), "fast": _binstats(pe["fast_mu"], pe["fast"], sel),
                  "sum_pe": {n: float(v[sel].sum()) for n, v in pe.items()},
                  "same_crossing": _same_crossing(pe, sel),
                  "secondary_only_channels": {"g4": int((sel & (pe["ref_mu"] == 0) & (pe["ref"] > 0)).sum()),
                                              "fast": int((sel & (pe["fast_mu"] == 0) & (pe["fast"] > 0)).sum())}}
    plots.write_json(outdir, "pulsematch_stats.json", {"events": int(len(seen)), "groups": out})
    print(f"events seen by all arms: {len(seen):,}")
    for g, o in out.items():
        s = o["sum_pe"]
        print(f"== {g}: total PE  G4 mu {s['ref_mu']:.3g} -> all {s['ref']:.3g} (+{s['ref'] / s['ref_mu'] - 1:.3f}); "
              f"fast mu {s['fast_mu']:.3g} -> all {s['fast']:.3g} (+{s['fast'] / s['fast_mu'] - 1:.3f}); "
              f"mu-only fast/G4 {s['fast_mu'] / s['ref_mu']:.4f}; secondary-only channels G4 "
              f"{o['secondary_only_channels']['g4']:,} fast {o['secondary_only_channels']['fast']:,}")
        print("   mu-only PE bin: channels G4/fast | added PE mean G4/fast | frac added>1 G4/fast")
        for b in o["g4"]:
            G, F = o["g4"][b], o["fast"][b]
            if not G["channels"]:
                continue
            print(f"   {b:>9s}: {G['channels']:>7,}/{F['channels']:<7,} | {G['added_mean']:6.2f}/{F['added_mean'] or 0:6.2f}"
                  f" | {G['frac_added_gt1']:.3f}/{F['frac_added_gt1'] or 0:.3f}")
        print("   same crossing (both muon-only arms >= 5 PE), by G4 muon-only PE: channels | median fast/G4 muon-only, "
              "total | added PE G4/fast")
        for b, r in o["same_crossing"].items():
            print(f"   {b:>9s}: {r['channels']:>7,} | {r['mu_ratio_median']:.3f} {r['tot_ratio_median'] or float('nan'):.3f} | "
                  f"{r['tot_minus_mu_g4_mean']:.2f}/{r['tot_minus_mu_fast_mean']:.2f}")
    return out
