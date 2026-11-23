"""Device trigger chain: StrawDigis -> hits -> time clusters -> helices -> per-path decisions.

`Chain(menu_cfg, conditions, tc_mva, precision="float64"|"float32")` compiles once per padded batch shape; `Chain.decide(batch)` returns per-path
decisions [E] for every tracker path it evaluates, all on the device; only the stateful D0-phi prescale counter is
applied on the host afterwards (`Chain.finish`). `Chain.decide_packed(batch)` returns the same decisions and stage
arrays stacked into one device array per dtype and shape (a few host copies per batch instead of one per array);
`Chain.unpack` turns them back into the dicts of `decide` on the host.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import (aprhelix, calhelix, calo, calogeom, cpr as cprmod, deltafinder as dfmod, helix, hits, menu, mpr, mva, paths,
               tzcluster,
               simpletc, timecluster)

HELIX_TYPES = ("RobustHelixFinder", "AgnosticHelixFinder", "CalHelixFinder")
KSF_TYPES = ("LoopHelixFit", "KinKalSeedFit", "LoopHelixDeFit")
PRECISIONS = ("float64", "float32")
MPR_DRAW_BASE = 500           # response-draw keys of the mpr paths (the tpr/apr paths use 0, 1, ...)
CPR_DRAW_BASE = 600           # response-draw keys of the cpr paths
CPR_CLUSTERS = 32             # calo-cluster slots of the cpr time peaks (fixed, so the per-slot outputs keep one shape)
CALO_KEYS = ("valid", "sipm", "t0", "peakpos", "wf_len", "s0", "s1", "s2", "s3", "s_peak", "n")


def pack_layout(out, aux):
    """Static layout of `pack`: [((dtype, shape), [(dict, name), ...]), ...], groups in sorted order, names sorted
    within a group ("out" before "aux"). Needs only shapes and dtypes (works on tracers)."""
    groups = {}
    for d, tree in (("out", out), ("aux", aux)):
        for k in sorted(tree):
            v = tree[k]
            groups.setdefault((str(v.dtype), tuple(v.shape)), []).append((d, k))
    return [(g, groups[g]) for g in sorted(groups)]


def pack(out, aux, layout):
    """The arrays of `out` and `aux` stacked into one array per layout group (on the device: a stack copies values,
    so the host gets the same bits)."""
    trees = {"out": out, "aux": aux}
    return [jnp.stack([trees[d][k] for d, k in names]) for _, names in layout]


def unpack(packed, layout):
    """Host: the dicts (out, aux) back from `pack`'s arrays (one row per name)."""
    trees = {"out": {}, "aux": {}}
    for arr, (_, names) in zip(packed, layout):
        for i, (d, k) in enumerate(names):
            trees[d][k] = arr[i]
    return trees["out"], trees["aux"]


def pair_bucket(n, total):
    """Padded pair count of the compact path: the smallest of total / 4, total / 2 and total (total = the batch's
    cluster slots) that holds n pairs, so each hit-axis size compiles at most three shapes per stage."""
    for k in (total // 4, total // 2):
        if n <= k:
            return max(k, 1)
    return total


class Chain:
    def __init__(self, cfg, cond, tc_mva_path, response=None, path_names=None, salt=20261107, debug=False,
                 precision="float64", compact=False, pairs=True, mpr=False, cpr=False, deltafinder=False, tz=False,
                 apr_helix=False):
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        if tz and not deltafinder:
            raise ValueError("tz: TZClusterFinder reads TrigFlagPH's flags, run it with deltafinder=True")
        if apr_helix and not tz:
            raise ValueError("apr_helix: AgnosticHelixFinder reads the TZClusterFinder clusters, run it with tz=True")
        # float32: the device arrays in 32-bit, except the stock double steps of the hit and mpr ports, which trace
        # in float64 in both precisions (exact.strict). JAX's x64 flag is process-wide, so set it before any device
        # array is made.
        jax.config.update("jax_enable_x64", precision == "float64")
        if precision == "float32":   # full float32 matmuls: XLA's default runs them on TF32 tensor cores (Ampere+)
            jax.config.update("jax_default_matmul_precision", "highest")
        self.precision = precision
        self.cfg = cfg
        self.debug = debug
        phys = cfg["physics"]
        self.mods = {**phys.get("producers", {}), **phys.get("filters", {})}
        self.dpaths = menu.device_paths(cfg, path_names)
        self.cond_dev, self.tot_bins = hits.device_conditions(cond)
        self.fz = helix.face_z(cond)
        self.net = mva.load(tc_mva_path)
        # {path: {"hs": f, "ks": f}}: response-layer pass probabilities; or, opt-in, {path: {"final": {"var": "tc_nsh",
        # "edges": [...], "f": [...]}}}: one pass probability for the path, binned in the straw hits of the event's
        # best time cluster of the path's producer, applied to the path's own device decision (then hs/ks are not used)
        self.response = response or {}
        self.salt = salt
        # stage settings, shared between paths that use the same producer
        self.sh = hits.sh_settings(self.mods["TrigMakeSH"])
        self.ph = hits.ph_settings(self.mods["TrigMakePH"])
        self.plan, self.skipped = {}, {}
        self.opaths = menu.other_paths(cfg, path_names)
        self.stc = {}                            # SimpleTimeCluster label -> settings (cst paths)
        self.mprs = {}                           # RobustMultiHelixFinder label -> (finder, merge) settings (mpr paths)
        self.mpr = mpr                           # mpr paths are opt-in: the finder costs ~0.5 s per event on a CPU
        self.cpr = cpr                           # cpr paths are opt-in: they need the event's CaloDigis in the batch
        # TZClusterFinder port (opt-in): the apr paths' time clusters from the stock-exact port (tzcluster.py), with
        # their calo clusters, in place of the TimeClusterFinder approximation; needs CaloDigis and DeltaFinder flags
        self.tz = tz
        # AgnosticHelixFinder port (opt-in, needs tz): the apr paths' helices from the stock-exact port (aprhelix.py)
        # and MergeHelices, in place of the RobustHelixFinder-style approximation; ~0.4 s per cosmic event on a CPU
        self.apr_helix = apr_helix
        self.ahfs = {}                           # MergeHelices label -> (AgnosticHelixFinder label, settings, merge)
        self.chfs = {}                           # MergeHelices label -> (CalHelixFinder label, settings) (cpr paths)
        self.chf_geom = None                     # CalHelixFinder geometry inputs (calhelix.geometry)
        # TrigFlagPH (DeltaFinder port, opt-in): its flags replace the panel hits' for every stage that reads them
        self.df = None
        if deltafinder:
            m = self.mods.get("TrigFlagPH", {})
            if m.get("module_type") != "DeltaFinder":
                raise ValueError("deltafinder: the menu has no DeltaFinder TrigFlagPH")
            self.df = (dfmod.settings(m), dfmod.geometry())
        for entry in phys["trigger_paths"]:
            name = entry.split(":", 1)[-1]
            if path_names and name not in path_names:
                continue
            ps = menu.prescale_factors(cfg, name)
            if ps is not None and all(f <= 0 for f in ps.values()):
                # PrescaleEvent first and disabled in every mode it configures (e.g. apr_proton): stock rejects every
                # event before any other filter
                self.plan[name] = {"family": "disabled", "ps": ps, "filters": []}
            elif ps is None:
                self.skipped[name] = ["first filter is not a PrescaleEvent"]
            elif name in self.dpaths:
                self._plan_tracker(name, self.dpaths[name], ps)
            elif name in self.opaths:
                o = self.opaths[name]
                if o["family"] is None:
                    self.skipped[name] = o["unsupported"]
                    continue
                self.plan[name] = {"family": o["family"], "ps": ps,
                                   "filters": [(t_, lbl, cf) for t_, lbl, cf in o["filters"] if t_ != "PrescaleEvent"]}
            else:
                self.skipped[name] = ["no TimeClusterFilter with a device time-cluster producer"]
        # cpr paths: the calo stage (calo.reconstruct, from the batch's packed CaloDigis "cd_*") and CalTimePeakFinder
        self.cpr_tp, self.calo_s = {}, None
        cpr_tc = sorted({p["tc"] for p in self.plan.values() if p["family"] == "cpr"})
        self.tz_s = {lbl: tzcluster.settings(self.mods[lbl]) for lbl in
                     sorted({p["tc"] for p in self.plan.values() if p["family"] == "apr"})} if tz else {}
        if cpr_tc or self.tz_s:
            calib = cfg["services"]["ProditionsService"]["calCalib"]
            if calib.get("useDb"):
                raise ValueError("cpr and tz paths need calCalib from the configuration (useDb false)")
            geo = calogeom.build()
            self.calo_s = calo.settings(self.mods["CaloHitMakerFast"], self.mods["CaloClusterFast"], calib, geo)
            if compact:
                raise ValueError("cpr and tz paths run on the one-call path only (compact=False)")
        if cpr_tc:
            self.cpr_tp = {lbl: cprmod.settings(self.mods[lbl], geo) for lbl in cpr_tc}
            self.chf_geom = calhelix.geometry(cond)
        # off-spill hit reconstruction (cst paths): the TrigOffSpMakeSH/PH settings on the same digis
        self.sh_off = self.ph_off = None
        if self.stc:
            ph_lbl = {self.mods[lbl]["ComboHitCollection"].split(":")[0] for lbl in self.stc}
            if len(ph_lbl) != 1:
                raise ValueError(f"cst paths read several ComboHit collections: {sorted(ph_lbl)}")
            ph_m = self.mods[ph_lbl.pop()]
            self.ph_off = hits.ph_settings(ph_m)
            self.sh_off = hits.sh_settings(self.mods[ph_m["ComboHitCollection"].split(":")[0]])
        self.lumi = any(p["family"] == "lumi" for p in self.plan.values())
        # response-layer draw keys: a path's index among the tpr/apr paths in name order (the keys before the other
        # families joined the plan, so their draws are unchanged); a family added later that draws takes its own range
        self.draw_index = {n: i for i, n in enumerate(sorted(n for n, p in self.plan.items()
                                                              if p["family"] in ("tpr", "apr")))}
        self.draw_index.update({n: MPR_DRAW_BASE + i for i, n in enumerate(sorted(n for n, p in self.plan.items()
                                                                                   if p["family"] == "mpr"))})
        self.draw_index.update({n: CPR_DRAW_BASE + i for i, n in enumerate(sorted(n for n, p in self.plan.items()
                                                                                   if p["family"] == "cpr"))})
        # compact (opt-in, for CPUs): time-cluster refinement and helix fits run only on the (event, cluster) pairs
        # that have a candidate / a valid cluster, in three device calls with one host read of the counts between
        # them; the same computation per cluster as the one-call path, and the same decisions
        self.compact = bool(compact)
        # pairs (one-call path, the default): cluster refinement and helix fits only on the (producer, event, slot)
        # pairs with a candidate / a valid cluster, up to one pair per event and producer, else the full batched path
        # in the same call; the same decisions, and the faster one-call variant (A100 A/B, both dtypes). pairs=False
        # keeps the full batched path (for comparisons). The compact path has its own pair gathering and ignores it.
        self.pairs = bool(pairs)
        self._fn = jax.jit(self._decide)
        self._fa = jax.jit(self._stage_candidates)
        self._fr = jax.jit(self._refine_one, static_argnames=("lbl", "k"))
        self._fh = jax.jit(self._helix_one, static_argnames=("key", "k"))
        self._fp = jax.jit(self._paths)
        # packed results: the layout is recorded when a batch shape is traced (it depends only on the plan and the
        # batch shape), so the host can unpack the arrays of every later call with that shape
        self._layouts = {}
        self._fnp = jax.jit(self._decide_packed)
        self._fpp = jax.jit(self._paths_packed)

    def _plan_tracker(self, name, p, ps):
        """Plan entry of a path with a TimeClusterFilter (tpr, apr, cst), or the reason it is skipped."""
        phys = self.cfg["physics"]
        if p["family"] == "cst":
            if p["unsupported"]:                 # cst_CosmicTrackSeed: LineFinder + CosmicSeedFilter, not on the device yet
                self.skipped[name] = p["unsupported"]
                return
            try:
                self.stc[p["tc_producer"]] = simpletc.settings(self.mods[p["tc_producer"]])
            except ValueError as e:
                self.skipped[name] = [str(e)]
                return
            self.plan[name] = {"tc": p["tc_producer"], "helix": None, "ksf": None, "filters": p["filters"],
                               "family": "cst", "min_helices": 1, "ps": ps}
            return
        if p["family"] == "cpr" and not self.cpr:
            self.skipped[name] = ["calorimeter-seeded: opt-in (Chain(cpr=True), with CaloDigis in the batch)"]
            return
        labels = phys[name]
        tcl = self._tc_label(p)
        mlbl = next((lbl for lbl in labels if self.mods.get(lbl, {}).get("module_type") == "RobustMultiHelixFinder"),
                    None)
        if mlbl is not None and not self.mpr:
            self.skipped[name] = ["multi-helix finder: opt-in (Chain(mpr=True), decide --mpr, TRIGGER_DEVICE_MPR=1)"]
            return
        if mlbl is not None:                     # mpr: RobustMultiHelixFinder + MergeHelices (mpr.py)
            merger = next((lbl for lbl in labels if self.mods.get(lbl, {}).get("module_type") == "MergeHelices"), None)
            fins = self.mods[merger]["HelixFinders"] if merger else []
            if fins != [f"{mlbl}:Positive", f"{mlbl}:Negative"] or self.mods[mlbl]["TimeClusterCollection"] != tcl:
                self.skipped[name] = [f"mpr layout not supported: {merger} {fins}"]
                return
            try:
                self.mprs[mlbl] = (mpr.settings(self.mods[mlbl]), mpr.merge_settings(self.mods[merger]))
            except ValueError as e:
                self.skipped[name] = [str(e)]
                return
            ksf = next((lbl for lbl in labels if lbl.startswith("Trig") and "KSF" in lbl and "MC" not in lbl), None)
            self.plan[name] = {"tc": tcl, "helix": None, "mpr": mlbl, "ksf": ksf, "filters": p["filters"],
                               "family": "mpr", "min_helices": p["min_helices"], "ps": ps}
            return
        hel = next((lbl for lbl in labels if self.mods.get(lbl, {}).get("module_type") in HELIX_TYPES), None)
        has_hs = any(role == "hs" for role, _, _ in p["filters"])
        if (has_hs and hel is None) or "proton" in name or p["unsupported"] or p["min_helices"] > 1:
            self.skipped[name] = p["unsupported"] or ["proton, multi-helix finder or minNHelices > 1"]
            return                               # e.g. MultiHelixFilter (several helices per cluster): not on the device
        chf = None
        if p["family"] == "cpr":                 # CalHelixFinder + MergeHelices (calhelix.py)
            chf = next((lbl for lbl in labels if self.mods.get(lbl, {}).get("module_type") == "MergeHelices"), None)
            fins = [f.split(":")[0] for f in self.mods[chf]["HelixFinders"]] if chf else []
            if hel is None or fins != [hel] or self.mods[hel].get("module_type") != "CalHelixFinder" or \
                    self.mods[hel]["TimeClusterCollectionLabel"] != tcl:
                self.skipped[name] = [f"cpr layout not supported: {hel} {chf} {fins}"]
                return
            try:
                self.chfs[chf] = (hel, calhelix.settings(self.mods[hel], self.mods[chf]))
            except ValueError as e:
                self.skipped[name] = [str(e)]
                return
        ahf = None
        if p["family"] == "apr" and self.apr_helix and hel and self.mods[hel].get("module_type") == "AgnosticHelixFinder":
            ahf = next((lbl for lbl in labels if self.mods.get(lbl, {}).get("module_type") == "MergeHelices"), None)
            fins = [f.split(":")[0] for f in self.mods[ahf]["HelixFinders"]] if ahf else []
            if fins != [hel] or self.mods[hel]["tcCollLabel"] != tcl:
                self.skipped[name] = [f"apr helix layout not supported: {hel} {ahf} {fins}"]
                return
            try:
                self.ahfs[ahf] = (hel, aprhelix.settings(self.mods[hel]), aprhelix.merge_settings(self.mods[ahf]))
            except ValueError as e:
                self.skipped[name] = [str(e)]
                return
        hcfg_label = self._helix_label(hel) if hel and chf is None and ahf is None else None
        if p["family"] == "cpr":                 # CalSeedFit (TrigCalSeedFitDe): found by its type
            ksf = next((lbl for lbl in labels if self.mods.get(lbl, {}).get("module_type") in KSF_TYPES), None)
        else:
            ksf = next((lbl for lbl in labels if lbl.startswith("Trig") and "KSF" in lbl and "MC" not in lbl), None)
        self.plan[name] = {"tc": tcl, "helix": hcfg_label, "chf": chf, "ahf": ahf, "ksf": ksf, "filters": p["filters"],
                           "family": p["family"], "min_helices": p["min_helices"], "ps": ps}

    def _tc_label(self, p):
        """Device time-cluster settings: the path's TimeClusterFinder, or the matching-pitch one for TZClusterFinder."""
        m = self.mods[p["tc_producer"]]
        if m["module_type"] in ("TimeClusterFinder", "CalTimePeakFinder") or (self.tz and m["module_type"] == "TZClusterFinder"):
            return p["tc_producer"]
        return "TrigTimeClusterFinder"

    def _helix_label(self, hel):
        m = self.mods[hel]
        if m["module_type"] == "RobustHelixFinder":
            return hel
        return "TrigHelixFinder"

    def _tc_settings(self, lbl):
        """TimeClusterFinder settings; with mpr paths, every producer also returns the stock hit order (`order`)."""
        return {**timecluster.settings(self.mods[lbl]), "order": bool(self.mprs)}

    def _tc_labels(self):
        """TimeClusterFinder settings the tpr/apr paths use (cst clusters come from SimpleTimeCluster)."""
        return sorted({p["tc"] for p in self.plan.values() if p["family"] in ("tpr", "apr", "mpr")} - set(self.tz_s))

    def _helix_keys(self):
        return sorted({(p["tc"], p["helix"]) for p in self.plan.values() if p.get("helix")})

    def _decide(self, batch):
        sh = hits.straw_hits(batch, self.cond_dev, self.sh, self.tot_bins)
        ph = self._flag(hits.panel_hits(sh, self.ph), batch)
        E = batch["valid"].shape[0]
        tcs, hxs = {}, {}
        for group in self._tc_groups():              # producers that differ only in pitch share one batched call
            ss = [self._tc_settings(lbl) for lbl in group]
            res = (timecluster.time_clusters_multi_pairs(ph, ss, self.net, K=E * len(group)) if self.pairs else
                   timecluster.time_clusters_multi(ph, ss, self.net))
            tcs.update(zip(group, res))
        cl = None
        if self.calo_s is not None:                  # calo clusters (cpr and tz paths)
            dg = {k: batch["cd_" + k] for k in CALO_KEYS}
            _, cl = calo.reconstruct(dg, batch["pbt"].astype(jnp.float32), self.calo_s, C=CPR_CLUSTERS)
        for lbl, s in self.tz_s.items():             # tz: TZClusterFinder port on the flagged panel hits + calo clusters
            # carry=False: the block's events need not be consecutive in file order; a calo-recovered chunk that needs
            # the stale energy selection of an earlier event is flagged (indeterminate) and not made an electron
            z = tzcluster.clusters({k: v for k, v in ph.items() if k != "df_err"}, cl, s, carry=False)
            tcs[lbl] = {"valid": z["valid"], "nsh": z["nsh"], "t0": z["t0"], "members": z["members"],
                        "calo": z["calo"], "order": z["order"], "overflow": (z["overflow"] != 0) | cl["overflow"],
                        "indeterminate": z["indeterminate"]}
        for mlbl, (hlbl, s, ms) in sorted(self.ahfs.items()):   # apr: AgnosticHelixFinder + MergeHelices
            tz = tcs[self.mods[hlbl]["tcCollLabel"]]
            seeds, _ = aprhelix.helices({k: v for k, v in ph.items() if k != "df_err"},
                                        {k: tz[k] for k in ("valid", "order", "calo")}, cl, s)
            hx = aprhelix.merge(seeds, ph["x"].shape[1], ms, self.fz)
            hxs[("ahf", mlbl)] = {**hx, "overflow": seeds["overflow"]}
        if self.cpr_tp:                              # cpr: CalTimePeakFinder on the panel hits and calo clusters
            for lbl, s in self.cpr_tp.items():
                pk = cprmod.time_peaks(ph, cl, s)
                # overflow: more clusters than CPR_CLUSTERS (the later ones are not seen), reported per path
                tcs[lbl] = {"valid": pk["tc"], "nsh": pk["nsh"], "t0": pk["t0"], "members": pk["hits"],
                            "overflow": cl["overflow"],
                            "cl": {k: cl[k] for k in ("disk", "time", "cog_x", "cog_y")}}
            # CalHelixFinder per (event, time peak), then MergeHelices per event over its seeds in peak order;
            # mergers whose settings agree share one batched call
            groups = {}
            for mlbl, (hlbl, s) in sorted(self.chfs.items()):
                groups.setdefault(tuple(sorted(s.items())), []).append(mlbl)
            for group in groups.values():
                s = self.chfs[group[0]][1]
                sets = [tcs[self.mods[self.chfs[m][0]]["TimeClusterCollectionLabel"]] for m in group]
                res = calhelix.helices(ph, sets, cl, s, self.chf_geom, self.fz)
                hxs.update({("cpr", m): r for m, r in zip(group, res)})
        for group in self._helix_groups():           # helix finders with the same settings share one batched call
            s = helix.settings(self.mods[group[0][1]])
            sets = [tcs[key[0]] for key in group]
            bkg = self._hbkg(group[0][1])
            res = (helix.helices_multi_pairs(ph, sets, s, self.fz, K=E * len(group), bkg_mask=bkg) if self.pairs else
                   helix.helices_multi(ph, sets, s, self.fz, bkg_mask=bkg))
            hxs.update(zip(group, res))
        return self._paths(batch, sh, ph, tcs, hxs)

    def _tc_groups(self):
        """Time-cluster producers grouped by their settings apart from the pitch, in label order."""
        groups = {}
        for lbl in self._tc_labels():
            s = self._tc_settings(lbl)
            groups.setdefault(tuple(sorted((k, v) for k, v in s.items() if k != "pitch")), []).append(lbl)
        return list(groups.values())

    def _helix_groups(self):
        """(time-cluster producer, helix finder) keys grouped by identical helix settings, in key order."""
        groups = {}
        for key in self._helix_keys():
            s = helix.settings(self.mods[key[1]])
            # cpr time peaks have another slot count than the time clusters: their own batched call
            groups.setdefault((key[0] in self.cpr_tp, key[0] in self.tz_s) + tuple(sorted(s.items())), []).append(key)
        return list(groups.values())

    # compact path: candidates -> (host: counts) -> refined clusters on candidate pairs -> (host: counts) -> helices on
    # valid pairs and the path decisions
    def _stage_candidates(self, batch):
        sh = hits.straw_hits(batch, self.cond_dev, self.sh, self.tot_bins)
        ph = self._flag(hits.panel_hits(sh, self.ph), batch)
        cands = {lbl: timecluster.candidates(ph, self._tc_settings(lbl)) for lbl in self._tc_labels()}
        return sh, ph, cands, {lbl: jnp.sum(c[2]) for lbl, c in cands.items()}

    def _refine_one(self, cand, lbl, k):
        tc = timecluster.refine_pairs(*cand, self._tc_settings(lbl), self.net, k)
        return tc, jnp.sum(tc["valid"])

    def _helix_one(self, ph, tc, key, k):
        return helix.helices_pairs(ph, tc, helix.settings(self.mods[key[1]]), self.fz, k, bkg_mask=self._hbkg(key[1]))

    def _flag(self, ph, batch):
        """TrigFlagPH: DeltaFinder's flags on the panel hits, when the port is on (its error bits ride along)."""
        if self.df is None:
            return ph
        fl, _, err, _ = dfmod.flag_hits({**ph, "spill": batch["spill"]}, *self.df)
        return {**ph, "flag": fl.astype(ph["flag"].dtype), "df_err": err}

    def _hbkg(self, lbl):
        """A helix finder's HitBackgroundBits as a mask, when DeltaFinder flags the hits (else 0, as before)."""
        if self.df is None:
            return 0
        return hits.bits([b for b in self.mods[lbl].get("HitBackgroundBits", []) if b in hits.STRAW_FLAGS])

    def _packed(self, batch, res):
        out, aux, _ = res
        layout = pack_layout(out, aux)
        self._layouts[tuple(batch["valid"].shape)] = layout      # trace time
        return pack(out, aux, layout)

    def _decide_packed(self, batch):
        return self._packed(batch, self._decide(batch))

    def _paths_packed(self, batch, sh, ph, tcs, hxs):
        return self._packed(batch, self._paths(batch, sh, ph, tcs, hxs))

    def _decide_compact(self, batch, packed=False):
        """The compact path (one jitted call per stage and producer; compiles add up over producers instead of
        multiplying): candidates, the counts to the host, refined clusters on the candidate pairs, the valid counts to
        the host, helices on the valid pairs, then the path decisions."""
        sh, ph, cands, n = self._fa(batch)
        n = jax.device_get(n)
        slots = cands[self._tc_labels()[0]][2].size if self._tc_labels() else 0
        tcs, nv = {}, {}
        for lbl in self._tc_labels():
            tcs[lbl], nv[lbl] = self._fr(cands[lbl], lbl=lbl, k=pair_bucket(int(n[lbl]), slots))
        nv = jax.device_get(nv)
        hxs = {key: self._fh(ph, tcs[key[0]], key=key, k=pair_bucket(int(nv[key[0]]), slots))
               for key in self._helix_keys()}
        return (self._fpp if packed else self._fp)(batch, sh, ph, tcs, hxs)

    def warmup(self, batch):
        """Compact path: compile every pair capacity of every stage for the shapes of `batch` (a packed block; its
        contents are not used), so a job pays the compiles of a hit-axis size once, up front, instead of whenever a
        batch first reaches a new capacity."""
        if not self.compact:
            return
        b = {k: jnp.zeros_like(jnp.asarray(v)) for k, v in batch.items()}
        sh, ph, cands, _ = self._fa(b)
        slots = cands[self._tc_labels()[0]][2].size if self._tc_labels() else 0
        ks = sorted({pair_bucket(n, slots) for n in (0, slots // 4 + 1, slots)})
        tcs = {}
        for lbl in self._tc_labels():
            for k in ks:
                tcs[lbl], _ = self._fr(cands[lbl], lbl=lbl, k=k)
        hxs = {}
        for key in self._helix_keys():
            for k in ks:
                hxs[key] = self._fh(ph, tcs[key[0]], key=key, k=k)
        jax.block_until_ready(self._fpp(b, sh, ph, tcs, hxs))

    def _paths(self, batch, sh, ph, tcs, hxs):
        out, aux = {}, {}
        if "df_err" in ph:                       # events where DeltaFinder reaches a case stock leaves undefined
            aux["TrigFlagPH:err"] = ph["df_err"]
        ev = batch["event"]
        diag_off = {}
        if self.stc:                             # cst paths: off-spill hits and SimpleTimeCluster clusters
            sh2 = hits.reflag_straw_hits(sh, self.sh_off)
            ph2 = hits.panel_hits_other(sh, {k: v for k, v in ph.items() if k != "df_err"}, self.ph, sh2, self.ph_off)
            tcs = {**tcs, **{lbl: simpletc.clusters(ph2, s) for lbl, s in self.stc.items()}}
            diag_off = {"sh_offsp": sh2, "ph_offsp": ph2}
        if self.mprs:                            # mpr paths: RobustMultiHelixFinder + MergeHelices on their clusters
            hxs = dict(hxs)
            for lbl, (sm, mm) in self.mprs.items():
                tcl = self.mods[lbl]["TimeClusterCollection"]
                hxs[("mpr", lbl)] = mpr.helices(ph, tcs[tcl], sm, mm, self.fz)
        nsd = jnp.sum(batch["valid"], axis=1)    # straw digis of the event (DigiFilter)
        tc_nsh = {lbl: jnp.max(jnp.where(tc["valid"], tc["nsh"], 0), axis=1) for lbl, tc in tcs.items()}
        if self.lumi:                            # the LumiStreamFilter counter runs on the host, per subrun
            aux["run"], aux["subrun"] = batch["run"], batch["subrun"]
        for name, p in sorted(self.plan.items()):
            k = self.draw_index.get(name)              # None for the families without draws
            ok = batch["valid_event"] & paths.prescale_pass(ev, batch["spill"], p["ps"])
            if p["family"] in ("disabled", "digi", "lumi"):
                for role, lbl, fc in p["filters"]:
                    if role == "DigiFilter":
                        ok = ok & paths.digi_pass(nsd, fc)
                out[name] = ok if p["family"] != "disabled" else jnp.zeros_like(ok)
                continue
            tc = tcs[p["tc"]]
            hx = hxs.get(("mpr", p["mpr"]) if p["family"] == "mpr" else
                         ("cpr", p["chf"]) if p.get("chf") else ("ahf", p["ahf"]) if p.get("ahf") else
                         (p["tc"], p["helix"]))
            if p["family"] == "mpr":
                aux[name + ":overflow"] = hx["overflow"]           # time clusters with more hits than the finder's slots
            if p["tc"] in self.tz_s:
                aux[name + ":tzoverflow"] = tc["overflow"]         # a TZ capacity, or calo clusters past CPR_CLUSTERS
                aux[name + ":tzindet"] = tc["indeterminate"]       # recovered chunks with an unknown stale selection
            if p.get("ahf"):
                aux[name + ":ahoverflow"] = hx["overflow"] != 0    # an AgnosticHelixFinder capacity (aprhelix.py)
            if p["family"] == "cpr":
                aux[name + ":overflow"] = tc["overflow"]           # events with more calo clusters than CPR_CLUSTERS
                aux[name + ":hxoverflow"] = hx["overflow"]        # time peaks with more hits than the finder's slots
                aux[name + ":hxerr"] = hx["err"]                 # stock behaviour the finder flags (calhelix.py)
            hs_mask = None
            binned = self.response.get(name, {}).get("final")
            for role, lbl, fc in p["filters"]:
                if role == "sd":
                    ok = ok & paths.digi_pass(nsd, fc)
                elif role == "tc":
                    if p["family"] == "apr":   # TZClusterFinder makes no cluster below clusterThresh straw hits
                        thr = int(self.mods[self.dpaths[name]["tc_producer"]].get("clusterThresh", 0))
                        fc = {**fc, "minNStrawHits": max(int(fc["minNStrawHits"]), thr)}
                    if p["family"] == "cpr":   # every time peak has its calo cluster
                        fc = {**fc, "requireCaloCluster": False}
                    tc_ok = paths.tc_pass(tc, fc)
                    ok = ok & tc_ok
                    aux[name + ":tc"] = ok
                elif role == "hs":
                    hp = paths.helix_pass(hx, fc)
                    hs_mask = hp
                    ok = ok & (jnp.sum(hp, axis=1) >= int(fc.get("minNHelices", 1)))
                    aux[name + ":hs_raw"] = ok
                    f = 1.0 if binned else float(self.response.get(name, {}).get("hs", 1.0))
                    if f < 1.0:
                        ok = ok & (paths.draw(ev, self.salt + 1000 + k) < f)
                    aux[name + ":hs"] = ok
                elif role == "ks":
                    mom = float(self.mods[p["ksf"]].get("HelixMask", {}).get("MinHelixMom", 0.0)) if p["ksf"] else 0.0
                    ok = ok & paths.kalseed_pass(hx, fc, mom, 1.0, ev, self.salt + k)
                    aux[name + ":ks_raw"] = ok
                    f = 1.0 if binned else float(self.response.get(name, {}).get("ks", 1.0))
                    if f < 1.0:
                        ok = ok & (paths.draw(ev, self.salt + k) < f)
            aux[name + ":tcnsh"] = tc_nsh[p["tc"]]
            fdt = jnp.float32 if self.precision == "float32" else jnp.float64
            hp = jnp.max(jnp.where(hx["ok"], hx["p"], 0.0), axis=1).astype(fdt) if hx is not None else None
            if hp is not None:
                aux[name + ":help"] = hp
            if binned:
                aux[name + ":raw"] = ok
                fb = jnp.asarray(binned["f"], fdt)
                edges = jnp.asarray(binned["edges"], fdt)
                ib = jnp.clip(jnp.searchsorted(edges, tc_nsh[p["tc"]].astype(fdt), side="right") - 1, 0, fb.shape[-1] - 1)
                if "edges_p" in binned and hp is not None:   # 2D: helix momentum rows x time-cluster hits
                    ep = jnp.asarray(binned["edges_p"], fdt)
                    jp = jnp.clip(jnp.searchsorted(ep, hp, side="right") - 1, 0, fb.shape[0] - 1)
                    fe = fb[jp, ib]
                else:
                    fe = fb[ib]
                ok = ok & (paths.draw(ev, self.salt + 2000 + k) < fe)
            out[name] = ok
            if hs_mask is not None:
                aux[name + ":hsmask"] = hs_mask
                aux[name + ":fcent"] = hx["fcent"]
        diag = {"sh": sh, "ph": ph, "tc": tcs, "hx": hxs, **diag_off} if self.debug else {}
        return out, aux, diag

    def decide(self, batch):
        b = {k: jnp.asarray(v) for k, v in batch.items()}
        return self._decide_compact(b) if self.compact else self._fn(b)

    def decide_packed(self, batch):
        """decide's decisions and stage arrays (no diagnostics), stacked on the device into one array per dtype and
        shape. Returns (key, arrays); after a device_get, `unpack(key, arrays)` gives decide's (out, aux) on the
        host, the same bits."""
        b = {k: jnp.asarray(v) for k, v in batch.items()}
        arrays = self._decide_compact(b, packed=True) if self.compact else self._fnp(b)
        return tuple(b["valid"].shape), arrays

    def unpack(self, key, arrays):
        return unpack(arrays, self._layouts[key])

    def finish(self, decisions, aux, n_events):
        """Host: the stateful D0-phi prescale of HelixFilter, counting invocations in event order."""
        out = {k: np.asarray(v)[:n_events] for k, v in decisions.items()}
        for name, p in self.plan.items():
            if p["family"] == "lumi":            # LumiStreamFilter: event counter per subrun, in event order
                fc = next(cf for t_, _, cf in p["filters"] if t_ == "LumiStreamFilter")
                out[name] = paths.lumi_stream(out[name], np.asarray(aux["run"])[:n_events],
                                              np.asarray(aux["subrun"])[:n_events], fc)
                continue
            for role, lbl, fc in p["filters"]:
                if role != "hs":
                    continue
                cuts = [fc[s] for s in ("posHelicitySelection", "negHelicitySelection")
                        if fc[s].get("configured") and fc[s].get("prescaleUsingD0Phi")]
                if not cuts:
                    continue
                reached = np.asarray(aux[name + ":tc"])[:n_events]
                calls = np.cumsum(reached)
                mask = np.asarray(aux[name + ":hsmask"])[:n_events]
                pres = paths.ipa_prescale(calls, np.asarray(aux[name + ":fcent"])[:n_events].astype(np.float32),
                                          cuts[0]["prescalerPar"])
                out[name] = out[name] & np.any(mask & pres, axis=1)
        return out
