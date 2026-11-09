"""Device trigger chain: StrawDigis -> hits -> time clusters -> helices -> per-path decisions.

`Chain(menu_cfg, conditions, tc_mva, precision="float64"|"float32")` compiles once per padded batch shape; `Chain.decide(batch)` returns per-path
decisions [E] for every tracker path it evaluates, all on the device; only the stateful D0-phi prescale counter is
applied on the host afterwards (`Chain.finish`).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import helix, hits, menu, mva, paths, timecluster

HELIX_TYPES = ("RobustHelixFinder", "AgnosticHelixFinder")
KSF_TYPES = ("LoopHelixFit", "KinKalSeedFit", "LoopHelixDeFit")
PRECISIONS = ("float64", "float32")


class Chain:
    def __init__(self, cfg, cond, tc_mva_path, response=None, path_names=None, salt=20261107, debug=False,
                 precision="float64"):
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        # float32: every device array, including the stock double-precision intermediates of the hit ports, in
        # 32-bit (JAX's x64 flag is process-wide, so set it before any device array is made)
        jax.config.update("jax_enable_x64", precision == "float64")
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
        for name, p in self.dpaths.items():
            if p["family"] == "cst":
                continue                         # cosmic SimpleTimeCluster paths: not on the device yet
            labels = phys[name]
            tcl = self._tc_label(p)
            hel = next((lbl for lbl in labels if self.mods.get(lbl, {}).get("module_type") in HELIX_TYPES), None)
            has_hs = any(role == "hs" for role, _, _ in p["filters"])
            if (has_hs and hel is None) or "proton" in name or p["unsupported"] or p["min_helices"] > 1:
                self.skipped[name] = p["unsupported"] or ["proton, multi-helix finder or minNHelices > 1"]
                continue                         # e.g. MultiHelixFilter (several helices per cluster): not on the device
            hcfg_label = self._helix_label(hel) if hel else None
            ksf = next((lbl for lbl in labels if lbl.startswith("Trig") and "KSF" in lbl and "MC" not in lbl), None)
            self.plan[name] = {"tc": tcl, "helix": hcfg_label, "ksf": ksf, "filters": p["filters"],
                               "family": p["family"], "min_helices": p["min_helices"]}
        self._fn = jax.jit(self._decide)

    def _tc_label(self, p):
        """Device time-cluster settings: the path's TimeClusterFinder, or the matching-pitch one for TZClusterFinder."""
        m = self.mods[p["tc_producer"]]
        if m["module_type"] == "TimeClusterFinder":
            return p["tc_producer"]
        return "TrigTimeClusterFinder"

    def _helix_label(self, hel):
        m = self.mods[hel]
        if m["module_type"] == "RobustHelixFinder":
            return hel
        return "TrigHelixFinder"

    def _decide(self, batch):
        sh = hits.straw_hits(batch, self.cond_dev, self.sh, self.tot_bins)
        ph = hits.panel_hits(sh, self.ph)
        tcs = {lbl: timecluster.time_clusters(ph, timecluster.settings(self.mods[lbl]), self.net)
               for lbl in {p["tc"] for p in self.plan.values()}}
        hxs = {}
        for p in self.plan.values():
            if p["helix"] and (p["tc"], p["helix"]) not in hxs:
                hxs[(p["tc"], p["helix"])] = helix.helices(ph, tcs[p["tc"]], helix.settings(self.mods[p["helix"]]), self.fz)
        out, aux = {}, {}
        ev = batch["event"]
        tc_nsh = {lbl: jnp.max(jnp.where(tc["valid"], tc["nsh"], 0), axis=1) for lbl, tc in tcs.items()}
        for k, (name, p) in enumerate(sorted(self.plan.items())):
            tc = tcs[p["tc"]]
            ok = batch["valid_event"]
            hx = hxs.get((p["tc"], p["helix"]))
            hs_mask = None
            binned = self.response.get(name, {}).get("final")
            for role, lbl, fc in p["filters"]:
                if role == "tc":
                    if p["family"] == "apr":   # TZClusterFinder makes no cluster below clusterThresh straw hits
                        thr = int(self.mods[self.dpaths[name]["tc_producer"]].get("clusterThresh", 0))
                        fc = {**fc, "minNStrawHits": max(int(fc["minNStrawHits"]), thr)}
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
            if binned:
                aux[name + ":raw"] = ok
                edges = jnp.asarray(binned["edges"], tc_nsh[p["tc"]].dtype)
                fb = jnp.asarray(binned["f"], jnp.float32 if self.precision == "float32" else jnp.float64)
                ib = jnp.clip(jnp.searchsorted(edges, tc_nsh[p["tc"]], side="right") - 1, 0, fb.shape[0] - 1)
                ok = ok & (paths.draw(ev, self.salt + 2000 + k) < fb[ib])
            out[name] = ok
            if hs_mask is not None:
                aux[name + ":hsmask"] = hs_mask
                aux[name + ":fcent"] = hx["fcent"]
        diag = {"sh": sh, "ph": ph, "tc": tcs, "hx": hxs} if self.debug else {}
        return out, aux, diag

    def decide(self, batch):
        return self._fn({k: jnp.asarray(v) for k, v in batch.items()})

    def finish(self, decisions, aux, n_events):
        """Host: the stateful D0-phi prescale of HelixFilter, counting invocations in event order."""
        out = {k: np.asarray(v)[:n_events] for k, v in decisions.items()}
        for name, p in self.plan.items():
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
