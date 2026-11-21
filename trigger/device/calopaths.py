"""The calorimeter trigger paths on the device: calo hits and clusters (`calo.reconstruct`) and the filters of each
path, from the menu.

Paths handled (each: PrescaleEvent, the calo stage, one filter, MergeTriggerInfo): CaloClusterCounterFilter
(calo_cluster_*, calo_RMC), CaloLikelihood (calo_photon), CaloNNEval + FilterEcalNNTrigger (calo_NNCE),
CaloCosmicCalib (calo_cosmic); and the digi-count paths (PrescaleEvent + DigiFilter on CaloDigis: calo_N0Source,
minBias_CDCount). The PrescaleEvent factor is applied on the event number (`paths.prescale_pass`).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import calo, calogeom, mva
from .menu import prescale_factors
from .paths import SPILL, prescale_pass

STAGE = ("CaloHitMakerFast", "CaloClusterFast")
FILTERS = ("CaloClusterCounterFilter", "CaloLikelihood", "CaloNNEval", "FilterEcalNNTrigger", "CaloCosmicCalib")
PASS_THROUGH = ("PrescaleEvent", "MergeTriggerInfo", "EventHeaderFromCFOFragment")


def _modules(cfg):
    ph = cfg["physics"]
    return {**ph.get("producers", {}), **ph.get("filters", {})}


def plan(cfg, names=None):
    """{path: (kind, filter labels)} for the menu's calo paths the device evaluates."""
    mods = _modules(cfg)
    out = {}
    for entry in cfg["physics"]["trigger_paths"]:
        name = str(entry).split(":")[-1]
        if names is not None and name not in names:
            continue
        seq = cfg["physics"][name]
        types = [mods[m]["module_type"] for m in seq]
        rest = [m for m, t in zip(seq, types) if t not in PASS_THROUGH + STAGE]
        rtypes = [mods[m]["module_type"] for m in rest]
        if rtypes == ["DigiFilter"] and mods[rest[0]].get("useCaloDigi"):
            out[name] = ("digicount", rest)
        elif all(s in types for s in STAGE) and rtypes and all(t in FILTERS for t in rtypes):
            kinds = {"CaloClusterCounterFilter": "counter", "CaloLikelihood": "likelihood",
                     "CaloCosmicCalib": "cosmic"}
            kind = "nn" if rtypes == ["CaloNNEval", "FilterEcalNNTrigger"] else kinds.get(rtypes[0]) if len(rtypes) == 1 else None
            if kind:
                out[name] = (kind, rest)
    return out


class CaloStage:
    """The calo paths of a menu, jitted. Inputs per block: packed digis (`calo.pack_digis`), ProtonBunchTime,
    event numbers and the spill type of each event."""

    def __init__(self, cfg, templates=None, nn_weights=None, names=None):
        mods = _modules(cfg)
        self.cfg, self.mods = cfg, mods
        self.paths = plan(cfg, names)
        calib = cfg["services"]["ProditionsService"]["calCalib"]
        if calib.get("useDb"):
            raise ValueError("device calo stage needs calCalib from the configuration (useDb false)")
        self.geo = calogeom.build()
        self.s = calo.settings(mods["CaloHitMakerFast"], mods["CaloClusterFast"], calib, self.geo)
        self.tpl = self.net = None
        kinds = {k for k, _ in self.paths.values()}
        if "likelihood" in kinds:
            lh = next(mods[r[0]] for k, r in self.paths.values() if k == "likelihood")
            if templates is None:
                raise ValueError("calo_photon needs the likelihood templates (signal, background)")
            self.tpl = calo.load_templates(*templates, float(lh["MinClusterRadialDist"]),
                                           float(lh["ClusterRadialDistStep"]))
        if "nn" in kinds:
            if nn_weights is None:
                raise ValueError("calo_NNCE needs the NN weights")
            self.net = mva.load(nn_weights)
        self.factors = {p: prescale_factors(cfg, p) for p in self.paths}
        self._fn = jax.jit(self._decide)

    def _decide(self, dg, pbt, events, spill):
        hits, cl = calo.reconstruct(dg, pbt, self.s)
        out, diag = {}, {}
        with calo.strict(dg["sipm"]):
            for p, (kind, labels) in self.paths.items():
                m = [self.mods[x] for x in labels]
                if kind == "digicount":
                    ok = calo.digi_count_pass(dg["n"], m[0])
                elif kind == "counter":
                    ok = calo.counter_pass(cl, m[0])
                elif kind == "likelihood":
                    ok = calo.likelihood_pass(cl, hits, self.s, self.tpl, m[0])
                elif kind == "nn":
                    ok = calo.nn_pass(cl, hits, self.s, self.net, m[0], m[1])
                else:
                    ok = calo.cosmic_pass(cl, hits, self.s, m[0])
                out[p] = ok & prescale_pass(events, spill, self.factors[p])
        diag["n_hits"], diag["n_clusters"] = hits["n"], cl["n"]
        for k in ("overflow", "tie"):
            diag[f"hits_{k}"] = hits[k]
        for k in ("overflow", "tie", "fallback"):
            diag[f"clusters_{k}"] = cl[k]
        return out, diag, cl

    def decide(self, dg, pbt, events, spill):
        keys = ("valid", "sipm", "t0", "peakpos", "wf_len", "s0", "s1", "s2", "s3", "s_peak", "n")
        dev = {k: jnp.asarray(dg[k]) for k in keys}
        return self._fn(dev, jnp.asarray(pbt, jnp.float32), jnp.asarray(events), jnp.asarray(spill))


def spill_code(mode):
    return SPILL[mode]
