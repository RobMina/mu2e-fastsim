"""VetoCard: a CRV veto rule as data. A track is vetoed if any coincidence of its event passes the coincidence cuts
and lies in the time window dt = t_track - t_coinc in (dt_lo, dt_hi). Event-level gates (the beam-flash cut: every
coincidence must start after start_lo and end before end_hi) are separate, because the analyses treat them as a
selection, not as part of the veto (mu2e_cosmic_ana "within_coinc_start_time"/"within_coinc_end_time")."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class VetoCard:
    name: str = "custom"
    dt_lo: float = 0.0                 # ns, exclusive (t_track - t_coinc > dt_lo)
    dt_hi: float = 150.0               # ns, exclusive
    min_pe: float = 0.0                # coincidence PEs >= min_pe
    min_layers: int = 0                # coincidence nLayers >= min_layers
    sectors: Optional[tuple] = None    # sectorType values allowed to veto (None: all)
    min_pe_by_sector: Dict[int, float] = field(default_factory=dict)   # overrides min_pe per sectorType
    gate_start_lo: Optional[float] = None   # event gate: all coincidences timeStart > this (None: off)
    gate_end_hi: Optional[float] = None     # event gate: all coincidences timeEnd < this (None: off)
    gate_mode: str = "event"           # "event": a gated coincidence removes the event; "coinc": it only cannot veto
    source: str = ""                   # where the card came from (file, cut set)

    def sha(self) -> str:
        """Identity of the rule (goes into a kit's measure_sha)."""
        d = asdict(self)
        d.pop("source")
        if d["gate_mode"] == "event":  # default: keeps the sha of cards made before gate_mode existed
            d.pop("gate_mode")
        d["min_pe_by_sector"] = sorted((int(k), float(v)) for k, v in d["min_pe_by_sector"].items())
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, d: dict) -> "VetoCard":
        d = dict(d)
        if d.get("gate_mode", "event") not in ("event", "coinc"):
            raise ValueError(f"gate_mode must be 'event' or 'coinc', not {d['gate_mode']!r}")
        if "sectors" in d and d["sectors"] is not None:
            d["sectors"] = tuple(int(s) for s in d["sectors"])
        if "min_pe_by_sector" in d:
            d["min_pe_by_sector"] = {int(k): float(v) for k, v in d["min_pe_by_sector"].items()}
        return cls(**d)

    @classmethod
    def from_mu2e_cosmic_ana(cls, cuts_yaml: str, cutset: str) -> "VetoCard":
        """S. Grant's mu2e_cosmic_ana (config/common/cuts.yaml): the defaults overridden by the named cut set. Only the
        veto keys are read; the track selection stays with the analysis (rule E11)."""
        import yaml
        cfg = yaml.safe_load(open(cuts_yaml))
        th = dict(cfg["defaults"]["thresholds"]); act = dict(cfg["defaults"]["active"])
        cs = (cfg.get("cutsets") or {}).get(cutset)
        if cs is None and cutset != "defaults":
            raise KeyError(f"cut set {cutset!r} not in {cuts_yaml}")
        th.update((cs or {}).get("thresholds") or {}); act.update((cs or {}).get("active") or {})
        if not act.get("unvetoed", True):
            raise ValueError(f"cut set {cutset!r} has the veto switched off (active.unvetoed: false)")
        return cls(name=f"mu2e_cosmic_ana:{cutset}", dt_lo=float(th["lo_veto_dt_ns"]), dt_hi=float(th["hi_veto_dt_ns"]),
                   gate_start_lo=float(th["lo_crv_start_ns"]) if act.get("within_coinc_start_time") else None,
                   gate_end_hi=float(th["hi_crv_end_ns"]) if act.get("within_coinc_end_time") else None,
                   source=f"{cuts_yaml}#{cutset}")
