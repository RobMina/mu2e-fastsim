"""Fixed device shapes for the runchain JAX path (ChainConfig.capacities): no recompiles on new events.

The default JAX path pads every device call to a power-of-two bucket of its own size (photon rows, substeps and
arrivals per chunk, SiPM photons per chunk, waveform clusters and samples per call, refit rows and samples). New
input files bring new bucket combinations, so a warm pass on fresh events recompiles (31-47 compiles, 11-14 s per
200k generated cosmics on an A100). With a capacity file every such call takes a shape from the file instead: one
shape per buffer, sized from measured demand (``python3 -m crvfastsim.capacity`` over runs that recorded their usage,
x margin), so a warm pass compiles nothing.

The draws are made independent of the buffer sizes (SiPM: one key per hit section, ``sipm_jax.device_sipm(draws=
"section")``; waveform noise: one key per charge cluster), so the charges and digis do not depend on the capacities.
That is another realization than the bucketed default (same distributions, validated separately); the default path
is unchanged. A call whose demand exceeds its capacity is not truncated: it runs with the bucketed shape (exact, one
compile) and is counted in ``timers["capacity_overflow"]`` per buffer, so a capacity file that is too small shows up
in the report.

File format (JSON)::

    {"format": "crvfastsim-capacities-1", "margin": 1.25, "census": "...",
     "steps": N,                                                       # CRV steps of a file (event keys per step)
     "photon_block_rows": B,                                           # (step, SiPM) rows of one lookup-table block
     "photon_rows": R, "photon_substeps": S, "photon_arrivals": A,     # per photons_jax kernel chunk
     "photon_buffer": B,                                               # all arrivals of a file + one SiPM chunk
     "sipm_photons": N,                       # SiPM chunk size (resident path; also its max_photons), set by hand
     "synth_rows": {"Q": C, ...}, "synth_samples": {"Q": L, ...},     # per power-of-two charges-per-cluster Q
     "refit_rows": R2, "refit_samples": L2}                            # device refit of flagged waveforms
"""
import argparse
import json
import math
from pathlib import Path

FORMAT = "crvfastsim-capacities-1"
SCALARS = ("steps", "photon_block_rows", "photon_rows", "photon_substeps", "photon_arrivals", "photon_buffer",
           "sipm_photons", "refit_rows", "refit_samples")
FIXED = ("sipm_photons",)   # set by hand (the SiPM chunk size), not derived from demand
CLAMP = {"photon_substeps": 1 << 22}   # hard per-chunk limits of the kernels (photons_jax max_sub): never above
PER_Q = ("synth_rows", "synth_samples")


def _p2(n, lo=1):
    return max(lo, 1 << max(int(n) - 1, 0).bit_length())


class Capacities:
    """Shapes from a capacity file plus a usage recorder (max demand per buffer) and overflow counts."""

    def __init__(self, spec=None, record=False):
        spec = dict(spec or {})
        if spec and spec.get("format") != FORMAT:
            raise ValueError(f"capacity file format {spec.get('format')!r}, expected {FORMAT!r}")
        unknown = set(spec) - set(SCALARS) - set(PER_Q) - {"format", "margin", "census", "note"}
        if unknown:
            raise ValueError(f"unknown capacity entries {sorted(unknown)}")
        self.spec = spec
        self.record = record or not spec
        self.usage = {}
        self.overflow = {}

    @classmethod
    def load(cls, path, record=False):
        return cls(json.loads(Path(path).read_text()), record=record)

    def note(self, name, demand):
        if self.record:
            self.usage[name] = max(self.usage.get(name, 0), int(demand))

    def cap(self, name, q=None):
        """The file's capacity for a buffer (per Q for synth_*), or None."""
        cap = self.spec.get(name)
        if isinstance(cap, dict):
            cap = cap.get(str(q))
        return None if cap is None else int(cap)

    def overflowed(self, name, count=1, q=None):
        key = name if q is None else f"{name}/{q}"
        self.overflow[key] = self.overflow.get(key, 0) + int(count)

    def size(self, name, demand, bucket, q=None):
        """Shape for a call needing `demand`: the file's capacity, or `bucket` (the default path's shape) when the
        file has none for this buffer or the demand exceeds it (counted as an overflow)."""
        key = name if q is None else f"{name}/{q}"
        self.note(key, demand)
        cap = self.spec.get(name)
        if isinstance(cap, dict):
            cap = cap.get(str(q))
        if cap is None:
            return bucket
        if demand > cap:
            self.overflow[key] = self.overflow.get(key, 0) + 1
            return bucket
        return int(cap)

    def report(self):
        return {"usage": dict(sorted(self.usage.items())), "overflow": dict(sorted(self.overflow.items()))}


def derive(usage, margin=1.25, census="", fixed=None):
    """Capacity file from the largest recorded demand per buffer (x margin; sample lengths rounded up to 8)."""
    if margin < 1:
        raise ValueError("margin must be at least 1")
    out = {"format": FORMAT, "margin": margin, "census": census}
    grow = lambda v, step=1: -(-int(math.ceil(margin * v)) // step) * step
    for name, value in usage.items():
        base, _, q = name.partition("/")
        if base in FIXED:
            continue
        if base in SCALARS:
            out[base] = min(grow(value, 8 if base == "refit_samples" else 1), CLAMP.get(base, 1 << 62))
        elif base in PER_Q:
            out.setdefault(base, {})[q] = grow(value, 8 if base == "synth_samples" else 1)
    out.update(fixed or {})
    for base in PER_Q:
        if base in out:
            out[base] = dict(sorted(out[base].items(), key=lambda kv: int(kv[0])))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Capacity file from runchain_bench reports that recorded their usage "
                                            "(--record-capacities): ceil(margin x largest demand) per buffer.")
    p.add_argument("reports", nargs="+", type=Path, help="runchain_bench --out JSON files")
    p.add_argument("--margin", type=float, default=1.25)
    p.add_argument("--sipm-photons", type=int, default=1 << 22, help="SiPM chunk size (fixed, not derived)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    usage = {}
    for r in a.reports:
        u = json.loads(r.read_text()).get("runchain", {}).get("capacity", {}).get("usage")
        if not u:
            raise SystemExit(f"{r}: no recorded capacity usage (run with --record-capacities)")
        for k, v in u.items():
            usage[k] = max(usage.get(k, 0), int(v))
    caps = derive(usage, a.margin, census=" ".join(r.name for r in a.reports),
                  fixed={"sipm_photons": a.sipm_photons})
    with a.out.open("x") as f:
        f.write(json.dumps(caps, indent=1) + "\n")
    print(json.dumps(caps, indent=1))


if __name__ == "__main__":
    main()
