"""Configuration manifests of a reco response build: the stock inputs a response table is measured with, a provenance
hash over them, and the FHiCL files of the stock chain rendered from them.

A manifest (JSON) holds:

  release        the stock software the chain ran with (e.g. the backing Musing and a hash of the muse area)
  geometry       GeometryService input file
  db             {"purpose": ..., "version": ...} (DbService)
  digi_seed      SeedService base seed of the digitisation
  digi, reco, ntuple
                 {"include": base FCL, "lines": [FHiCL lines appended after the include]}: output names, and any
                 change of a stock input (a straw-status list, a fit setting)
  sample         the Geant4 reference sample: {"dataset": ..., "development": [sequencers], "held_out": [...]}
  table          build options (`pools`, `min_count`)

The hash covers every field (canonical JSON: sorted keys, no whitespace), so a change of any stock input gives a new
hash. `table.build(..., config=manifest)` stores the manifest and its hash in the table, and
`DeviceChain(response=..., response_config=...)` refuses a table whose hash differs from the requested one.

CLI:
  python3 -m trkfastsim.recoresp.config hash MANIFEST
  python3 -m trkfastsim.recoresp.config render MANIFEST OUTDIR    (OUTDIR/{digi,reco,ntuple}.fcl and manifest.json)
  python3 -m trkfastsim.recoresp.config register TABLE [--registry PATH] [--url-base URL]
                 (a registry entry for a table built with config=: file name, sha256, URL of the release asset)
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

STAGES = ("digi", "reco", "ntuple")
URL_BASE = "https://github.com/RobMina/mu2e-fastsim/releases/download/resp-{hash}/"   # release assets per hash
FIELDS = ("release", "geometry", "db", "digi_seed", "digi", "reco", "ntuple", "sample", "table")
HASH_LEN = 16


def load(m):
    """A manifest from a dict, a JSON text or a path; checks its fields."""
    if isinstance(m, dict):
        m = copy.deepcopy(m)
    else:
        s = str(m)
        m = json.loads(s) if s.lstrip().startswith("{") else json.loads(Path(s).read_text())
    missing = [k for k in FIELDS if k not in m]
    if missing:
        raise ValueError(f"manifest misses {missing}")
    for st in STAGES:
        if "include" not in m[st]:
            raise ValueError(f"manifest stage {st} has no include")
        m[st].setdefault("lines", [])
    return m


def canonical(m):
    return json.dumps(load(m), sort_keys=True, separators=(",", ":"))


def digest(m):
    """Provenance hash of a manifest (first HASH_LEN hex digits of the SHA-256 of its canonical JSON)."""
    return hashlib.sha256(canonical(m).encode()).hexdigest()[:HASH_LEN]


def expected_hash(c):
    """The hash a table must carry for c: a manifest (dict, JSON text or path) or a hash string itself."""
    s = str(c) if not isinstance(c, dict) else None
    if s is not None and len(s) == HASH_LEN and all(ch in "0123456789abcdef" for ch in s):
        return s
    return digest(c)


def render(m, stage):
    """FHiCL text of one stage of the stock chain: the include, the manifest's lines, geometry, DbService (and the
    digitisation seed)."""
    m = load(m)
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    out = [f'#include "{m[stage]["include"]}"', *m[stage]["lines"],
           f'services.GeometryService.inputFile : "{m["geometry"]}"',
           f'services.DbService.purpose : {m["db"]["purpose"]}', f'services.DbService.version : {m["db"]["version"]}']
    if stage == "digi":
        out.append(f'services.SeedService.baseSeed : {int(m["digi_seed"])}')
    return "\n".join(out) + "\n"


def register(table_path, registry_path=None, url_base=URL_BASE):
    """Add the table at table_path (built with table.build(..., config=)) to the registry under its config hash:
    file name ce_loophelix_<hash>.npz, sha256, URL (url_base with {hash}), reference text and knobs. Returns the entry."""
    import numpy as np
    from . import table as RT
    with np.load(table_path) as f:
        if "config_hash" not in f.files:
            raise ValueError(f"{table_path}: built without a config (no config_hash)")
        h, m = str(f["config_hash"]), load(str(f["config"]))
        ref = str(f["reference"]) if "reference" in f.files else ""
    name = f"ce_loophelix_{h}.npz"
    e = dict(file=name, sha256=RT._sha256(table_path), url=url_base.format(hash=h) + name, reference=ref,
             knobs=m.get("knobs", {}), release=m["release"], sample=m["sample"]["dataset"])
    p = Path(registry_path) if registry_path else RT.REGISTRY
    reg = json.loads(p.read_text()) if p.exists() else {}
    reg[h] = e
    p.write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")
    return e


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m trkfastsim.recoresp.config", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("hash").add_argument("manifest")
    r = sub.add_parser("render")
    r.add_argument("manifest")
    r.add_argument("outdir")
    g = sub.add_parser("register")
    g.add_argument("table")
    g.add_argument("--registry", default=None, help="registry JSON (default: the shipped recoresp/data/registry.json)")
    g.add_argument("--url-base", default=URL_BASE)
    a = ap.parse_args(argv)
    if a.cmd == "register":
        print(json.dumps(register(a.table, a.registry, a.url_base)))
        return
    m = load(a.manifest)
    if a.cmd == "render":
        o = Path(a.outdir)
        o.mkdir(parents=True, exist_ok=True)
        for st in STAGES:
            (o / f"{st}.fcl").write_text(render(m, st))
        (o / "manifest.json").write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    print(digest(m))


if __name__ == "__main__":
    main()
