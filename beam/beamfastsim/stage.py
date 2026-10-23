"""Stage the beam stage's public inputs from CVMFS (over HTTP from a Stratum-1, no mount) into a local directory.

For endpoints without /cvmfs (UVA): the field map (.header + .bin) and a geometry file with its whole #include tree,
laid out as on CVMFS so the defaults of beamfastsim work with BEAMFS_OFFLINE_ROOT and BEAMFS_MAPS pointed at it.
  python3 -m beamfastsim.stage OUT [--geom Offline/Mu2eG4/geom/geom_common.txt] [--map DSMap] [--workers 16]
Prints the two environment settings to export. An overlay geometry (absolute path, e.g. beam/geom/*.txt) is read from
disk and its Offline includes are fetched.
"""
import argparse
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from mu2efastsim import cvmfsget

REPO = "mu2e.opensciencegrid.org"
OFFLINE = "/Musings/Offline/v10_47_00"
MAPS = "/DataFiles/BFieldMaps/Mau13"
_INC = re.compile(r'^\s*#include\s+"([^"]+)"', re.M)


def _includes(text):
    return _INC.findall(re.sub(r'//[^\n]*', '', text))


class _Repos:
    """One cvmfsget.Repo per thread (its catalog cache holds sqlite connections, which are bound to their thread)."""

    def __init__(self):
        self._tl = threading.local()

    def _r(self):
        if not hasattr(self._tl, "r"):
            self._tl.r = cvmfsget.Repo(REPO)
        return self._tl.r

    def read(self, path):
        return self._r().read(path)

    def fetch(self, path, out):
        return self._r().fetch(path, out)


def stage(out, geom="Offline/Mu2eG4/geom/geom_common.txt", map_name="DSMap", workers=16, offline=OFFLINE):
    """Fetch map and geometry tree into out/{offline,maps}; returns (offline_root, maps_dir, n_files, seconds)."""
    t0 = time.time()
    repo = _Repos()
    off_root, maps_dir = os.path.join(out, "offline"), os.path.join(out, "maps")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(repo.fetch, f"{MAPS}/{map_name}.{ext}", os.path.join(maps_dir, f"{map_name}.{ext}"))
                for ext in ("header", "bin")]
        seen, frontier = set(), []
        if os.path.isabs(geom):
            with open(geom) as f:
                frontier = [i for i in _includes(f.read()) if i.startswith("Offline/")]
        else:
            frontier = [geom]
        while frontier:   # breadth-first: each level of the include tree fetched in parallel
            level = [p for p in dict.fromkeys(frontier) if p not in seen]
            seen.update(level)
            texts = list(ex.map(lambda p: repo.read(f"{offline}/{p}").decode(), level))
            for p, txt in zip(level, texts):
                dest = os.path.join(off_root, p)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "w") as f:
                    f.write(txt)
            frontier = [i for txt in texts for i in _includes(txt) if i not in seen]
        for f in futs:
            f.result()
    return off_root, maps_dir, len(seen) + 2, time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--geom", default="Offline/Mu2eG4/geom/geom_common.txt")
    ap.add_argument("--map", default="DSMap")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    off, maps, n, dt = stage(a.out, a.geom, a.map, a.workers)
    print(f"export BEAMFS_OFFLINE_ROOT={off} BEAMFS_MAPS={maps}   # {n} files in {dt:.1f} s")


if __name__ == "__main__":
    main()
