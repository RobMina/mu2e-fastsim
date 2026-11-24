"""Build a Codabench bundle (results submission, one phase) for the fastsim benchmark: hep-simbench's builder
(hepsimbench.codabench) with this repository's synthetic example (benchmarks.fastsim_score.synthetic, which includes
the B1 calorimeter card), benchmarks/README.md as the overview page and the fastsim titles.

The reference data is the stock side and stays private. By default the bundle is built on the synthetic example, so
the bundle can be built and tested anywhere. With --reference DIR it holds a real reference tree (kept outside this
repository), and the bundle is for an unpublished competition or local scoring only. SU per event is never scored.

    python -m benchmarks.codabench.build_bundle OUTDIR [--reference DIR] [--solution DIR] [--title TEXT]

writes OUTDIR/bundle/ and OUTDIR/competition_bundle.zip (plus OUTDIR/example/ when the synthetic example is used).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hepsimbench import codabench as _codabench
from hepsimbench.codabench import SCORING, competition_yaml, zip_tree  # noqa: F401  (names kept for importers)

HERE = Path(__file__).resolve().parent
TITLE = "mu2e-fastsim three metrics (PLACEHOLDER TITLE)"


def build(out: Path, reference: Path | None = None, title: str = TITLE, solution: Path | None = None) -> Path:
    from benchmarks.fastsim_score import synthetic
    return _codabench.build(out, reference, title, solution, synthetic=synthetic,
                            overview=HERE.parent / "README.md", commit_title="mu2e-fastsim commit",
                            task_name="fastsim three metrics")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--solution", type=Path)
    ap.add_argument("--title", default=TITLE)
    a = ap.parse_args(argv)
    print(build(a.out, a.reference, a.title, a.solution))
    return 0


if __name__ == "__main__":
    sys.exit(main())
