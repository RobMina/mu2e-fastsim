"""A partial or duplicated G4 matrix is not an independent complete comparison."""
import json

import numpy as np
import pytest

from beamfastsim import foilcells
from beamfastsim.reference_audit import _identity


def campaign(tmp_path):
    root = tmp_path / "campaign"
    run = root / "moved_edge_loss_finest_mup"
    run.mkdir(parents=True)
    path = run / "scan.npz"
    np.savez(path, other=np.zeros(3, bool), exit_MeV=np.ones(3))
    manifest = dict(complete=True, runs={run.name: dict(scan="moved_edge", mode="loss_finest", pdg=-13,
                                                        seed_offset=27, scan_npz=_identity(path))})
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_duplicate_seed_not_pooled(tmp_path):
    root = campaign(tmp_path)
    with pytest.raises(ValueError, match="duplicate response sample"):
        foilcells.load_runs([root, root])


def test_missing_reference_rejected_before_sampling(tmp_path):
    root = campaign(tmp_path)
    with pytest.raises(ValueError, match="missing required references"):
        foilcells.main(["--campaign", str(root), "--slab-tables", "not-needed", "--out", str(tmp_path / "out"),
                        "--cells", "moved_edge", "--charges", "-13", "--samples", "10"])
    assert not (tmp_path / "out/summary.json").exists()


def test_changed_sample_rejected(tmp_path):
    root = campaign(tmp_path)
    np.savez(root / "moved_edge_loss_finest_mup/scan.npz", other=np.zeros(4, bool))
    with pytest.raises(ValueError):
        foilcells.load_runs([root])
