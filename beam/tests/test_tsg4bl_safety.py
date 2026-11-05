"""Additional crossing-receipt safety tests; production coordinates remain unverified."""
import os

import numpy as np
import pytest
import uproot

from beamfastsim import tsg4bl as ref


def write(path, pdgs=(13, 11)):
    n = len(pdgs)
    data = {k: np.arange(n, dtype=np.float32) for k in ref.BRANCHES}
    data.update(EventID=np.arange(n, dtype=np.float32), TrackID=np.ones(n, dtype=np.float32),
                ParentID=np.zeros(n, dtype=np.float32), PDGid=np.array(pdgs, dtype=np.float32),
                Weight=np.ones(n, dtype=np.float32))
    with uproot.recreate(path) as root:
        root.mktree(ref.PLANE_PATHS["COL1in"], {k: v.dtype for k, v in data.items()}).extend(data)


@pytest.mark.parametrize("kind", ["uppercase", "uppercase_actual_input", "symlink_outputs", "hardlink_outputs"])
def test_output_aliases_and_numpy_implicit_extension_rejected(tmp_path, kind):
    source = tmp_path / "input.root"
    output = tmp_path / "out.npz"
    write(source)
    if kind.startswith("uppercase"):
        output = tmp_path / "out.NPZ"
        if kind == "uppercase_actual_input":
            source = tmp_path / "out.NPZ.npz"
            write(source)
    else:
        output.write_bytes(b"preserve existing output")
        sidecar = output.with_suffix(".json")
        if kind == "symlink_outputs":
            sidecar.symlink_to(output)
        else:
            os.link(output, sidecar)
    before = source.read_bytes()
    old_out = output.read_bytes() if output.exists() else None
    with pytest.raises(SystemExit):
        ref.main([str(source), "--plane", "COL1in", "--out", str(output)])
    assert source.read_bytes() == before
    if old_out is not None:
        assert output.read_bytes() == old_out
    else:
        assert not output.exists()


def test_raw_observations_scopes_and_empty_ranges(tmp_path):
    source = tmp_path / "input.root"
    write(source)
    _, receipt = ref.read([source], planes=["COL1in"])
    plane = receipt["files"][0]["planes"][0]
    selected = plane["selected_raw_observations"]
    all_species = plane["all_species_raw_observations"]
    assert selected["event_id_range"] == [0, 0]
    assert all_species["event_id_range"] == [0, 1]
    assert selected["raw_ranges"]["x"] == {"min": 0.0, "max": 0.0}
    assert all_species["raw_ranges"]["x"] == {"min": 0.0, "max": 1.0}
    assert all_species["raw_momentum_signs"]["Pz"] == {"negative": 0, "zero": 1, "positive": 1}
    _, empty = ref.read([source], planes=["COL1in"], pdgs=[-211])
    summary = empty["files"][0]["planes"][0]["selected_raw_observations"]
    assert summary["event_id_range"] is None
    assert summary["raw_ranges"]["t"] == {"min": None, "max": None}


def test_float32_nuclear_species_are_flagged_not_used_as_exact_isotopes(tmp_path):
    source = tmp_path / "nuclei.root"
    write(source, pdgs=(13, 1000020040))
    out, receipt = ref.read([source], planes=["COL1in"])
    assert out["PDGid"].tolist() == [13]
    assert receipt["pdg_precision_unsafe_rows"] == 1
    assert receipt["files"][0]["planes"][0]["pdg_precision_unsafe_rows"] == 1
    assert "not certified isotope" in receipt["pdg_precision"]
