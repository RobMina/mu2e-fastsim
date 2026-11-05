"""Saved-chain audits preserve photons and terminal parents without runtime transport."""
import copy

import numpy as np
import pytest

from beamfastsim import flash_checkpoint as C
from beamfastsim.flash_checkpoint_report import summarize, FROZEN, COUNTERS


def _arrays():
    data = dict(schema_version=np.array(1, np.int32), capacity=np.array(2, np.int64),
                parent_index=np.arange(3), pdg=np.array([11, -11, 11], np.int32), t0=np.arange(3.),
                species_codes=np.array([11, -11], np.int32), iterations=np.array([5, 5], np.int32),
                total_iterations=np.array([5, 5], np.int32))
    for code, ids in ((11, [0, 2]), (-11, [1])):
        ids = np.array(ids, np.int64)
        n = len(ids)
        prefix = f"species/{code}/"
        data[prefix + "ids"] = ids
        data[prefix + "key"] = np.array([1, 2], np.uint32)
        root = prefix + "state/"
        fields = {name: np.zeros(n) for name in FROZEN}
        fields.update({name: np.zeros(n, np.int32) for name in COUNTERS})
        fields.update(x=np.zeros((n, 3)), u=np.tile([0., 0., 1.], (n, 1)), cnt=np.zeros((n, 5), np.int32),
                      fate=np.array([0, 1] if code == 11 else [2], np.int32),
                      **{"slab/it": np.array(5, np.int32)})
        fields["count"][0] = fields["nph"][0] = 1
        fields.update({"records/valid": np.arange(2)[None, :] < fields["count"][:, None],
                       "records/energy": np.zeros((n, 2)), "records/mechanism": np.zeros((n, 2), np.int32),
                       "records/position": np.zeros((n, 2, 3)), "records/direction": np.zeros((n, 2, 3)),
                       "records/path": np.zeros((n, 2)), "records/time": np.zeros((n, 2))})
        fields["records/energy"][0, 0] = .5
        fields["records/mechanism"][0, 0] = 1
        data.update({root + name: value for name, value in fields.items()})
    return data


def _chain(tmp_path, change=None):
    initial = _arrays()
    binding = dict(source="test", settings=dict(h=1))
    first = tmp_path / "first.json"
    receipt = C.save(first, initial, binding)
    child = copy.deepcopy(initial)
    child["total_iterations"] += [3, 0]
    child["iterations"][:] = [3, 0]
    child["species/11/state/slab/it"] += 3
    child["species/11/state/fate"][0] = 3
    if change:
        change(child)
    second = tmp_path / "second.json"
    C.save(second, child, binding, parent_sha256=receipt["sha256"])
    return [first, second], child, binding


def test_chain_preserves_terminal_rows_and_existing_photons(tmp_path):
    paths, _, _ = _chain(tmp_path)
    report = summarize(paths)
    assert report["parents"] == 3
    assert report["initial_live_parents"] == 1
    assert report["initially_live_final_fates"]["upstream"] == 1
    assert report["checkpoints"][-1]["fate_counts"]["live"] == 0
    assert report["checkpoints"][-1]["retained_energy_mev"] == 1.
    assert report["checkpoints"][-1]["emitted_photons"] == 2


@pytest.mark.parametrize("name,value,match", [
    ("t0", np.array([1., 1., 2.]), "identity"),
    ("species/11/key", np.array([1, 9], np.uint32), "RNG key"),
    ("species/11/state/x", np.array([[0., 0., 0.], [1., 0., 0.]]), "Terminal"),
    ("species/11/state/fate", np.array([3, 2], np.int32), "Terminal"),
    ("species/11/state/records/energy", np.array([[.6, 0.], [0., 0.]]), "photon changed"),
    ("species/11/state/slab/it", np.array(7, np.int32), "iteration accounting"),
    ("iterations", np.array([2, 0], np.int32), "increment mismatch"),
    ("species/11/ids", np.array([0, 0]), "row/fate identity"),
    ("species/11/state/nvac", np.array([-1, 0], np.int32), "counter decreased"),
])
def test_chain_mutations_fail_closed(tmp_path, name, value, match):
    paths, _, _ = _chain(tmp_path, lambda child: child.__setitem__(name, value))
    with pytest.raises(ValueError, match=match):
        summarize(paths)


def test_new_birth_appends_without_changing_old_prefix(tmp_path):
    def append(child):
        root = "species/11/state/"
        child[root + "count"][0] = child[root + "nph"][0] = 2
        child[root + "records/valid"][0, 1] = True
        child[root + "records/energy"][0, 1] = .25
        child[root + "records/mechanism"][0, 1] = 1
        child[root + "eph"][0] += .25
    paths, _, _ = _chain(tmp_path, append)
    report = summarize(paths)
    assert report["checkpoints"][-1]["emitted_photons"] == 3
    assert report["checkpoints"][-1]["retained_energy_mev"] == 1.25


def test_removing_old_birth_is_not_hidden_by_counter_accounting(tmp_path):
    def remove(child):
        root = "species/11/state/"
        child[root + "count"][0] = child[root + "nph"][0] = 0
        child[root + "records/valid"][0, 0] = False
    paths, _, _ = _chain(tmp_path, remove)
    with pytest.raises(ValueError, match="counter decreased"):
        summarize(paths)


def test_foreign_binding_or_missing_link_rejected(tmp_path):
    paths, arrays, binding = _chain(tmp_path)
    with pytest.raises(ValueError, match="lineage"):
        summarize(paths[1:])
    foreign = tmp_path / "foreign.json"
    C.save(foreign, arrays, dict(source="other"), parent_sha256=C.sha256(paths[0].with_suffix(".npz")))
    with pytest.raises(ValueError, match="binding mismatch"):
        summarize([paths[0], foreign])


def test_missing_parent_schema_or_empty_chain_rejected(tmp_path):
    with pytest.raises(ValueError, match="At least"):
        summarize([])
    paths, _, _ = _chain(tmp_path, lambda child: child.pop("species/11/state/T"))
    with pytest.raises(ValueError, match="schema mismatch"):
        summarize(paths)


def test_cli_no_overwrite(tmp_path):
    from beamfastsim.flash_checkpoint_report import main
    paths, _, _ = _chain(tmp_path)
    out = tmp_path / "report.json"
    assert main([str(p) for p in paths] + ["--out", str(out)]) == 0
    with pytest.raises(FileExistsError):
        main([str(p) for p in paths] + ["--out", str(out)])
