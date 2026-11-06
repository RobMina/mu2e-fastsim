"""Synthetic bookkeeping tests; no transported detector-response claim."""
import json

import numpy as np
import pytest

from beamfastsim.flash_inclusive_check import FIELDS, join, keys, main, summarize

BASE = 2**63


def table(tree, n, **columns):
    out = {field: np.zeros(n, dtype=np.uint64) for field in FIELDS[tree]}
    for field in ("label", "instance", "name", "material"):
        if field in out:
            out[field] = np.full(n, "", dtype="U20")
    for field in ("time", "total_edep", "nonionizing_edep", "px", "py", "pz"):
        if field in out:
            out[field] = np.zeros(n, dtype=np.float64)
    if "run" in out:
        out["run"][:] = 1
        out["subrun"][:] = 7
    out.update({field: np.broadcast_to(value, (n,)).copy() for field, value in columns.items()})
    return out


@pytest.fixture
def sample():
    pids = np.arange(1, 7, dtype=np.uint64) + np.uint64(BASE)
    products = table("products", 19,
                     event=np.r_[np.repeat([1, 2, 3], 6), 0],
                     product=np.r_[np.tile(pids, 3), np.uint64(BASE + 7)],
                     kind=np.r_[np.tile([3, 3, 3, 1, 4, 1], 3), 5],
                     scope=np.r_[np.zeros(18, int), 1],
                     size=[2, 7, 2, 4, 1, 2, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 3],
                     label=np.concatenate((np.tile(["g4run", "g4run", "BeamFilter", "g4run", "g4run", "BeamFilter"], 3), ["g4run"])),
                     instance=np.concatenate((np.tile(["photonSteps", "tracker", "Beam", "", "", ""], 3), [""])))
    particles = table("particles", 9, event=[1, 1, 1, 1, 2, 3, 1, 1, 3],
                      product=np.array([BASE+4]*6 + [BASE+6]*3, dtype=np.uint64),
                      key=[10, 11, 12, 13, 10, 11, 11, 12, 11], pdg=[22, 22, 22, 11, 22, 22, 11, -11, 13],
                      stage=[1, 2, 2, 2, 1, 2, 1, 1, 1], creation=[1, 3, 4, 5, 1, 3, 1, 1, 1],
                      start_volume=[9, 9, 99, 9, 9, 9, 9, 9, 9])
    steps = table("steps", 14, event=[1, 1, 3] + [1]*7 + [3] + [1, 1, 3],
                  product=np.array([BASE+1]*3 + [BASE+2]*8 + [BASE+3]*3, dtype=np.uint64),
                  particle_product=np.array([BASE+4]*11 + [BASE+6]*3, dtype=np.uint64),
                  particle_key=[11]*3 + [13]*7 + [11] + [11, 12, 11], particle_state=np.ones(14, np.uint32),
                  time=np.array([0]*3 + [-1, 10, 250, 251, 1695, 1705, 1945, 1946] + [0]*3, float),
                  total_edep=np.array([0]*3 + list(range(1, 9)) + [0]*3, np.float32),
                  nonionizing_edep=np.array([0]*3 + [.25]*8 + [0]*3, np.float32),
                  px=np.array([0]*11 + [3, 0, 0], float),
                  py=np.array([0]*11 + [4, 0, 0], float),
                  pz=np.array([0]*11 + [0, 2, 3], float))
    return dict(events=table("events", 3, event=[1, 2, 3], ordinal=[0, 1, 2], ptr_failures=[0, 2, 0]),
                products=products, particles=particles, steps=steps,
                statuses=table("statuses", 3, event=[1, 2, 3], product=np.full(3, BASE+5, np.uint64),
                               status=[0, 1, 0], overflow=[0, 1, 0], killed_step=[0, 2, 0], killed_field=[0, 0, 3], ntracks=[5, 0, 1]),
                volumes=table("volumes", 3, product=np.full(3, BASE+7, np.uint64), stage=[0, 1, 2],
                              index=[9, 9, 9], name=["upstream", "copied", "birthFoil"], material=["Al"]*3),
                completion=table("completion", 1, complete=np.array([1], np.uint32), events=[3], ordinal_base=[0], subrun_products=[1]))


def test_stages_zero_events_and_exact_product_keys(sample):
    report = summarize(sample, 3)
    assert report["events"]["observed"] == 3
    assert report["events"]["zero_photon_steps"] == 1
    assert report["events"]["zero_tracker_steps"] == 1
    assert report["completion"]["ok"]
    assert report["photons"]["newly_simulated"] == 3
    assert report["photons"]["copied_earlier_stage"] == 2
    assert report["per_event"]["new_photons"] == [2, 0, 1]
    assert report["per_event"]["recorded_photon_tracks"] == [1, 0, 1]
    assert report["photons"]["max_recorded_photon_steps_per_event"] == 2
    assert report["photons"]["by_stage_creation_birthVolume"] == {
        "stage": [2, 2], "creation": [3, 4], "birthVolumeIndex": [9, 99],
        "birthVolume": ["birthFoil", "<unresolved>"], "count": [2, 1]}
    assert report["joins"]["new_photons_without_birth_volume"] == 1
    assert report["source"]["pdg_counts"] == {"pdg": [-11, 11, 13], "count": [1, 1, 1]}
    assert report["source"]["momentum_magnitude_sum_mev_c"] == 10
    assert report["source"]["unresolved_particle_rows"] == 0
    assert report["products"]["particles"]["size_mismatches"] == 0
    assert "NOT valid S2" in report["scope"]


def test_strict_stock_fmod_all_species_and_status(sample):
    report = summarize(sample, 3)
    assert report["tracker"]["raw"] == dict(steps=8, total_mev=36., nonionizing_mev=2., ionizing_mev=34.)
    assert report["tracker"]["stock_time_selected"] == dict(steps=4, total_mev=18., nonionizing_mev=1., ionizing_mev=17.)
    assert report["per_event"]["tracker_stock_time_ionizing_mev"] == [9.25, 0., 7.75]
    assert report["statuses"]["overflow"] == 1
    assert report["statuses"]["killed_step"] == 2
    assert report["statuses"]["killed_field"] == 3
    assert report["pointers"]["exporter_failures"] == 2


def test_missing_joins_do_not_silently_resolve_or_invent_empty_products(sample):
    sample["steps"]["particle_key"][-1] = 999
    sample["steps"]["particle_state"][-2] = 2
    sample["steps"]["product"][0] = np.uint64(BASE+88)
    sample["products"]["label"][7] = "notTracker"
    report = summarize(sample, 4)
    assert not report["events"]["matches_expected"]
    assert report["source"]["unresolved_particle_rows"] == 2
    assert report["source"]["pdg_counts"] == {"pdg": [11], "count": [1]}
    assert report["joins"]["steps_without_product"] == 1
    assert report["joins"]["available_step_pointers_without_particle"] == 1
    assert report["products"]["tracker"]["missing_events"] == 1
    assert report["products"]["tracker"]["zero_events_with_product"] == 0
    assert report["products"]["photonSteps"]["size_mismatches"] == 1


def test_no_stage_metadata_never_calls_copied_photons_new(sample):
    sample["products"]["label"][-1] = "unknown"
    report = summarize(sample, 3)
    assert report["photons"]["newly_simulated"] == 0
    assert report["joins"]["g4run_photons_without_stage_metadata"] == 5


def test_cap_and_incomplete_marker(sample):
    sample["products"]["size"][0] = 1_000_000
    sample["completion"] = {field: arr[:0] for field, arr in sample["completion"].items()}
    report = summarize(sample, 3)
    assert report["photons"]["at_or_above_1M"]
    assert not report["completion"]["ok"]


def test_completely_empty_export(sample):
    empty = {name: {field: array[:0] for field, array in fields.items()} for name, fields in sample.items()}
    report = summarize(empty, 3)
    assert report["events"]["observed"] == 0
    assert not report["completion"]["ok"]
    assert report["photons"]["newly_simulated"] == 0
    assert report["tracker"]["raw"]["ionizing_mev"] == 0


def test_integer_identity_and_ambiguous_join():
    with pytest.raises(ValueError, match="integer"):
        keys({"key": np.array([1.])}, ("key",))
    target = keys({"key": np.array([BASE+1, BASE+2], np.uint64)}, ("key",))
    np.testing.assert_array_equal(join(target[::-1], target), [1, 0])
    with pytest.raises(ValueError, match="ambiguous"):
        join(target, np.repeat(target, 2))


def test_root_cli_roundtrip(sample, tmp_path, capsys):
    import uproot

    path, out = tmp_path / "inclusive.root", tmp_path / "summary.json"
    with uproot.recreate(path) as root:
        for name, arrays in sample.items():
            root.mktree("export/" + name, arrays)
    assert main(["--input", str(path), "--out", str(out), "--events", "3"]) == 0
    report = json.loads(out.read_text())
    assert report["photons"]["newly_simulated"] == 3
    assert report["tracker"]["stock_time_selected"]["ionizing_mev"] == 17.
    assert "s/file" in capsys.readouterr().out
    assert main(["--input", str(path), "--out", str(out), "--events", "4"]) == 1


def test_event_without_rows_and_orphan_event_are_not_renormalized(sample):
    sample["events"]["event"][-1] = 4
    report = summarize(sample, 3)
    assert report["events"]["observed"] == 3
    assert report["products"]["tracker"]["missing_events"] == 1
    assert report["joins"]["event_products_without_event"] == 6
    assert report["joins"]["steps_without_event"] == 3
    assert report["per_event"]["new_photons"] == [2, 0, 0]
    assert report["per_event"]["tracker_raw_ionizing_mev"] == [26.25, 0., 0.]


def test_subrun_qualified_volumes_and_full_uint64_particle_keys(sample):
    # Same EventID component/key/product in another SubRun must not alias.
    for name in ("events", "products", "particles", "steps", "statuses"):
        sample[name]["subrun"][sample[name]["event"] == 3] = 8
    sample["events"]["event"][-1] = 1
    for name in ("products", "particles", "steps", "statuses"):
        sample[name]["event"][sample[name]["event"] == 3] = 1
    # No destination volume metadata exists for SubRun 8.
    for name, field in (("particles", "key"), ("steps", "particle_key")):
        sample[name][field] = sample[name][field].astype(np.uint64) + np.uint64(BASE)
    report = summarize(sample, 3)
    assert report["joins"]["available_step_pointers_without_particle"] == 0
    assert report["photons"]["newly_simulated"] == 2
    assert report["joins"]["g4run_photons_without_stage_metadata"] == 1
    assert report["source"]["pdg_counts"] == {"pdg": [-11, 11, 13], "count": [1, 1, 1]}


def test_only_empty_collections_retain_all_events(sample):
    for name in ("particles", "steps"):
        sample[name] = {field: array[:0] for field, array in sample[name].items()}
    sample["products"]["size"][np.isin(sample["products"]["kind"], [1, 3])] = 0
    report = summarize(sample, 3)
    assert report["events"]["observed"] == report["events"]["zero_new_photons"] == 3
    assert report["products"]["photonSteps"]["zero_events_with_product"] == 3
    assert report["products"]["photonSteps"]["missing_events"] == 0
    assert report["per_event"]["tracker_raw_total_mev"] == [0., 0., 0.]
