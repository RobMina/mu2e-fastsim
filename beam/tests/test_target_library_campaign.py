"""Synthetic offline inventory contracts; heavy ROOT metadata IO is monkeypatched."""
import copy
import json
import numpy as np
import pytest

from beamfastsim import target_library_audit as A
from beamfastsim import target_library_campaign as C


def event(run, evt, process, trigger):
    text = (f"PrintModule Run/Subrun/Event {run} 1 {evt}\n"
            f"ProductPrint mu2e::StatusG4_g4run__{process}\n"
            "status: 0 nG4Tracks: 7 overflowSimP: 0 nStepLimit: 0\n")
    if trigger:
        text += f"ProductPrint art::TriggerResults_TriggerResults__{process}\npass 0 PrimaryPath\n"
    return text


def make_arm(base, name, index, floating=False):
    arm = dict(arm_id=name, arm_complete=f"{name}.done")
    (base / arm["arm_complete"]).touch()
    for offset, (stage, process) in enumerate((("pot", "R1LibraryPOT"), ("stops", "R1LibraryStops"))):
        directory = base / name / stage
        directory.mkdir(parents=True)
        run = index * 10 + offset + 1
        spec = dict(cap=2, run=run, subrun=1, base_seed=index * 100 + offset * 10, max_unique_engines=10)
        files = dict(generation_log="generation.log", read_log="read.log", admission="admission.json",
                     selected_art="selected.art", generation_complete="generation.done",
                     read_complete="read.done", selected_read_complete="selected-read.done")
        for key, filename in files.items():
            path = directory / filename
            path.write_text(f"{name}-{stage}-{key}")
            spec[key] = str(path.relative_to(base))
        (directory / "generation.log").write_text("".join(event(run, n, process, False) for n in (1, 2)))
        (directory / "read.log").write_text("".join(event(run, n, process, True) for n in (2, 1)))
        _, receipt = A.reconcile(directory / "generation.log", directory / "read.log", process, 2)
        receipt.update(parent_art_sha256=C.checksum(directory / "selected.art"), selected_art_events=2,
                       selected_parent_verified=stage == "stops")
        if stage == 'pot':
            import uproot
            tree = directory / 'selected.root'
            beam = dict(run=np.full(2, run), subrun=np.ones(2, int), evt=np.array([1, 2]),
                        trk=np.ones(2, int), pdg=np.full(2, 13), codestop=np.full(2, 54),
                        pstop=np.ones(2), xstop=np.zeros(2), ystop=np.zeros(2), zstop=np.zeros(2), tstop=np.full(2, 350.))
            with uproot.recreate(tree) as root:
                root.mktree(A.TREE, {k: v.dtype for k, v in beam.items()}).extend(beam)
            spec['selected_tree'] = str(tree.relative_to(base))
        if stage == "stops":
            tree = directory / "selected.root"
            tree.write_text(name + "-tree")
            spec["selected_tree"] = str(tree.relative_to(base))
            spec["parent_npz"] = str((directory / "parent_stops.npz").relative_to(base))
            dtype = np.float32 if floating else np.int64
            parent = dict(run=np.full(3, run, dtype=dtype), subrun=np.ones(3, dtype=dtype),
                          evt=np.array([1, 1, 2], dtype=dtype), trk=np.array([1, 2, 1], dtype=dtype),
                          event_index=np.array([0, 0, 1]), event_keys=np.array([[run, 1, 1], [run, 1, 2]], dtype=dtype),
                          event_multiplicity=np.array([2, 1]), pdg=np.full(3, 13, dtype=dtype),
                          codestop=np.full(3, 32, dtype=dtype), at_rest=np.ones(3, bool), foil=np.array([0, 1, 1]),
                          pstop=np.zeros(3), xstop=np.zeros(3), ystop=np.zeros(3), zstop=np.zeros(3),
                          tstop=np.array([1., 2., 3.]))
            receipt.update(parent_tree_sha256=C.checksum(tree), parent_tree=A.TREE,
                           library_muons=3, beam_parent_events=2,
                           parent=dict(parent_events=2, eligible_muons=3, electrons_per_parent_event=1.5,
                                       multiplicity={"1": 1, "2": 1}, max_end_momentum=0.,
                                       event_order_preserved=True, expected_event_count_verified=True),
                           physical_event_order=dict(method="exact event-key join to FileIndex, sorted by physical Events entry",
                                                     reordered_events=0, art_events=2, explicit_uproot_metadata_byteswap=False))
            np.savez_compressed(base / spec["parent_npz"], **parent, metadata=json.dumps(receipt))
        (directory / "admission.json").write_text(json.dumps(receipt))
        spec['stage_checksums'] = str((directory / 'SHA256').relative_to(base))
        arm[stage] = spec
        refresh_stage_hashes(base, spec)
    return arm


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(C.A, "art_entries", lambda path: 2)
    monkeypatch.setattr(C, "physical_event_order", lambda parent, art, swap: (dict(parent), {}))
    monkeypatch.setattr(C, "verify_tree_rows", lambda parent, tree: None)
    manifest = dict(arms=[make_arm(tmp_path, "A", 0), make_arm(tmp_path, "B", 1, floating=True)])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def refresh_stage_hashes(base, spec):
    keys = ('selected_art', 'selected_tree', 'generation_log', 'read_log', 'admission')
    lines = [f'{C.checksum(base / spec[k])}  {(base / spec[k]).name}\n' for k in keys]
    (base / spec['stage_checksums']).write_text(''.join(lines))


def save_manifest(path, manifest):
    path.write_text(json.dumps(manifest))


def change_receipt(path, spec, field, value, embedded=False):
    receipt_path = path.parent / spec["admission"]
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = value
    receipt_path.write_text(json.dumps(receipt))
    if embedded:
        change_parent(path.parent / spec["parent_npz"], metadata=np.asarray(json.dumps(receipt)))
    refresh_stage_hashes(path.parent, spec)


def change_parent(path, **changes):
    with np.load(path, allow_pickle=False) as saved:
        parent = {key: saved[key] for key in saved.files}
    parent.update(changes)
    np.savez_compressed(path, **parent)


def test_inventory_descriptive_totals_and_honest_scope(campaign):
    path, _ = campaign
    report = C.inventory(path)
    assert report["totals"] == dict(POT_events=4, beam_draw_events=4, POT_accepted_events=4,
                                    stop_accepted_events=4, stops=6)
    assert report["support"]["multiplicity"] == {"1": 2, "2": 2}
    assert report["support"]["foil"] == {"0": 2, "1": 4}
    assert report["support"]["time"]["mean"] == 2.
    assert report["support"]["time"]["sd"] == pytest.approx(np.std([1, 2, 3]))
    for key in ("physics_validated", "generated_yield_validated", "actual_configuration_verified",
                "actual_seed_schedule_verified", "upstream_sourceEventIDSequence_audited",
                "historical_equivalence_established", "independent_parent_provenance_established",
                "automatic_refill", "exclusions_applied", "certificate_established"):
        assert report[key] is False
    assert "manifest-declared" in report["seed_scope"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("stage,key", [(None, "arm_complete"), ("pot", "generation_complete"),
    ("pot", "read_complete"), ("pot", "selected_read_complete"), ("stops", "generation_complete"),
    ("stops", "read_complete"), ("stops", "selected_read_complete"), ("pot", "admission"),
    ("stops", "admission"), ("stops", "parent_npz"), ("stops", "selected_tree")])
def test_partial_missing_receipts_and_markers(campaign, stage, key):
    path, manifest = campaign
    arm = manifest["arms"][0]
    spec = arm[stage] if stage else arm
    (path.parent / spec[key]).unlink()
    with pytest.raises(ValueError, match="missing artifact"):
        C.inventory(path)


@pytest.mark.parametrize("stage,key", [("pot", "selected_art"), ("stops", "selected_art"),
                                      ("stops", "selected_tree"), ("pot", "read_log")])
def test_hash_or_log_tampering(campaign, stage, key):
    path, manifest = campaign
    artifact = path.parent / manifest["arms"][0][stage][key]
    artifact.write_bytes(artifact.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="mismatch"):
        C.inventory(path)


@pytest.mark.parametrize("field,value", [("library_muons", 9), ("selected_parent_verified", False),
    ("selected_art_events", 1), ("accepted_events", 1), ("parent_tree", "wrong")])
def test_receipt_tampering_even_when_embedded_agrees(campaign, field, value):
    path, manifest = campaign
    change_receipt(path, manifest["arms"][0]["stops"], field, value, embedded=True)
    with pytest.raises(ValueError, match="mismatch"):
        C.inventory(path)


def test_embedded_receipt_must_equal_json(campaign):
    path, manifest = campaign
    change_parent(path.parent / manifest["arms"][0]["stops"]["parent_npz"], metadata=np.asarray("{}"))
    with pytest.raises(ValueError, match="embedded"):
        C.inventory(path)


@pytest.mark.parametrize("changes", [dict(trk=np.array([1, 1, 1])), dict(event_multiplicity=np.array([1, 2])),
    dict(event_keys=np.array([[2, 1, 1], [2, 1, 1]])), dict(evt=np.array([1., 1.5, 2.])),
    dict(pdg=np.full(3, -13)), dict(tstop=np.array([1., np.nan, 2.])), dict(foil=np.array([0, -1, 1]))])
def test_bad_parent_keys_multiplicity_and_support(campaign, changes):
    path, manifest = campaign
    change_parent(path.parent / manifest["arms"][0]["stops"]["parent_npz"], **changes)
    with pytest.raises(ValueError):
        C.inventory(path)


@pytest.mark.parametrize("field,value", [("cap", 3), ("cap", 0), ("cap", True), ("run", 99), ("subrun", 9)])
def test_wrong_cap_or_identity(campaign, field, value):
    path, manifest = campaign
    manifest["arms"][0]["pot"][field] = value
    save_manifest(path, manifest)
    with pytest.raises(ValueError):
        C.inventory(path)


@pytest.mark.parametrize("stage", ["pot", "stops"])
def test_seed_overlap_including_both_stages(campaign, stage):
    path, manifest = campaign
    manifest["arms"][1][stage]["base_seed"] = 5
    save_manifest(path, manifest)
    with pytest.raises(ValueError, match="seed intervals"):
        C.inventory(path)


def test_same_arm_seed_overlap(campaign):
    path, manifest = campaign
    manifest["arms"][0]["stops"]["base_seed"] = 9
    save_manifest(path, manifest)
    with pytest.raises(ValueError, match="seed intervals"):
        C.inventory(path)


@pytest.mark.parametrize("stage,artifact,hash_field", [("pot", "selected_art", "parent_art_sha256"),
    ("stops", "selected_art", "parent_art_sha256"), ("stops", "selected_tree", "parent_tree_sha256")])
def test_duplicate_artifact_hash_across_arms(campaign, stage, artifact, hash_field):
    path, manifest = campaign
    first, second = (arm[stage] for arm in manifest["arms"])
    (path.parent / second[artifact]).write_bytes((path.parent / first[artifact]).read_bytes())
    change_receipt(path, second, hash_field, C.checksum(path.parent / second[artifact]), embedded=stage == "stops")
    with pytest.raises(ValueError, match="duplicate selected"):
        C.inventory(path)


def test_duplicate_event_identity_across_arms(campaign):
    path, manifest = campaign
    first, second = (arm["pot"] for arm in manifest["arms"])
    old, new = second["run"], first["run"]
    second["run"] = new
    for key in ("generation_log", "read_log"):
        file = path.parent / second[key]
        file.write_text(file.read_text().replace(f"Event {old} ", f"Event {new} "))
    refresh_stage_hashes(path.parent, second)
    save_manifest(path, manifest)
    with pytest.raises(ValueError, match="duplicate event"):
        C.inventory(path)


def test_duplicate_stop_guard():
    seen = []
    rows = np.array([[1, 2, 3, 4]])
    C.disjoint(rows, seen, "stop identities across arms")
    with pytest.raises(ValueError, match="duplicate stop"):
        C.disjoint(rows, seen, "stop identities across arms")


def test_physical_order_recomputed(campaign, monkeypatch):
    path, _ = campaign
    def reorder(parent, art, swap):
        result = dict(parent)
        result["event_keys"] = result["event_keys"][::-1]
        return result, {}
    monkeypatch.setattr(C, "physical_event_order", reorder)
    with pytest.raises(ValueError, match="physical ordering mismatch"):
        C.inventory(path)


@pytest.mark.parametrize("bad", [[], [{"arm_id": ""}], [{"arm_id": "A"}, {"arm_id": "A"}]])
def test_bad_arm_list(campaign, bad):
    path, manifest = campaign
    if bad and bad[0].get("arm_id") == "A":
        bad = [manifest["arms"][0], copy.deepcopy(manifest["arms"][0])]
        bad[1]["pot"]["base_seed"] = 1000
        bad[1]["stops"]["base_seed"] = 1010
    save_manifest(path, dict(arms=bad))
    with pytest.raises(ValueError):
        C.inventory(path)


def test_cli_exclusive_and_strict_json(campaign, capsys):
    path, _ = campaign
    out = path.parent / "inventory.json"
    args = ["--manifest", str(path), "--out", str(out)]
    C.main(args)
    before = out.read_bytes()
    assert json.loads(before) == json.loads(capsys.readouterr().out)
    with pytest.raises(FileExistsError):
        C.main(args)
    assert out.read_bytes() == before
    with pytest.raises(FileExistsError):
        C.main(["--manifest", str(path), "--out", str(path)])


def test_invalid_campaign_leaves_no_output(campaign):
    path, manifest = campaign
    (path.parent / manifest["arms"][1]["arm_complete"]).unlink()
    out = path.parent / "inventory.json"
    with pytest.raises(ValueError):
        C.main(["--manifest", str(path), "--out", str(out)])
    assert not out.exists()


@pytest.mark.parametrize("text", ['{"arms": [], "arms": []}', '{"arms": NaN}'])
def test_non_strict_input_json(campaign, text):
    path, _ = campaign
    path.write_text(text)
    with pytest.raises(ValueError):
        C.inventory(path)


def test_raw_selected_tree_rows_bound_to_parent(tmp_path):
    uproot = pytest.importorskip('uproot')
    columns = dict(run=np.ones(3, int), subrun=np.full(3, 2), evt=np.array([1, 1, 2]),
        trk=np.array([1, 2, 1]), pdg=np.full(3, 13), codestop=np.full(3, 32),
        pstop=np.zeros(3), xstop=np.array([1., 2., 3.]), ystop=np.zeros(3), zstop=np.ones(3),
        tstop=np.full(3, 350.))
    tree = tmp_path / 'selected.root'
    with uproot.recreate(tree) as root:
        root.mktree(C.A.TREE, {k: v.dtype for k, v in columns.items()}).extend(columns)
    shuffled = {k: v[[2, 0, 1]] for k, v in columns.items()}
    C.verify_tree_rows(shuffled, tree)
    shuffled['xstop'][0] += .25
    with pytest.raises(ValueError, match='differs from selected tree'):
        C.verify_tree_rows(shuffled, tree)


def test_pot_tree_replacement_with_same_keys_is_blocked_by_closed_manifest(campaign):
    import uproot
    path, manifest = campaign
    spec = manifest['arms'][0]['pot']
    tree = path.parent / spec['selected_tree']
    with uproot.open(tree) as root:
        columns = root[A.TREE].arrays(library='np')
    columns['pstop'] = np.full(2, 77.)
    with uproot.recreate(tree) as root:
        root.mktree(A.TREE, {k: v.dtype for k, v in columns.items()}).extend(columns)
    with pytest.raises(ValueError, match='extraction hash mismatch'):
        C.inventory(path)
