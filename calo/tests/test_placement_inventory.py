"""Synthetic compact-v2 contracts; no native Geant4/containment claims."""
from array import array
from copy import deepcopy
import hashlib
import io
import json

import pytest

from calofastsim.placement_inventory import (
    AuditReport, CompletionReceipt, FRAME, IDENTITY, InventoryError, Limits,
    NAME_ENCODING, SCHEMA, SCOPE, UNITS, UNSUPPORTED_REASON, audit_file, read,
)


def fixture():
    begin = dict(type="inventory_begin", schema=SCHEMA, g4_run=0, scope=SCOPE, units=UNITS,
                 frame=FRAME, name_encoding=NAME_ENCODING, identity=IDENTITY, certificate=False,
                 max_depth=128, max_placements=1_000_000, max_bytes=1073741824, max_rows=1_000_000,
                 helper_allocator_max_bytes=268435456, stream_info_max_bytes=65536,
                 polycone_max_planes=16384, candidate_roots=["Calo", "Inner"])
    rows = [begin, dict(type="material", id=0, name="Vacuum", native_index=0),
            dict(type="solid", id=0, name="Box", entity_type="G4Box", parameter_status="native_parameters",
                 parameters=dict(half_x=10, half_y=20, half_z=30), native_stream_info="native box\xff"),
            dict(type="solid", id=1, name="Boolean", entity_type="G4SubtractionSolid",
                 parameter_status="unsupported_metadata_only", parameters=None, native_stream_info=None,
                 reason=UNSUPPORTED_REASON)]
    rows.extend(dict(type="logical_volume", id=i, name=name, material_id=0, solid_id=solid)
                for i, (name, solid) in enumerate((("WorldLV", 0), ("CaloLV", 0), ("MotherLV", 0),
                                                 ("SharedLV", 1), ("InnerLV", 0))))

    def pv(parent, depth, name, lv=1, copy=0, root=None, selected=True):
        i = sum(r["type"] == "placement" for r in rows)
        rows.append(dict(type="placement", id=i, parent_id=parent, depth=depth, pv=name, copy=copy,
                         logical_volume_id=lv, selected=selected, candidate_root_id=root,
                         local_translation=[1.25, -2, 3], local_object_rotation=[[0, -1, 0], [1, 0, 0], [0, 0, 1]]))

    pv(None, 0, "World", lv=0, selected=False)  # World is the sole null parent.
    pv(0, 1, "Calo", root=0)
    pv(1, 2, "Mother", lv=2, copy=0)
    pv(2, 3, 'Shared/"\\\x00\xff', lv=3)
    pv(1, 2, "Mother", lv=2, copy=1)
    pv(4, 3, 'Shared/"\\\x00\xff', lv=3)  # Same conceptual daughter, distinct expanded path.
    pv(1, 2, "Inner", lv=4, root=1)
    pv(6, 3, "Leaf", lv=3)
    pv(0, 1, "OutsideCable", lv=3, selected=False)
    rows.append(dict(type="inventory_end", complete_physical_tree=True, resolved_all_solids=False,
                     placements=9, unique_paths=9, materials=1, solids=2, logical_volumes=5,
                     candidate_placements=7, unsupported_solid_placements=4, deepest=3,
                     rows_including_footer=len(rows) + 1, bytes_including_footer_decimal="0" * 20,
                     root_placements=[dict(root_id=0, name="Calo", count=1, placement_id=1),
                                      dict(root_id=1, name="Inner", count=1, placement_id=6)]))
    return rows


def encoded(rows, *, seal=True):
    rows = deepcopy(rows)

    def dump():
        return b"".join((json.dumps(r, ensure_ascii=True, separators=(",", ":")) + "\n").encode("ascii") for r in rows)

    if seal and rows[-1].get("type") == "inventory_end":
        rows[-1]["rows_including_footer"] = len(rows)
        rows[-1]["bytes_including_footer_decimal"] = "0" * 20
        rows[-1]["bytes_including_footer_decimal"] = f"{len(dump()):020d}"
    return dump()


def run(rows=None, **kwargs):
    rows = fixture() if rows is None else rows
    return read(io.BytesIO(rows if isinstance(rows, bytes) else encoded(rows)), **kwargs)


def placements(rows):
    return [r for r in rows if r["type"] == "placement"]


def test_tree_paths_transform_shared_occurrences_and_counts(tmp_path):
    raw = encoded(fixture())
    report = run(raw)
    assert isinstance(report, AuditReport)
    assert report.protocol_complete and not report.accepted
    assert report.counts == dict(placements=9, unique_paths=9, materials=1, solids=2, logical_volumes=5,
                                 candidate_placements=7, unsupported_solid_placements=4,
                                 deepest=3, rows=19, bytes=len(raw))
    assert report.sha256 == hashlib.sha256(raw).hexdigest()
    assert report.root_placements == (1, 6)
    p = report.placements
    assert isinstance(p.parent_id, array) and p.parent_id.typecode == "i"
    assert p.selected == bytes([0, 1, 1, 1, 1, 1, 1, 1, 0])
    assert p.candidate_root_id.tolist() == [-1, 0, -1, -1, -1, -1, 1, -1, -1]
    assert p.local_transform[:12].tolist() == [1.25, -2, 3, 0, -1, 0, 1, 0, 0, 0, 0, 1]
    assert report.path(3) == (("World", 0), ("Calo", 0), ("Mother", 0), ('Shared/"\\\x00\xff', 0))
    assert report.path(5)[-1] == report.path(3)[-1] and report.path(5) != report.path(3)
    assert report.path(0) == (("World", 0),)
    assert p.pv[3].encode("latin1") == b'Shared/"\\\x00\xff'
    assert b"\\u00ff" in raw and b"\\u0000" in raw
    assert report.solids[1]["parameters"] is None
    assert report.placements.logical_volume_id[3] == report.placements.logical_volume_id[5]
    path = tmp_path / "inventory.jsonl"
    path.write_bytes(raw)
    assert audit_file(path).sha256 == report.sha256
    with pytest.raises(InventoryError):
        report.path(True)
    with pytest.raises(InventoryError):
        report.path(9)


@pytest.mark.parametrize("closed,success,matching,accepted", [
    (True, True, True, True), (False, True, True, False), (True, False, True, False),
    (False, False, True, False), (True, True, False, False),
])
def test_external_receipt_and_permanently_closed_native_library_gates(closed, success, matching, accepted):
    raw = encoded(fixture())
    receipt = CompletionReceipt(hashlib.sha256(raw).hexdigest() if matching else "0" * 64, closed, success)
    report = run(raw, receipt=receipt)
    assert report.accepted is accepted and report.external_completion_verified is accepted
    assert report.diagnostic_only
    assert not any((report.certificate, report.library_eligible, report.native_validation, report.root_store_verified))


@pytest.mark.parametrize("mutate", [
    lambda r: r[0].update(schema="mu2e-placement-inventory-v1"),
    lambda r: r[0].update(certificate=True),
    lambda r: r[0].update(max_depth=129),
    lambda r: r[0].update(max_placements=True),
    lambda r: r[0].update(helper_allocator_max_bytes=12),
    lambda r: r[0].update(scope="candidate_only"),
    lambda r: r[0].update(candidate_roots=[]),
    lambda r: r[0].update(candidate_roots=["Calo", "Calo"]),
    lambda r: r[0].update(candidate_roots=[""]),
    lambda r: r[0].update(candidate_roots=["Absent", "Inner"]),
    lambda r: r[0].update(extra=True),
    lambda r: r[1].update(id=1),
    lambda r: r[1].update(id=True),
    lambda r: r[1].update(native_index=-1),
    lambda r: r[2].update(id=1),
    lambda r: r[4].update(id=2),
    lambda r: r[4].update(material_id=1),
    lambda r: r[4].update(solid_id=2),
    lambda r: r[4].update(solid_id=True),
    lambda r: placements(r)[0].update(parent_id=0),
    lambda r: placements(r)[1].update(parent_id=None),
    lambda r: placements(r)[1].update(parent_id=1),
    lambda r: placements(r)[1].update(parent_id=8),
    lambda r: placements(r)[1].update(depth=0),
    lambda r: placements(r)[1].update(depth=True),
    lambda r: placements(r)[1].update(id=0),
    lambda r: placements(r)[1].update(logical_volume_id=5),
    lambda r: placements(r)[1].update(selected=1),
    lambda r: placements(r)[2].update(selected=False),
    lambda r: placements(r)[8].update(selected=True),
    lambda r: placements(r)[1].update(candidate_root_id=None),
    lambda r: placements(r)[2].update(candidate_root_id=0),
    lambda r: placements(r)[1].update(candidate_root_id=True),
    lambda r: placements(r)[5].update(pv="Calo", candidate_root_id=0),
    lambda r: placements(r)[4].update(copy=0),
    lambda r: placements(r)[6].update(pv="Mother", copy=0, candidate_root_id=None),
    lambda r: placements(r)[7].update(parent_id=2),
    lambda r: placements(r)[5].update(parent_id=2),
    lambda r: placements(r)[3].update(copy=2**31),
    lambda r: placements(r)[3].update(pv="\u0100"),
    lambda r: placements(r)[3].update(pv="\ud800"),
    lambda r: placements(r)[3].update(local_translation=[1, 2]),
    lambda r: placements(r)[3].update(local_translation=[1, True, 3]),
    lambda r: placements(r)[3].update(local_object_rotation=[[1, 0, 0]]),
    lambda r: r[2]["parameters"].update(half_x=True),
    lambda r: r[2]["parameters"].update(extra=0),
    lambda r: r[2].update(entity_type="G4UnionSolid"),
    lambda r: r[2].update(native_stream_info="x" * 65537),
    lambda r: r[3].update(parameters={"fabricated_operands": []}),
    lambda r: r[3].update(native_stream_info="not allowed"),
    lambda r: r[3].update(reason="guessed"),
    lambda r: r[-1].update(complete_physical_tree=False),
    lambda r: r[-1].update(resolved_all_solids=True),
    lambda r: r[-1]["root_placements"][0].update(root_id=1),
    lambda r: r[-1]["root_placements"][0].update(name="Other"),
    lambda r: r[-1]["root_placements"][0].update(count=0),
    lambda r: r[-1]["root_placements"][0].update(placement_id=2),
    lambda r: r[-1]["root_placements"][0].update(count=True),
    lambda r: r[-1].update(root_placements=[]),
])
def test_reject_structural_mutations(mutate):
    rows = fixture()
    mutate(rows)
    with pytest.raises(InventoryError):
        run(rows)


@pytest.mark.parametrize("field", ["placements", "unique_paths", "materials", "solids", "logical_volumes",
                                   "candidate_placements", "unsupported_solid_placements", "deepest"])
def test_footer_count_reconciliation(field):
    rows = fixture()
    rows[-1][field] += 1
    with pytest.raises(InventoryError, match=field):
        run(rows)


@pytest.mark.parametrize("value", ["1", "0" * 19, "0" * 21, "x" * 20, 1, True])
def test_fixed_width_footer_bytes(value):
    rows = fixture()
    rows[-1]["bytes_including_footer_decimal"] = value
    with pytest.raises(InventoryError, match="byte total"):
        run(encoded(rows, seal=False))


def test_footer_rows_and_bytes_exact():
    raw = encoded(fixture())
    assert run(raw).protocol_complete
    for bad in (raw.replace(b'"rows_including_footer":19', b'"rows_including_footer":20'),
                raw.replace(b'"placements":9', b'"placements":9 ')):
        with pytest.raises(InventoryError):
            run(bad)


@pytest.mark.parametrize("mutation", [
    lambda r: r[1:], lambda r: r[:-1], lambda r: r + [r[-1]], lambda r: [r[0], r[0]] + r[1:],
    lambda r: r[:2] + [r[4], r[2], r[3]] + r[5:],
    lambda r: r[:7] + [r[8], r[7]] + r[9:],
    lambda r: r[:8] + [r[1]] + r[8:], lambda r: r[:7] + [r[-1]],
])
def test_required_sequence_and_nonempty_world(mutation):
    with pytest.raises(InventoryError):
        run(mutation(fixture()))


@pytest.mark.parametrize("raw", [b"", b"\n", b"[]\n", b"{}\n", b"{\n", b"null\n", b'{"type":[]}\n'])
def test_bad_json_and_missing_footer(raw):
    with pytest.raises(InventoryError):
        run(raw)


def test_truncation_extra_data_and_unescaped_nonascii():
    raw = encoded(fixture())
    for bad in (raw[:-1], raw + b"\n", raw + b"garbage", raw.replace(b"\\u00ff", b"\xc3\xbf")):
        with pytest.raises(InventoryError):
            run(bad)


@pytest.mark.parametrize("old,new", [
    (b'"half_x":10', b'"half_x":NaN'), (b'"half_x":10', b'"half_x":Infinity'),
    (b'"half_x":10', b'"half_x":-Infinity'), (b'"half_x":10', b'"half_x":1e999'),
    (b'"half_x":10', b'"half_x":10,"half_x":11'),
    (b'"id":0', b'"id":0,"id":0'),
    (b'"half_x":10', b'"half_x":' + b"9" * 21),
])
def test_nonfinite_duplicate_keys_and_integer_token_bound(old, new):
    with pytest.raises(InventoryError):
        run(encoded(fixture()).replace(old, new, 1))


@pytest.mark.parametrize("kind", ["bytes", "rows", "line_bytes", "placements"])
def test_independent_caps_and_exact_boundaries(kind):
    raw = encoded(fixture())
    maxima = dict(max_bytes=len(raw), max_rows=19,
                  max_line_bytes=max(map(len, raw.splitlines(keepends=True))), max_placements=9)
    assert run(raw, limits=Limits(**maxima)).protocol_complete
    maxima["max_" + kind] -= 1
    with pytest.raises(InventoryError, match="cap"):
        run(raw, limits=Limits(**maxima))


def test_declared_caps_not_only_reader_caps():
    for field, value in (("max_bytes", 1), ("max_rows", 3)):
        rows = fixture()
        rows[0][field] = value
        if field == "max_bytes":
            rows[0]["helper_allocator_max_bytes"] = value
        with pytest.raises(InventoryError, match="cap"):
            run(rows)


def test_bounded_read_never_requests_whole_file():
    class CheckedStream(io.BytesIO):
        def readline(self, size=-1):
            assert 0 < size <= 4097
            return super().readline(size)

        def read(self, *args):
            raise AssertionError("whole-file read forbidden")

    assert read(CheckedStream(encoded(fixture())), limits=Limits(max_line_bytes=4096)).protocol_complete
    with pytest.raises(InventoryError, match="binary"):
        read(io.StringIO("{}\n"))


@pytest.mark.parametrize("entity,params", [
    ("G4Tubs", dict(rmin=0, rmax=5, half_z=10, start_phi=0.1, delta_phi=6.2)),
    ("G4Polycone", dict(representation="native_original_z_planes", start_phi=0.1, delta_phi=6.2,
                       planes=[[-3, 0, 1], [0, 0.2, 2], [3, 1, 2]])),
])
def test_known_shape_records(entity, params):
    rows = fixture()
    rows[2].update(entity_type=entity, parameters=params)
    assert run(rows).solids[0]["parameters"] == params
    bad = deepcopy(rows)
    bad[2]["parameters"]["start_phi"] = False
    with pytest.raises(InventoryError):
        run(bad)
    if entity == "G4Polycone":
        for planes in ([], [[0, 0, 1]], [[0, 0, 1], [1, False, 1]], [[0, 0], [1, 0, 1]], [[0, 0, 1]] * 16385):
            bad = deepcopy(rows)
            bad[2]["parameters"]["planes"] = planes
            with pytest.raises(InventoryError):
                run(bad)


def test_entity_type_alone_cannot_establish_native_dynamic_type():
    rows = fixture()
    rows[3]["entity_type"] = "G4Box"
    report = run(rows)
    assert report.solids[1]["parameter_status"] == "unsupported_metadata_only"
    assert report.counts["unsupported_solid_placements"] == 4
    assert not report.native_validation


def test_resolved_solids_is_still_not_native_validation():
    rows = fixture()
    rows[3] = deepcopy(rows[2])
    rows[3]["id"] = 1
    rows[-1].update(resolved_all_solids=True, unsupported_solid_placements=0)
    report = run(rows)
    assert report.counts["unsupported_solid_placements"] == 0
    assert not report.library_eligible and not report.native_validation and not report.accepted


@pytest.mark.parametrize("kwargs", [dict(max_bytes=True), dict(max_rows=0), dict(max_line_bytes=-1),
                                    dict(max_placements=1_000_001)])
def test_invalid_reader_limits(kwargs):
    with pytest.raises(InventoryError):
        Limits(**kwargs)


def test_minimal_world_root_and_empty_dictionary_stages_rejected():
    rows = fixture()
    rows[0]["candidate_roots"] = ["World"]
    rows = [rows[0], rows[1], rows[2], rows[4], placements(rows)[0], rows[-1]]
    rows[4].update(selected=True, candidate_root_id=0)
    rows[-1].update(placements=1, unique_paths=1, solids=1, logical_volumes=1,
                    candidate_placements=1, unsupported_solid_placements=0, deepest=0,
                    resolved_all_solids=True,
                    root_placements=[dict(root_id=0, name="World", count=1, placement_id=0)])
    report = run(rows)
    assert report.root_placements == (0,) and report.path(0) == (("World", 0),)
    assert report.placements.selected == b"\x01"
    for index in (1, 2, 3):
        with pytest.raises(InventoryError):
            run(rows[:index] + rows[index + 1:])


def test_depth_limit_and_malformed_nested_json():
    rows = fixture()
    placements(rows)[3]["depth"] = 129
    with pytest.raises(InventoryError, match="depth"):
        run(rows)
    with pytest.raises(InventoryError):
        run(b'{"type":"inventory_begin","nested":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}\n")


def test_invalid_receipt_flags_are_not_truthy_integers():
    with pytest.raises(InventoryError):
        CompletionReceipt("0" * 64, 1, True)
    with pytest.raises(InventoryError):
        CompletionReceipt("not-a-hash", True, True)


def reconcile(rows):
    """Reseal synthetic counts/IDs after edits, without masking semantics."""
    nodes = placements(rows)
    logicals = [r for r in rows if r["type"] == "logical_volume"]
    solids = [r for r in rows if r["type"] == "solid"]
    for i, node in enumerate(nodes):
        node["id"] = i
    footer = rows[-1]
    unsupported = sum(solids[logicals[p["logical_volume_id"]]["solid_id"]]["parameter_status"] ==
                      "unsupported_metadata_only" for p in nodes)
    footer.update(placements=len(nodes), unique_paths=len(nodes),
                  materials=sum(r["type"] == "material" for r in rows), solids=len(solids),
                  logical_volumes=len(logicals), candidate_placements=sum(p["selected"] for p in nodes),
                  unsupported_solid_placements=unsupported, resolved_all_solids=unsupported == 0,
                  deepest=max(p["depth"] for p in nodes))
    for root in footer["root_placements"]:
        root["placement_id"] = next(p["id"] for p in nodes if p["candidate_root_id"] == root["root_id"])
    return rows


def matching_receipt(raw):
    return CompletionReceipt(hashlib.sha256(raw).hexdigest(), True, True)


@pytest.mark.parametrize("kind", ["material", "solid", "logical_volume"])
def test_unreachable_dictionary_rejected_even_with_matching_success_receipt(kind):
    rows = fixture()
    table = [r for r in rows if r["type"] == kind]
    extra = deepcopy(table[-1])
    extra.update(id=len(table), name="Unused")
    rows.insert(rows.index(table[-1]) + 1, extra)
    raw = encoded(reconcile(rows))
    with pytest.raises(InventoryError, match="unreachable"):
        run(raw, receipt=matching_receipt(raw))


@pytest.mark.parametrize("kind", ["material", "solid", "logical_volume"])
def test_remapped_dictionary_reordering_rejected(kind):
    rows = fixture()
    if kind == "material":
        extra = deepcopy(rows[1])
        extra.update(id=1, name="Other", native_index=1)
        rows.insert(2, extra)
        # Material 1 is first encountered with SharedLV, after material 0.
        next(r for r in rows if r["type"] == "logical_volume" and r["id"] == 3)["material_id"] = 1
    reconcile(rows)
    assert run(rows).protocol_complete
    table = [r for r in rows if r["type"] == kind]
    a, b = deepcopy(table[1]), deepcopy(table[0])
    a["id"], b["id"] = 0, 1
    index = rows.index(table[0])
    rows[index:index + 2] = [a, b]
    owners, field = ((placements(rows), "logical_volume_id") if kind == "logical_volume" else
                     ([r for r in rows if r["type"] == "logical_volume"], kind + "_id"))
    for owner in owners:
        if owner[field] in (0, 1):
            owner[field] = 1 - owner[field]
    raw = encoded(rows)
    with pytest.raises(InventoryError, match="first-encounter"):
        run(raw, receipt=matching_receipt(raw))


@pytest.mark.parametrize("change", ["name", "copy", "logical", "translation", "rotation", "missing", "extra", "order"])
def test_repeated_logical_direct_daughters_must_match(change):
    rows = fixture()
    p = placements(rows)
    if change == "name":
        p[5]["pv"] += "\x00\xff"
    elif change == "copy":
        p[5]["copy"] = 1
    elif change == "logical":
        p[5]["logical_volume_id"] = 2
    elif change == "translation":
        p[5]["local_translation"][0] += 0.25
    elif change == "rotation":
        p[5]["local_object_rotation"][0][0] = 1
    elif change == "missing":
        rows.remove(p[5])
        p[7]["parent_id"] = 5
    else:
        second = deepcopy(p[5])
        second.update(pv="SecondDaughter")
        rows.insert(rows.index(p[5]) + 1, second)
        if change == "order":
            first = deepcopy(p[3])
            first.update(pv="SecondDaughter")
            rows.insert(rows.index(p[3]) + 1, first)
            p[5]["pv"], second["pv"] = second["pv"], p[5]["pv"]
            p[5]["parent_id"] = second["parent_id"] = 5
            p[7]["parent_id"] = 8
        else:
            p[7]["parent_id"] = 7
    raw = encoded(reconcile(rows))
    with pytest.raises(InventoryError, match="direct daughter"):
        run(raw, receipt=matching_receipt(raw))


def test_repeated_logical_signature_excludes_occurrence_context():
    rows = fixture()
    p = placements(rows)
    p[8]["logical_volume_id"] = 2
    daughter = deepcopy(p[3])
    daughter.update(parent_id=8, depth=2, selected=False)
    rows.insert(-1, daughter)
    raw = encoded(reconcile(rows))
    report = run(raw, receipt=matching_receipt(raw))
    assert report.accepted and report.external_completion_verified
    assert not report.native_validation
    # Mother expansions differ in their own names, copies, depth and selection.
    assert report.placements.depth[8] != report.placements.depth[2]
    assert report.placements.selected[8] != report.placements.selected[2]


def test_exact_transform_comparison_does_not_round_integer_differences():
    rows = fixture()
    p = placements(rows)
    p[3]["local_translation"][0] = 2**53
    p[5]["local_translation"][0] = 2**53 + 1
    raw = encoded(rows)
    with pytest.raises(InventoryError, match="direct daughters"):
        run(raw, receipt=matching_receipt(raw))
