"""Public synthetic GDML structure fixtures; no geometry/navigation certificate."""
import hashlib
import json

import pytest

from beamfastsim import tsgdml


def fixture(tmp_path, *, structure=None, solids=None, define="", extra=""):
    path = tmp_path / "toy.gdml"
    path.write_text(f'''<gdml><define>{define}</define>
      <materials><material name="vac"><D value="0" unit="g/cm3"/></material>
      <material name="unused"/></materials>
      <solids>{solids or '<box name="box" x="1" y="2" z="3"/>'}</solids>
      <structure>{structure or hierarchy()}</structure>
      <setup name="default" version="1"><world ref="world"/></setup>{extra}</gdml>''')
    return path


def volume(name, children="", solid="box"):
    return f'<volume name="{name}"><materialref ref="vac"/><solidref ref="{solid}"/>{children}</volume>'


def placement(name, target, extra=""):
    return f'<physvol name="{name}"><volumeref ref="{target}"/>{extra}</physvol>'


def hierarchy():
    return (volume("world", placement("TS1Vacuum0xabc", "branch") + placement("second", "branch"))
            + volume("branch", placement("daughter", "leaf")) + volume("leaf") + volume("orphan"))


def test_complete_hierarchy_reuse_and_scopes(tmp_path):
    path = fixture(tmp_path)
    report = tsgdml.audit(path, ["TS1Vacuum"])
    assert report["source"]["bytes"] == len(path.read_bytes())
    assert report["source"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["world_volume"] == "world"
    assert report["reachable"]["physical_volume_instances"] == 4
    assert report["reachable"]["volume_instances"] == 5
    assert report["reachable"]["repeated_logical_placements"] == {"branch": 2, "leaf": 2}
    assert report["reachable"]["material_instance_counts"] == {"vac": 5}
    assert report["unreachable"]["logical_volumes"] == ["orphan"]
    assert report["unreachable"]["materials"] == ["unused"]
    records = report["scopes"]["TS1Vacuum"]["records"]
    assert [r["logical_name"] for r in records] == ["branch", "leaf"]
    assert records[0]["physical_name"] == "TS1Vacuum0xabc"
    assert records[0]["child_count"] == 1
    assert records[1]["path"][-2]["physical"] == "TS1Vacuum0xabc"
    assert records[1]["material"] == "vac"
    assert records[1]["solid_tag"] == "box"
    assert report["navigation_supported"] is False
    assert tsgdml.audit(path)["scopes"] == {}
    json.dumps(report, allow_nan=False)


def test_pointer_transform_raw_specs_and_copy_number(tmp_path):
    structure = volume("world", '<physvol name="one" copynumber="7"><volumeref ref="leaf"/>'
                       '<positionref ref="p"/><rotationref ref="r"/><scale x="2"/></physvol>') + volume("leaf")
    path = fixture(tmp_path, structure=structure,
                   define='<position name="p" x="offset+2" unit="cm"/>'
                          '<rotation name="r" z="pi/2" unit="rad"/>')
    report = tsgdml.audit(path, ["leaf"])
    record = report["scopes"]["leaf"]["records"][0]
    assert record["copy_number"]["attribute"] == "7"
    assert record["transforms"]["position"]["reference"]["attributes"] == {"ref": "p"}
    assert record["transforms"]["rotation"]["definition"]["attributes"]["z"] == "pi/2"
    assert report["transforms_evaluated"] is False
    assert report["transform_tag_counts"]["positionref"] == 1
    assert any(o["symbolic_values"] == {"x": "offset+2"} and o["nonstandard_unit"] == "cm"
               for o in report["transform_observations"])


@pytest.mark.parametrize("change, message", [
    (lambda s: s.replace('world ref="world"', 'world ref="absent"'), "reference"),
    (lambda s: s.replace('materialref ref="vac"', 'materialref ref="missing"'), "reference"),
    (lambda s: s.replace('solidref ref="box"', 'solidref ref="missing"'), "reference"),
    (lambda s: s.replace('volumeref ref="leaf"', 'volumeref ref="missing"'), "reference"),
    (lambda s: s.replace('volumeref ref="leaf"', 'volumeref ref="world"'), "cycle"),
    (lambda s: s.replace('<volume name="leaf">', '<volume name="leaf"><replicavol/>'), "Unsupported"),
])
def test_missing_refs_cycles_and_invalid_xml(tmp_path, change, message):
    path = fixture(tmp_path)
    path.write_text(change(path.read_text()))
    with pytest.raises(ValueError, match=message):
        tsgdml.audit(path)


def test_root_ambiguity_missing_and_instance_bound(tmp_path):
    path = fixture(tmp_path)
    for root in ("branch", "leaf", "daughter"):
        with pytest.raises(ValueError, match="Ambiguous"):
            tsgdml.audit(path, [root])
    with pytest.raises(ValueError, match="not found"):
        tsgdml.audit(path, ["orphan"])
    with pytest.raises(ValueError, match="max_instances"):
        tsgdml.audit(path, max_instances=4)
    assert tsgdml.audit(path, ["TS1Vacuum0x123"])["scopes"]["TS1Vacuum0x123"]["volume_instances"] == 2


def test_csg_dag_closure_and_unreachable_definitions(tmp_path):
    solids = ('<box name="box"/><union name="u"><first ref="box"/><second ref="box"/></union>'
              '<subtraction name="c"><first ref="u"/><second ref="box"/></subtraction>'
              '<union name="unreachable"><first ref="missing"/><second ref="missing"/></union>')
    path = fixture(tmp_path, structure=volume("world", solid="c"), solids=solids)
    report = tsgdml.audit(path)
    assert report["reachable"]["solid_definitions"] == ["box", "c", "u"]
    assert report["unreachable"]["solids"] == ["unreachable"]
    assert report["solid_tag_counts"] == {"box": 1, "union": 2, "subtraction": 1}
    path.write_text(path.read_text().replace('<second ref="box"/>', '<second ref="missing"/>'))
    with pytest.raises(ValueError, match="reference"):
        tsgdml.audit(path)


def test_csg_cycle_and_missing_operand(tmp_path):
    for operands, message in (('<first ref="c"/><second ref="box"/>', "cycle"),
                               ('<first ref="box"/>', "expected one second")):
        path = fixture(tmp_path, structure=volume("world", solid="c"),
                       solids=f'<box name="box"/><union name="c">{operands}</union>')
        with pytest.raises(ValueError, match=message):
            tsgdml.audit(path)


def test_unknown_solid_inventory_and_unsupported_placements(tmp_path):
    path = fixture(tmp_path, solids='<futureShape name="box" unusual="symbol"/>')
    report = tsgdml.audit(path)
    assert report["solid_tag_counts"] == {"futureShape": 1}
    assert report["solids"]["box"]["navigation"] == "unsupported-for-navigation"
    assert report["solids"]["box"]["definition"]["attributes"]["unusual"] == "symbol"
    path.write_text(path.read_text().replace('<volume name="world">', '<volume name="world"><replicavol/>'))
    with pytest.raises(ValueError, match="Unsupported"):
        tsgdml.audit(path)


@pytest.mark.parametrize("extra", ['<positionref ref="absent"/>', '<position x="1+"/>',
                                    '<position x="nan"/>', '<position x="1e999"/>',
                                    '<position x="a[0]"/>', '<position x="1"/><position x="2"/>',
                                    '<positionref ref="r"/>'])
def test_bad_transforms_fail_not_evaluate(tmp_path, extra):
    path = fixture(tmp_path, structure=volume("world", placement("one", "leaf", extra)) + volume("leaf"),
                   define='<rotation name="r"/>')
    with pytest.raises(ValueError):
        tsgdml.audit(path)


@pytest.mark.parametrize("prefix, extra", [
    ('<!DOCTYPE gdml [<!ENTITY x SYSTEM "file:///etc/passwd">]>', ""),
    ('<!DOCTYPE gdml SYSTEM "https://example.invalid/schema">', ""),
    ("", '<file name="part.gdml"/>'),
])
def test_no_dtd_external_or_modular_xml(tmp_path, prefix, extra):
    path = fixture(tmp_path, extra=extra)
    path.write_text(prefix + path.read_text())
    with pytest.raises(ValueError):
        tsgdml.audit(path)


def test_selected_scope_does_not_hide_invalid_other_branch(tmp_path):
    structure = (volume("world", placement("selected", "good") + placement("other", "bad"))
                 + volume("good") + volume("bad", solid="missing"))
    path = fixture(tmp_path, structure=structure)
    with pytest.raises(ValueError, match="reference"):
        tsgdml.audit(path, ["selected"])


def test_normalized_physical_name_collisions_are_ambiguous(tmp_path):
    structure = (volume("world", placement("same0x123", "a") + placement("same0x456", "b"))
                 + volume("a") + volume("b"))
    path = fixture(tmp_path, structure=structure)
    with pytest.raises(ValueError, match="Ambiguous"):
        tsgdml.audit(path, ["same"])


def test_no_remote_or_nonfile_input(tmp_path):
    with pytest.raises(ValueError, match="local"):
        tsgdml.audit("https://example.invalid/world.gdml")
    with pytest.raises(OSError):
        tsgdml.audit(tmp_path)


def test_cli_and_alias_clobber_guards(tmp_path):
    path = fixture(tmp_path)
    out = tmp_path / "receipt.json"
    tsgdml.main([str(path), "--root", "TS1Vacuum", "--out", str(out)])
    assert json.loads(out.read_text())["world_volume"] == "world"
    before = path.read_bytes()
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(path)
    hardlink = tmp_path / "hardlink.json"
    hardlink.hardlink_to(path)
    for alias in (path, symlink, hardlink):
        with pytest.raises(SystemExit):
            tsgdml.main([str(path), "--out", str(alias)])
        assert path.read_bytes() == before


@pytest.mark.parametrize("root", ["0xabc", "0x0", ""])
def test_empty_normalized_root_cannot_alias_unnamed_world(tmp_path, root):
    path = fixture(tmp_path)
    with pytest.raises(ValueError, match="nonempty"):
        tsgdml.audit(path, [root])


@pytest.mark.parametrize("tag, target", [
    ("world", "world"), ("volumeref", "leaf"), ("solidref", "c"),
    ("materialref", "vac"), ("first", "box"), ("second", "box"),
    ("positionref", "p"), ("firstpositionref", "p"), ("firstrotationref", "r"),
])
@pytest.mark.parametrize("suffix, contents", [
    ("", "nested"), ("", "<position x='1'/>"),
    (' unexpected="1"', ""),
])
def test_reference_elements_require_only_ref_and_empty_contents(tmp_path, tag, target, suffix, contents):
    path = fixture(tmp_path, structure=volume("world", placement("one", "leaf", '<positionref ref="p"/>'), solid="c")
                   + volume("leaf", solid="c"), define='<position name="p" x="1"/><rotation name="r"/>',
                   solids='<box name="box"/><union name="c"><first ref="box"/>'
                          '<second ref="box"/><firstpositionref ref="p"/><firstrotationref ref="r"/></union>')
    old = f'<{tag} ref="{target}"/>'
    replacement = f'<{tag} ref="{target}"{suffix}>{contents}</{tag}>'
    path.write_text(path.read_text().replace(old, replacement))
    with pytest.raises(ValueError, match="reference syntax"):
        tsgdml.audit(path)


def test_file_size_bounds_and_bounded_read(tmp_path, monkeypatch):
    path = fixture(tmp_path)
    size = path.stat().st_size
    assert tsgdml.audit(path, max_bytes=size)["source"]["bytes"] == size
    with pytest.raises(ValueError, match="max_bytes"):
        tsgdml.audit(path, max_bytes=size - 1)
    # Simulate a file growing after fstat: the bounded read still rejects it.
    monkeypatch.setattr(tsgdml.os, "fstat", lambda fd: type("Stat", (), {"st_size": 0})())
    with pytest.raises(ValueError, match="max_bytes"):
        tsgdml.audit(path, max_bytes=size - 1)


@pytest.mark.parametrize("limit, ceiling", [
    ("max_bytes", 64 * 1024 * 1024), ("max_nodes", 500000),
    ("max_depth", 128), ("max_physical_depth", 128),
    ("max_path_items", 2000000), ("max_instances", 1000000),
])
@pytest.mark.parametrize("bad", [True, 1.0, "1", None, 0, -1, "above"])
def test_numeric_limits_require_bounded_integers(tmp_path, limit, ceiling, bad):
    path = fixture(tmp_path)
    with pytest.raises(ValueError, match=limit):
        tsgdml.audit(path, **{limit: ceiling + 1 if bad == "above" else bad})


def test_xml_node_and_depth_limits_precede_raw_serialization(tmp_path):
    path = fixture(tmp_path)
    count = tsgdml.audit(path)["xml_node_count"]
    assert tsgdml.audit(path, max_nodes=count)["xml_node_count"] == count
    with pytest.raises(ValueError, match="max_nodes"):
        tsgdml.audit(path, max_nodes=count - 1)
    with pytest.raises(ValueError, match="max_depth"):
        tsgdml.audit(path, max_depth=3)
    path = fixture(tmp_path, solids='<futureShape name="box">' + '<node>' * 126
                   + '</node>' * 126 + '</futureShape>')
    with pytest.raises(ValueError, match="max_depth"):
        tsgdml.audit(path)


def test_physical_depth_limit_independent_of_xml_depth(tmp_path):
    structure = volume("world", placement("p0", "v0")) + ''.join(
        volume(f"v{i}", placement(f"p{i + 1}", f"v{i + 1}") if i < 127 else "")
        for i in range(128))
    path = fixture(tmp_path, structure=structure)
    with pytest.raises(ValueError, match="max_physical_depth"):
        tsgdml.audit(path)
    path = fixture(tmp_path)
    assert tsgdml.audit(path, max_physical_depth=3)["reachable"]["volume_instances"] == 5
    with pytest.raises(ValueError, match="max_physical_depth"):
        tsgdml.audit(path, max_physical_depth=2)


def test_scope_path_budget_counts_each_scope_including_root_path(tmp_path):
    path = fixture(tmp_path)
    report = tsgdml.audit(path, ["world", "TS1Vacuum"], max_path_items=16)
    assert report["scope_path_items"] == 16
    with pytest.raises(ValueError, match="max_path_items"):
        tsgdml.audit(path, ["world", "TS1Vacuum"], max_path_items=15)
    assert tsgdml.audit(path, max_path_items=1)["scope_path_items"] == 0


def test_scope_route_retains_ancestor_transforms_copy_and_instance_ids(tmp_path):
    structure = (volume("world", '<physvol name="outer" copynumber="4"><volumeref ref="branch"/>'
                        '<positionref ref="p"/></physvol>')
                 + volume("branch", placement("inner", "leaf", '<rotation z="angle"/>'))
                 + volume("leaf"))
    path = fixture(tmp_path, structure=structure,
                   define='<constant name="offset" value="2"/><position name="p" x="offset"/>'
                          '<matrix name="matrix" coldim="2" values="1 2 3 4"/>')
    report = tsgdml.audit(path, ["leaf"])
    record, = report["scopes"]["leaf"]["records"]
    world, outer, inner = record["path"]
    assert world["transforms"] == {}
    assert world["parent_instance"] is None
    assert outer["copy_number"]["attribute"] == "4"
    assert outer["transforms"]["position"]["reference"]["attributes"] == {"ref": "p"}
    assert inner["transforms"] == record["transforms"]
    assert inner["instance"] == record["instance"] == 3
    assert inner["parent_instance"] == record["parent_instance"] == outer["instance"] == 2
    assert report["definitions"]["matrix"]["attributes"]["values"] == "1 2 3 4"
    assert [c["attributes"]["name"] for c in report["define"]["children"]] == ["offset", "p", "matrix"]
    assert report["standalone_geometry_compilation_supported"] is False
    assert tsgdml.summary(report)["standalone_geometry_compilation_supported"] is False


@pytest.mark.parametrize("transforms", [
    '<firstposition x="offset" unit="cm"/><firstrotation z="angle" unit="rad"/>',
    '<firstpositionref ref="p"/><firstrotationref ref="r"/>',
])
def test_csg_first_operand_transforms_preserved_raw_and_observed(tmp_path, transforms):
    path = fixture(tmp_path, structure=volume("world", solid="c"),
                   define='<constant name="offset" value="2"/><position name="p" x="offset" unit="cm"/>'
                          '<rotation name="r" z="angle" unit="rad"/>',
                   solids='<box name="box"/><union name="c"><first ref="box"/>'
                          '<second ref="box"/>' + transforms + '</union>')
    report = tsgdml.audit(path)
    raw = report["solids"]["c"]["definition"]["children"]
    assert [c["tag"] for c in raw][2:] == (["firstpositionref", "firstrotationref"]
             if "ref" in transforms else ["firstposition", "firstrotation"])
    occurrences = [o for o in report["transform_observations"] if o["context"] == "solid:c:first"]
    assert len(occurrences) == 2
    assert occurrences[0]["symbolic_values"] == {"x": "offset"}
    assert occurrences[1]["nonstandard_unit"] == "rad"
    assert report["transforms_evaluated"] is False


@pytest.mark.parametrize("extra", ['<firstscale x="2"/>', '<firstscaleref ref="p"/>',
    '<firstpositionref ref="r"/>', '<firstrotationref ref="p"/>',
    '<firstposition x="1"/><firstpositionref ref="p"/>'])
def test_csg_invalid_first_operand_transform_fails_closed(tmp_path, extra):
    path = fixture(tmp_path, structure=volume("world", solid="c"),
                   define='<position name="p"/><rotation name="r"/>',
                   solids='<box name="box"/><union name="c"><first ref="box"/>'
                          '<second ref="box"/>' + extra + '</union>')
    with pytest.raises(ValueError):
        tsgdml.audit(path)
