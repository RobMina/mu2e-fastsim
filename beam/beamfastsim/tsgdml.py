"""Read-only local GDML structural inventory; NOT a geometry parser or navigator."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

TRANSFORMS = ("position", "rotation", "scale")
CSG = {"union", "intersection", "subtraction"}


def _raw(node):
    return dict(tag=node.tag, attributes=dict(node.attrib),
                text=(node.text or "").strip(), children=[_raw(c) for c in node])


def _index(nodes):
    result = {}
    for node in nodes:
        name = node.get("name")
        if not name or name in result:
            raise ValueError(f"Missing or duplicate definition name: {name!r}")
        result[name] = node
    return result


def _one(node, tag):
    children = node.findall(tag)
    if len(children) != 1:
        raise ValueError(f"{node.tag} {node.get('name')}: expected one {tag}")
    return children[0]


def _ref(node, definitions):
    if set(node.attrib) != {"ref"} or len(node) or (node.text or "").strip():
        raise ValueError(f"Invalid {node.tag} reference syntax")
    name = node.get("ref")
    if not name or name not in definitions:
        raise ValueError(f"Missing {node.tag} reference: {name!r}")
    return name


def _scalar(value):
    """Check expression syntax, never evaluate a GDML expression."""
    try:
        tree = ast.parse(value, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Malformed scalar expression: {value!r}") from exc
    allowed = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.BinOp,
               ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
               ast.Mod, ast.UAdd, ast.USub, ast.Call)
    for item in ast.walk(tree):
        if not isinstance(item, allowed):
            raise ValueError(f"Unsupported scalar syntax: {value!r}")
        if isinstance(item, ast.Constant) and (type(item.value) not in (int, float)
                                              or (type(item.value) is float and not math.isfinite(item.value))):
            raise ValueError(f"Nonfinite/nonnumeric scalar: {value!r}")
        if isinstance(item, ast.Call) and (not isinstance(item.func, ast.Name) or item.keywords):
            raise ValueError(f"Unsupported scalar call: {value!r}")
    names = sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)})
    if any(n.lower() in {"nan", "inf", "infinity"} for n in names):
        raise ValueError(f"Nonfinite scalar: {value!r}")
    return names


def audit(path, roots=(), *, max_instances=1000000, max_bytes=64 * 1024 * 1024,
          max_nodes=500000, max_depth=128, max_physical_depth=128,
          max_path_items=2000000):
    """Inventory all world-reachable placements; scopes include their root and descendants.

    Root matching uses physical OR logical names, stripping only final 0xHEX.
    Multiple matching physical instances are ambiguous (even shared logical volumes).
    The world counts as one logical instance but not as a physical placement.
    No dimensions, transforms, material physics, overlaps or navigation are certified.
    Limits may be lowered, never raised beyond their safety ceilings. XML and
    physical depth count the document/world root as depth one. Path items count
    every route entry for every scope record, including overlapping scopes.
    """
    normalize = lambda s: re.sub(r"0x[0-9a-fA-F]+$", "", s)
    if isinstance(roots, str):
        raise ValueError("roots must be a sequence of names")
    roots = tuple(roots)
    if any(not isinstance(r, str) or not r or not normalize(r) for r in roots):
        raise ValueError("roots must be distinct nonempty names")
    if len(set(roots)) != len(roots):
        raise ValueError("roots must be distinct nonempty names")
    for label, value, ceiling in (
            ("max_instances", max_instances, 1000000),
            ("max_bytes", max_bytes, 64 * 1024 * 1024),
            ("max_nodes", max_nodes, 500000), ("max_depth", max_depth, 128),
            ("max_physical_depth", max_physical_depth, 128),
            ("max_path_items", max_path_items, 2000000)):
        if type(value) is not int or not 1 <= value <= ceiling:
            raise ValueError(f"{label} must be an integer in [1, {ceiling}]")
    if "://" in str(path):
        raise ValueError("An explicit local GDML file is required")
    path = Path(path).expanduser().resolve(strict=True)
    with path.open("rb") as source:
        if os.fstat(source.fileno()).st_size > max_bytes:
            raise ValueError("max_bytes exceeded")
        data = source.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("max_bytes exceeded")
    if b"\x00" in data or re.search(br"<!\s*(DOCTYPE|ENTITY)\b", data, re.I):
        raise ValueError("DTD/entities and non-ASCII-compatible XML encodings are forbidden")
    try:
        xml = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"Malformed GDML XML: {exc}") from exc
    # Bound recursive raw serialization before inspecting or copying any subtrees.
    pending, node_count = [(xml, 1)], 0
    while pending:
        node, depth = pending.pop()
        node_count += 1
        if node_count > max_nodes:
            raise ValueError("max_nodes exceeded")
        if depth > max_depth:
            raise ValueError("max_depth exceeded")
        pending.extend((child, depth + 1) for child in node)
    if xml.tag != "gdml" or any(not isinstance(n.tag, str) or "{" in n.tag for n in xml.iter()):
        raise ValueError("Expected unnamespaced gdml elements")
    if any(n.tag == "file" for n in xml.iter()):
        raise ValueError("Modular file includes are forbidden")
    if any(n.tag not in {"define", "materials", "solids", "structure", "setup"} for n in xml):
        raise ValueError("Unsupported top-level GDML construct")
    define = _one(xml, "define") if xml.find("define") is not None else None
    definitions = _index(define) if define is not None else {}
    materials = _index(_one(xml, "materials"))
    solids = _index(_one(xml, "solids"))
    volumes = _index(_one(xml, "structure"))
    world = _ref(_one(_one(xml, "setup"), "world"), volumes)
    observations = []

    def transform(node, context, kind=None):
        kind = kind or node.tag
        if kind not in TRANSFORMS or node.tag not in (kind, "first" + kind) or len(node) or (node.text or "").strip():
            raise ValueError(f"Invalid scalar transform in {context}")
        if set(node.attrib) - {"name", "unit", "x", "y", "z"}:
            raise ValueError(f"Unsupported transform attributes in {context}")
        symbolic = {k: _scalar(v) for k, v in node.attrib.items() if k in {"x", "y", "z"}}
        unit = node.get("unit")
        expected = {"position": "mm", "rotation": "deg", "scale": None}[kind]
        observations.append(dict(context=context, spec=_raw(node),
                                 symbolic_values={k: node.get(k) for k, v in symbolic.items() if v},
                                 nonstandard_unit=unit if unit is not None and unit != expected else None))
        return _raw(node)

    for name, node in definitions.items():
        if node.tag in TRANSFORMS:
            transform(node, f"define:{name}")

    def specs(node, context, prefix="", kinds=TRANSFORMS):
        result = {}
        for kind in kinds:
            tag = prefix + kind
            entries = [c for c in node if c.tag in (tag, tag + "ref")]
            if len(entries) > 1:
                raise ValueError(f"Duplicate {tag} specification in {context}")
            if entries:
                entry = entries[0]
                if entry.tag.endswith("ref"):
                    name = _ref(entry, definitions)
                    if definitions[name].tag != kind:
                        raise ValueError(f"Wrong {tag} reference type/syntax in {context}")
                    transform(definitions[name], context, kind)
                    observations[-1]["reference"] = _raw(entry)
                    result[tag] = dict(reference=_raw(entry), definition=_raw(definitions[name]))
                else:
                    result[tag] = transform(entry, context, kind)
        return result

    reached_solids, solid_active = set(), set()

    def solid_closure(start):
        stack = [(start, False)]
        while stack:
            name, leaving = stack.pop()
            if leaving:
                solid_active.remove(name)
                reached_solids.add(name)
                continue
            if name in solid_active:
                raise ValueError(f"Solid reference cycle: {name}")
            if name in reached_solids:
                continue
            node = solids[name]
            solid_active.add(name)
            stack.append((name, True))
            if node.tag in CSG:
                if any(c.tag not in {"first", "second", "position", "positionref", "rotation", "rotationref", "scale", "scaleref", "firstposition", "firstpositionref", "firstrotation", "firstrotationref"} for c in node):
                    raise ValueError(f"Unsupported CSG structure: {name}")
                specs(node, f"solid:{name}")
                specs(node, f"solid:{name}:first", prefix="first", kinds=("position", "rotation"))
                for tag in ("second", "first"):
                    stack.append((_ref(_one(node, tag), solids), False))
            else:
                # Unknown solids are inventoried, not interpreted. Still close any explicit solid links.
                for child in node.iter():
                    if child.tag in {"first", "second", "solid", "solidref"}:
                        stack.append((_ref(child, solids), False))

    cache = {}
    def volume(name):
        if name not in cache:
            node = volumes[name]
            if node.tag != "volume" or any(c.tag not in {"materialref", "solidref", "physvol", "auxiliary"} for c in node):
                raise ValueError(f"Unsupported volume/placement construct: {name}")
            material = _ref(_one(node, "materialref"), materials)
            if materials[material].tag != "material":
                raise ValueError(f"materialref must reference a material: {material}")
            solid = _ref(_one(node, "solidref"), solids)
            solid_closure(solid)
            children = []
            for p in node.findall("physvol"):
                if any(c.tag not in {"volumeref", "position", "positionref", "rotation", "rotationref", "scale", "scaleref", "copynumber"} for c in p):
                    raise ValueError(f"Unsupported physvol construct in {name}")
                target = _ref(_one(p, "volumeref"), volumes)
                children.append((target, p.get("name"), specs(p, f"physvol:{name}/{p.get('name')}"),
                                 dict(attribute=p.get("copynumber"), elements=[_raw(c) for c in p.findall("copynumber")])))
            cache[name] = material, solid, children
        return cache[name]

    counts, material_counts, solid_counts, tag_counts = (Counter() for _ in range(4))
    scopes = {r: dict(records=[], matches=0) for r in roots}
    stack = [(world, None, {}, None, (), (), ())]
    total = path_items = 0
    while stack:
        name, physical, transforms, copy, ancestors, route, active = stack.pop()
        if len(ancestors) + 1 > max_physical_depth:
            raise ValueError("max_physical_depth exceeded")
        if name in ancestors:
            raise ValueError(f"Physical placement cycle: {name}")
        total += 1
        if total > max_instances:
            raise ValueError("max_instances exceeded; no truncated receipt is produced")
        material, solid, children = volume(name)
        counts[name] += 1
        material_counts[material] += 1
        solid_counts[solid] += 1
        tag_counts[solids[solid].tag] += 1
        parent_instance = route[-1]["instance"] if route else None
        route = route + (dict(logical=name, physical=physical, instance=total,
                              parent_instance=parent_instance, transforms=transforms,
                              copy_number=copy),)
        candidates = {normalize(n) for n in (name, physical) if n is not None}
        selected = tuple(r for r in roots if normalize(r) in candidates)
        for r in selected:
            scopes[r]["matches"] += 1
            if scopes[r]["matches"] > 1:
                raise ValueError(f"Ambiguous root: {r}")
        active = tuple(dict.fromkeys(active + selected))
        if active:
            path_items += len(route) * len(active)
            if path_items > max_path_items:
                raise ValueError("max_path_items exceeded; no truncated receipt is produced")
            record = dict(instance=total, parent_instance=parent_instance, path=list(route), logical_name=name, physical_name=physical,
                          material=material, solid=solid, solid_tag=solids[solid].tag,
                          transforms=transforms, copy_number=copy, child_count=len(children))
            for r in active:
                scopes[r]["records"].append(record)
        for child in reversed(children):
            stack.append((*child, ancestors + (name,), route, active))
    for r, scope in scopes.items():
        if not scope["matches"]:
            raise ValueError(f"Root not found in world subtree: {r}")
        scope["volume_instances"] = len(scope["records"])
        scope["material_instance_counts"] = dict(Counter(n["material"] for n in scope["records"]))
        scope["solid_tag_instance_counts"] = dict(Counter(n["solid_tag"] for n in scope["records"]))
    transform_tags = {prefix + kind + suffix
                      for prefix, kinds in (("", TRANSFORMS), ("first", ("position", "rotation")))
                      for kind in kinds for suffix in ("", "ref")}
    return dict(schema="ts-gdml-structure-v1",
                source=dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()),
                world_volume=world, structural_audit_complete=True, navigation_supported=False,
                standalone_geometry_compilation_supported=False,
                limits=dict(max_bytes=max_bytes, max_nodes=max_nodes, max_depth=max_depth,
                            max_instances=max_instances, max_physical_depth=max_physical_depth,
                            max_path_items=max_path_items),
                xml_node_count=node_count, scope_path_items=path_items,
                define=_raw(define) if define is not None else None,
                definitions={k: _raw(v) for k, v in definitions.items()},
                scope="World-reachable structural references only; no evaluated transforms, extents, overlaps, solid equivalence or geometry validity",
                reachable=dict(volume_instances=total, physical_volume_instances=total - 1,
                               logical_volume_instances=dict(counts), repeated_logical_placements={k: v for k, v in counts.items() if v > 1},
                               material_instance_counts=dict(material_counts), solid_instance_counts=dict(solid_counts), solid_tag_instance_counts=dict(tag_counts),
                               solid_definitions=sorted(reached_solids)),
                unreachable=dict(logical_volumes=sorted(set(volumes) - set(counts)), solids=sorted(set(solids) - reached_solids),
                                 materials=sorted({k for k, v in materials.items() if v.tag == "material"} - set(material_counts))),
                materials={k: _raw(v) for k, v in materials.items()},
                solids={k: dict(definition=_raw(v), navigation="unsupported-for-navigation") for k, v in solids.items()},
                solid_tag_counts=dict(Counter(s.tag for s in solids.values())),
                transform_tag_counts=dict(Counter(n.tag for n in xml.iter() if n.tag in transform_tags)),
                transform_observations=observations, transforms_evaluated=False, scopes=scopes)


def summary(report):
    """Small structural summary; never a navigation acceptance or geometry check."""
    return dict(source=report["source"], world_volume=report["world_volume"],
                volume_instances=report["reachable"]["volume_instances"],
                logical_volumes=len(report["reachable"]["logical_volume_instances"]),
                solid_tag_counts=report["solid_tag_counts"],
                scopes={key: {name: value[name] for name in
                        ("volume_instances", "material_instance_counts", "solid_tag_instance_counts")}
                        for key, value in report["scopes"].items()},
                symbolic_transform_count=sum(bool(o["symbolic_values"]) for o in report["transform_observations"]),
                nonstandard_transform_unit_count=sum(o["nonstandard_unit"] is not None for o in report["transform_observations"]),
                navigation_supported=False, standalone_geometry_compilation_supported=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        source = Path(args.input).expanduser().resolve(strict=True)
        if args.out.resolve() == source or (args.out.exists() and args.out.samefile(source)):
            raise ValueError("--out must not overwrite input (including aliases)")
        report = audit(args.input, args.root)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary(report), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
