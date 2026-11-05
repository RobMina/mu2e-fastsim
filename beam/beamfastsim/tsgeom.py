"""Restricted GDML-to-clearance compiler for vacuum mothers with leaf daughters.

This compiles only explicitly selected, structurally audited subtrees. It is not
an arbitrary GDML loader or Geant4 navigator. All daughter solids are retained;
unknown solids, symbolic dimensions, scales, and nested daughters fail closed.
The caller explicitly names accepted vacuum materials. Export world coordinates
are used, NOT presumed Mu2e coordinates. No material transport or boundary/seam
crossing algorithm is supplied. Numerical equivalence to native solid queries is
not certified by successful compilation.

GDML conventions follow Geant4 v11.3.2 G4GDMLReadDefine::GetRotationMatrix,
G4GDMLReadStructure::PhysvolRead and G4GDMLReadSolids::BooleanRead: frame Euler
Rz*Ry*Rx is inverted for physical placements, but Boolean operand transforms
use it directly. See https://github.com/Geant4/geant4/tree/v11.3.2/source/persistency/gdml/src.
These are source-level conventions, not a matched-binary roundtrip certificate.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

from . import tsgdml, tslocate, tsnav, tssolid


def _name(value):
    return re.sub(r"0x[0-9a-fA-F]+$", "", value)


def _number(attributes, key, default=None):
    value = attributes.get(key, default)
    if value is None or not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", str(value)):
        raise ValueError(f"{key} requires a finite numeric literal")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{key} requires a finite numeric literal")
    return result


def _angles(attributes):
    unit = attributes.get("unit", "rad")
    if unit not in ("deg", "rad"):
        raise ValueError("rotation requires deg or rad")
    return np.array([_number(attributes, k, 0) for k in "xyz"]) * (np.pi / 180 if unit == "deg" else 1.)


def _matrix(angles):
    x, y, z = angles
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    rx = np.array([[1., 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.]])
    return rz @ ry @ rx


def _transform(specs, *, physical):
    if set(specs) - {"position", "rotation"}:
        raise ValueError("scale or unsupported placement transform")
    def attributes(key):
        node = specs.get(key)
        return (node.get("definition", node)["attributes"] if node else {})
    pos, rot = attributes("position"), attributes("rotation")
    if pos.get("unit", "mm") != "mm":
        raise ValueError("position requires explicit/default mm")
    t = np.array([_number(pos, k, 0) for k in "xyz"])
    matrix = _matrix(_angles(rot))
    return (matrix.T if physical else matrix), t


def _raw_specs(node, definitions, prefix=""):
    specs = {}
    allowed = {"first", "second", "position", "rotation", "positionref", "rotationref",
               "firstposition", "firstrotation", "firstpositionref", "firstrotationref"}
    if any(c["tag"] not in allowed for c in node["children"]):
        raise ValueError("unsupported CSG transform construct")
    for key in ("position", "rotation"):
        tag = prefix + key
        matches = [c for c in node["children"] if c["tag"] in (tag, tag + "ref")]
        if len(matches) > 1:
            raise ValueError("duplicate CSG transform")
        if matches:
            value = matches[0]
            if value["tag"].endswith("ref"):
                ref = value["attributes"]["ref"]
                if ref not in definitions or definitions[ref]["tag"] != key:
                    raise ValueError("missing/wrong CSG transform reference")
                value = definitions[ref]
            specs[key] = value
    return specs


class _Solids:
    def __init__(self, report):
        self.nodes = {k: v["definition"] for k, v in report["solids"].items()}
        self.definitions = report.get("definitions", {})
        self.cache = {}
        self.active = set()

    def get(self, name):
        if name in self.active:
            raise ValueError("solid reference cycle")
        if name in self.cache:
            return self.cache[name]
        self.active.add(name)
        try:
            result = self._compile(self.nodes[name])
        finally:
            self.active.remove(name)
        self.cache[name] = result
        return result

    def _compile(self, node):
        tag, a, children = node["tag"], node["attributes"], node["children"]
        if tag in ("union", "intersection", "subtraction"):
            if set(a) != {"name"}:
                raise ValueError("unsupported Boolean solid attributes")
            operands = []
            for key, prefix in (("first", "first"), ("second", "")):
                refs = [c for c in children if c["tag"] == key]
                if len(refs) != 1:
                    raise ValueError("CSG requires exactly two operands")
                solid = self.get(refs[0]["attributes"]["ref"])
                matrix, position = _transform(_raw_specs(node, self.definitions, prefix), physical=False)
                operands.append(tsnav.Placed(solid, position, matrix))
            cls = {"union": tsnav.Union, "intersection": tsnav.Intersection, "subtraction": tsnav.Difference}[tag]
            return cls(*operands)
        if a.get("lunit", "mm") != "mm":
            raise ValueError("solid lengths must use mm")
        allowed = {"box": {"x", "y", "z"},
                   "tube": {"rmax", "rmin", "z", "startphi", "deltaphi", "aunit"},
                   "cone": {"rmax1", "rmax2", "rmin1", "rmin2", "z", "startphi", "deltaphi", "aunit"},
                   "torus": {"rtor", "rmax", "rmin", "startphi", "deltaphi", "aunit"}, "xtru": set()}
        if tag not in allowed or set(a) - allowed[tag] - {"name", "lunit"}:
            raise ValueError(f"unsupported solid/attributes: {tag}")
        if tag != "xtru" and children:
            raise ValueError(f"unexpected children in {tag}")
        if tag == "box":
            return tsnav.Box([_number(a, k) / 2 for k in "xyz"])
        if tag == "xtru":
            if any(c["tag"] not in ("twoDimVertex", "section") or c["children"] for c in children):
                raise ValueError("unsupported extrusion structure")
            vertices = [c["attributes"] for c in children if c["tag"] == "twoDimVertex"]
            sections = [c["attributes"] for c in children if c["tag"] == "section"]
            if any(set(v) != {"x", "y"} for v in vertices):
                raise ValueError("unsupported extrusion vertex attributes")
            if len(sections) != 2 or any(set(s) != {"zOrder", "zPosition", "xOffset", "yOffset", "scalingFactor"} for s in sections):
                raise ValueError("exactly two explicit extrusion sections required")
            sections.sort(key=lambda s: _number(s, "zOrder"))
            if [_number(s, "zOrder") for s in sections] != [0., 1.]:
                raise ValueError("extrusion section order must be 0,1")
            if any(_number(s, "scalingFactor") != 1 or _number(s, "xOffset") != 0 or _number(s, "yOffset") != 0 for s in sections):
                raise ValueError("scaled/offset extrusion sections unsupported")
            z0, z1 = [_number(s, "zPosition") for s in sections]
            if z1 <= z0:
                raise ValueError("extrusion sections must increase in z")
            solid = tssolid.Extrusion([[_number(v, "x"), _number(v, "y")] for v in vertices], (z1 - z0) / 2)
            return tsnav.Placed(solid, [0, 0, (z0 + z1) / 2], np.eye(3))
        if tag == "tube":
            solid = tsnav.Tube(_number(a, "rmax"), _number(a, "z") / 2, _number(a, "rmin", 0))
        elif tag == "cone":
            solid = tssolid.Cone(_number(a, "rmax1"), _number(a, "rmax2"), _number(a, "z") / 2,
                                  _number(a, "rmin1", 0), _number(a, "rmin2", 0))
        else:
            if _number(a, "rmin", 0) != 0:
                raise ValueError("hollow torus is unsupported")
            solid = tsnav.Torus(_number(a, "rtor"), _number(a, "rmax"))
        unit = a.get("aunit", "rad")
        if unit not in ("deg", "rad"):
            raise ValueError("solid angles must use deg or rad")
        scale = np.pi / 180 if unit == "deg" else 1.
        return tssolid.Sector(solid, _number(a, "startphi", 0) * scale,
                              _number(a, "deltaphi") * scale)


@dataclass(frozen=True)
class Scene:
    """World-frame conservative vacuum region plus a placed-leaf receipt.

    clearance zero may be a nonphysical CSG seam. Never interpret it as material.
    The receipt documents selected roots only; other world solids are NOT modeled.
    The required external exclusion of sibling overlaps is not certified here.
    """

    region: object
    receipt: dict
    partition: tslocate.Partition

    def clearance(self, points):
        return self.region.clearance(points)

    def locate(self, points):
        return self.partition.locate(points)


def load(path, roots, *, vacuum_materials):
    """Compile selected vacuum mothers with all direct leaf daughters.

    No defaults: roots and accepted material names are caller decisions. Exact
    material references win; suffix-stripped aliases must be unambiguous. A vacuum
    daughter is geometrically coalesced into its accepted vacuum mother; material
    daughters remain exact CSG exclusions. Nested daughters fail, not flatten.
    """
    if isinstance(vacuum_materials, str):
        raise ValueError("vacuum_materials must be an explicit sequence")
    vacuum = tuple(vacuum_materials)
    if not vacuum or any(not isinstance(m, str) or not _name(m) for m in vacuum):
        raise ValueError("nonempty explicit vacuum material allowlist required")
    if not roots:
        raise ValueError("explicit roots required")
    report = tsgdml.audit(path, roots)
    material_names = {k for k, v in report["materials"].items() if v["tag"] == "material"}
    vacuum_refs = set()
    for request in vacuum:
        matches = {request} if request in material_names else {k for k in material_names if _name(k) == request}
        if len(matches) != 1:
            raise ValueError("vacuum material name is missing or ambiguous; use exact reference")
        vacuum_refs.update(matches)
    compiler = _Solids(report)
    regions, ledger, cells = [], [], []
    selected_instances = set()
    for key, scope in report["scopes"].items():
        records = scope["records"]
        mother = records[0]
        if mother["instance"] in selected_instances:
            raise ValueError("duplicate selected physical instance through root aliases")
        selected_instances.add(mother["instance"])
        if len(records) != mother["child_count"] + 1 or any(r["child_count"] != 0 for r in records[1:]):
            raise ValueError("only vacuum mothers with direct leaf daughters are supported")
        if mother["material"] not in vacuum_refs:
            raise ValueError("selected mother material is not in vacuum allowlist")
        placed, obstacles, volumes = [], [], []
        for record in records:
            matrix, position = np.eye(3), np.zeros(3)
            for ancestor in record["path"]:
                local_r, local_t = _transform(ancestor["transforms"], physical=True)
                position = position + matrix @ local_t
                matrix = matrix @ local_r
            shape = tsnav.Placed(compiler.get(record["solid"]), position, matrix)
            is_vacuum = record["material"] in vacuum_refs
            placed.append(shape)
            volumes.append(tslocate.Volume(shape, record["material"], record["physical_name"], key,
                                           is_vacuum, instance=record["instance"],
                                           logical=record["logical_name"],
                                           instance_path=tuple(a["instance"] for a in record["path"])))
            if not is_vacuum:
                obstacles.append(shape)
            ledger.append(dict(root=key, logical=record["logical_name"], physical=record["physical_name"],
                               instance=record["instance"], parent_instance=record["parent_instance"],
                               copy_number=record["copy_number"], path=record["path"],
                               material=record["material"], vacuum=is_vacuum,
                               center_world_mm=position.tolist(), rotation_local_to_world=matrix.tolist(),
                               solid=record["solid"], solid_tag=record["solid_tag"]))
        cells.append(tslocate.Cell(volumes[0], tuple(volumes[1:])))
        regions.append(tsnav.Difference(placed[0], tsnav.Union(*obstacles)) if obstacles else placed[0])
    receipt = dict(schema="ts-selected-gdml-clearance-v1", source=report["source"],
                   roots=list(roots), vacuum_materials=list(vacuum),
                   resolved_vacuum_materials=sorted(vacuum_refs), placed_volumes=ledger,
                   compiled_solid_definitions=len(compiler.cache),
                   coordinates="Export world frame; no inferred Mu2e coordinate translation",
                   native_query_certified=False, material_transport=False,
                   scope="Selected vacuum mothers and all leaf daughters only; other siblings/overlaps and floating tolerances not certified; zero-set seams unresolved")
    return Scene(tsnav.Union(*regions), receipt, tslocate.Partition(tuple(cells)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--vacuum-material", action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--point", type=float, nargs=3, action="append", default=[],
                    help="world-frame point [mm] for a numerical clearance query (not native validation)")
    args = ap.parse_args(argv)
    try:
        source = Path(args.input).resolve(strict=True)
        if args.out.resolve() == source or (args.out.exists() and args.out.samefile(source)):
            raise ValueError("output must not alias GDML input")
        scene = load(source, args.root, vacuum_materials=args.vacuum_material)
        receipt = dict(scene.receipt)
        if args.point:
            points = np.asarray(args.point, dtype=float)
            location = scene.locate(points)
            volumes = scene.partition.volumes
            receipt["numerical_clearance_queries"] = dict(points_world_mm=points.tolist(),
                                                          clearance_mm=scene.clearance(points).tolist(),
                                                          native_validation=False)
            receipt["numerical_locations"] = dict(
                status=[tslocate.STATUS[int(s)] for s in location.status],
                volume_index=location.volume_index.tolist(),
                boundary_index=location.boundary_index.tolist(),
                material=[volumes[i].material if i >= 0 else None for i in location.volume_index],
                local_clearance_mm=location.clearance.tolist(),
                root_count=location.root_count.tolist(), material_count=location.material_count.tolist(),
                boundary_count=location.boundary_count.tolist(), native_validation=False)
        text = json.dumps(receipt, indent=2, allow_nan=False)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(text)
    except (ValueError, OSError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
