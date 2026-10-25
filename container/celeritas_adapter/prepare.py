"""Prepare a hash-pinned standalone application extension; never configure or build."""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMIT = "ddf6c690e37d53b56d001e6bf0c26cb1b14963fa"


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Patch anchor is not unique: {before!r}")
    return text.replace(before, after)


def patch_sources(files):
    files = dict(files)
    def change(name, before, after):
        key = f"app/celer-g4/{name}"
        files[key] = replace_once(files[key], before, after)

    change("DetectorConstruction.cc", '#include "DetectorConstruction.hh"',
           '#include "DetectorConstruction.hh"\n#include "CartMapIO.hh"\n'
           '#include "accel/CartMapMagneticField.hh"')
    change("DetectorConstruction.cc", "    auto field = this->construct_field();",
           '    if (GlobalSetup::Instance()->GetFieldType() == "cartmap")\n'
           '        m2fs::require_world_inside(m2fs::read_cartmap(GlobalSetup::Instance()->GetFieldFile()), *loaded.world);\n'
           "    auto field = this->construct_field();")
    change("DetectorConstruction.cc", '    if (field_type == "rzmap")',
           '    if (field_type == "cartmap")\n'
           '    {\n'
           '        auto map = m2fs::read_cartmap(GlobalSetup::Instance()->GetFieldFile());\n'
           '        map.driver_options = GlobalSetup::Instance()->GetFieldOptions();\n'
           '        auto params = std::make_shared<CartMapFieldParams>(map);\n'
           '        return {CartMapFieldAlongStepFactory([map] { return map; }),\n'
           '                std::make_shared<CartMapMagneticField>(std::move(params))};\n'
           '    }\n'
           '    if (field_type == "rzmap")')
    change("RunInputIO.json.cc", 'v.field != RunInput::no_field() || v.field_type == "rzmap")',
           'v.field != RunInput::no_field() || v.field_type == "rzmap" || v.field_type == "cartmap")')
    change("RunInputIO.json.cc", 'if (v.field_type == "rzmap")',
           'if (v.field_type == "rzmap" || v.field_type == "cartmap")')
    change("RunInput.cc", '#include "RunInput.hh"', '#include "RunInput.hh"\n#include "CartMapIO.hh"')
    change("RunInput.cc", '    if (ri.field_type == "rzmap")',
           '    if (ri.field_type == "cartmap")\n'
           '    {\n'
           '        auto map = m2fs::read_cartmap(ri.field_file);\n'
           '        map.driver_options = ri.field_options;\n'
           '        p.field = std::move(map);\n'
           '    }\n'
           '    else if (ri.field_type == "rzmap")')
    change("GlobalSetup.cc", '[rzmap|uniform]', '[cartmap|rzmap|uniform]')
    change("GlobalSetup.cc", 'Filename of the rz-map loaded by RZMapFieldInput',
           'Filename of the selected RZ or Cartesian field map')
    change("EventAction.cc", '#include "EventAction.hh"', '#include "EventAction.hh"\n#include "EventJson.hh"')
    change("EventAction.cc", '    if (GlobalSetup::Instance()->root_sd_io())',
           '    m2fs::write_event(*event);\n\n    if (GlobalSetup::Instance()->root_sd_io())')
    change("celer-g4.cc", '#include "GlobalSetup.hh"',
           '#include "GlobalSetup.hh"\n#include "EventJson.hh"')
    change("celer-g4.cc", '    std::vector<std::string> ignore_processes = {"CoulombScat"};',
           '    if (auto* path = std::getenv("M2FS_EVENT_OUTPUT"); path && *path)\n'
           '    {\n'
           '        CELER_VALIDATE(setup.input().sd_type == SensitiveDetectorType::event_hit,\n'
           '                       << "M2FS_EVENT_OUTPUT requires sd_type=event_hit");\n'
           '        m2fs::initialize_event_output();\n'
           '    }\n'
           '    std::vector<std::string> ignore_processes = {"CoulombScat"};')
    return files


def prepare(source, output):
    pin = json.loads((HERE / "upstream-files.json").read_text())
    if pin["commit"] != COMMIT:
        raise ValueError("Wrong upstream pin")
    files = {}
    for name, digest in pin["sha256"].items():
        data = (source / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Unexpected upstream source: {name}")
        files[name] = data.decode()
    files = patch_sources(files)
    output.mkdir(parents=False, exist_ok=False)
    for name, text in files.items():
        dest = output / ("app/" + Path(name).name if name.startswith("app/") else name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
    local = ("CartMapIO.hh", "CartMapIO.cc", "EventJson.hh", "EventJson.cc", "field-check.cc", "CMakeLists.txt", "LICENSE-MIT")
    for name in local:
        (output / name).write_bytes((HERE / name).read_bytes())
    manifest = {"upstream": pin, "status": "prepared only; not compiled",
                "adapter_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                   for name in (*local, "prepare.py")}}
    manifest["prepared_sha256"] = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "adapter-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
    print(f"Prepared {args.output}; no configure/build was run")


if __name__ == "__main__":
    main()
