"""Synthetic pair evidence: strict syntax, complete execution, immutable diagnostics."""
from decimal import Decimal
import gzip
import json
import stat

import pytest

from valhist import generator_bank_provenance as P

pytestmark = pytest.mark.usefixtures("dcache_site")


def fhicl(value):
    if isinstance(value, dict):
        return "{ " + " ".join(k + ": " + fhicl(v) for k, v in value.items()) + " }"
    if isinstance(value, list):
        return "[" + ", ".join(map(fhicl, value)) + "]"
    return json.dumps(value)


def config(role):
    cry = role == "crygen"
    n, sub, seed = (1000000, 2001, 20263000) if cry else (50000, 1, 20261016)
    primary = dict(outputs={"GenOutput": dict(module_type="RootOutput",
        fileName=f"gen.mu2e.CosmicCRY{'Gen' if cry else 'Ref'}.v1.001400_{sub:08d}.art",
        outputCommands=P.COMMANDS[:])}, source=dict(module_type="EmptyEvent", maxEvents=n,
        firstRun=1400, firstSubRun=sub), services=dict(SeedService=dict(policy="autoIncrement",
        maxUniqueEngines=100, baseSeed=seed), GeometryService=dict(inputFile="geom.txt"),
        GlobalConstantsService=dict(inputFile="constants.txt"), RandomNumberGenerator=dict(module_type="RandomNumberGenerator")),
        physics=dict(producers={name: dict(module_type=name, nested=dict(x=[1, True, "escaped\\\"}"]))
                               for name in P.GENERATORS}, EndPath=["genCountLogger", "GenOutput"], end_paths=["EndPath"]))
    if cry:
        primary["outputs"]["GenOutput"]["SelectEvents"] = ["GenPath"]
        primary["physics"].update(GenPath=P.GENERATORS[:], G4InitPath=["noG4", "g4run"],
            filters=dict(noG4=dict(module_type="BlockingPrescaler", blockSize=1,
                                  stepSize=1000000000, offset=1000000000)))
        paths = ["GenPath", "G4InitPath"]
    else:
        primary["physics"]["PrimaryPath"] = P.GENERATORS + ["g4run", "filter"]
        paths = ["PrimaryPath"]
    primary["trigger_paths"] = {"trigger_paths": paths}
    return primary


def dump(primary):
    return "Processing file: /dcache-test/mu2e/" + primary["outputs"]["GenOutput"]["fileName"] + "\n# Read SQLiteDB from input file\nPrimary: " + fhicl(primary) + "\n"


def log(primary, role):
    n = primary["source"]["maxEvents"]
    lines = ["mu2e build: MDC2025ay Offline abc123", "art version 3.14.00"]
    for tag in ("GlobalConstants", "Geometry", "BField"):
        lines += [tag + " file: " + tag + ".txt", tag + " lines: 123  hash: 1234567890"]
    lines += ["CRYSetup: returnMuons 1", "Ref.point: 0, 0, 0", "Targetbox: 1, 2, 3", "World: -1, 1",
              "2026-10-22 12:00:00 Closed output file " + primary["outputs"]["GenOutput"]["fileName"],
              f"TrigReport Events total = {n} passed = {n} failed = 0", "TrigReport ---------- Module Summary ------------",
              "TrigReport    Visited    Run    Passed    Failed    Error Name"]
    for name in P.GENERATORS + ["GenOutput"]:
        lines.append(f"TrigReport    {n}    {n}    {n}          0          0 {name}")
    if role == "crygen":
        lines += [f"TrigReport {n} {n} 0 {n} 0 noG4", "TrigReport 0 0 0 0 0 g4run"]
    return "\n".join(lines + ["TimeReport ---------- Time summary ------------", "\tExit status: 0", ""])


@pytest.fixture
def evidence(tmp_path):
    paths = []
    for role in ("crygen", "g4ref"):
        primary = config(role)
        for kind, text in (("config", dump(primary)), ("log", log(primary, role))):
            path = tmp_path / (role + "." + kind)
            path.write_text(text)
            paths.append(path)
    return paths


def args(paths, output):
    return ["--crygen-config", str(paths[0]), "--crygen-log", str(paths[1]),
            "--g4ref-config", str(paths[2]), "--g4ref-log", str(paths[3]), "--output", str(output)]


def test_positive_and_immutable_cli(evidence, tmp_path):
    out = tmp_path / "report.json"
    before = [p.read_bytes() for p in evidence]
    assert P.main(args(evidence, out)) == 0
    report = json.loads(out.read_text())
    assert report["downstream_authorized"] is False
    assert report["whole_bank_certified"] is False
    assert report["actual_engine_seed_mapping_certified"] is False
    assert report["source_provenance_equality_claimed"] is False
    assert report["status"] == "observed_pair_matches"
    assert len(report["equal_blocks"]) == 6
    assert len(report["inputs"]) == 5
    assert all(v["sha256_before"] == v["sha256_after"] for v in report["inputs"].values())
    assert stat.S_IMODE(out.stat().st_mode) == 0o444
    retained = out.read_bytes()
    assert P.main(args(evidence, out)) == 1
    assert out.read_bytes() == retained
    assert [p.read_bytes() for p in evidence] == before


def test_gzip_and_scientific_seed(evidence):
    evidence[0].write_text(evidence[0].read_text().replace("20263000", "2.0263e7"))
    evidence[1].write_bytes(gzip.compress(evidence[1].read_bytes()))
    assert P.compare(*evidence)["configurations"]["crygen"]["SeedService"]["baseSeed"] == 20263000


def test_parser_exact_numbers_strings_and_structures():
    parsed = P.parse_fhicl(r'A: { x: 2.0263e7, y: [true, false, "}\\\"", -1.234567890123456789e-5] }')
    assert parsed["A"]["x"] == 20263000
    assert type(parsed["A"]["x"]) is int
    assert parsed["A"]["y"][-1] == Decimal("-0.00001234567890123456789")
    assert not P.exact_equal(True, 1)
    assert not P.exact_equal([True], [1])
    assert P.exact_equal(Decimal("1.0"), 1)
    assert P._jsonable(Decimal("0.1234567890123456789")) == {"decimal": "0.1234567890123456789"}


@pytest.mark.parametrize("text", [
    "x: 1 x: 2", "x: { y: 1 y: 2 }", "x: [1 2]", "x: [1,]", "x: { y: 1, }",
    "x: { y: 1", "x: 1 }", "x: @local::a", "x.y: 1", '#include "a"', "x: abc",
    "x: truejunk", "x: 1e", "x: 0x10", "x: nan", "x: 1 garbage", "x: 1;",
    'x: "bad\\q"', 'x: "a"y: 1', "x: [1,,2]", "x: +", "x: 1e10000000",
])
def test_parser_rejects_unexpected_syntax(text):
    with pytest.raises(P.ProvenanceError):
        P.parse_fhicl(text)


@pytest.mark.parametrize("suffix", [" Other: {}", " Primary: {}", "\n# unsupported", " !"])
def test_primary_rejects_unconsumed_envelope(suffix):
    with pytest.raises(P.ProvenanceError):
        P.extract_primary(dump(config("crygen")) + suffix)


@pytest.mark.parametrize("path,value", [
    ("source.maxEvents", 999999), ("source.firstRun", 1), ("source.firstSubRun", 1001),
    ("services.SeedService.policy", "random"), ("services.SeedService.maxUniqueEngines", 99),
    ("services.SeedService.baseSeed", 20263001), ("physics.GenPath", P.GENERATORS[::-1]),
    ("physics.G4InitPath", ["g4run", "noG4"]), ("outputs.GenOutput.SelectEvents", []),
    ("outputs.GenOutput.outputCommands", ["keep *_*_*_*"]),
    ("trigger_paths.trigger_paths", ["GenPath"]), ("physics.filters.noG4.offset", 0),
    ("physics.EndPath", ["genCountLogger"]), ("physics.producers.generate.nested.x", [1, 1, "escaped\\\"}"]),
    ("services.RandomNumberGenerator.module_type", "Other"),
])
def test_bad_config_fails_closed(evidence, tmp_path, path, value):
    primary = config("crygen")
    table = primary
    keys = path.split(".")
    for key in keys[:-1]:
        table = table[key]
    table[keys[-1]] = value
    evidence[0].write_text(dump(primary))
    output = tmp_path / "bad.json"
    assert P.main(args(evidence, output)) == 1
    assert not output.exists()


def test_reference_selection_and_prefix(evidence):
    primary = config("g4ref")
    primary["outputs"]["GenOutput"]["SelectEvents"] = []
    evidence[2].write_text(dump(primary))
    with pytest.raises(P.ProvenanceError, match="unrestricted"):
        P.compare(*evidence)
    primary = config("g4ref")
    primary["physics"]["PrimaryPath"] = ["filter"] + P.GENERATORS
    evidence[2].write_text(dump(primary))
    with pytest.raises(P.ProvenanceError, match="prefix"):
        P.compare(*evidence)


@pytest.mark.parametrize("old,new", [
    ("Exit status: 0", "Exit status: 1"), ("Exit status: 0", "done"),
    ("Exit status: 0", "Exit status: 0\nExit status: 0"),
    ("Events total = 1000000", "Events total = 999999"),
    ("Closed output file gen.", "Closed output file othergen."),
    ("00002001.art", "00002001.art.extra"),
    ("Module Summary", "Path Summary"),
    ("TrigReport 0 0 0 0 0 g4run", "TrigReport 1 1 1 0 0 g4run"),
    ("TrigReport 1000000 1000000 0 1000000 0 noG4", "TrigReport 1000000 1000000 1 999999 0 noG4"),
    ("0 generate", "1 generate"),
    ("CRYSetup: returnMuons 1", "CRYSetup: returnMuons 0"),
    ("BField lines: 123  hash: 1234567890", "BField lines: 123  hash: 1234567891"),
    ("Geometry file: Geometry.txt", "Geometry file: other.txt"),
    ("Targetbox: 1, 2, 3", ""), ("Ref.point: 0, 0, 0", "Ref.point: 0, 1, 0"),
    ("World: -1, 1", "World: -2, 1"),
    ("mu2e build: MDC2025ay Offline abc123", "mu2e build: MDC2025ay Offline xyz"),
    ("TimeReport", "ERROR: broken\nTimeReport"),
])
def test_bad_log_fails_closed(evidence, old, new):
    text = evidence[1].read_text()
    assert old in text
    evidence[1].write_text(text.replace(old, new))
    with pytest.raises(P.ProvenanceError):
        P.compare(*evidence)


@pytest.mark.parametrize("prefix", ["CRYSetup", "mu2e build", "art version", "World:", "Ref.point:", "Targetbox:"])
def test_common_missing_runtime_not_equality(evidence, prefix):
    # Missing evidence on BOTH sides must never be counted as equal evidence.
    for path in (evidence[1], evidence[3]):
        text = path.read_text()
        if prefix in ("mu2e build", "art version"):
            text = "\n".join(line for line in text.splitlines() if not (line.startswith("mu2e build") or line.startswith("art version")))
        else:
            text = "\n".join(line for line in text.splitlines() if not line.startswith(prefix))
        path.write_text(text)
    with pytest.raises(P.ProvenanceError, match="missing runtime"):
        P.compare(*evidence)


def test_changed_input_during_comparison(evidence, monkeypatch):
    original = P.check_log
    def changing(text, cfg, role):
        result = original(text, cfg, role)
        if role == "g4ref":
            evidence[0].write_text(evidence[0].read_text() + "\n")
        return result
    monkeypatch.setattr(P, "check_log", changing)
    with pytest.raises(P.ProvenanceError, match="input changed"):
        P.compare(*evidence)


def test_duplicate_summary_row_and_nonmodule_spoof(evidence):
    text = evidence[1].read_text()
    row = "TrigReport 0 0 0 0 0 g4run"
    evidence[1].write_text(text.replace(row, row + "\n" + row))
    with pytest.raises(P.ProvenanceError, match="duplicate module"):
        P.compare(*evidence)
    evidence[1].write_text(row + "\n" + text.replace(row, ""))
    with pytest.raises(P.ProvenanceError, match="g4run"):
        P.compare(*evidence)


def test_actual_geometry_line_spellings(evidence):
    for path in (evidence[1], evidence[3]):
        path.write_text(path.read_text().replace("Ref.point:", "Ref. point:").replace("Targetbox:", "Target box:"))
    assert P.compare(*evidence)["status"] == "observed_pair_matches"


def test_error_cannot_hide_in_trigreport_line(evidence):
    evidence[1].write_text(evidence[1].read_text() + "TrigReport ERROR: unexpected failure\n")
    with pytest.raises(P.ProvenanceError, match="error indication"):
        P.compare(*evidence)


def test_output_symlink_is_not_followed(evidence, tmp_path):
    target = tmp_path / "retained"
    target.write_text("unchanged")
    link = tmp_path / "report.json"
    link.symlink_to(target)
    assert P.main(args(evidence, link)) == 1
    assert target.read_text() == "unchanged"


def test_positive_offset_subruns(evidence):
    for ci, li, role in ((0, 1, "crygen"), (2, 3, "g4ref")):
        primary = config(role)
        old = primary["source"]["firstSubRun"]
        primary["source"]["firstSubRun"] += 4
        primary["services"]["SeedService"]["baseSeed"] += 4
        output = primary["outputs"]["GenOutput"]
        output["fileName"] = output["fileName"].replace(f"_{old:08d}.", f"_{old+4:08d}.")
        evidence[ci].write_text(dump(primary))
        evidence[li].write_text(log(primary, role))
    assert P.compare(*evidence)["configurations"]["crygen"]["subrun"] == 2005


@pytest.mark.parametrize("block", P.BLOCKS)
def test_each_compared_block_is_required(evidence, block):
    primary = config("crygen")
    keys = block.split(".")
    table = primary
    for key in keys[:-1]:
        table = table[key]
    del table[keys[-1]]
    evidence[0].write_text(dump(primary))
    with pytest.raises(P.ProvenanceError, match="missing block/key"):
        P.compare(*evidence)


def test_input_source_code_binding(evidence):
    import hashlib
    from pathlib import Path
    report = P.compare(*evidence)
    binding = report["inputs"]["comparator_source"]
    assert binding["sha256_before"] == hashlib.sha256(Path(P.__file__).read_bytes()).hexdigest()


def test_same_changed_commands_are_not_authorized(evidence):
    for index, role in ((0, "crygen"), (2, "g4ref")):
        primary = config(role)
        primary["outputs"]["GenOutput"]["outputCommands"] = ["keep *_*_*_*"]
        evidence[index].write_text(dump(primary))
    with pytest.raises(P.ProvenanceError, match="outputCommands"):
        P.compare(*evidence)


def test_partial_reference_log_fails(evidence):
    evidence[3].write_text(evidence[3].read_text().replace("Events total = 50000", "Events total = 100"))
    with pytest.raises(P.ProvenanceError, match="complete TrigReport total"):
        P.compare(*evidence)


def test_processing_banner_identity_and_numbered_paths(evidence):
    primary = config("crygen")
    primary["trigger_paths"]["trigger_paths"] = ["0:GenPath", "1:G4InitPath"]
    evidence[0].write_text("=============================================\n" + dump(primary))
    assert P.compare(*evidence)["status"] == "observed_pair_matches"
    text = evidence[0].read_text().replace("001400_00002001.art", "001400_00002003.art", 1)
    evidence[0].write_text(text)
    with pytest.raises(P.ProvenanceError, match="processing path"):
        P.compare(*evidence)


def test_real_log_headers_not_errors(evidence):
    text = evidence[1].read_text().replace("TrigReport ---------- Module Summary",
        "TrigReport Path ID Run Passed Failed Error Name\n"
        "TrigReport Path ID Visited Passed Failed Error Name\n"
        "TrigReport Run Success Error\nTrigReport Run Success Error Name\n"
        "TrigReport ---------- Module Summary")
    evidence[1].write_text(text)
    assert P.compare(*evidence)["status"] == "observed_pair_matches"


def test_bad_end_paths_structure_fails(evidence):
    primary = config("crygen")
    primary["physics"]["end_paths"] = [["EndPath"]]
    evidence[0].write_text(dump(primary))
    with pytest.raises(P.ProvenanceError, match="end_paths"):
        P.compare(*evidence)
