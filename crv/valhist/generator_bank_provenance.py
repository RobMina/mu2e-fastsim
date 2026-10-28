"""Compare two saved generator configurations/logs, never authorize a bank.

Only expanded FHiCL is accepted (no includes, references, dotted keys, comments
inside the document, or unquoted string values).  Numeric comparison uses Decimal,
not binary floats.  ``parse_fhicl`` and ``extract_primary`` are public helpers for
later *separate* initialization-probe tooling; this module never runs art.

CLI::

    python -m valhist.generator_bank_provenance \\
      --crygen-config crygen.P.txt --crygen-log crygen.log.gz \\
      --g4ref-config g4ref.P.txt --g4ref-log g4ref.log.gz --output diagnostic.json

Output is exclusive-create and read-only (0444), not a cryptographic signature or
filesystem-level immutability guarantee.  Inputs and this implementation are
SHA256-bound before and after evaluation. Failure emits no successful report.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys


class ProvenanceError(ValueError):
    """Missing, ambiguous, unsupported, or contradictory evidence."""


def require(condition, message):
    if not condition:
        raise ProvenanceError(message)


_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


class _Parser:
    def __init__(self, text):
        self.text, self.pos = text, 0

    def ws(self):
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def take(self, token):
        self.ws()
        if self.text.startswith(token, self.pos):
            self.pos += len(token)
            return True
        return False

    def need(self, token):
        require(self.take(token), f"expected {token!r} at offset {self.pos}")

    def table(self, closing=None):
        result = {}
        while True:
            self.ws()
            if closing and self.take(closing):
                return result
            if self.pos == len(self.text):
                require(closing is None, "unterminated table")
                return result
            key = _IDENT.match(self.text, self.pos)
            require(key is not None, f"expected identifier at offset {self.pos}")
            self.pos = key.end()
            key = key.group()
            require(key not in result, f"duplicate key {key!r}")
            self.need(":")
            result[key] = self.value()
            # Expanded tables allow whitespace or commas, never doubled or trailing commas.
            before_ws = self.pos
            self.ws()
            separated = self.pos > before_ws
            if self.take(","):
                self.ws()
                require(self.pos < len(self.text) and self.text[self.pos] != closing,
                        "trailing table comma")
            elif self.pos < len(self.text) and (not closing or self.text[self.pos] != closing):
                require(separated, f"missing table separator at offset {self.pos}")

    def value(self):
        self.ws()
        require(self.pos < len(self.text), "missing value")
        if self.take("{"):
            return self.table("}")
        if self.take("["):
            result = []
            if self.take("]"):
                return result
            while True:
                result.append(self.value())
                if self.take("]"):
                    return result
                self.need(",")
        if self.text[self.pos] == '"':
            try:
                value, end = json.JSONDecoder().raw_decode(self.text, self.pos)
            except ValueError as exc:
                raise ProvenanceError(f"invalid quoted string at offset {self.pos}") from exc
            require(isinstance(value, str), "expected string")
            self.pos = end
            return value
        for token, value in (("true", True), ("false", False)):
            if self.text.startswith(token, self.pos):
                end = self.pos + len(token)
                require(end == len(self.text) or self.text[end] in " \t\r\n,]}",
                        "invalid boolean token")
                self.pos = end
                return value
        number = _NUMBER.match(self.text, self.pos)
        require(number is not None, f"unsupported value at offset {self.pos}")
        self.pos = number.end()
        require(self.pos == len(self.text) or self.text[self.pos] in " \t\r\n,]}",
                "invalid numeric token")
        try:
            value = Decimal(number.group())
        except InvalidOperation as exc:
            raise ProvenanceError("invalid number") from exc
        # Bound resource use when an exponent is expanded to an integer.
        require(value.is_finite() and abs(value.adjusted()) < 10000, "number out of range")
        return int(value) if value == value.to_integral_value() else value


def parse_fhicl(text):
    """Parse one complete expanded document to dict/list/str/bool/int/Decimal.

    All input must be consumed. Unsupported FHiCL features are errors, not ignored.
    Decimal nonintegers are retained exactly; e.g. 2.0263e7 becomes integer 20263000.
    """
    parser = _Parser(text)
    result = parser.table()
    parser.ws()
    require(parser.pos == len(text), "unconsumed FHiCL syntax")
    return result


def extract_primary(text):
    """Validate a saved config_dumper -P envelope and return its Primary table.

    The processing path is evidence only: it is not opened, fetched, or hashed.
    """
    lines = text.splitlines()
    if lines and re.fullmatch(r"=+", lines[0]):
        lines = lines[1:]
    require(bool(lines) and re.fullmatch(r"Processing file: /pnfs/\S+", lines[0]) is not None,
            "missing config_dumper Processing file: /pnfs/... banner")
    index = 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    require(index < len(lines) and lines[index].startswith("# Read SQLiteDB"),
            "missing Read SQLiteDB header")
    index += 1
    document = parse_fhicl("\n".join(lines[index:]))
    require(set(document) == {"Primary"} and isinstance(document["Primary"], dict),
            "expected exactly one Primary table")
    return document["Primary"]


def exact_equal(a, b):
    """FHiCL structural equality without Python's True == 1 coercion."""
    if isinstance(a, (int, Decimal)) and not isinstance(a, bool):
        return isinstance(b, (int, Decimal)) and not isinstance(b, bool) and a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(exact_equal(x, y) for x, y in zip(a, b))
    return a == b


def at(table, dotted):
    for key in dotted.split("."):
        require(isinstance(table, dict) and key in table, "missing block/key: " + dotted)
        table = table[key]
    return table


def same(value, expected, label):
    require(exact_equal(value, expected), "mismatch: " + label)


def integer(value, label):
    require(type(value) is int, "expected integer: " + label)
    return value


GENERATORS = ["generate", "genCounter", "cosmicTimeOffset"]
BLOCKS = ["physics.producers." + name for name in GENERATORS] + [
    "services." + name for name in ("GeometryService", "GlobalConstantsService", "RandomNumberGenerator")]
COMMANDS = ["drop *_*_*_*", "keep mu2e::GenParticles_*_*_*",
            "keep mu2e::CosmicLivetime_*_*_*", "keep mu2e::GenEventCount_*_*_*"]
LIMITATIONS = [
    "Diagnostic comparison of the supplied pair only; not a whole-bank certificate or downstream authorization.",
    "No full-bank file/event census, uniqueness, persisted generator payload or livetime validation.",
    "No census of missing jobs, selection failures, failed outputs or survivorship bias.",
    "Configured SeedService policy/baseSeed is not actual RNG engine seed assignment or mapping certification.",
    "Runtime external-config hashes are observed implementation-specific log fingerprints, not independently "
    "hashed external files, authenticated provenance, or source-code equality.",
    "Build banners and selected configuration/runtime fields do not prove complete software/source provenance equality.",
    "Saved config/log association is filename-based; art/ROOT files are not opened or independently bound by content.",
    "Read-only exclusive output prevents accidental overwrite, not privileged mutation or forgery.",
]


def check_config(primary, role):
    n, first, base = (1000000, 2001, 20263000) if role == "crygen" else (50000, 1, 20261016)
    source = at(primary, "source")
    same(at(source, "module_type"), "EmptyEvent", role + " source type")
    same(at(source, "maxEvents"), n, role + " requested events")
    same(at(source, "firstRun"), 1400, role + " run")
    subrun = integer(at(source, "firstSubRun"), role + " subrun")
    require(first <= subrun <= (2100 if role == "crygen" else 220), role + " subrun outside bank range")
    seed = at(primary, "services.SeedService")
    same(at(seed, "policy"), "autoIncrement", role + " seed policy")
    same(at(seed, "maxUniqueEngines"), 100, role + " maxUniqueEngines")
    same(at(seed, "baseSeed"), base + subrun - first, role + " baseSeed")
    output = at(primary, "outputs.GenOutput")
    same(at(output, "module_type"), "RootOutput", role + " output type")
    same(at(output, "outputCommands"), COMMANDS, role + " outputCommands")
    filename = at(output, "fileName")
    require(isinstance(filename, str) and re.fullmatch(
        rf"gen\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.001400_{subrun:08d}\.art", filename),
        role + " output filename/run/subrun")
    physics = at(primary, "physics")
    if role == "crygen":
        same(at(output, "SelectEvents"), ["GenPath"], "crygen SelectEvents")
        same(at(physics, "GenPath"), GENERATORS, "crygen GenPath")
        same(at(physics, "G4InitPath"), ["noG4", "g4run"], "crygen G4InitPath")
        require("PrimaryPath" not in physics, "crygen unexpectedly retains PrimaryPath")
        blocking = at(physics, "filters.noG4")
        for key, value in {"module_type": "BlockingPrescaler", "blockSize": 1,
                           "stepSize": 1000000000, "offset": 1000000000}.items():
            same(at(blocking, key), value, "crygen noG4." + key)
        paths = ["GenPath", "G4InitPath"]
    else:
        require("SelectEvents" not in output, "g4ref GenOutput must be unrestricted")
        path = at(physics, "PrimaryPath")
        require(isinstance(path, list) and len(path) >= 3, "g4ref PrimaryPath")
        same(path[:3], GENERATORS, "g4ref generator prefix")
        paths = ["PrimaryPath"]
    # art persists the trigger path names in this separate ParameterSet.
    persisted = at(primary, "trigger_paths")
    numbered = {"trigger_paths": [f"{i}:{name}" for i, name in enumerate(paths)]}
    require(exact_equal(persisted, {"trigger_paths": paths}) or exact_equal(persisted, numbered),
            "mismatch: " + role + " trigger_paths")
    if "trigger_paths" in physics:
        same(physics["trigger_paths"], paths, role + " physics.trigger_paths")
    end_paths = at(physics, "end_paths")
    require(isinstance(end_paths, list) and end_paths
            and all(isinstance(name, str) for name in end_paths)
            and len(end_paths) == len(set(end_paths)),
            role + " end_paths")
    output_uses = 0
    for path_name in end_paths:
        require(isinstance(path_name, str), "invalid end-path name")
        sequence = at(physics, path_name)
        require(isinstance(sequence, list), "invalid end-path sequence")
        output_uses += sequence.count("GenOutput")
    same(output_uses, 1, role + " scheduled GenOutput")
    return dict(source=source, SeedService=seed, GenOutput=output, requested_events=n,
                filename=filename, subrun=subrun, trigger_paths=paths)


_EVENT = re.compile(r"^\s*TrigReport\s+Events total\s*=\s*(\d+)\s+passed\s*=\s*(\d+)\s+failed\s*=\s*(\d+)\s*$")
_MODULE = re.compile(r"^\s*TrigReport\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([A-Za-z_][A-Za-z_0-9]*)\s*$")
# GNU /usr/bin/time -v status in the original mu2e.log; a worker 'done' is insufficient.
_EXIT = re.compile(r"^\s*Exit status:\s*(\d+)\s*$")
_ERROR = re.compile(r"(?:\b(?:error|fatal|exception|failed|failure|segmentation fault|aborted)\b|%MSG-[esf])", re.I)
_BUILD = re.compile(r"^(?:.*\b(?:build(?:ing|t)?|BuildInfo)\b.*|(?:art|mu2e|Offline|Geant4|CLHEP|CRY)\s+(?:version|release)\b.*)$", re.I)


def check_log(text, config, role):
    lines = text.splitlines()
    exits = [(i, _EXIT.fullmatch(line)) for i, line in enumerate(lines) if _EXIT.fullmatch(line)]
    require(len(exits) == 1 and exits[0][1][1] == "0", role + " missing/ambiguous/nonzero art exit")
    totals = [(i, _EVENT.fullmatch(line)) for i, line in enumerate(lines) if _EVENT.fullmatch(line)]
    require(len(totals) == 1, role + " missing/ambiguous TrigReport event total")
    total_index, total_match = totals[0]
    total, passed, failed = map(int, total_match.groups())
    same(total, config["requested_events"], role + " complete TrigReport total")
    same(passed + failed, total, role + " TrigReport total balance")
    require(total_index < exits[0][0], role + " completion order")
    for line in lines:
        if _EVENT.fullmatch(line) or re.fullmatch(
                r"\s*TrigReport\s+(?:(?:Path ID\s+)?(?:Visited\s+)?Run\s+(?:Passed\s+Failed|Success)\s+Error(?:\s+Name)?|Path ID\s+Visited\s+Passed\s+Failed\s+Error\s+Name)\s*",
                line, re.I):
            continue  # Report column headers and failed selections are not errors.
        require(_ERROR.search(line) is None, role + " error indication: " + line[:250])
    # Require exact output closure despite the timestamp prefix.
    closed = []
    for i, line in enumerate(lines):
        match = re.search(r"\bClosed output file\s+['\"]?([^\s'\"]+)['\"]?\s*$", line)
        if match and match[1] == config["filename"]:
            closed.append(i)
    require(len(closed) == 1 and closed[0] < exits[0][0], role + " exact output filename closure")
    in_modules = False
    sections = 0
    modules = {}
    for index, line in enumerate(lines):
        if re.fullmatch(r"\s*TrigReport\s*-*\s*Module Summary\s*-*\s*", line, re.I):
            require(index < exits[0][0], role + " module summary after exit")
            in_modules = True
            sections += 1
            continue
        if in_modules and (re.search(r"\b(?:Summary|TimeReport)\b", line) or _EXIT.fullmatch(line)):
            in_modules = False
        match = _MODULE.fullmatch(line) if in_modules else None
        if match:
            name = match[6]
            require(name not in modules, role + " duplicate module summary: " + name)
            modules[name] = list(map(int, match.groups()[:5]))
    require(sections == 1, role + " missing/ambiguous module summary")
    n = config["requested_events"]
    # Module columns: visited, run, passed, failed, error.
    for name in GENERATORS + ["GenOutput"]:
        same(at(modules, name), [n, n, n, 0, 0], role + " module " + name)
    if role == "crygen":
        same(at(modules, "noG4"), [n, n, 0, n, 0], "crygen noG4 module")
        same(at(modules, "g4run"), [0, 0, 0, 0, 0], "crygen g4run module")
    for name, counts in modules.items():
        require(counts[-1] == 0, role + " module errors: " + name)
    fingerprints = {}
    for tag in ("GlobalConstants", "Geometry", "BField"):
        files = [m[1] for line in lines if (m := re.fullmatch(rf"\s*{tag} file:\s*(\S+)\s*", line))]
        hashes = [m.groups() for line in lines if (m := re.fullmatch(
            rf"\s*{tag} lines:\s*(\d+)\s+hash:\s*(\S+)\s*", line))]
        require(len(files) == len(hashes) == 1 and int(hashes[0][0]) > 0,
                role + " missing/ambiguous runtime fingerprint: " + tag)
        require(re.fullmatch(r"(?:[0-9]+|0x[0-9a-fA-F]+|[0-9a-fA-F]{32,64})", hashes[0][1]) is not None,
                role + " invalid runtime hash: " + tag)
        fingerprints[tag] = dict(file=files[0], lines=int(hashes[0][0]), hash=hashes[0][1])
    observed = {"fingerprints": fingerprints}
    observed["build_banner_lines"] = [line.strip() for line in lines if _BUILD.fullmatch(line.strip())]
    observed["CRYSetup_lines"] = [line.strip() for line in lines if "CRYSetup" in line]
    for label, pattern in (("Ref.point", r"Ref(?:\.\s*|\s+)point\s*:"),
                           ("Targetbox", r"Target\s*box\s*:"), ("World", r"World\s*:")):
        observed[label + "_lines"] = [line.strip() for line in lines if re.search(pattern, line, re.I)]
    for name, value in observed.items():
        require(bool(value), role + " missing runtime field: " + name)
    return dict(events=dict(total=total, passed=passed, failed=failed), modules=modules,
                output_closed=config["filename"], art_exit=0, runtime=observed)


def _jsonable(value):
    # Decimal tags preserve exact noninteger values without silently rounding JSON floats.
    if isinstance(value, Decimal):
        return {"decimal": str(value)}
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _text(data):
    if data.startswith(b"\x1f\x8b"):
        data = gzip.decompress(data)
    return data.decode("utf-8", errors="strict")


def compare(crygen_config, crygen_log, g4ref_config, g4ref_log):
    """Read four local files and return a JSON-safe, nonauthorizing diagnostic.

    Raises ProvenanceError on absent/mismatched evidence. No subprocesses/network,
    art-file access, output mutation, or true downstream_authorized state exists.
    """
    paths = {"crygen_config": Path(crygen_config), "crygen_log": Path(crygen_log),
             "g4ref_config": Path(g4ref_config), "g4ref_log": Path(g4ref_log),
             "comparator_source": Path(__file__)}
    paths = {key: path.resolve(strict=True) for key, path in paths.items()}
    require(len(set(paths.values())) == len(paths), "evidence/source paths must be distinct")
    data = {key: path.read_bytes() for key, path in paths.items()}
    bindings = {key: dict(path=str(paths[key]), size=len(raw), sha256_before=_hash(raw))
                for key, raw in data.items()}
    primary, configs, logs = {}, {}, {}
    for role in ("crygen", "g4ref"):
        primary[role] = extract_primary(_text(data[role + "_config"]))
        configs[role] = check_config(primary[role], role)
        processing = re.findall(r"(?m)^Processing file: (/pnfs/\S+)$", _text(data[role + "_config"]))
        require(len(processing) == 1 and Path(processing[0]).name == configs[role]["filename"],
                role + " processing path differs from configured output identity")
        configs[role]["processing_path"] = processing[0]
        logs[role] = check_log(_text(data[role + "_log"]), configs[role], role)
    blocks = {}
    for block in BLOCKS:
        a, b = at(primary["crygen"], block), at(primary["g4ref"], block)
        require(isinstance(a, dict) and a and isinstance(b, dict) and b, "empty/non-table block: " + block)
        same(a, b, block)
        blocks[block] = a
    same(configs["crygen"]["GenOutput"]["outputCommands"],
         configs["g4ref"]["GenOutput"]["outputCommands"], "GenOutput outputCommands equality")
    same(logs["crygen"]["runtime"], logs["g4ref"]["runtime"], "runtime fingerprints/banner/CRY/geometry")
    for key, path in paths.items():
        bindings[key]["sha256_after"] = _hash(path.read_bytes())
        same(bindings[key]["sha256_before"], bindings[key]["sha256_after"], "input changed: " + key)
    return _jsonable(dict(schema_version=1, status="observed_pair_matches", diagnostic_only=True,
                         downstream_authorized=False, whole_bank_certified=False,
                         actual_engine_seed_mapping_certified=False, source_provenance_equality_claimed=False,
                         inputs=bindings, equal_blocks=blocks, configurations=configs, logs=logs,
                         limitations=LIMITATIONS))


def write_report(path, report):
    """Exclusive-create a read-only JSON artifact; never replace an existing path."""
    require(report.get("downstream_authorized") is False, "report must not authorize downstream use")
    payload = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    # O_EXCL refuses existing paths and symlinks; failed writes retain evidence.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o444)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("crygen", "g4ref"):
        parser.add_argument("--" + role + "-config", required=True, type=Path)
        parser.add_argument("--" + role + "-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        require(not args.output.exists() and not args.output.is_symlink(), "output already exists")
        report = compare(args.crygen_config, args.crygen_log, args.g4ref_config, args.g4ref_log)
        write_report(args.output, report)
    except (ValueError, OSError, EOFError, RecursionError, TypeError) as exc:
        print("generator provenance diagnostic REFUSED: " + str(exc), file=sys.stderr)
        return 1
    print("Observed pair matches; diagnostic only; downstream_authorized=false: " + str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
