"""Approximation switches: one scheme for every workflow, over the switches the chains already have.

Every workflow package declares its approximations in ``<package>/approximations.py`` as a :class:`Workflow` (or a
``WORKFLOWS`` dict of them): one :class:`Approximation` per model that stands in for stock, with the levels it can
take, the level that is today's behaviour (``default``), and what each other level sets on the chain. A run selects
levels with a preset and per-approximation overrides:

    approximations: {preset: fidelity, pulse_fit: short_window}   # a config section (JSON, or YAML with PyYAML)
    --approx-preset fidelity --approx pulse_fit=short_window      # the same on a command line

Presets:
  default    every approximation at its default level: what the chain does with no approximation options
  fast       every approximation at its fastest level
  fidelity   every approximation at the level closest to stock that the code has; an approximation with no such
             path yet stays at its default and is listed as ``no_fidelity_path`` in the record

The default level never sets anything, so a run with no options, or with ``preset: default``, passes exactly the
arguments it passed before this scheme existed. A level an approximation does not have (for example "off" where
there is no off path yet) is refused with the stock alternative in the message; nothing is invented.

:meth:`Selection.record` is the provenance entry for an output: the workflow, the preset and overrides, every
level (``levels``: what ran; ``selected``: what was asked, where the run's other options decided otherwise, with
the reason under ``effective``), the ones that differ from the default, the approximations without a fidelity level
or without any stock path, and the ones with no switch (``fixed``). :func:`table_markdown` renders the
documentation table from the declarations. Standard library only.
"""
import argparse
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

SCHEMA = 1
PRESETS = ("default", "fast", "fidelity")
# stock-equivalent alternative of an approximation, as it stands in code
PORT = "exact port"           # a level of the chain runs the stock algorithm (ported)
STOCK_ART = "stock art"       # stock products read from art / EventNtuple replace the approximated stage
NONE = "none"                 # no stock-equivalent path in the package yet


@dataclass(frozen=True)
class Approximation:
    """One approximation of a workflow.

    ``sets`` maps each non-default level to the arguments it sets on the chain, as ``{name: value}``; the names are
    the keyword arguments of the chain's entry points, which the package keeps equal to its command-line ``dest``
    names; ``cli`` overrides that for a level whose command-line value differs from the keyword value (e.g. the
    string "none" for None). The default level sets nothing. ``fast`` and ``fidelity`` name the levels the presets select (``fidelity``
    None: no level closer to stock exists yet). ``alternative`` is PORT, STOCK_ART or NONE, ``stock`` says in words
    what stock does or which path gives it, and ``switch`` names the existing switch the levels drive.
    """
    name: str
    description: str
    levels: Tuple[str, ...]
    default: str
    sets: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    cli: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    fast: Optional[str] = None
    fidelity: Optional[str] = None
    alternative: str = NONE
    stock: str = ""
    switch: str = ""
    performance_only: bool = False   # the levels give the same output (a speed switch listed for completeness)

    def __post_init__(self):
        if self.default not in self.levels:
            raise ValueError(f"{self.name}: default {self.default!r} is not one of {self.levels}")
        for level in (self.fast, self.fidelity):
            if level is not None and level not in self.levels:
                raise ValueError(f"{self.name}: preset level {level!r} is not one of {self.levels}")
        if self.default in self.sets and self.sets[self.default]:
            raise ValueError(f"{self.name}: the default level must set nothing")
        for level in self.levels:
            if level != self.default and level not in self.sets:
                raise ValueError(f"{self.name}: level {level!r} sets nothing; give its arguments in sets")
        if set(self.cli) - set(self.sets):
            raise ValueError(f"{self.name}: cli levels must be levels of sets")
        if self.alternative not in (PORT, STOCK_ART, NONE):
            raise ValueError(f"{self.name}: alternative must be PORT, STOCK_ART or NONE")

    def level_for(self, preset):
        if preset == "default":
            return self.default
        if preset == "fast":
            return self.fast or self.default
        if preset == "fidelity":
            return self.fidelity or self.default
        raise ValueError(f"unknown preset {preset!r}; choose from {PRESETS}")


@dataclass(frozen=True)
class Workflow:
    """The approximations of one workflow (W1 ... W8)."""
    name: str
    title: str
    approximations: Tuple[Approximation, ...]
    # approximations the workflow has with no switch at all (name -> what stock does instead); listed in the table
    # and the record so a reader sees them, never selectable
    fixed: Mapping[str, str] = field(default_factory=dict)
    aliases: Tuple[str, ...] = ()    # other workflow names the declaration serves (W3, W4 on the W2 chain)

    def __post_init__(self):
        names = [a.name for a in self.approximations]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: duplicate approximation names")
        if set(names) & set(self.fixed):
            raise ValueError(f"{self.name}: a fixed approximation has the name of a switchable one")

    def get(self, name):
        for a in self.approximations:
            if a.name == name:
                return a
        if name in self.fixed:
            raise ValueError(f"{self.name}: {name!r} has no switch yet (stock: {self.fixed[name]})")
        raise ValueError(f"{self.name}: unknown approximation {name!r}; choose from "
                         f"{[a.name for a in self.approximations]}")

    def select(self, preset="default", overrides=None):
        """The levels of ``preset`` with ``overrides`` ({name: level}) on top, as a :class:`Selection`."""
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {PRESETS}")
        levels = {a.name: a.level_for(preset) for a in self.approximations}
        for name, level in (overrides or {}).items():
            a = self.get(name)
            level = _level_name(level)
            if level not in a.levels:
                hint = f"; stock alternative: {a.stock}" if a.stock else ""
                raise ValueError(f"{self.name}: {name} has no level {level!r} (levels: {', '.join(a.levels)})"
                                 + (" — no such path in code yet" if a.alternative == NONE else "") + hint)
            levels[name] = level
        return Selection(self, preset, levels, {k: _level_name(v) for k, v in (overrides or {}).items()})


def _level_name(level):
    """YAML reads on/off as booleans: map them back."""
    if level is True:
        return "on"
    if level is False:
        return "off"
    return str(level)


@dataclass
class Selection:
    workflow: Workflow
    preset: str
    levels: Dict[str, str]
    overrides: Dict[str, object]
    effective: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def arguments(self, surface="kwargs"):
        """{name: value} the selected non-default levels set (empty at the default levels); ``surface`` "cli" gives
        the command-line values."""
        out = {}
        for a in self.workflow.approximations:
            level = self.levels[a.name]
            sets = a.cli.get(level, a.sets.get(level, {})) if surface == "cli" else a.sets.get(level, {})
            for key, value in sets.items():
                if key in out and out[key] != value:
                    raise ValueError(f"{self.workflow.name}: approximations disagree on {key!r}")
                out[key] = value
        return out

    def set_effective(self, name, level, reason):
        """Record that the run's other options put ``name`` at ``level`` whatever was selected (for example no
        albedo on the NumPy tracker); the record keeps both."""
        a = self.workflow.get(name)
        if level not in a.levels:
            raise ValueError(f"{name}: no level {level!r}")
        if level != self.levels[name]:
            self.effective[name] = {"level": level, "reason": reason}
        return self

    def effective_levels(self):
        return {name: self.effective.get(name, {}).get("level", level) for name, level in self.levels.items()}

    def non_default(self):
        levels = self.effective_levels()
        return sorted(a.name for a in self.workflow.approximations if levels[a.name] != a.default)

    def no_fidelity_path(self):
        return sorted(a.name for a in self.workflow.approximations if a.fidelity is None)

    def no_stock_path(self):
        return sorted(a.name for a in self.workflow.approximations if a.alternative == NONE)

    def record(self):
        """Provenance entry for an output (JSON-serialisable)."""
        return {"schema": SCHEMA, "workflow": self.workflow.name, "preset": self.preset,
                "overrides": dict(self.overrides), "levels": self.effective_levels(),
                "selected": dict(self.levels), "effective": {k: dict(v) for k, v in self.effective.items()},
                "non_default": self.non_default(), "no_fidelity_path": self.no_fidelity_path(),
                "no_stock_path": self.no_stock_path(), "fixed": sorted(self.workflow.fixed)}

    def apply_kwargs(self, kwargs, accepted=None):
        """``kwargs`` with the selected arguments added. A key already in ``kwargs`` with a different value is
        refused, and so is a non-default level whose argument the entry point does not take (``accepted``: the names
        it takes, None to skip the check)."""
        out = dict(kwargs)
        for key, value in self.arguments().items():
            if accepted is not None and key not in accepted:
                raise ValueError(f"{self.workflow.name}: this entry point takes no {key!r}, so the selected level "
                                 f"cannot reach it; use the workflow's chain entry point")
            if key in out and out[key] != value:
                raise ValueError(f"{self.workflow.name}: {key}={out[key]!r} given, but the approximations set "
                                 f"{key}={value!r}")
            out[key] = value
        return out

    def apply_namespace(self, args, parser):
        """Set the selected arguments on an argparse namespace (``parser``: the one that parsed it, for the
        defaults). An option the user gave explicitly with another value is refused (detected as differing from the
        parser default)."""
        for key, value in self.arguments("cli").items():
            if not hasattr(args, key):
                raise ValueError(f"{self.workflow.name}: this command has no option for {key!r}")
            given, default = getattr(args, key), parser.get_default(key)
            if given != default and given != value:
                raise ValueError(f"{self.workflow.name}: --{key.replace('_', '-')} {given!r} given, but the "
                                 f"approximations set {value!r}")
            setattr(args, key, value)
        return args


# ------------------------------------------------------------------ configuration input


def read_section(source, workflow=None, aliases=()):
    """The ``approximations`` section from a file path or a JSON string.

    Accepted shapes: ``{approximations: {...}}`` or the section itself; inside, either one workflow's
    ``{preset: ..., name: level, ...}`` or ``{W5: {...}, W2: {...}}`` keyed by workflow (then ``workflow``, or the
    first of ``aliases`` present, picks one). Returns ``(preset, overrides)``.
    """
    if isinstance(source, Mapping):
        data = source
    else:
        text = str(source)
        path = Path(text)
        if text.lstrip().startswith("{"):
            data = json.loads(text)
        elif path.suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as e:
                raise ValueError(f"{path}: reading YAML needs PyYAML; use a .json file") from e
            data = yaml.safe_load(path.read_text())
        else:
            data = json.loads(path.read_text())
    if not isinstance(data, Mapping):
        raise ValueError("approximations: expected a mapping")
    section = data.get("approximations", data)
    if not isinstance(section, Mapping):
        raise ValueError("approximations: the section must be a mapping")
    keys = [k for k in (workflow, *aliases) if k is not None and isinstance(section.get(k), Mapping)]
    if keys:
        section = section[keys[0]]
    elif any(isinstance(v, Mapping) for v in section.values()):
        raise ValueError(f"approximations: the section is keyed by workflow; no entry for {workflow!r}")
    section = dict(section)
    preset = section.pop("preset", "default")
    return preset, section


def parse_pairs(pairs):
    out = {}
    for pair in pairs or ():
        name, sep, level = pair.partition("=")
        if not sep or not name or not level:
            raise ValueError(f"--approx expects NAME=LEVEL, got {pair!r}")
        out[name.strip()] = level.strip()
    return out


def add_arguments(parser):
    """The three approximation options of a chain command or the bench."""
    g = parser.add_argument_group("approximations")
    g.add_argument("--approx-preset", choices=PRESETS, default=None,
                   help="default (as without these options), fast, or fidelity (closest to stock the code has)")
    g.add_argument("--approx", action="append", metavar="NAME=LEVEL", default=None,
                   help="set one approximation's level (repeatable; applied after the preset)")
    g.add_argument("--approximations", metavar="FILE|JSON", default=None,
                   help="config with an approximations section ({preset: ..., NAME: LEVEL}, optionally keyed by "
                        "workflow)")
    return parser


def requested(args):
    """True when any approximation option was given."""
    return bool(getattr(args, "approx_preset", None) or getattr(args, "approx", None)
                or getattr(args, "approximations", None))


def from_args(args, workflow):
    """The :class:`Selection` of a parsed command line (config first, then --approx-preset, then --approx)."""
    preset, overrides = "default", {}
    if getattr(args, "approximations", None):
        preset, overrides = read_section(args.approximations, workflow.name, workflow.aliases)
    if getattr(args, "approx_preset", None):
        preset = args.approx_preset
    overrides.update(parse_pairs(getattr(args, "approx", None)))
    return workflow.select(preset, overrides)


# ------------------------------------------------------------------ declarations


def declared(package):
    """{name: Workflow} declared by ``<package>.approximations``; {} if the package declares none."""
    name = f"{package}.approximations"
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name in (name, package):
            return {}
        raise
    workflows = getattr(module, "WORKFLOWS", None)
    if workflows is None:
        single = getattr(module, "WORKFLOW", None)
        workflows = {} if single is None else {single.name: single}   # (this module itself declares none)
    return dict(workflows)


def load(package, workflow=None, required=True):
    """The :class:`Workflow` of ``<package>.approximations``: ``workflow`` (a name or an alias) picks one of several.
    None when the package declares none and ``required`` is False."""
    workflows = declared(package)
    if not workflows:
        if required:
            raise ValueError(f"{package} declares no approximations ({package}.approximations)")
        return None
    if workflow is None:
        if len(workflows) == 1:
            return next(iter(workflows.values()))
        if not required:
            return None
        raise ValueError(f"{package} declares workflows {sorted(workflows)}; name one (--workflow)")
    for w in workflows.values():
        if workflow == w.name or workflow in w.aliases:
            return w
    if required:
        raise ValueError(f"{package} declares no workflow {workflow!r}; choose from {sorted(workflows)}")
    return None


def table_markdown(workflows):
    """The documentation table of ``workflows`` (an iterable of :class:`Workflow`)."""
    rows = ["| workflow | approximation | levels (default first) | fast | fidelity | stock alternative | switch |",
            "|---|---|---|---|---|---|---|"]
    for w in workflows:
        for a in w.approximations:
            levels = [a.default] + [x for x in a.levels if x != a.default]
            note = " (speed only)" if a.performance_only else ""
            rows.append(f"| {w.name} | `{a.name}`: {a.description}{note} | {', '.join(levels)} | {a.fast or a.default} "
                        f"| {a.fidelity or 'no path yet'} | {a.alternative}: {a.stock} | {a.switch} |")
        for name, stock in w.fixed.items():
            rows.append(f"| {w.name} | `{name}` (no switch) | always on | - | no path yet | none: {stock} | - |")
    return "\n".join(rows) + "\n"


def main(argv=None):
    """``python3 -m mu2efastsim.approximations PACKAGE ...``: the documentation table of the packages' declarations,
    or with ``--record WORKFLOW --approx-preset P [--approx N=L]`` the record a run would write."""
    import sys
    p = argparse.ArgumentParser(prog="python3 -m mu2efastsim.approximations", description=main.__doc__)
    p.add_argument("packages", nargs="+", help="packages with an approximations module, e.g. crvfastsim")
    p.add_argument("--record", metavar="WORKFLOW", help="print the record of this workflow instead of the table")
    add_arguments(p)
    a = p.parse_args(argv)
    if a.record:
        print(json.dumps(from_args(a, load(a.packages[0], a.record)).record(), indent=1))
        return 0
    workflows = []
    for package in a.packages:
        found = declared(package)
        if not found:
            print(f"{package}: no approximations declared", file=sys.stderr)
            return 2
        workflows.extend(found.values())
    sys.stdout.write(table_markdown(sorted(workflows, key=lambda w: w.name)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
