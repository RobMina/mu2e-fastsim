"""``python -m fastsim_mcp_server``: start the server (stdio by default)."""
from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from pathlib import Path

from .server import run_server

ASSIGN = re.compile(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def load_site_env(path: str | Path, environ=os.environ) -> list[str]:
    """Set the plain ``NAME=value`` (or ``export NAME=value``) lines of a site file in ``environ``.

    Comments and blank lines are skipped; a value is read as one shell word (quotes removed) with ``$NAME`` and
    ``${NAME}`` of variables set earlier expanded. Anything else (commands, functions) is refused: the file is
    configuration, not a script. Variables already set in the environment win, so an MCP client's ``env`` block
    overrides the file. Returns the names it set."""
    names = []
    for n, line in enumerate(Path(path).expanduser().read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = ASSIGN.match(line)
        if m is None:
            raise ValueError(f"{path}:{n}: not a NAME=value line")
        name, raw = m.groups()
        if name == "PYTHONPATH":   # read only at interpreter start: set it in the client's env instead
            continue
        words = shlex.split(raw, comments=True)
        if len(words) > 1:
            raise ValueError(f"{path}:{n}: value of {name} is more than one word")
        value = words[0] if words else ""
        value = re.sub(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", lambda k: environ.get(k.group(1), ""), value)
        if not environ.get(name):
            environ[name] = value
            names.append(name)
    return names


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m fastsim_mcp_server", description="Run the Mu2e fastsim MCP server.")
    p.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    p.add_argument("--host", default="127.0.0.1", help="host for streamable-http")
    p.add_argument("--port", type=int, default=8000, help="port for streamable-http")
    p.add_argument("--site-env", help="a site file of NAME=value lines (site.env.example): the endpoint, hosts, "
                                      "account, work areas and image the kit uses; the environment wins over it")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.site_env:
        try:
            names = load_site_env(args.site_env)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"site settings from {args.site_env}: {len(names)} variables", file=sys.stderr, flush=True)
    code = os.environ.get("M2FS_CODE")
    if code:   # the kit (core) and the beam tasks it submits (beam) of that checkout
        for sub in ("beam", "core"):
            path = str(Path(code) / sub)
            if path not in sys.path:
                sys.path.insert(0, path)
    run_server(transport=args.transport, host=args.host, port=args.port)
    return 0
