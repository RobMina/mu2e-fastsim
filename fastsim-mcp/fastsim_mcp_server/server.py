"""Generic MCP wrapper: registers every function in each configured tool module's __all__ as an MCP tool.

The tool modules come from ``[tool.mcp-server] tool_modules`` in this directory's pyproject.toml. Type hints,
pydantic Field constraints and docstrings are the schema agents see. Nothing here knows about Mu2e.
"""
from __future__ import annotations

import importlib
import tomllib
from pathlib import Path
from typing import Literal

# mcp 2.x renamed FastMCP to MCPServer and moved host/port from the constructor to run()
try:
    from mcp.server.mcpserver import MCPServer as FastMCP
    _HOST_PORT_IN_RUN = True
except ImportError:
    from mcp.server.fastmcp import FastMCP
    _HOST_PORT_IN_RUN = False

Transport = Literal["stdio", "streamable-http"]

INSTRUCTIONS = (
    "Mu2e fast simulation (JAX, GPUs and CPUs) as tools. Call list_workflows to see the fast workflows (beam, "
    "tracker, CRV, mixing, calorimeter, trigger, the conversion-electron chain and the EdepAna / EventNtuple "
    "writers), describe_approximations for what each replaces in the stock simulation and the switches. "
    "run_design_point runs one stopping-target design, asynchronously, in float64 and float32, on CPU by default: "
    "it returns job handles at once; poll job_status (at most once a minute), then fetch_outputs, which returns "
    "the EdepAna file and the run_analysis call that scores it with the Mu2e analysis server's "
    "approx_ce_sensitivity. GPU work (device='gpu', run_crv_chain) is sized first and returns needs_approval: show "
    "the request to the user and repeat with allow_gpu=true only after they approve it."
)


def pyproject_toml() -> Path:
    for directory in Path(__file__).resolve().parents:
        path = directory / "pyproject.toml"
        if path.exists():
            return path
    raise FileNotFoundError("no pyproject.toml above the server package")


def configured_tool_module_names() -> list[str]:
    path = pyproject_toml()
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    try:
        return list(config["tool"]["mcp-server"]["tool_modules"])
    except KeyError as exc:
        raise RuntimeError(f"{path} needs a [tool.mcp-server] section with a tool_modules list") from exc


def tool_functions():
    """(name, function) of every exported tool, in order."""
    out = []
    for module_name in configured_tool_module_names():
        module = importlib.import_module(module_name)
        if not hasattr(module, "__all__"):
            raise RuntimeError(f"tool module {module_name!r} must define __all__")
        out += [(name, getattr(module, name)) for name in module.__all__]
    return out


def create_server(*, host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    if _HOST_PORT_IN_RUN:
        mcp = FastMCP("Mu2e Fastsim MCP Server", instructions=INSTRUCTIONS)
    else:
        mcp = FastMCP("Mu2e Fastsim MCP Server", instructions=INSTRUCTIONS, host=host, port=port)
    for _, function in tool_functions():
        mcp.tool()(function)
    return mcp


def run_server(*, transport: Transport = "stdio", host: str = "127.0.0.1", port: int = 8000) -> None:
    mcp = create_server(host=host, port=port)
    if _HOST_PORT_IN_RUN and transport != "stdio":
        mcp.run(transport=transport, host=host, port=port)
    else:
        mcp.run(transport=transport)
