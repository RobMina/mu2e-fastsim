"""Chain two MCP servers as an agent would: fastsim fetch_outputs -> the analysis server's approx_ce_sensitivity.

Starts this server (``python -m fastsim_mcp_server``) and the Mu2e analysis server (the command in
$ANALYSIS_MCP_CMD, a JSON list, e.g. ``["python3", "-m", "analysis_mcp_server", "--transport", "stdio"]`` or a
relay that runs it on another host), calls fetch_outputs(NAME) on the first and runs every returned ``score_with``
call on the second. Writes a transcript.

usage: ANALYSIS_MCP_CMD='[...]' python chain_client.py NAME TRANSCRIPT.json [-- fastsim server args]
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from smoke_client import HERE, reply


async def main(argv):
    server_args = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    name, out = argv[0], Path(argv[1])
    cmd = json.loads(os.environ["ANALYSIS_MCP_CMD"])
    fast = StdioServerParameters(command=sys.executable, args=["-m", "fastsim_mcp_server", *server_args],
                                 cwd=str(HERE), env=dict(os.environ))
    ana = StdioServerParameters(command=cmd[0], args=cmd[1:], env=dict(os.environ))
    log = {"calls": []}
    async with stdio_client(fast) as (r1, w1), ClientSession(r1, w1) as fs:
        await fs.initialize()
        t0 = time.time()
        got = reply(await fs.call_tool("fetch_outputs", {"name": name}))
        log["calls"].append({"server": "fastsim", "tool": "fetch_outputs", "request": {"name": name},
                             "wall_s": round(time.time() - t0, 2), "reply": got})
        print(f"[fetch_outputs] {got.get('status')}: {got.get('message')}", flush=True)
        calls = [o["score_with"] for o in got.get("metadata", {}).get("outputs", []) if "score_with" in o]
        dtypes = [o["dtype"] for o in got.get("metadata", {}).get("outputs", []) if "score_with" in o]
        async with stdio_client(ana) as (r2, w2), ClientSession(r2, w2) as an:
            await an.initialize()
            for dtype, call in zip(dtypes, calls):
                t0 = time.time()
                rep = reply(await an.call_tool(call["tool"], call["arguments"]))
                log["calls"].append({"server": "analysis", "dtype": dtype, "tool": call["tool"],
                                     "request": call["arguments"], "wall_s": round(time.time() - t0, 2),
                                     "reply": rep})
                out.write_text(json.dumps(log, indent=1, default=str))
                print(f"[{call['arguments']['analysis']} {dtype}] {rep.get('status')} ({round(time.time() - t0, 1)} s): "
                      f"{str(rep.get('message'))[:400]}", flush=True)
    out.write_text(json.dumps(log, indent=1, default=str))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    asyncio.run(main(sys.argv[1:]))
