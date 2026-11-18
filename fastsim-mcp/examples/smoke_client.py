"""Smoke test of the fastsim MCP server as an MCP client sees it (Python mcp SDK, stdio).

Starts ``python -m fastsim_mcp_server`` as a subprocess, lists the tools, calls list_workflows and
describe_approximations, then (with --point) run_design_point, job_status once a minute until every job has left
the queue, and fetch_outputs. Keeps the session open meanwhile: the server streams the jobs' inputs in. Writes a
transcript of every call (request, reply, wall time).

usage: python smoke_client.py TRANSCRIPT.json [--point '{"name": "demo1", "electrons": 5000}'] [--poll-s 60]
       [--max-wait-h 6] [-- server args, e.g. --site-env site.env]
The server gets this process's environment (the SDK's default would keep only a few variables).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parents[1]


def reply(res):
    if getattr(res, "structuredContent", None):
        sc = res.structuredContent
        return sc.get("result", sc) if isinstance(sc, dict) and set(sc) == {"result"} else sc
    texts = [getattr(c, "text", None) for c in res.content]
    try:
        return json.loads(texts[0])
    except (TypeError, ValueError, IndexError):
        return {"status": "text", "message": "\n".join(t for t in texts if t)}


async def main(argv):
    if "--" in argv:
        i = argv.index("--")
        argv, server_args = argv[:i], argv[i + 1:]
    else:
        server_args = []
    p = argparse.ArgumentParser()
    p.add_argument("transcript", type=Path)
    p.add_argument("--point", help="run_design_point arguments as JSON")
    p.add_argument("--poll-s", type=float, default=60.0)
    p.add_argument("--max-wait-h", type=float, default=6.0)
    a = p.parse_args(argv)
    params = StdioServerParameters(command=sys.executable, args=["-m", "fastsim_mcp_server", *server_args],
                                   cwd=str(HERE), env=dict(os.environ))
    log = {"server": {"args": server_args}, "calls": []}

    def save():
        a.transcript.write_text(json.dumps(log, indent=1, default=str))

    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            init = await s.initialize()
            log["server"]["name"] = init.serverInfo.name
            log["tools"] = [{"name": t.name, "description": (t.description or "").split("\n")[0],
                             "parameters": sorted(t.inputSchema.get("properties", {}))}
                            for t in (await s.list_tools()).tools]
            print("tools:", [t["name"] for t in log["tools"]], flush=True)

            async def call(tool, args, quiet=False):
                t0 = time.time()
                try:
                    rep = reply(await s.call_tool(tool, args))
                except Exception as e:   # noqa: BLE001 - a smoke test records every failure
                    rep = {"status": "client_error", "message": f"{type(e).__name__}: {e}"}
                entry = {"tool": tool, "request": args, "at": time.strftime("%H:%M:%S"),
                         "wall_s": round(time.time() - t0, 2), "reply": rep}
                log["calls"].append(entry)
                save()
                if not quiet:
                    print(f"[{tool}] {rep.get('status')} ({entry['wall_s']} s): {str(rep.get('message'))[:400]}",
                          flush=True)
                return rep

            await call("list_workflows", {})
            await call("describe_approximations", {"workflow": "edepana_writer", "preset": "fidelity"})
            if a.point:
                args = json.loads(a.point)
                sub = await call("run_design_point", args)
                if sub.get("status") == "submitted":
                    t_end = time.time() + 3600 * a.max_wait_h
                    while time.time() < t_end:
                        st = await call("job_status", {"name": args["name"]})
                        if st.get("status") in ("success", "error"):
                            break
                        await asyncio.sleep(a.poll_s)
                    await call("fetch_outputs", {"name": args["name"]})
    save()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
