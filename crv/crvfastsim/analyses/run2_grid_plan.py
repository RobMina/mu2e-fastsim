"""Prepare a frozen, pilot-gated Run 2 grid extension; never stage files or submit jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

FILES = ("ref_sim_extracted.fcl", "ref_sim_run2.fcl", "geom_cosmic_run2_v10.txt")
PHASES = {"test": (121, 1), "pilot": (122, 2), "remainder": (124, 377)}
BASESEED = 20265100
NEVTS = 50000
FROZEN_FCL_SHA256 = {
    "ref_sim_extracted.fcl": "15dd5c0930a6d4013e9850a3746fe9a78746b0749d06450024dddd62d8fdacb1",
    "ref_sim_run2.fcl": "4828bcb649da2381533eb94710b726d02a508c17dd6c3ae8b02add924d8f87a9",
    "geom_cosmic_run2_v10.txt": "9d666ac19bb86240d2204cc118c4458d198b8c41d4b540448aa94a331acd23a1",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(state, outdir, owner, source=None):
    """Create only local payload/manifest/commands with disjoint known reference identities."""
    import re
    from .. import dcache
    if not re.fullmatch(re.escape(dcache.area() + "/") + r"scratch/users/[A-Za-z0-9_-]+/[A-Za-z0-9_./-]+", outdir):
        raise ValueError("explicit Mu2e pnfs scratch output required")
    if ".." in Path(outdir).parts or not re.fullmatch(r"[A-Za-z0-9_-]+", owner):
        raise ValueError("unsafe output or owner")
    source = Path(source or Path(__file__).resolve().parents[2] / "grid").resolve()
    inputs = [source / "fcl" / name for name in FILES] + [source / "grid_run2_extension.sh"]
    if any(not p.is_file() for p in inputs):
        raise ValueError("missing source payload")
    if any(digest(source / "fcl" / name) != sha for name, sha in FROZEN_FCL_SHA256.items()):
        raise ValueError("FCL differs from the original frozen Run 2 campaign")
    state = Path(state).resolve()
    state.mkdir(parents=True, exist_ok=False)
    payload = state / "payload"
    payload.mkdir()
    for path in inputs:
        shutil.copy2(path, payload / path.name)
    (payload / "grid_run2_extension.sh").chmod(0o755)
    hashes = {p.name: digest(p) for p in inputs}
    campaign = {
        "schema": "crv-run2-grid-extension-v1", "run": 1400, "config": "run2", "owner": owner,
        "tag": "v1", "musing": "MDC2025ay", "offline": "v13_39_00", "stub": "al9-prof-e29-p107",
        "outdir": outdir.rstrip("/"), "write_scope": dcache.token_scope(outdir).rstrip("/"),
        "events_per_job": NEVTS, "baseseed": BASESEED, "first_subrun": 121, "last_subrun": 500,
        "additional_generated_events": 19000000, "reused_generated_events": 1000000,
        "total_generated_events": 20000000,
        "resources": {"cpus": 1, "memory_mb": 3000, "disk_mb": 4000, "expected_lifetime": "4h"},
        "prior_run2_reference": {"subruns": [101, 120], "seeds": [20265200, 20265219]},
        "phases": {name: {"first_subrun": first, "jobs": jobs, "last_subrun": first + jobs - 1}
                   for name, (first, jobs) in PHASES.items()},
        "shards": [{"subrun": s, "seed": BASESEED + s - 1, "generated_events": NEVTS}
                   for s in range(121, 501)],
        "payload_sha256": hashes,
        "output_policy": "art, logs and prospective completion receipts directly to pnfs; no bulk exp copy",
        "postprocessing_policy": "bounded direct-pnfs dumping or UVA analysis; no full derived set in exp",
        "scope": "frozen G4 transport reference, not fixed-stock response V2 or scatter-in promotion",
    }
    (payload / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")
    manifest_sha = digest(payload / "campaign.json")
    (payload / "payload.sha256").write_text("".join(f"{value}  {name}\n" for name, value in hashes.items())
                                            + f"{manifest_sha}  campaign.json\n")
    (state / "campaign.sha256").write_text(f"{manifest_sha}  payload/campaign.json\n")
    lines = ["#!/usr/bin/env bash", "# Default commands print only; add --submit after approval and live preflight.",
             'STATE=$(cd "$(dirname "$0")" && pwd)', ': "${CRVFS_CODE:?explicit CRV checkout}"']
    for phase in PHASES:
        lines.append(f'bash "$CRVFS_CODE/grid/submit_run2_extension.sh" "$STATE" {phase}')
    (state / "commands.sh").write_text("\n".join(lines) + "\n")
    (state / "README.txt").write_text(
        "Prepared only; no job or pnfs write has occurred.\n"
        "Each phase is dry-run by default. test -> validate persisted counts/copyout -> pilot -> validate -> remainder.\n"
        "--submit requires explicit approval; a prior attempted phase is never silently resubmitted.\n"
        "Queue/seed/output checks must be current on the submitting host.\n"
        "Do not use the historical submit_ref_sim.sh: it removes/replaces shared FCL before submitting.\n")
    return campaign


def validate_gate(path, campaign_sha, expected):
    """Require successful persisted-count checks for every prior phase shard, not log totals alone."""
    row = json.loads(Path(path).read_text())
    if row.get("schema") != "crv-run2-extension-gate-v1" or row.get("campaign_sha256") != campaign_sha:
        raise ValueError("gate schema or campaign mismatch")
    shards = row.get("shards", [])
    if len(shards) != len(expected) or sorted(s["subrun"] for s in shards) != expected:
        raise ValueError("gate requires every prior phase shard exactly once")
    for s in shards:
        if (s.get("generated_events") != NEVTS or s.get("seed") != BASESEED + s["subrun"] - 1
                or not isinstance(s.get("accepted_events"), int) or isinstance(s.get("accepted_events"), bool)
                or not 0 < s["accepted_events"] <= NEVTS
                or any(s.get(k) is not True for k in ("highest_cycle_counts_verified", "generator_values_verified",
                                                     "copyout_verified", "runtime_verified"))):
            raise ValueError("gate missing required persisted/copyout/runtime evidence")
        evidence = s.get("evidence", [])
        if not evidence or any(not Path(e["path"]).is_file() or digest(e["path"]) != e["sha256"] for e in evidence):
            raise ValueError("gate evidence changed or missing")
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--owner", required=True)
    args = parser.parse_args(argv)
    c = prepare(args.state, args.outdir, args.owner)
    print(f"Prepared {len(c['shards'])} jobs; seeds {c['shards'][0]['seed']}..{c['shards'][-1]['seed']}; no submission")


if __name__ == "__main__":
    main()
