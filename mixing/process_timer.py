"""Time a fresh subprocess through exit, before importing any analysis or GPU package."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time


def run(command, receipt):
    """Save elapsed wall time and child resource usage, including unsuccessful exits."""
    receipt = Path(receipt)
    if not command:
        raise ValueError("a subprocess command is required")
    with receipt.open("x") as stream:
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        begin = time.perf_counter()
        result = subprocess.run(command, check=False)
        elapsed = time.perf_counter() - begin
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        json.dump({"wall_s": elapsed, "user_s": after.ru_utime - before.ru_utime,
                   "system_s": after.ru_stime - before.ru_stime,
                   "max_rss_kb": after.ru_maxrss // (1024 if sys.platform == "darwin" else 1),
                   "exit_status": result.returncode}, stream)
        stream.write("\n")
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return run(command, args.receipt)


if __name__ == "__main__":
    raise SystemExit(main())
