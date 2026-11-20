"""A stand-in for the worker's apptainer in the grid-route tests: ``exec [options] IMAGE command...`` and
``build --sandbox DIR IMAGE``. It models only what the grid job relies on:

- exec of a SIF file fails like a worker without /dev/fuse when FAKEAPPT_NO_FUSE is set; a directory (sandbox) runs;
- ``python3 -c "import jax..."`` (the image check) prints the check line; ``python -c "import ..."`` succeeds;
- ``test -r <path>`` answers from FAKEAPPT_CVMFS (set: visible);
- ``python -m beamfastsim.stage OUT ...`` prints a stage line;
- ``python -m beamfastsim.edepana point ... --out D`` writes a metrics.json, the EdepAna file and its run record into D
  (exit FAKEAPPT_POINT_RC when set), and records its argv and --env list in D/fake_point.json.
"""
import json
import os
import sys


def main(argv):
    if argv[0] == "build":
        assert argv[1] == "--sandbox"
        os.makedirs(argv[2], exist_ok=True)
        open(os.path.join(argv[2], "image"), "w").write(open(argv[3], "rb").read().hex()[:16])
        return 0
    assert argv[0] == "exec", argv
    args, env = argv[1:], ""
    binds = []
    while args and args[0].startswith("-"):
        if args[0] == "--env":
            env = args[1]
        if args[0] == "-B":
            binds.append(args[1])
        args = args[2:] if args[0] in ("--env", "-B", "--pwd") else args[1:]
    image, cmd = args[0], args[1:]
    if os.path.isfile(image) and os.environ.get("FAKEAPPT_NO_FUSE"):
        print("FATAL: squashfuse: fuse: device not found", file=sys.stderr)
        return 255
    if cmd[0] == "test":
        return 0 if os.environ.get("FAKEAPPT_CVMFS") else 1
    if cmd[:2] == ["python3", "-c"] and "import jax" in cmd[2]:
        print("nested apptainer ok 0.0 [CpuDevice(id=0)]")
        return 0
    if cmd[:2] == ["python", "-c"]:
        return 0
    if cmd[:3] == ["python", "-m", "beamfastsim.stage"]:
        print(f"export BEAMFS_OFFLINE_ROOT={cmd[3]}/offline BEAMFS_MAPS={cmd[3]}/maps   # 3 files in 0.1 s")
        return 0
    if cmd[:4] == ["python", "-m", "beamfastsim.edepana", "point"]:
        out = cmd[cmd.index("--out") + 1]
        for flag in ("--g4", "--library", "--edge-library", "--gdml"):
            assert os.path.isfile(cmd[cmd.index(flag) + 1]), flag
        n = int(cmd[cmd.index("--electrons") + 1])
        json.dump({"n_events": n // 2, "n_gen_events": n, "ce_acceptance": 0.5, "stops_per_pot": 1.2e-3},
                  open(os.path.join(out, "metrics.json"), "w"))
        open(os.path.join(out, "nts.mu2efastsim.edep.root"), "wb").write(b"root")
        open(os.path.join(out, "nts.mu2efastsim.edep.root.edep.json"), "w").write("{}")
        json.dump({"argv": cmd, "env": env, "binds": binds, "image": image},
                  open(os.path.join(out, "fake_point.json"), "w"))
        print("point done")
        return int(os.environ.get("FAKEAPPT_POINT_RC", "0"))
    print(f"fakeapptainer: unsupported {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
