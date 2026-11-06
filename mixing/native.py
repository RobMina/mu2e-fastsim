"""Configure stock native-art frame mixing before the existing detector digitizers."""
import argparse
import json
import re
from pathlib import Path

from .sampling import STREAMS


LABELS = {name: name + "Mixer" for name in STREAMS}


def _quoted(value):
    value = str(value)
    if not value or not value.isascii() or any(c in value for c in '\n\r\x00"\\'):
        raise ValueError("FHiCL strings must be nonempty ASCII without quotes or backslashes")
    return '"' + value + '"'


def render_mix(donor_files, *, parent_fcl, bunch_fcl, signal_tag="compressDetStepMCs",
               max_skip=0, write_event_ids=True):
    """Keep stock laws/conditions/filters; only bind inputs and the signal step tag.

    parent_fcl is a campaign wrapper of Production's Mix.fcl with matching
    conditions, geometry and primary selection. bunch_fcl is its stock bunch
    configuration (e.g. OneBB.fcl). No response parameters or rates are fitted.
    """
    if set(donor_files) != set(STREAMS):
        raise ValueError("all four stock streams must have donor files")
    if not isinstance(max_skip, int) or isinstance(max_skip, bool) or max_skip < 0:
        raise ValueError("max_skip must be a nonnegative integer")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", signal_tag):
        raise ValueError("signal_tag must be a module label without instance/process")
    if str(parent_fcl).startswith("/") or str(bunch_fcl).startswith("/"):
        raise ValueError("FHiCL include names must be relative to FHICL_FILE_PATH")
    lines = [f"#include {_quoted(parent_fcl)}", f"#include {_quoted(bunch_fcl)}"]
    for stream in STREAMS:
        files = donor_files[stream]
        if isinstance(files, str) or not files:
            raise ValueError(f"{stream} needs a nonempty file list")
        prefix = f"physics.filters.{LABELS[stream]}"
        lines += [f"{prefix}.fileNames: [{', '.join(_quoted(f) for f in files)}]",
                  f"{prefix}.readMode: sequential", f"{prefix}.wrapFiles: true",
                  f"{prefix}.mu2e.MaxEventsToSkip: {max_skip}",
                  f"{prefix}.mu2e.writeEventIDs: {str(bool(write_event_ids)).lower()}"]
    filtered = ", ".join(stream + "Filter" for stream in STREAMS)
    mixed = ", ".join(LABELS[stream] for stream in STREAMS)
    truth = "compressDetStepMCs" if signal_tag == "compressDetStepMCs" else f"compressDetStepMCs, {signal_tag}"
    lines += [f"physics.producers.makeSD.StrawGasStepModules: [{signal_tag}, {filtered}]",
              f"physics.producers.CrvPhotons.crvStepModuleLabels: [{_quoted(signal_tag)}, {filtered}]",
              'physics.producers.CrvPhotons.crvStepProcessNames: ["", "", "", "", ""]',
              f"physics.producers.compressDigiMCs.simParticleTags: [{truth}, {mixed}]"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("donors", type=Path, help="JSON mapping all four stream names to art file lists")
    parser.add_argument("--parent-fcl", required=True)
    parser.add_argument("--bunch-fcl", required=True)
    parser.add_argument("--signal-tag", default="compressDetStepMCs")
    parser.add_argument("--max-skip", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    text = render_mix(json.loads(args.donors.read_text()), parent_fcl=args.parent_fcl,
                      bunch_fcl=args.bunch_fcl, signal_tag=args.signal_tag, max_skip=args.max_skip)
    args.out.write_text(text)


if __name__ == "__main__":
    main()
