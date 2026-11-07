"""Create a direct-frame development manifest from explicit campaign exports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .sampling import CHAINS, STREAMS


def campaign(expanded, efficiency_csv, *, beam_primary=False):
    """Use actual SimEfficiencies2 values, including its stock float32 parse."""
    if beam_primary is not True:
        raise ValueError("explicit beam-primary signal declaration required")
    lines = [line.split("#", 1)[0].strip() for line in Path(efficiency_csv).read_text().splitlines()]
    lines = [line for line in lines if line]
    if lines and lines[0].startswith("TABLE"):
        if lines.pop(0) != "TABLE SimEfficiencies2":
            raise ValueError("wrong efficiency table")
    rows = list(csv.reader(lines))
    if not rows or any(len(row) != 4 for row in rows):
        raise ValueError("expected SimEfficiencies2 tag,numerator,denominator,eff CSV")
    tags = [row[0] for row in rows]
    if len(set(tags)) != len(tags):
        raise ValueError("duplicate efficiency tags")
    efficiencies = {row[0]: float(np.float32(row[3])) for row in rows}
    required = {tag for chain in CHAINS.values() for tag in chain}
    if not required <= efficiencies.keys():
        raise ValueError("missing campaign efficiency tags")
    if not all(np.isfinite(efficiencies[k]) and 0 <= efficiencies[k] <= 1 for k in required):
        raise ValueError("invalid campaign efficiencies")
    physics = expanded["physics"]
    pbi = physics["producers"]["PBISim"]
    if pbi["module_type"] != "ProtonBunchIntensityLogNormal" or pbi["PrimaryParticle"] != "FindMCPrimary":
        raise ValueError("unsupported bunch law or primary declaration")
    skips = {}
    for stream in STREAMS:
        selection = physics["producers"].get(stream + "Filter")
        if selection is None or selection != {"module_type": "MixingFilter", "MixingModule": stream + "Mixer"}:
            raise ValueError("direct frames require the explicit universal postmix filter contract")
        mixer = physics["filters"][stream + "Mixer"]
        cfg = mixer["mu2e"]
        if (mixer["module_type"] != "MixBackgroundFrames" or mixer["readMode"] != "sequential"
                or mixer["wrapFiles"] is not True or cfg["simStageEfficiencyTags"] != list(CHAINS[stream])):
            raise ValueError("unsupported mixer law")
        factors = cfg["meanEventsPerPOTFactors"]
        if factors != ([1000] if stream == STREAMS[0] else []):
            raise ValueError("unsupported campaign prescale")
        if cfg.get("timeOffset") is not None or {"simTimeOffset", "simTimeOffsetMixer"} & cfg["products"].keys():
            raise ValueError("additional donor timing requires explicit support")
        skip = cfg["MaxEventsToSkip"]
        if not np.isfinite(skip) or skip < 0 or int(skip) != skip:
            raise ValueError("invalid initial skip")
        skips[stream] = int(skip)
    from .sampling import BunchLaw
    cut_min = pbi.get("cutMin", 0.0)
    BunchLaw.from_sdf(pbi["extendedMean"], pbi["SDF"], pbi["cutMax"], cut_min)
    return {"extended_mean": pbi["extendedMean"], "sdf": pbi["SDF"], "cut_max": pbi["cutMax"], "cut_min": cut_min,
            "beam_primary": True, "efficiencies": {tag: efficiencies[tag] for tag in sorted(required)},
            "stop_prescale": 1000, "max_skip": skips, "mode": "sequential",
            "postmix_selection": "universal", "additional_time_offset_ns": 0}


def main(argv=None):
    from trkfastsim.digi.digicond import parse_fhicl_dump
    from .frames import read_event_log, read_frames
    from .inspect import summarize
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--beam-primary", action="store_true", required=True,
                        help="explicit declaration that the signal census is entirely beam-derived")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("nonnegative sampling seed required")
    cache = args.cache.resolve()
    sources, census = {}, {}
    for name in ("signal", *STREAMS):
        log, path = cache / (name + ".log"), cache / (name + ".root")
        text = log.read_text()
        if "Art has completed and will exit with status 0" not in text:
            raise ValueError(f"missing successful complete extraction: {name}")
        batch = read_frames(path, event_ids=read_event_log(log))
        sources[name] = {"path": str(path), "event_log": str(log)}
        census[name] = summarize(batch)
        del batch
    csv_path = cache / "db_SimEfficiencies2.csv"
    manifest = {"version": 1, "signal": sources.pop("signal"), "pools": sources,
                "campaign": campaign(parse_fhicl_dump(args.expanded.read_text()), csv_path, beam_primary=args.beam_primary), "seed": args.seed}
    receipt = {"census": census, "configuration": str(args.expanded.resolve()),
               "configuration_sha256": hashlib.sha256(args.expanded.read_bytes()).hexdigest(),
               "efficiencies": str(csv_path), "efficiencies_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()}
    for filename, content in (("direct_manifest.json", manifest), ("cache_census.json", receipt)):
        with (cache / filename).open("x") as stream:
            stream.write(json.dumps(content, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
