"""On-device occupancy output for the resident backend.

Opt-in (`--device-output occupancy`). The compiled batch reduces its own detector output to
the observables of `counts.read_counts`: straw digis per tracker plane and reconstructed CRV
pulses per sector, per event, plus per-event CRV digi and coincidence-cluster counts. Only
these small dense arrays, the event identities and a compact sampling receipt leave the
device. Detector digis, ADC samples and pulses are not persisted in this mode.
"""
import json
from pathlib import Path

import numpy as np

PLANES = 36
FIELDS = ("planes", "sectors", "crv_digis", "crv_clusters")


def load_sector_map(path):
    """Bar -> sector map with the same JSON contract as `counts.main --geometry`."""
    geometry = json.loads(Path(path).read_text())
    names = list(geometry["sector_names"])
    if "bar_to_sector" in geometry:
        mapping = np.asarray(geometry["bar_to_sector"])
    else:
        mapping = np.repeat(np.arange(len(names)), geometry["bars_per_sector"])
    if not names or mapping.ndim != 1 or not mapping.size or mapping.dtype.kind not in "iu":
        raise ValueError("invalid sector geometry map")
    if np.any(mapping < 0) or np.any(mapping >= len(names)):
        raise ValueError("sector map contains out-of-range bins")
    return mapping.astype(np.int64), names, {k: geometry[k] for k in ("geometry", "coverage") if k in geometry}


def reduce(output, bar_to_sector, n_sectors, batch_events):
    """Pure JAX: dense per-event occupancy from the masked detector tables, plus error flags."""
    import jax.numpy as jnp

    b = batch_events
    trk = output["tracker"]
    plane = (trk["sid"].astype(jnp.int64) & 0xFC00) >> 10
    tvalid = trk["mask"] & (plane < PLANES)
    planes = jnp.zeros((b, PLANES), jnp.int32).at[
        jnp.where(tvalid, trk["ev"], b), jnp.clip(plane, 0, PLANES - 1)].add(1, mode="drop")
    pulses = output["pulses"]
    bar = pulses["bar"].astype(jnp.int64)
    inside = (bar >= 0) & (bar < bar_to_sector.shape[0])
    pvalid = pulses["mask"] & inside
    sector = bar_to_sector[jnp.clip(bar, 0, bar_to_sector.shape[0] - 1)]
    sectors = jnp.zeros((b, n_sectors), jnp.int32).at[jnp.where(pvalid, pulses["ev"], b), sector].add(1, mode="drop")

    def per_event(table):
        return jnp.zeros(b, jnp.int32).at[jnp.where(table["mask"], table["ev"], b)].add(1, mode="drop")

    result = {"planes": planes, "sectors": sectors,
              "crv_digis": per_event(output["digis"]), "crv_clusters": per_event(output["clusters"])}
    errors = {"occupancy/plane": jnp.any(trk["mask"] & (plane >= PLANES)),
              "occupancy/bar": jnp.any(pulses["mask"] & ~inside)}
    return result, errors


def compact_receipt(receipt):
    """Drop per-donor arrays; sequential donors are recoverable from counts and cursors."""
    selections = {stream: {key: value for key, value in selection.items()
                           if key not in ("donors", "mask", "source_event_ids")}
                  for stream, selection in receipt["selections"].items()}
    return {**receipt, "selections": selections}


def write(path, keys, ordinal, occupancy, sector_names):
    """One closed NPZ per batch: identities, realization ordinals and the occupancy arrays."""
    n = len(keys)
    arrays = {"event_ids": np.ascontiguousarray(keys, np.uint64),
              "realization": np.arange(n, dtype=np.uint64) + np.uint64(ordinal),
              "sector_names": np.asarray(sector_names)}
    for name in FIELDS:
        value = np.asarray(occupancy[name])[:n]
        if np.any(value < 0):
            raise ValueError(f"negative occupancy count: {name}")
        arrays[name] = value
    with Path(path).open("xb") as stream:
        np.savez(stream, **arrays)


def read(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def census(path, ordinal_start, events, expected_ids):
    """Verify one finished occupancy batch against the sampled identities and ordinals."""
    data = read(path)
    ids = data["event_ids"]
    if ids.shape != (events, 3) or not np.array_equal(ids, expected_ids):
        raise ValueError("finished occupancy identities differ from sampled source")
    if not np.array_equal(data["realization"], np.arange(ordinal_start, ordinal_start + events, dtype=np.uint64)):
        raise ValueError("persisted realization identities differ")
    if data["planes"].shape != (events, PLANES) or data["sectors"].shape[0] != events:
        raise ValueError("occupancy arrays do not match the event census")
    return {"events": events, "tracker_digis": int(data["planes"].sum()),
            "crv": {"pulses": int(data["sectors"].sum()), "digis": int(data["crv_digis"].sum()),
                    "clusters": int(data["crv_clusters"].sum())},
            "tracker_empty_events": int(np.count_nonzero(data["planes"].sum(axis=1) == 0))}


def collect(run_directory, phase="measure"):
    """Concatenate one phase's finished batch NPZs, in ledger order, as `counts.align` input."""
    run_directory = Path(run_directory)
    rows = [json.loads(line) for line in (run_directory / "batches.jsonl").read_text().splitlines()]
    parts = [read(run_directory / row["directory"] / "occupancy.npz") for row in rows if row["phase"] == phase]
    if not parts:
        raise ValueError(f"no {phase} batches in {run_directory}")
    return {"event_ids": np.concatenate([p["event_ids"] for p in parts]),
            "planes": np.concatenate([p["planes"] for p in parts]),
            "sectors": np.concatenate([p["sectors"] for p in parts]),
            "realization": np.concatenate([p["realization"] for p in parts])}


def main(argv=None):
    """Pair a finished occupancy run with saved stock counts (`counts.py` NPZ) by full event ID."""
    import argparse
    from .counts import align, json_values
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("run", type=Path, help="finished stream_benchmark output directory")
    parser.add_argument("--stock", type=Path, required=True, help="saved stock counts.npz")
    parser.add_argument("--selection", required=True, help="explicit description of the paired selection")
    parser.add_argument("--out", type=Path, required=True, help="new NPZ; JSON summary written beside it")
    args = parser.parse_args(argv)
    data = collect(args.run)
    with args.out.open("xb") as stream:
        np.savez_compressed(stream, **data)
    with np.load(args.stock, allow_pickle=False) as stock:
        paired = align({k: stock[k] for k in ("event_ids", "planes", "sectors")}, data)
    report = {"events": len(data["event_ids"]), "selection": args.selection, "paired": paired,
              "straw_digis_per_plane": data["planes"].sum(axis=0), "crv_pulses_per_sector": data["sectors"].sum(axis=0),
              "comparison_note": "Event intersection only; independent pileup draws and response; selection and coverage not corrected."}
    text = json.dumps(json_values(report), indent=2, allow_nan=False) + "\n"
    args.out.with_suffix(".json").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
