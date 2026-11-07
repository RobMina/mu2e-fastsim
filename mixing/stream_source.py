"""Bounded-batch mixing over complete resident source-frame snapshots.

This is benchmark infrastructure, not additional independent signal statistics.
Signal slots and sequential donors wrap; original signal IDs are deliberately
reused across batches, never within a batch. Each batch uses NumPy default_rng
with SeedSequence([seed, batch_index]). This domain differs from direct_frames
and C++ replay: changing batch sizes changes draws and subsequent donor cursors.

All five full pools remain resident. max_selected_rows limits the sum of selected
rows in ALL collections (truth included), not bytes, source-pool memory, selection
arrays or overlay/adapter workspace. No rows are truncated. The source retains
no expanded batches; callers must consume/release each result rather than collect
results if they require one-batch residency. Full mixed truth is released after
adaptation; this is not a native truth-product output.

Sources are read, hashed and validated once at construction via direct_frames._read.
Later batches use those in-memory snapshots, not current disk bytes: no repeated
file hashing or rereading is performed. Overlay and adapter reference checks still
run afresh on every batch. No timing is synthesized or detector response run here.
"""
from dataclasses import dataclass
import time

import awkward as ak
import numpy as np
import pandas as pd

from . import direct_frames
from .frames import FrameBatch, overlay
from .sampling import STREAMS, draw_multiplicities, select_frames


@dataclass(frozen=True)
class StreamBatch:
    """One detector-input batch with original Nx3 IDs and a sampling receipt.

    ordinal_start is the zero-based global stream-event offset, not a source ID;
    batch_index is zero-based. pdg is aligned to crv rows by validated references.
    provenance is accepted by fast_detectors.save_sampling. Source configuration
    and fingerprints are stored once in the producing StreamSource.metadata.
    """

    keys: np.ndarray
    tracker: pd.DataFrame
    crv: pd.DataFrame
    pdg: np.ndarray
    provenance: dict
    ordinal_start: int
    batch_index: int


class StreamSource:
    """Load a v1 manifest once; generate fresh bounded batches from resident pools.

    seed is an explicit nonnegative stream seed (default 0), independent of the
    validated manifest seed. max_selected_rows must be a positive integer.
    metadata is a JSON-friendly source/configuration summary; persist it once
    alongside batch receipts. Empty signal or donor censuses are rejected, but
    empty detector events remain valid slots. This mutable iterator is not
    thread-safe. A failed batch does not advance signal, donor or RNG state.
    """

    def __init__(self, manifest, *, seed=0, max_selected_rows):
        seed = direct_frames._integer(seed, "seed")
        max_selected_rows = direct_frames._integer(max_selected_rows, "max_selected_rows")
        if max_selected_rows == 0:
            raise ValueError("max_selected_rows must be a positive integer")
        path, manifest_seed, _, campaign, law, rates, sources = direct_frames._load_manifest_config(manifest)
        start = time.perf_counter()
        signal, signal_fingerprint = direct_frames._read(sources["signal"])
        pools, pool_fingerprints = {}, {}
        for stream in STREAMS:
            pools[stream], pool_fingerprints[stream] = direct_frames._read(sources["pools"][stream])
        read_s = time.perf_counter() - start
        pool_counts = {stream: len(pool.event_ids) for stream, pool in pools.items()}
        if not all(pool_counts.values()):
            raise ValueError("all four full donor pools must have a nonempty event census")
        if not len(signal.event_ids):
            raise ValueError("signal must have a nonempty event census")
        self._signal, self._pools = signal, pools
        self._seed, self._max_selected_rows = seed, max_selected_rows
        self._campaign, self._law, self._rates = campaign, law, rates
        self._pool_counts = pool_counts
        self._cursors = dict.fromkeys(STREAMS)
        self._ordinal = 0
        self._batch_index = 0
        self.metadata = {
            "version": 1, "manifest": str(path), "sources": sources, "campaign": campaign,
            "source_fingerprints": {"signal": signal_fingerprint, "pools": pool_fingerprints},
            "seed": seed, "manifest_seed": manifest_seed, "stream_order": list(STREAMS),
            "rates": rates, "pool_event_counts": pool_counts, "signal_events_full": self.population,
            "max_selected_rows": max_selected_rows, "read_validate_s": read_s,
            "rng": "NumPy default_rng(SeedSequence([seed, batch_index])); not direct/C++ RNG replay",
            "snapshot_policy": "all five sources read/hash/validated once; no per-batch disk recheck",
            "memory_policy": "full pools resident; selected-row cap is not a byte-memory limit",
            "signal_policy": "contiguous wrapping slots; repeated IDs across batches are not independent statistics",
        }

    @property
    def signal(self):
        """Resident source for timing alignment; consumers must not mutate its arrays."""
        return self._signal

    @property
    def population(self):
        """Full original signal census size, including empty detector events."""
        return len(self._signal.event_ids)

    def next_batch(self, count) -> StreamBatch:
        """Return 1 <= count <= population consecutive signal slots, wrapping.

        One shared bunch intensity per signal slot feeds four conditional Poissons.
        Each donor stream gets its stock inclusive uniform initial skip once, on
        the first successful batch; later batches continue its modulo-pool cursor.
        The row guard runs before overlay, without truncation or state advancement.
        """
        count = direct_frames._integer(count, "count")
        if not 1 <= count <= self.population:
            raise ValueError("count must satisfy 1 <= count <= population")
        start = time.perf_counter()
        indices = (np.arange(count, dtype=np.int64) + self._ordinal % self.population) % self.population
        rng = np.random.default_rng(np.random.SeedSequence([self._seed, self._batch_index]))
        intensity = self._law.draw(rng, count, beam_primary=True)
        counts = draw_multiplicities(rng, intensity, self._rates)
        estimated_rows = {name: int(ak.sum(ak.num(rows)[indices]))
                          for name, rows in self._signal.collections.items()}
        backgrounds, selections, next_cursors = [], {}, {}
        for column, stream in enumerate(STREAMS):
            pool = self._pools[stream]
            selection, next_cursor = select_frames(
                rng, counts[:, column], self._pool_counts[stream], cursor=self._cursors[stream],
                mode=self._campaign["mode"], max_events_to_skip=self._campaign["max_skip"][stream])
            backgrounds.append((pool, selection))
            next_cursors[stream] = next_cursor
            for name, rows in pool.collections.items():
                estimated_rows[name] = estimated_rows.get(name, 0) + int(ak.sum(ak.num(rows)[selection.donors]))
            selections[stream] = {
                "counts": selection.counts, "donors": selection.donors,
                "source_event_ids": pool.event_ids[selection.donors].copy(),
                "event_index": selection.event_index,
                "initial_cursor": (next_cursor - len(selection.donors)) % self._pool_counts[stream],
                "next_cursor": next_cursor,
            }
        selected_rows = sum(estimated_rows.values())
        if selected_rows > self._max_selected_rows:
            raise ValueError(f"selected rows {selected_rows} exceed max_selected_rows={self._max_selected_rows}")
        signal = FrameBatch(self._signal.event_ids[indices],
                            {name: rows[indices] for name, rows in self._signal.collections.items()})
        sampling_s = time.perf_counter() - start
        start = time.perf_counter()
        mixed = overlay(signal, backgrounds)
        overlay_s = time.perf_counter() - start
        start = time.perf_counter()
        tracker, crv, pdg = direct_frames.to_detector_inputs(mixed)
        adapter_s = time.perf_counter() - start
        actual_rows = {name: int(ak.sum(ak.num(rows))) for name, rows in mixed.collections.items()}
        keys = mixed.event_ids.copy()
        del mixed
        provenance = {
            "version": 1, "source_metadata_ref": "StreamSource.metadata",
            "seed": self._seed, "rng": self.metadata["rng"], "stream_order": STREAMS,
            "batch_index": self._batch_index, "ordinal_start": self._ordinal,
            "signal_events_full": self.population, "signal_indices": indices,
            "signal_event_ids": keys.copy(), "rates": self._rates,
            "intensity": intensity, "multiplicities": counts, "selections": selections,
            "pool_event_counts": self._pool_counts, "estimated_selected_rows": estimated_rows,
            "selected_rows_total": selected_rows, "actual_rows": actual_rows,
            "timing_s": {"sample": sampling_s, "overlay_validate": overlay_s,
                         "adapter_validate": adapter_s},
        }
        result = StreamBatch(keys, tracker, crv, pdg, provenance, self._ordinal, self._batch_index)
        self._cursors = next_cursors
        self._ordinal += count
        self._batch_index += 1
        return result
