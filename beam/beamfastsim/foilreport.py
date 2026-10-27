"""Identity-bound single-foil G4 census and material/table audit, independent of fast response.

The published study readback has float step IDs and pre-step states only. Chord sums are geometric diagnostics,
not Geant4 true path lengths or deposited energy. SimParticle isstop flags only mean a known end volume here.
"""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryFile

import numpy as np
import uproot

from . import physics_tables as pt
from .foilclosure import MMU, macro_errors
from .reference_audit import _identity
from .slicestudy import verify_hashes


ME = .51099895


def kinetic(p, mass=MMU):
    p = np.asarray(p, float)
    return p * p / (np.sqrt(p * p + mass * mass) + mass)


def highest(root, name, *, arrays=True):
    """Select the recorded highest cycle; streaming callers request the tree, not arrays."""
    keys = [k for k in root.keys(recursive=True, cycle=True) if k.rsplit(";", 1)[0] == name]
    if not keys:
        raise ValueError(f"missing tree {name}")
    key = max(keys, key=lambda x: int(x.rsplit(";", 1)[1]))
    tree = root[key]
    return (tree.arrays(library="np") if arrays else tree), dict(key=key, entries=int(tree.num_entries), cycles=keys)


def integer(values, name):
    if np.any(~np.isfinite(values)) or np.any(values != np.floor(values)):
        raise ValueError(f"invalid integer IDs: {name}")
    return np.asarray(values, np.int64)


def census(particles, steps, case):
    n, pdg, half = case["events"], case["pdg"], case["thickness_mm"] / 2
    primary = (particles["parent_id"] == -1) & (particles["pdg"] == pdg)
    if int(primary.sum()) != n or np.any((particles["parent_id"] == -1) & ~primary):
        raise ValueError("one requested primary muon per event required")
    p = {k: v[primary] for k, v in particles.items()}
    order = np.argsort(p["evt"])
    p = {k: v[order] for k, v in p.items()}
    if not np.array_equal(integer(p["evt"], "primary event"), np.arange(1, n + 1)) or np.any(p["run"] != 1):
        raise ValueError("persisted primary event coverage mismatch")
    if np.any(p["code"] != 56):
        raise ValueError("unexpected primary creation code")
    events = integer(steps["evt"], "step event") - 1
    tracks = integer(steps["trk"], "step track")
    if np.any(events < 0) or np.any(events >= n) or np.any(steps["run"] != 1):
        raise ValueError("step event outside primary population")
    matched = tracks == p["trk"][events]
    s = {k: v[matched] for k, v in steps.items()}
    events = events[matched]
    if np.any(s["pdg"] != pdg) or np.any(np.diff(events) < 0):
        raise ValueError("primary step identity/order mismatch")
    if any(np.any(~np.isfinite(s[k])) for k in ("x", "y", "z", "px", "py", "pz", "ke", "time")):
        raise ValueError("nonfinite primary steps")
    at = np.arange(len(events))
    first = np.full(n, len(events), np.int64)
    np.minimum.at(first, events[s["sid"] == 2], at[s["sid"] == 2])
    if np.any(first == len(events)):
        raise ValueError("primary never enters foil")
    outside = (s["sid"] != 2) & (at > first[events])
    exit_at = np.full(n, len(events), np.int64)
    np.minimum.at(exit_at, events[outside], at[outside])
    exits = exit_at < len(events)
    defined = np.isin(p["codestop"], [14, 32]) & np.isfinite(p["tstop"]) & (p["tstop"] >= s["time"][first])
    stop = (~exits) & defined & (p["pstop"] < 1e-6) & (np.abs(p["zstop"]) <= half + 2e-6)
    other = ~(exits | stop)
    ix = np.minimum(exit_at, len(events) - 1)
    out = np.where(exits, kinetic(np.sqrt(s["px"][ix] ** 2 + s["py"][ix] ** 2 + s["pz"][ix] ** 2)),
                   np.where(stop, 0., np.nan))
    pin = np.sqrt(s["px"][first] ** 2 + s["py"][first] ** 2 + s["pz"][first] ** 2)
    entry = kinetic(pin)
    entry_cos = s["pz"][first] / pin
    emax = case.get("energy_max_MeV", case["energy_MeV"])
    cmin = case.get("cosine_min", case["cosine"])
    if np.any(entry < case["energy_MeV"] - 1e-5) or np.any(entry > emax + 1e-5):
        raise ValueError("measured foil entry outside the configured gun energy range")
    if np.any(entry_cos < cmin - 1e-6) or np.any(entry_cos > case["cosine"] + 1e-6):
        raise ValueError("measured foil entry direction outside the configured gun range")
    time = np.where(exits, s["time"][ix], p["tstop"])
    step_count = np.bincount(events[(s["sid"] == 2) & (at < exit_at[events])], minlength=n)
    next_at = np.minimum(at + 1, len(events) - 1)
    use = (s["sid"] == 2) & (at < exit_at[events]) & (events == events[next_at]) & (at != next_at)
    chord = np.sqrt(sum((s[k][next_at] - s[k]) ** 2 for k in ("x", "y", "z")))
    chords = np.bincount(events[use], weights=chord[use], minlength=n)
    secondary_events = integer(particles["evt"], "particle event") - 1
    if np.any(secondary_events < 0) or np.any(secondary_events >= n):
        raise ValueError("secondary event outside primary population")
    direct = particles["parent_id"] == p["trk"][secondary_events]
    inside = (np.abs(particles["z"]) <= half + 2e-6) & (particles["time"] <= time[secondary_events] + 1e-6)
    hard = direct & inside & (particles["pdg"] == 11) & (particles["code"] == 31)
    gamma = direct & inside & (particles["pdg"] == 22) & (particles["code"] == 30)
    hard_energy = kinetic(particles["p"][hard], ME)
    hard_sum = np.bincount(secondary_events[hard], weights=hard_energy, minlength=n)
    rad_sum = np.bincount(secondary_events[gamma], weights=particles["p"][gamma], minlength=n)
    hard_count = np.bincount(secondary_events[hard], minlength=n)
    store = dict(event=p["evt"], track=p["trk"], entry_MeV=entry, entry_cos=entry_cos, exit_MeV=out,
        exit=exits, physical_stop=stop, other=other, exit_z_mm=np.where(exits, s["z"][ix], p["zstop"]),
        primary_end_code=p["codestop"], foil_steps=step_count, chord_sum_mm=chords,
        hard_count=hard_count, hard_MeV=hard_sum, brems_MeV=rad_sum,
        nonhard_muon_loss_MeV=entry - out - hard_sum - rad_sum,
        layer_time_ns=time - s["time"][first])
    finite = ~other
    codes, counts = np.unique(p["codestop"], return_counts=True)
    report = dict(primaries=n, pdg=pdg, exits=int(exits.sum()), physical_stops=int(stop.sum()),
        threshold_stops=int(np.sum(out[finite] < .1)), subthreshold_exits=int(np.sum(exits & (out < .1))),
        other=int(other.sum()), back_exits=int(np.sum(exits & (store["exit_z_mm"] < 0))),
        sid_values=np.unique(s["sid"]).tolist(), primary_end_codes=dict(zip(codes.astype(int).astype(str), counts.tolist())),
        mean_exit_MeV=float(np.mean(out[finite])), var_exit_MeV2=float(np.var(out[finite], ddof=1)) if finite.sum() > 1 else None,
        hard_electrons=int(hard.sum()), mean_hard_MeV=float(hard_sum.mean()),
        hard_energy_MeV_quantiles=np.quantile(hard_energy, [0, .5, 1]).tolist() if len(hard_energy) else [],
        mean_nonhard_muon_loss_MeV=float(np.mean(store["nonhard_muon_loss_MeV"][finite])),
        primary_step_codes={str(int(k)): int(v) for k, v in zip(*np.unique(s["code"], return_counts=True))},
        foil_steps_quantiles=np.quantile(step_count, [0, .5, .9, 1]).tolist(),
        exit_chord_over_projected_quantiles=np.quantile(chords[exits] / (2 * half / entry_cos[exits]), [0, .5, .9, 1]).tolist() if exits.any() else [],
        note="First primary foil exit by identity; no isstop interpretation; chords are not G4 true path, nonhard loss is not deposited energy")
    if case.get("no_msc") and np.any(np.isin(s["code"], [29, 33, 59])):
        raise ValueError("scattering steps present in a no-MSC diagnostic")
    return report, store


# Working memory is O(events + chunk rows); the byte budget covers decoded branches, not all scratch.
CHUNK_SIZE = "16 MB"
PRIMARY_FIELDS = "run evt trk pdg parent_id code pstop zstop codestop tstop".split()
STEP_FIELDS = "run evt trk pdg sid code x y z px py pz ke time".split()
SECONDARY_FIELDS = "evt parent_id pdg code z time p".split()


class _HardEnergies:
    """Exact nonnegative-energy quantiles with bounded RAM and temporary disk storage.

    Four radix passes find the two central float64 values without loading/sorting
    all secondaries or relying on a whole-file memmap's resident-page behaviour.
    The temporary file uses the standard temporary directory (TMPDIR may select
    worker scratch), costs eight bytes per hard electron, and is removed on
    success or failure by the caller.
    """
    def __init__(self, file):
        self.file = file
        self.count = 0
        self.minimum, self.maximum = np.inf, -np.inf
        self.nan = False

    def add(self, energy):
        energy = np.asarray(energy, dtype=np.float64)
        self.count += len(energy)
        if len(energy):
            self.nan |= bool(np.isnan(energy).any())
            self.minimum = min(self.minimum, float(energy.min()))
            self.maximum = max(self.maximum, float(energy.max()))
            energy.tofile(self.file)

    def quantiles(self):
        if not self.count:
            return []
        if self.nan:
            return [float("nan")] * 3  # np.quantile propagates any NaN
        ranks = np.array([(self.count - 1) // 2, self.count // 2], dtype=np.int64)
        prefixes = np.zeros(2, dtype=np.uint64)
        # Positive float64 bit patterns have the same ordering as their values.
        for shift in (48, 32, 16, 0):
            hist = np.zeros((2, 65536), dtype=np.int64)
            self.file.seek(0)
            while True:
                bits = np.fromfile(self.file, dtype=np.uint64, count=65536)
                if not len(bits):
                    break
                for i in range(2):  # two order statistics, not a per-event loop
                    selected = bits if shift == 48 else bits[(bits >> (shift + 16)) == (prefixes[i] >> (shift + 16))]
                    digit = ((selected >> shift) & 65535).astype(np.int64)
                    hist[i] += np.bincount(digit, minlength=65536)
            cumulative = np.cumsum(hist, axis=1)
            for i in range(2):
                digit = np.searchsorted(cumulative[i], ranks[i], side="right")
                if digit:
                    ranks[i] -= cumulative[i, digit - 1]
                prefixes[i] |= np.uint64(digit) << np.uint64(shift)
        middle = prefixes.view(np.float64)
        return [self.minimum, float(np.quantile(middle, .5)), self.maximum]


def _primary_tree(tree, case, step_size):
    """Only primary endpoint rows survive this first pass over tpart."""
    n, pdg = case["events"], case["pdg"]
    p, count = None, 0
    for rows in tree.iterate(PRIMARY_FIELDS, step_size=step_size, library="np"):
        primary = (rows["parent_id"] == -1) & (rows["pdg"] == pdg)
        size = int(primary.sum())
        if count + size > n or np.any((rows["parent_id"] == -1) & ~primary):
            raise ValueError("one requested primary muon per event required")
        if p is None:
            p = {k: np.empty(n, dtype=rows[k].dtype) for k in PRIMARY_FIELDS}
        for k in p:
            p[k][count:count + size] = rows[k][primary]
        count += size
    if count != n:
        raise ValueError("one requested primary muon per event required")
    order = np.argsort(p["evt"])
    p = {k: v[order] for k, v in p.items()}
    if not np.array_equal(integer(p["evt"], "primary event"), np.arange(1, n + 1)) or np.any(p["run"] != 1):
        raise ValueError("persisted primary event coverage mismatch")
    if np.any(p["code"] != 56):
        raise ValueError("unexpected primary creation code")
    return p


def census_streaming(particles, steps, case, *, step_size=CHUNK_SIZE):
    """Same census as :func:`census`, with projected, chunked ROOT-tree reads.

    Primary rows are retained per event; tpart is reread for direct secondaries
    after first-exit times are known. One last *matched primary* row connects
    chords across arbitrary chunk boundaries, including secondary-only chunks.
    No whole-event buffer is needed even for a single exceptionally long track.
    """
    n, pdg, half = case["events"], case["pdg"], case["thickness_mm"] / 2
    p = _primary_tree(particles, case, step_size)
    sentinel = np.iinfo(np.int64).max
    first, exit_at = np.full(n, sentinel), np.full(n, sentinel)
    entry_state, exit_state = None, None
    step_count, chords = np.zeros(n, np.int64), np.zeros(n)
    sid_values, code_values, code_counts = np.array([]), np.array([]), np.array([], dtype=np.int64)
    offset, previous, has_chords = 0, None, False
    for rows in steps.iterate(STEP_FIELDS, step_size=step_size, library="np"):
        events = integer(rows["evt"], "step event") - 1
        tracks = integer(rows["trk"], "step track")
        if np.any(events < 0) or np.any(events >= n) or np.any(rows["run"] != 1):
            raise ValueError("step event outside primary population")
        matched = tracks == p["trk"][events]
        s = {k: rows[k][matched] for k in STEP_FIELDS}
        events = events[matched]
        if not len(events):
            continue
        if (np.any(s["pdg"] != pdg) or np.any(np.diff(events) < 0)
                or (previous is not None and events[0] < previous["event"])):
            raise ValueError("primary step identity/order mismatch")
        if any(np.any(~np.isfinite(s[k])) for k in ("x", "y", "z", "px", "py", "pz", "ke", "time")):
            raise ValueError("nonfinite primary steps")
        if case.get("no_msc") and np.any(np.isin(s["code"], [29, 33, 59])):
            raise ValueError("scattering steps present in a no-MSC diagnostic")
        sid_values = np.union1d(sid_values, np.unique(s["sid"]))
        codes, counts = np.unique(s["code"], return_counts=True)
        code_values, inverse = np.unique(np.concatenate((code_values, codes)), return_inverse=True)
        combined = np.zeros(len(code_values), dtype=np.int64)
        np.add.at(combined, inverse, np.concatenate((code_counts, counts)))
        code_counts = combined
        if entry_state is None:
            entry_state = {k: np.zeros(n, dtype=s[k].dtype) for k in ("px", "py", "pz", "time")}
            exit_state = {k: np.zeros(n, dtype=s[k].dtype) for k in ("px", "py", "pz", "time", "z")}
        at = offset + np.arange(len(events))
        inside = np.flatnonzero(s["sid"] == 2)
        ev, ix = np.unique(events[inside], return_index=True)
        new = first[ev] == sentinel
        ev, ix = ev[new], inside[ix[new]]
        first[ev] = at[ix]
        for k in entry_state:
            entry_state[k][ev] = s[k][ix]
        outside = np.flatnonzero((s["sid"] != 2) & (at > first[events]))
        ev, ix = np.unique(events[outside], return_index=True)
        new = exit_at[ev] == sentinel
        ev, ix = ev[new], outside[ix[new]]
        exit_at[ev] = at[ix]
        for k in exit_state:
            exit_state[k][ev] = s[k][ix]
        use = (s["sid"] == 2) & (at < exit_at[events])
        np.add.at(step_count, events[use], 1)
        # Carry the last matched primary across intervening secondary-only chunks.
        xyz = {k: s[k] for k in ("x", "y", "z")}
        ce, ca, cs = events, at, s["sid"]
        if previous is not None:
            xyz = {k: np.concatenate((previous[k], s[k])) for k in xyz}
            ce = np.concatenate(([previous["event"]], events))
            ca = np.concatenate(([offset - 1], at))
            cs = np.concatenate((previous["sid"], s["sid"]))
        use = (cs[:-1] == 2) & (ca[:-1] < exit_at[ce[:-1]]) & (ce[:-1] == ce[1:])
        chord = np.sqrt(sum((xyz[k][1:] - xyz[k][:-1]) ** 2 for k in xyz))
        # Accumulate in row order for chunk-size-independent rounding.
        np.add.at(chords, ce[:-1][use], chord[use])
        has_chords |= bool(use.any())
        previous = {k: s[k][-1:].copy() for k in ("x", "y", "z", "sid")}
        previous["event"] = events[-1]
        offset += len(events)
    if np.any(first == sentinel):
        raise ValueError("primary never enters foil")
    exits = exit_at != sentinel
    defined = np.isin(p["codestop"], [14, 32]) & np.isfinite(p["tstop"]) & (p["tstop"] >= entry_state["time"])
    stop = (~exits) & defined & (p["pstop"] < 1e-6) & (np.abs(p["zstop"]) <= half + 2e-6)
    other = ~(exits | stop)
    out = np.where(exits, kinetic(np.sqrt(sum(exit_state[k] ** 2 for k in ("px", "py", "pz")))),
                   np.where(stop, 0., np.nan))
    pin = np.sqrt(sum(entry_state[k] ** 2 for k in ("px", "py", "pz")))
    entry = kinetic(pin)
    entry_cos = entry_state["pz"] / pin
    emax = case.get("energy_max_MeV", case["energy_MeV"])
    cmin = case.get("cosine_min", case["cosine"])
    if np.any(entry < case["energy_MeV"] - 1e-5) or np.any(entry > emax + 1e-5):
        raise ValueError("measured foil entry outside the configured gun energy range")
    if np.any(entry_cos < cmin - 1e-6) or np.any(entry_cos > case["cosine"] + 1e-6):
        raise ValueError("measured foil entry direction outside the configured gun range")
    time = np.where(exits, exit_state["time"], p["tstop"])
    hard_sum, rad_sum, hard_count = np.zeros(n), np.zeros(n), np.zeros(n, np.int64)
    has_gamma = False
    with TemporaryFile() as file:
        energies = _HardEnergies(file)
        for rows in particles.iterate(SECONDARY_FIELDS, step_size=step_size, library="np"):
            events = integer(rows["evt"], "particle event") - 1
            if np.any(events < 0) or np.any(events >= n):
                raise ValueError("secondary event outside primary population")
            direct = rows["parent_id"] == p["trk"][events]
            inside = (np.abs(rows["z"]) <= half + 2e-6) & (rows["time"] <= time[events] + 1e-6)
            hard = direct & inside & (rows["pdg"] == 11) & (rows["code"] == 31)
            gamma = direct & inside & (rows["pdg"] == 22) & (rows["code"] == 30)
            has_gamma |= bool(gamma.any())
            energy = kinetic(rows["p"][hard], ME)
            np.add.at(hard_sum, events[hard], energy)
            np.add.at(rad_sum, events[gamma], rows["p"][gamma])
            np.add.at(hard_count, events[hard], 1)
            energies.add(energy)
        hard_quantiles = energies.quantiles()
    # Empty weighted bincounts have integer dtype; preserve it in the saved arrays.
    if not has_chords:
        chords = chords.astype(np.int64)
    if not hard_count.any():
        hard_sum = hard_sum.astype(np.int64)
    if not has_gamma:
        rad_sum = rad_sum.astype(np.int64)
    store = dict(event=p["evt"], track=p["trk"], entry_MeV=entry, entry_cos=entry_cos, exit_MeV=out,
        exit=exits, physical_stop=stop, other=other, exit_z_mm=np.where(exits, exit_state["z"], p["zstop"]),
        primary_end_code=p["codestop"], foil_steps=step_count, chord_sum_mm=chords,
        hard_count=hard_count, hard_MeV=hard_sum, brems_MeV=rad_sum,
        nonhard_muon_loss_MeV=entry - out - hard_sum - rad_sum,
        layer_time_ns=time - entry_state["time"])
    finite = ~other
    codes, counts = np.unique(p["codestop"], return_counts=True)
    report = dict(primaries=n, pdg=pdg, exits=int(exits.sum()), physical_stops=int(stop.sum()),
        threshold_stops=int(np.sum(out[finite] < .1)), subthreshold_exits=int(np.sum(exits & (out < .1))),
        other=int(other.sum()), back_exits=int(np.sum(exits & (store["exit_z_mm"] < 0))),
        sid_values=sid_values.tolist(), primary_end_codes=dict(zip(codes.astype(int).astype(str), counts.tolist())),
        mean_exit_MeV=float(np.mean(out[finite])), var_exit_MeV2=float(np.var(out[finite], ddof=1)) if finite.sum() > 1 else None,
        hard_electrons=int(hard_count.sum()), mean_hard_MeV=float(hard_sum.mean()),
        hard_energy_MeV_quantiles=hard_quantiles,
        mean_nonhard_muon_loss_MeV=float(np.mean(store["nonhard_muon_loss_MeV"][finite])),
        primary_step_codes={str(int(k)): int(v) for k, v in zip(code_values, code_counts)},
        foil_steps_quantiles=np.quantile(step_count, [0, .5, .9, 1]).tolist(),
        exit_chord_over_projected_quantiles=np.quantile(chords[exits] / (2 * half / entry_cos[exits]), [0, .5, .9, 1]).tolist() if exits.any() else [],
        note="First primary foil exit by identity; no isstop interpretation; chords are not G4 true path, nonhard loss is not deposited energy")
    return report, store


def _local(path, directory):
    """A receipt path, or the same file name inside `directory` when the run directory was moved (grid copy-back)."""
    return path if Path(path).exists() else str(Path(directory) / Path(path).name)


def audit(directory, *, step_size=CHUNK_SIZE):
    """Verify the receipt and stream highest-cycle trees within uproot's chunk budget.

    ``step_size`` is an uproot byte-budget string or positive row count. The
    returned per-primary arrays remain resident; persisted step rows do not.
    """
    directory = Path(directory)
    receipt = json.loads((directory / "receipt.json").read_text())
    if not receipt["complete"] or macro_errors((directory / "job.log").read_text()):
        raise ValueError("completed G4 receipt and error-free macro execution required")
    for item in receipt["artifacts"] + [receipt["output"]]:
        item["path"] = _local(item["path"], directory)
        verify_hashes([item], [_identity(item["path"])], "G4 artifact")
    with uproot.open(receipt["output"]["path"], array_cache=None) as root:
        particles, pc = highest(root, "readback/tpart", arrays=False)
        steps, sc = highest(root, "readback/ntstepper", arrays=False)
        report, store = census_streaming(particles, steps, receipt["case"], step_size=step_size)
    report.update(trees=dict(particles=pc, steps=sc), case=receipt["case"], wall_s=receipt["wall_s"])
    return report, store, receipt


def table_audit(export, reference):
    export, reference = Path(export), Path(reference)
    ref = json.loads((reference / "manifest.json").read_text())
    for item in ref["artifacts"]:
        verify_hashes([item], [_identity(item["path"])], "reference table")
    refdir = Path(next(p["path"] for p in ref["artifacts"] if Path(p["path"]).name == "couple.dat")).parent
    export_receipt = json.loads((export / "receipt.json").read_text())
    if not export_receipt["complete"] or macro_errors((export / "job.log").read_text()):
        raise ValueError("completed error-free table export required")
    for item in export_receipt["artifacts"] + export_receipt["tables"]:
        verify_hashes([item], [_identity(item["path"])], "slab table export")
    slab_recorded = {str(Path(p["path"]).resolve()) for p in export_receipt["tables"]}
    ref_recorded = {str(Path(p["path"]).resolve()) for p in ref["artifacts"]}
    report = {}
    for particle in ("mu-", "mu+"):
        slab, si, paths = pt.load_target(export / "tables", material="G4_Al", particle=particle)
        target, ti, target_paths = pt.load_target(refdir, particle=particle)
        if not {str(p.resolve()) for p in paths} <= slab_recorded:
            raise ValueError("slab manifest omits consumed tables")
        reference_bound = {str(p.resolve()) for p in target_paths} <= ref_recorded
        if particle == "mu-" and not reference_bound:
            raise ValueError("reference manifest omits consumed negative-muon tables")
        grid = np.geomspace(.1, 40., 10001)
        ratios = {name: pt.interpolate(slab[name], grid) / pt.interpolate(target[name], grid) - 1 for name in pt.TABLES}
        differences = {name: float(np.max(np.abs(ratios[name]))) for name in pt.TABLES}
        equal_cut = si["energy_cuts_MeV"] == ti["energy_cuts_MeV"] and si["range_cuts_mm"] == ti["range_cuts_mm"]
        report[particle] = dict(slab=si, target=ti, max_relative=differences, cuts_identical=equal_cut,
            max_relative_at_MeV={name: float(grid[np.argmax(np.abs(ratios[name]))]) for name in pt.TABLES},
            relative_above_point2={name: float(np.max(np.abs(ratios[name][grid >= .2]))) for name in pt.TABLES},
            relative_above_1={name: float(np.max(np.abs(ratios[name][grid >= 1.]))) for name in pt.TABLES},
            reference_manifest_bound=reference_bound,
            compatible=reference_bound and equal_cut and si["density_g_cm3"] == ti["density_g_cm3"] and max(differences.values()) < 1e-5,
            table_identities=[_identity(p) for p in paths + target_paths])
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--table-reference")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    report, store, receipt = audit(a.run)
    if a.table_reference:
        report["material_check"] = table_audit(a.run, a.table_reference)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "primary.npz", **store)
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (out / "manifest.json").write_text(json.dumps(dict(receipt=_identity(Path(a.run) / "receipt.json"),
        sources=[_identity(Path(__file__)), _identity(Path(pt.__file__))], output=_identity(out / "primary.npz")), indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "material_check"}, indent=2, allow_nan=False))
    if a.table_reference:
        print(json.dumps({k: {x: v[x] for x in ("max_relative", "compatible", "cuts_identical")} for k, v in report["material_check"].items()}))


if __name__ == "__main__":
    main()
