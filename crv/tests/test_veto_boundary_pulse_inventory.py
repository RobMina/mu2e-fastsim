"""Synthetic local-boundary admission only: never run response, refits, or finder."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import uproot

from crvfastsim.reco import coinc, pulsefit, waveform
from crvfastsim.response import sipm
from crvfastsim.veto import boundary_pulse_inventory as B

I, M, A = B.I, B.M, B.A
sha256 = M.S.C.sha256


def forbidden(*args, **kwargs):
    pytest.fail("local inventory attempted finder, response, fit, raw access, or verifier rerun")


@pytest.fixture(autouse=True)
def no_physics(monkeypatch):
    for module, names in (
        (coinc, ("find_event",)), (pulsefit, ("reconstruct", "reconstruct_full")),
        (waveform, ("synthesize",)), (sipm, ("simulate",)),
        (I.P, ("run",)), (M.C, ("recover", "prepare_pulses")),
        (I.Q, ("run", "admission")), (I.Q.R, ("run_arm",)),
        (I.Q.R.Q, ("run_arm",)), (I.D, ("run", "run_arm")),
        (I.Q.M.K, ("make_photons",)), (I.Q.M.P, ("read_input",)),
        (I.SR, ("verify",)), (I.V, ("verify",)), (uproot, ("open",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)


def make_domain(event_ids=(7, 9), input_sha="a" * 64):
    events = pd.DataFrame({"run": 1430, "subrun": 23, "event": list(event_ids),
                           "pbtime": [-224.46] * len(event_ids)})
    p = pd.DataFrame({
        "wf": [10, 10, 11, 12], "seq": [0, 1, 0, 0], "chan": [4, 4, 4, 8],
        "event": event_ids[0], "time": np.array([500.125, 501.125, 502.125, 503.125]),
        "pe": np.array([0., 10.123456789, 12., 8.]),
        "failed": [False, True, False, False], "zerondf": [False, True, False, False],
        "channel_ignored": False,
    })
    w = pd.DataFrame({"wf": [10, 11, 12], "event": event_ids[0], "chan": [4, 4, 8],
                      "starttdc": [90, 110, 80]})
    c = pd.DataFrame({"event": event_ids[0], "chan": [4, 8], "timeoffset": [2.375, -1.125],
                      "w": -float(np.float32(events.pbtime.iloc[0])), "status": [0, 0]})
    p["time_relative"] = p.time + p.chan.map(c.set_index("chan").timeoffset)
    p["time_reco"] = p.time_relative - float(np.float32(events.pbtime.iloc[0]))
    return SimpleNamespace(p=p, w=w, c=c, e=events, sha=input_sha)


@pytest.fixture
def domain():
    return make_domain()


def inspect(d):
    return B.inspect(d.p, d.w, d.c, d.e, d.sha)


def test_exact_saved_links_full_identity_quiet_frame_and_no_mutation(domain):
    d = domain
    before = [t.copy(deep=True) for t in (d.p, d.w, d.c, d.e)]
    # Waveform table order and pandas row labels are not pulse order contracts.
    d.w = d.w.iloc[::-1]
    d.p.index = [40, 20, 60, 10]
    out = inspect(d)
    assert out["pulse_rows"] == 4 and out["waveform_rows"] == 3
    assert out["original_relative_and_forward_reco_bits_exact"]
    assert out["native_structural_order_exact"] and out["waveform_start_recovered_by_saved_link"]
    assert not out["full_registry_admitted"] and not out["nonempty_membership_established"]
    assert out["frames"] == [
        dict(run=1430, subrun=23, event=7, input_sha256=d.sha,
             saved_pulses=4, ignored_pulses=0, nonlive_bit_pulses=0),
        dict(run=1430, subrun=23, event=9, input_sha256=d.sha,
             saved_pulses=0, ignored_pulses=0, nonlive_bit_pulses=0),
    ]
    assert "starttdc" not in d.p and "frame_id" not in d.e
    for actual, expected in zip((d.p.reset_index(drop=True), d.w.iloc[::-1], d.c, d.e), before):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_typed_empty_population_keeps_quiet_census(domain):
    domain.p, domain.w, domain.c = domain.p.iloc[:0], domain.w.iloc[:0], domain.c.iloc[:0]
    out = inspect(domain)
    assert out["pulse_rows"] == out["waveform_rows"] == 0
    assert len(out["frames"]) == 2
    assert all(r["saved_pulses"] == r["ignored_pulses"] == r["nonlive_bit_pulses"] == 0
               for r in out["frames"])


def test_ignored_nonlive_secondary_and_failed_rows_are_retained(domain):
    domain.c["status"] = [3, 4]  # ignored+dead, then nonlive without ignore bit
    domain.p["channel_ignored"] = domain.p.chan.eq(4)
    out = inspect(domain)
    assert out["frames"][0]["saved_pulses"] == 4
    assert out["frames"][0]["ignored_pulses"] == 3
    assert out["frames"][0]["nonlive_bit_pulses"] == 4


@pytest.mark.parametrize("case", ["missing", "wf", "event", "chan"])
def test_start_recovery_requires_exact_waveform_event_channel_not_nearest(domain, case):
    if case == "missing":
        domain.w = domain.w.iloc[1:]
    else:
        domain.w.loc[0, case] = {"wf": 99, "event": 9, "chan": 5}[case]
    # Another saved waveform on channel 4 must never substitute for wf 10.
    with pytest.raises(ValueError, match="waveform link missing"):
        inspect(domain)


@pytest.mark.parametrize("case", ["wf", "structure"])
def test_waveform_mapping_must_be_bijective(domain, case):
    if case == "wf":
        domain.w.loc[1, "wf"] = 10
    else:
        domain.w.loc[1, "starttdc"] = 90
    with pytest.raises(ValueError, match="not bijective"):
        inspect(domain)


def test_duplicate_structural_pulses_are_not_secondary_pulses(domain):
    domain.p.loc[1, "seq"] = 0
    with pytest.raises(ValueError, match="duplicate.*structural"):
        inspect(domain)


@pytest.mark.parametrize("case", ["stored", "native", "starttdc"])
def test_native_structural_stored_order_is_checked_without_sorting_repair(domain, case):
    if case == "stored":
        domain.p = domain.p.iloc[::-1].reset_index(drop=True)
    elif case == "native":
        domain.p["wf"] = domain.p.wf.map({10: 12, 11: 11, 12: 10})
        domain.w["wf"] = domain.w.wf.map({10: 12, 11: 11, 12: 10})
    else:
        domain.w.loc[0, "starttdc"] = 120
    with pytest.raises(ValueError, match="order"):
        inspect(domain)


@pytest.mark.parametrize("table,column,value", [
    ("p", "wf", True), ("p", "seq", -1), ("w", "starttdc", 90.),
    ("p", "chan", 5120), ("e", "run", 2**32), ("e", "event", 7.),
])
def test_identity_fields_are_bounded_integers(domain, table, column, value):
    getattr(domain, table)[column] = value
    with pytest.raises(ValueError, match="integer|ambiguous"):
        inspect(domain)


@pytest.mark.parametrize("column", ["failed", "zerondf", "channel_ignored"])
@pytest.mark.parametrize("representation", ["integer", "nullable"])
def test_flags_must_be_real_nonnullable_booleans(domain, column, representation):
    domain.p[column] = 0 if representation == "integer" else pd.array([False, False, False, pd.NA], dtype="boolean")
    with pytest.raises(ValueError, match="invalid boundary flag"):
        inspect(domain)


@pytest.mark.parametrize("column", ["time", "time_relative", "time_reco", "pe"])
@pytest.mark.parametrize("case", ["float32", "nan"])
def test_time_and_pe_require_finite_float64(domain, column, case):
    if case == "float32":
        domain.p[column] = domain.p[column].astype(np.float32)
    else:
        domain.p.loc[0, column] = np.nan
    with pytest.raises(ValueError, match="floating field"):
        inspect(domain)


@pytest.mark.parametrize("case", ["pulse_field", "waveform_field", "invented_start", "negative_pe"])
def test_required_legacy_schema_and_nonnegative_pe(domain, case):
    if case == "pulse_field":
        domain.p = domain.p.drop(columns="time_relative")
    elif case == "waveform_field":
        domain.w = domain.w.drop(columns="starttdc")
    elif case == "invented_start":
        domain.p["starttdc"] = 90
    else:
        domain.p.loc[0, "pe"] = -1.
    with pytest.raises(ValueError, match="missing|unexpectedly present|negative"):
        inspect(domain)


@pytest.mark.parametrize("case", ["ulp", "negative_zero", "omit_offset"])
def test_original_relative_time_is_exact_peak_plus_saved_offset(domain, case):
    if case == "ulp":
        domain.p.loc[0, "time_relative"] = np.nextafter(domain.p.time_relative.iloc[0], np.inf)
    elif case == "negative_zero":
        domain.p.loc[0, "time"] = -domain.c.timeoffset.iloc[0]
        domain.p.loc[0, "time_relative"] = -0.
        assert domain.p.time_relative.iloc[0] == domain.p.time.iloc[0] + domain.c.timeoffset.iloc[0]
    else:
        domain.p["time_relative"] = domain.p.time
    domain.p["time_reco"] = domain.p.time_relative - float(np.float32(domain.e.pbtime.iloc[0]))
    with pytest.raises(ValueError, match="relative time bits"):
        inspect(domain)


@pytest.mark.parametrize("case", ["ulp", "negative_zero", "unrounded_pb"])
def test_forward_reco_shift_requires_rounded_bunch_time_and_exact_bits(domain, case):
    pb = float(np.float32(domain.e.pbtime.iloc[0]))
    if case == "ulp":
        domain.p.loc[0, "time_reco"] = np.nextafter(domain.p.time_reco.iloc[0], np.inf)
    elif case == "negative_zero":
        domain.p["time"] = pb
        domain.c["timeoffset"] = 0.
        domain.p["time_relative"] = pb
        domain.p["time_reco"] = 0.
        domain.p.loc[0, "time_reco"] = -0.
        assert np.array_equal(domain.p.time_reco, domain.p.time_relative - pb)
    else:
        domain.p["time_reco"] = domain.p.time_relative - domain.e.pbtime.iloc[0]
    with pytest.raises(ValueError, match="reco time bits"):
        inspect(domain)


@pytest.mark.parametrize("case", ["missing", "duplicate", "float_status", "negative_status",
                                   "status_overflow", "ignored", "shift", "nan_offset"])
def test_authoritative_conditions_join_status_and_bunch_shift(domain, case):
    if case == "missing":
        domain.c = domain.c.iloc[1:]
    elif case == "duplicate":
        domain.c = pd.concat([domain.c, domain.c.iloc[:1]], ignore_index=True)
    elif case == "float_status":
        domain.c["status"] = 0.
    elif case in ("negative_status", "status_overflow", "ignored"):
        domain.c.loc[0, "status"] = {"negative_status": -1, "status_overflow": 2**16, "ignored": 2}[case]
    elif case == "shift":
        domain.c["w"] = -domain.e.pbtime.iloc[0]  # unrounded is not the saved shift
    else:
        domain.c.loc[0, "timeoffset"] = np.nan
    with pytest.raises(ValueError, match="conditions|integer status|ignored flag|bunch shift"):
        inspect(domain)


@pytest.mark.parametrize("case", ["duplicate_event", "multiple_runs", "multiple_subruns", "unknown_event"])
def test_full_event_census_cannot_be_ambiguous_or_guessed(domain, case):
    if case == "unknown_event":
        domain.p["event"] = 99
    else:
        column, value = {"duplicate_event": ("event", 7), "multiple_runs": ("run", 1431),
                         "multiple_subruns": ("subrun", 24)}[case]
        domain.e.loc[1, column] = value
    with pytest.raises(ValueError, match="census"):
        inspect(domain)


@pytest.mark.parametrize("case", ["p", "w", "c", "e", "per_frame"])
def test_inspection_caps(domain, case):
    if case == "per_frame":
        domain.p = domain.p.iloc[np.zeros(513, dtype=np.int64)].reset_index(drop=True)
        domain.p["seq"] = np.arange(513, dtype=np.int64)
    else:
        setattr(domain, case, pd.DataFrame(index=range({"p": 2049, "w": 2049, "c": 20481, "e": 4}[case])))
    with pytest.raises(ValueError, match="cap"):
        inspect(domain)


class LocalSavedControls:
    """Tiny two-block/three-event fixture; all JSON/Parquet pins and end-of-run rehashes are real."""

    def __init__(self, root, monkeypatch):
        evidence = root / "evidence"
        self.downstream, self.boundary = evidence / "downstream", evidence / "boundary"
        self.downstream.mkdir(parents=True)
        self.boundary.mkdir()
        self.verified, self.out = evidence / "verified.json", root / "inventory"
        self.d = dict(schema="thread-d-saved-photon-downstream-v1", seed=37, blocks=[], output_sha256={})
        self.b = dict(blocks=[], output_sha256={})
        self.v = dict(schema="thread-d-downstream-report-v1", original_inputs_verified=True)
        self.payloads, self.reads = set(), []
        for number, events in enumerate(((7, 9), (7,))):
            d = make_domain(events, ("a" if number == 0 else "b") * 64)
            d.e["run"] += number  # event 7 in a distinct source/run is not the same frame
            self.persist(self.boundary, self.b, f"block{number}_events.parquet", d.e.drop(columns=["run", "subrun"]))
            self.persist(self.downstream, self.d, f"block{number}_stock_pulses.parquet", d.e[B.IDS])
            self.persist(self.downstream, self.d, f"block{number}_conditions.parquet", d.c)
            models = {f"{policy}_thermal{t}": [{"arm": arm} for arm in ("step_only", "arrival_only", "stock_timing")]
                      for policy in ("legacy_round", "fixed_integer") for t in (0, 1)}
            self.d["blocks"].append(dict(block=number, targets=list(events), models=models))
            self.b["blocks"].append(dict(block=number, targets=list(events),
                                        input_sha256={f"/unavailable/remote/input{number}/mixed.root": d.sha}))
            tags = [f"block{number}_stock_adc_refit", *[
                f"block{number}_{model}_{arm['arm']}" for model, arms in models.items() for arm in arms]]
            for kind, table in (("pulses", d.p), ("waveforms", d.w)):
                source = None
                for tag in tags:
                    name = f"{tag}_{kind}.parquet"
                    if source is None:
                        self.persist(self.downstream, self.d, name, table)
                        source = self.downstream / name
                    else:
                        path = self.downstream / name
                        os.link(source, path)
                        self.d["output_sha256"][name] = sha256(path)
                        self.payloads.add(path)
        self.commit()
        read = pd.read_parquet

        def bounded_read(path, *args, **kwargs):
            path = Path(path)
            assert path in self.payloads, f"unexpected decode: {path}"
            self.reads.append(path)
            return read(path, *args, **kwargs)

        monkeypatch.setattr(pd, "read_parquet", bounded_read)

    def persist(self, root, report, name, table):
        path = root / name
        table.to_parquet(path, index=False)
        report["output_sha256"][name] = sha256(path)
        self.payloads.add(path)

    def commit(self):
        def write(path, obj):
            path.write_bytes(A._canonical(obj))
            return sha256(path)
        bsha = write(self.boundary / "summary.json", self.b)
        self.d["input_summary_sha256"] = bsha
        dsha = write(self.downstream / "summary.json", self.d)
        self.v.update(boundary_summary_sha256=bsha, input_summary_sha256=dsha)
        vsha = write(self.verified, self.v)
        self.pins = dict(expected_downstream_sha256=dsha, expected_boundary_sha256=bsha,
                         expected_verified_sha256=vsha)

    def run(self, **pins):
        return B.run(self.downstream, self.boundary, self.verified, self.out, **(self.pins | pins))


@pytest.fixture
def saved(tmp_path, monkeypatch):
    return LocalSavedControls(tmp_path, monkeypatch)


def test_runner_persists_26_bound_populations_and_exact_code_snapshots(saved):
    out = saved.run()
    persisted = json.loads(saved.out.read_text())
    snapshots = persisted.pop("code_snapshots")
    assert persisted.pop("output_cap_bytes") == I.CAP
    assert persisted.pop("publication") == "exclusive JSON file; no completion protocol or crash-durability claim"
    assert out == persisted | {"persisted_path_bytes": out["persisted_path_bytes"]}
    assert out["schema"] == B.SCHEMA and out["unique_frames"] == 3
    assert len(out["records"]) == len({r["tag"] for r in out["records"]}) == 26
    assert sum(r["stock_adc_refit"] for r in out["records"]) == 2
    for r in out["records"]:
        assert r["pulse_rows"] == 4 and r["waveform_rows"] == 3
        assert r["frames"][0]["input_sha256"] == ("a" if r["tag"].startswith("block0_") else "b") * 64
        assert r["original_relative_and_forward_reco_bits_exact"] and r["native_structural_order_exact"]
        assert not r["full_registry_admitted"] and not r["nonempty_membership_established"]
    for field in ("regenerated_finder", "regenerated_response", "regenerated_fit", "inherited_raw_admission_repeated",
                  "full_registry_admitted", "execution_admitted", "timing_closure"):
        assert out[field] is False
    assert out["F"] is out["finite_library_uncertainty"] is None
    assert set(saved.reads) == saved.payloads and len(saved.reads) == 58
    assert len(out["inputs"]) == 61  # 52 population tables, six identity/condition tables, three JSON pins
    for path, digest in (out["inputs"] | out["code_sha256"]).items():
        assert sha256(path) == digest
    assert saved.out.stat().st_size == out["persisted_path_bytes"] <= I.CAP
    assert set(snapshots) == set(out["code_sha256"])
    for path, text in snapshots.items():
        assert text == Path(path).read_text()


@pytest.mark.parametrize("name", ["downstream", "boundary", "verified"])
def test_runner_requires_all_three_external_pins_before_decode(saved, name):
    with pytest.raises(ValueError, match="changed bound input"):
        saved.run(**{f"expected_{name}_sha256": "0" * 64})
    assert not saved.reads and not saved.out.exists()


def test_runner_rejects_pinned_report_without_original_admission(saved):
    saved.v["original_inputs_verified"] = False
    saved.commit()
    with pytest.raises(ValueError, match="provenance"):
        saved.run()
    assert not saved.reads and not saved.out.exists()


@pytest.mark.parametrize("case", ["missing_binding", "changed_payload"])
def test_runner_requires_actual_payload_binding(saved, case):
    name = "block0_stock_adc_refit_waveforms.parquet"
    if case == "missing_binding":
        del saved.d["output_sha256"][name]
        saved.commit()
    else:
        with (saved.downstream / name).open("ab") as stream:
            stream.write(b"changed saved payload")
    with pytest.raises(ValueError, match="binding|changed bound input"):
        saved.run()
    assert not saved.out.exists()


def test_runner_late_rehash_rejects_mutation_before_publication(saved, monkeypatch):
    recheck = M.check_bindings

    def mutate_then_check(bindings):
        assert not saved.out.exists()
        with (saved.boundary / "block0_events.parquet").open("ab") as stream:
            stream.write(b"late mutation")
        return recheck(bindings)

    monkeypatch.setattr(M, "check_bindings", mutate_then_check)
    with pytest.raises(ValueError, match="changed bound input"):
        saved.run()
    assert not saved.out.exists()


def test_runner_never_overwrites_completed_inventory(saved):
    saved.run()
    before = saved.out.read_bytes()
    reads = list(saved.reads)
    with pytest.raises(ValueError, match="choose a new output directory"):
        saved.run()
    assert before == saved.out.read_bytes()
    assert reads == saved.reads


def test_runner_caps_total_frames_not_just_each_block(saved):
    name = "block1_events.parquet"
    events = make_domain((7, 11)).e.assign(run=1431)
    saved.persist(saved.boundary, saved.b, name, events)
    for report in (saved.d, saved.b):
        report["blocks"][1]["targets"] = [7, 11]
    saved.commit()
    with pytest.raises(ValueError, match="frame|schedule|cap"):
        saved.run()


def test_input_digest_is_not_an_arbitrary_label(domain):
    domain.sha = "not-a-sha256"
    with pytest.raises(ValueError, match="SHA256|digest|sha256"):
        inspect(domain)


@pytest.mark.parametrize("value", [np.nan, np.inf, 1e300])
def test_quiet_census_bunch_time_is_validated(domain, value):
    domain.e.loc[1, "pbtime"] = value
    with pytest.raises(ValueError, match="census bunch time"):
        inspect(domain)


def test_census_restores_full_keys_from_bound_rows_not_constants(domain):
    summary = domain.e.drop(columns=["run", "subrun"])
    restored = B.event_census(summary, domain.e[B.IDS])
    pd.testing.assert_frame_equal(restored[domain.e.columns], domain.e)


@pytest.mark.parametrize("case", ["missing_quiet", "ambiguous", "wrong_run"])
def test_census_does_not_guess_missing_or_ambiguous_keys(domain, case):
    summary = domain.e.copy()
    stock = domain.e[B.IDS].copy()
    if case == "missing_quiet":
        stock = stock.iloc[:1]
    elif case == "ambiguous":
        stock = pd.concat([stock, stock.iloc[:1].assign(run=1431)], ignore_index=True)
    else:
        stock["run"] = 1431
    with pytest.raises(ValueError, match="identity"):
        B.event_census(summary, stock)
