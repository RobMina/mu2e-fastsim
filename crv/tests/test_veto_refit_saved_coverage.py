"""Synthetic saved certificates, real Parquets/receipts, and no prior verifier execution."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import refit_saved_coverage as C
from crvfastsim.veto.card import VetoCard
from test_veto_saved_qualified_report import reseal

R, N, M, V, G, A = C.R, C.N, C.M, C.V, C.G, C.A


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_bytes(A._canonical(json.loads(json.dumps(value))))
    return sha(path)


def empty(types):
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in types.items()})


def typed(table, types):
    return table[list(types)].astype(types)


def forbidden(*args, **kwargs):
    pytest.fail("coverage attempted physics, prior verification, admission, raw or selected-track access")


class SavedCoverage:
    """Manually asserted synthetic authority is not evidence about retained physics."""

    def __init__(self, root, monkeypatch):
        self.root, self.monkeypatch = root, monkeypatch
        self.sources = root / "sources"
        self.sources.mkdir()
        self.membership, self.directory = root / "membership", root / "refits"
        self.verified = self.sources / "refit_verified.json"
        self.mv_path = self.sources / "membership_verified.json"
        self.reference_path, self.pair_path = self.sources / "reference.json", self.sources / "paired.json"
        self.selection = root / "unread_selection" / "selected.parquet"
        self.selection.parent.mkdir()
        self.selection.write_bytes(b"not a readable selection; provenance only")
        self.registry = N.F.freeze(pd.DataFrame(dict(
            input_sha256=np.repeat([N.INPUTS[0], N.INPUTS[1], "f" * 64], 3000),
            run=7, subrun=2, event=np.tile(np.arange(1, 3001, dtype=np.int64), 3),
            pbtime=-224.46, intensity=1.)))
        monkeypatch.setattr(N, "REGISTRY_SHA", N.F.digest(self.registry))
        self.card = VetoCard(dt_lo=0., dt_hi=.5, min_pe=10., min_layers=3,
                             gate_start_lo=0., gate_end_hi=20.)
        self.reference = dict(card=asdict(self.card) | {"sha": self.card.sha()}, selected=1929421,
                              pmin=85, pmax=200, F={"grid_ns": .25})
        reference_sha = dump(self.reference_path, self.reference)
        self.pair = dict(reference=str(self.reference_path), reference_sha256=reference_sha,
                         selection=str(self.selection), selection_sha256="d" * 64)
        pair_sha = dump(self.pair_path, self.pair)
        self.grid_meta = dict(C.GRID_COUNTS, contract=G.CONTRACT, grid_ns=.25, card_sha=self.card.sha(),
                              scope="synthetic retained-count metadata; not a measurement")
        self.gcode = {str(p): sha(p) for p in M.code_paths()}
        self.rcode = {str(p): sha(p) for p in R.generation_paths()}
        self.summary = dict(schema=M.SCHEMA, scope=M.SCOPE,
            hashes=dict(expected_pilot_sha256="a" * 64, expected_verified_sha256="b" * 64,
                        expected_paired_sha256=pair_sha),
            inputs={str(self.pair_path): pair_sha, str(self.reference_path): reference_sha,
                    str(self.selection): self.pair["selection_sha256"]}, code_sha256=self.gcode,
            registry_sha256=N.REGISTRY_SHA, unique_frames=3, ledger_rows=0,
            seeds=list(M.S.SEEDS), arms=list(M.S.ARMS), counts={}, comparisons={}, grid=self.grid_meta,
            finder_config=asdict(M.C.finder_config()), coincidence_limits=asdict(N.P.Limits()),
            ignore_channels=True, ignored_pulse_policy=V.IGNORED_POLICY, output_cap_bytes=M.OUTPUT_CAP,
            memory_before={}, memory_after={}, coincidence_coverage_seconds=0.,
            **dict.fromkeys(V.FALSE_FIELDS, False), **dict.fromkeys(V.NULL_FIELDS))
        assert set(self.summary) == V.SUMMARY_KEYS
        edges = np.arange(4043, dtype=float) * .25
        weight = np.zeros(4042)
        weight[[4, 5, 8, 9]] = [.125, .375, .25, .25]
        tables = {name: empty(types) for name, types in V.TABLE_TYPES.items()}
        tables.update(registry=self.registry, selected=self.registry.iloc[:3].copy(),
                      grid=pd.DataFrame(dict(left=edges[:-1], right=edges[1:], weight=weight)))
        self.batch(self.membership, self.summary, tables, self.gcode,
                   dict(registry_sha256=N.REGISTRY_SHA, batch_identity={"contract": M.SCHEMA, **self.summary["hashes"]},
                        arms=list(M.S.ARMS), seeds=list(M.S.SEEDS)))
        self.mv = dict(schema=V.SCHEMA, summary_sha256=sha(self.membership / "summary.json"),
            registry_sha256=N.REGISTRY_SHA, registry_frames=9000, grid=dict(self.grid_meta), code_sha256=sha(V.__file__),
            **dict.fromkeys(("saved_artifact_protocol_exact", "source_transactions_exact",
                            "qualified_frame_census_exact", "frozen_grid_exact"), True),
            **dict.fromkeys(V.FALSE_FIELDS | {"regenerated_finder", "cluster_scalars_independently_recomputed"}, False),
            F=None, finite_library_uncertainty=None)
        dump(self.mv_path, self.mv)
        roles = {}
        for role, schema in R.ROLES.items():
            path = {"membership": self.membership / "summary.json", "membership_verified": self.mv_path}.get(role)
            if path is None:
                path = self.sources / (role + ".json")
                dump(path, {"schema": schema, "synthetic": True})
            roles[role] = {"path": str(path), "sha256": sha(path)}
        self.frames, self.clusters = [], []
        for number in (0, 1):
            f = self.registry.loc[self.registry.input_sha256.eq(N.INPUTS[number]) &
                                  self.registry.event.isin(N.TARGETS[number])].reset_index(drop=True)
            f = f.assign(pulses=[25] if number == 0 else [31, 31],
                         clusters=[1] if number == 0 else [1, 0], members=[4] if number == 0 else [4, 0])
            self.frames.append(typed(f, R.FRAME_TYPES))
            c = pd.DataFrame({name: np.zeros(1, dtype=dtype) for name, dtype in V.CLUSTERS.items()})
            for key in G.KEYS:
                c[key] = f[key].iloc[:1].to_numpy()
            c = c.assign(idx=0, avgtime=1. if number == 0 else 2., start=0. if number == 0 else 1.5,
                         end=1.25 if number == 0 else 2.25, pe=20., nLayers=3, sectortype=0)
            self.clusters.append(typed(c, V.CLUSTERS))
        records = self.records()
        self.refit = dict(schema=N.SCHEMA, scope=N.SCOPE,
            inputs={r["path"]: r["sha256"] for r in roles.values()} | M.transaction_bindings(
                self.membership, json.loads((self.membership / A.COMPLETE).read_text())), code_sha256=self.rcode,
            registry_sha256=N.REGISTRY_SHA, registry_frames=9000, unique_frames=3,
            saved_pulses=87, saved_waveforms=80, finder_calls=2, records=records,
            finder_config=asdict(M.C.finder_config()), coincidence_limits=asdict(N.P.Limits()),
            nonempty_real_refit_membership=True, full_registry_admitted=True, stock_refit_pulse_bridge_exact=True,
            output_cap_bytes=N.CAP, seconds=0., memory={"peak_rss_bytes_cumulative": 0, "current_rss_bytes": None},
            **dict.fromkeys(R.FALSE_FIELDS, False), **dict.fromkeys(R.NULL_FIELDS))
        assert set(self.refit) == R.SUMMARY_KEYS
        tables = {"registry": self.registry, "calibration": pd.DataFrame({"synthetic": [True]})}
        for number in (0, 1):
            tables.update({f"block{number}_frames": self.frames[number], f"block{number}_clusters": self.clusters[number]})
            for name in ("pulses", "members", "saved_bridge", "saved_waveforms", "saved_pulses"):
                tables[f"block{number}_{name}"] = pd.DataFrame({"synthetic_unread_payload": [True]})
        self.batch(self.directory, self.refit, tables, self.rcode,
            dict(registry_sha256=N.REGISTRY_SHA, batch_identity={"contract": N.SCHEMA,
                "downstream_sha256": roles["downstream"]["sha256"], "membership_sha256": roles["membership"]["sha256"],
                "seed_role": "protocol-only label; deterministic saved ADC refit; no random draw"},
                 arms=["saved_adc_refit"], seeds=[0]))
        code = {str(p): sha(p) for p in [Path(R.__file__), *R.generation_paths()]}
        self.prior = dict(schema=R.SCHEMA, scope=R.SCOPE, summary_sha256=sha(self.directory / "summary.json"),
            registry_sha256=N.REGISTRY_SHA, registry_frames=9000, unique_frames=3,
            code_sha256=code, code_snapshots={p: Path(p).read_text() for p in code}, source_reports=roles,
            counts=[dict(block=r["block"], **r["counts"]) for r in records],
            frames=pd.concat(self.frames).to_dict("records"), nonempty_real_refit_membership=True,
            **dict.fromkeys(("saved_artifact_protocol_exact", "inherited_membership_transaction_exact", "source_bridge_rechecked",
                "source_tables_exact", "canonical_pulses_exact", "native_member_structure_exact",
                "positive_pe_side_counts_exact", "qualified_frame_census_exact"), True),
            **dict.fromkeys(R.FALSE_FIELDS | {"regenerated_finder", "cluster_scalars_independently_recomputed",
                                            "native_member_order_independently_recomputed"}, False),
            F=None, finite_library_uncertainty=None)
        dump(self.verified, self.prior)
        self.args = dict(directory=self.directory, verified=self.verified, out=root / "coverage.json",
            expected_summary_sha256=sha(self.directory / "summary.json"), expected_verified_sha256=sha(self.verified),
            expected_complete_sha256=sha(self.directory / A.COMPLETE))

    def batch(self, directory, summary, tables, code, context):
        names = ["summary.json", *(name + ".parquet" for name in tables), *(f"code{i}.txt" for i in range(len(code)))]
        with A.BatchArtifacts(directory, max_bytes=N.CAP, context=context, expected_artifacts=names) as tx:
            tx.write_json("summary.json", summary)
            for name, table in tables.items():
                tx.write_parquet(name + ".parquet", table)
            for i, path in enumerate(code):
                with tx.open_artifact(f"code{i}.txt") as stream:
                    stream.write(Path(path).read_bytes())

    def records(self):
        return [dict(block=i, frames=f.to_dict("records"), counts=dict(frames=len(f), pulses=int(f.pulses.sum()),
                    clusters=len(c), members=int(f.members.sum()))) for i, (f, c) in enumerate(zip(self.frames, self.clusters))]

    def repin(self, grid=False):
        if grid:
            dump(self.membership / "summary.json", self.summary)
            self.recontext(self.membership, {"contract": M.SCHEMA, **self.summary["hashes"]})
            self.mv["summary_sha256"] = sha(self.membership / "summary.json")
            dump(self.mv_path, self.mv)
            for role, path in (("membership", self.membership / "summary.json"), ("membership_verified", self.mv_path)):
                self.prior["source_reports"][role]["sha256"] = sha(path)
                self.refit["inputs"][str(path)] = sha(path)
        if grid:
            self.refit["inputs"].update(M.transaction_bindings(
                self.membership, json.loads((self.membership / A.COMPLETE).read_text())))
        dump(self.directory / "summary.json", self.refit)
        identity = json.loads((self.directory / A.INCOMPLETE).read_text())["context"]["batch_identity"]
        identity["membership_sha256"] = self.prior["source_reports"]["membership"]["sha256"]
        self.recontext(self.directory, identity)
        self.prior["summary_sha256"] = sha(self.directory / "summary.json")
        self.args["expected_summary_sha256"] = self.prior["summary_sha256"]
        self.args["expected_complete_sha256"] = sha(self.directory / A.COMPLETE)
        self.repin_prior()

    def recontext(self, directory, identity):
        for name in (A.INCOMPLETE, A.COMPLETE):
            path = directory / name
            value = json.loads(path.read_text())
            value["context"]["batch_identity"] = identity
            value["context_sha256"] = A._provenance(value["context"])[1]
            dump(path, value)
        reseal(directory)

    def repin_prior(self):
        self.args["expected_verified_sha256"] = dump(self.verified, self.prior)

    def repin_lineage(self):
        refsha = dump(self.reference_path, self.reference)
        self.pair["reference_sha256"] = refsha
        pairsha = dump(self.pair_path, self.pair)
        self.summary["inputs"][str(self.reference_path)] = refsha
        self.summary["inputs"][str(self.pair_path)] = pairsha
        self.summary["hashes"]["expected_paired_sha256"] = pairsha
        self.repin(grid=True)

    def table(self, directory, name, change):
        path = directory / (name + ".parquet")
        table = pd.read_parquet(path)
        result = change(table)
        (table if result is None else result).to_parquet(path, index=False)
        self.repin(grid=directory == self.membership)

    def run(self, **changes):
        self.reads = []
        with self.monkeypatch.context() as guard:
            self.guards(guard)
            return C.run(**(self.args | changes))

    def guards(self, guard):
        modules = ((R, ("verify", "sources", "products")), (N, ("run", "dispatch", "admission", "authority")),
            (N.B, ("prepare",)), (N.P, ("run",)), (N.P.R, ("recover", "prepare_pulses")),
            (N.P.R.C, ("find_event", "_properties", "_find_clusters", "_check_coincidence_vec", "_check_coincidence")),
            (V, ("verify",)), (N.SR, ("verify",)), (M, ("run", "geometry_only")), (M.S, ("run",)),
            (N.I.Q, ("run", "admission")), (N.I.Q.R, ("run_arm",)), (N.I.Q.R.Q, ("run_arm",)),
            (N.I.Q.R.W, ("synthesize",)), (N.I.Q.R.P, ("reconstruct", "reconstruct_matrix", "fit_buckets")),
            (N.I.Q.R.P.P, ("reconstruct", "reconstruct_full", "fit_gumbel")), (N.I.Q.R.P.LM, ("fit_gumbel",)),
            (N.I.Q.R.S, ("simulate_candidates", "optical_candidates", "thermal_candidates")),
            (N.I.Q.M.K, ("make_photons",)), (N.I.Q.M.P, ("read_input", "run", "verify_metadata", "block_probe")),
            (N.I.D, ("run", "run_arm")), (M.T, ("verify",)), (G, ("frozen_grid", "track_histogram")))
        for module, names in modules:
            for name in names:
                if hasattr(module, name):
                    guard.setattr(module, name, forbidden)
        import awkward as ak
        import uproot
        guard.setattr(uproot, "open", forbidden)
        guard.setattr(ak, "from_parquet", forbidden)
        allowed = {self.membership / "registry.parquet", self.membership / "grid.parquet", self.directory / "registry.parquet"}
        allowed |= {self.directory / f"block{i}_{name}.parquet" for i in (0, 1) for name in ("frames", "clusters")}
        read, path_open = pd.read_parquet, Path.open

        def bounded_read(path, *args, **kwargs):
            if Path(path) not in allowed:
                pytest.fail(f"unexpected table decode: {path}")
            self.reads.append(Path(path))
            return read(path, *args, **kwargs)

        def no_selection(path, *args, **kwargs):
            if Path(path) == self.selection or Path(path).suffix == ".root":
                pytest.fail(f"forbidden open/hash: {path}")
            return path_open(path, *args, **kwargs)

        parquet_file = V.pq.ParquetFile

        def bounded_metadata(path, *args, **kwargs):
            if Path(path) not in allowed:
                pytest.fail(f"unexpected Parquet metadata access: {path}")
            return parquet_file(path, *args, **kwargs)

        guard.setattr(V.pq, "ParquetFile", bounded_metadata)
        guard.setattr(pd, "read_parquet", bounded_read)
        guard.setattr(Path, "open", no_selection)


@pytest.fixture
def saved(tmp_path, monkeypatch):
    return SavedCoverage(tmp_path, monkeypatch)


def reject(saved, match=None):
    with pytest.raises((ValueError, AssertionError, KeyError, FileNotFoundError), match=match):
        saved.run()
    assert not saved.args["out"].exists()


def test_saved_only_real_transactions_full_grid_and_gate_diagnostic(saved):
    before = {p: sha(p) for p in saved.root.rglob("*") if p.is_file()}
    result = saved.run()
    assert json.loads(saved.args["out"].read_text()) == result
    assert before == {p: sha(p) for p in before}
    assert result["schema"] == C.SCHEMA and result["scope"] == C.SCOPE
    assert (result["registry_frames"], result["unique_frames"], result["coincidences"], result["native_links"]) == (9000, 3, 2, 8)
    assert {k: result["grid"][k] for k in C.GRID_COUNTS} == C.GRID_COUNTS
    assert result["grid"]["grid_ns"] == .25 and result["grid"]["bins"] == 4042
    rows = pd.DataFrame(result["rows"])
    expected = pd.concat(saved.frames, ignore_index=True)
    pd.testing.assert_frame_equal(rows[list(expected)].astype(expected.dtypes), expected, check_exact=True)
    np.testing.assert_array_equal(rows.track_grid_coverage, [1., .5, 0.])
    np.testing.assert_array_equal(rows.eligible_window_coverage, [.5, .5, 0.])
    np.testing.assert_array_equal(rows.event_gate_override, [True, False, False])
    np.testing.assert_array_equal(rows.n_veto_eligible, [1, 1, 0])
    assert result["F"] is result["finite_library_uncertainty"] is None
    assert all(result[k] is False for k in C.FALSE_FIELDS)
    assert len(saved.reads) == 7 and len(set(saved.reads)) == 7
    assert result["selected_track_provenance"]["sha256"] != sha(saved.selection)
    assert str(saved.selection) not in result["inputs"]
    assert saved.args["out"].stat().st_size <= C.CAP
    for path, payload in result["code_snapshots"].items():
        assert hashlib.sha256(payload.encode()).hexdigest() == result["code_sha256"][path] == sha(path)
    assert "mean" not in result and "uncertainty" not in result


@pytest.mark.parametrize("pin", ["expected_summary_sha256", "expected_verified_sha256", "expected_complete_sha256"])
def test_external_pin_rejection(saved, pin):
    saved.args[pin] = "0" * 64
    reject(saved, "changed bound")


@pytest.mark.parametrize("field,value", [("scope", "promoted"), ("execution_admitted", 0), ("F", 0.),
    ("registry_frames", 8999), ("unique_frames", True), ("saved_pulses", 86), ("finder_calls", 3),
    ("stock_refit_pulse_bridge_exact", False), ("nonempty_real_refit_membership", 1), ("code_sha256", {})])
def test_refit_summary_scope_and_code(saved, field, value):
    saved.refit[field] = value
    saved.repin()
    reject(saved)


@pytest.mark.parametrize("field,value", [("scope", "promoted"), ("summary_sha256", "0" * 64), ("schema", "wrong"),
    ("saved_artifact_protocol_exact", 1), ("source_bridge_rechecked", False), ("native_member_structure_exact", False),
    ("regenerated_finder", True), ("native_member_order_independently_recomputed", True), ("F", 0.),
    ("finite_library_uncertainty", 0.), ("registry_frames", 8999), ("unique_frames", 4), ("code_sha256", {})])
def test_inherited_refit_certificate(saved, field, value):
    saved.prior[field] = value
    saved.repin_prior()
    reject(saved)


@pytest.mark.parametrize("case", ["snapshot_missing", "snapshot_text", "role_missing", "role_extra", "role_fields", "role_unbound"])
def test_inherited_snapshots_and_source_roles(saved, case):
    if case.startswith("snapshot"):
        path = next(iter(saved.prior["code_snapshots"]))
        if case == "snapshot_missing":
            saved.prior["code_snapshots"].pop(path)
        else:
            saved.prior["code_snapshots"][path] += "\n"
    else:
        roles = saved.prior["source_reports"]
        if case == "role_missing":
            roles.pop("boundary")
        elif case == "role_extra":
            roles["extra"] = dict(roles["boundary"])
        elif case == "role_fields":
            roles["boundary"]["extra"] = True
        else:
            roles["boundary"]["sha256"] = "0" * 64
    saved.repin_prior()
    reject(saved)


@pytest.mark.parametrize("which", ["directory", "membership"])
@pytest.mark.parametrize("case", ["extra", "failed", "missing_complete", "receipt", "pending", "snapshot", "snapshot_resealed"])
def test_transaction_rejection(saved, which, case):
    root = getattr(saved, which)
    if case == "missing_complete":
        (root / A.COMPLETE).unlink()
    else:
        name = {"extra": "extra.txt", "failed": A.FAILED, "receipt": A.COMPLETE, "pending": A.PENDING,
                "snapshot": "code0.txt", "snapshot_resealed": "code0.txt"}[case]
        (root / name).write_bytes(b"changed")
        if case == "snapshot_resealed":
            reseal(root)
    reject(saved)


@pytest.mark.parametrize("field,value", [("frozen_grid_exact", False), ("source_transactions_exact", 1),
    ("regenerated_finder", True), ("F", 0.), ("finite_library_uncertainty", 0.), ("registry_frames", 8999),
    ("schema", "wrong"), ("code_sha256", "0" * 64)])
def test_grid_certificate_rejection(saved, field, value):
    saved.mv[field] = value
    saved.repin(grid=True)
    reject(saved)


@pytest.mark.parametrize("field,value", [("selected", 1929420), ("total_tracks", 1930162), ("selection_events", 1930162),
    ("bins", 4041), ("bins", 4042.), ("grid_ns", .5), ("contract", "centre-exact"), ("card_sha", "wrong")])
def test_frozen_grid_metadata(saved, field, value):
    saved.summary["grid"][field] = saved.mv["grid"][field] = value
    saved.repin(grid=True)
    reject(saved)


@pytest.mark.parametrize("case", ["discontinuity", "negative", "unnormalized", "nonfinite", "spacing", "short", "dtype"])
def test_grid_parquet_rejection(saved, case):
    def change(t):
        if case == "discontinuity":
            t.loc[2, "left"] += .125
        elif case in ("negative", "unnormalized", "nonfinite"):
            t.loc[0, "weight"] = {"negative": -.1, "unnormalized": .1, "nonfinite": np.nan}[case]
        elif case == "spacing":
            t[["left", "right"]] *= 2
        elif case == "short":
            return t.iloc[:-1]
        else:
            return t.astype({"weight": "float32"})
    saved.table(saved.membership, "grid", change)
    reject(saved)


@pytest.mark.parametrize("case", ["card_sha", "card_mode", "card_window", "card_type", "selected", "pmin", "pmax", "step"])
def test_reference_card_and_selection_metadata(saved, case):
    if case == "card_sha":
        saved.reference["card"]["sha"] = "wrong"
    elif case == "card_mode":
        card = replace(saved.card, gate_mode="coinc")
        saved.reference["card"] = asdict(card) | {"sha": card.sha()}
        saved.summary["grid"]["card_sha"] = saved.mv["grid"]["card_sha"] = card.sha()
    elif case == "card_window":
        saved.reference["card"]["dt_hi"] = -1.
    elif case == "card_type":
        saved.reference["card"]["dt_hi"] = True
    elif case == "step":
        saved.reference["F"]["grid_ns"] = .5
    else:
        saved.reference[case] = {"selected": 1929420, "pmin": 86, "pmax": 199}[case]
    saved.repin_lineage()
    reject(saved)


@pytest.mark.parametrize("case", ["selection_pin", "selection_relative", "selection_digest", "reference_pin", "ambiguous_pair"])
def test_selection_lineage_without_reading_selection(saved, case):
    if case == "selection_pin":
        saved.pair["selection_sha256"] = "e" * 64
    elif case == "selection_relative":
        saved.pair["selection"] = "relative.parquet"
        saved.summary["inputs"]["relative.parquet"] = saved.pair["selection_sha256"]
    elif case == "selection_digest":
        saved.pair["selection_sha256"] = "malformed"
        saved.summary["inputs"][str(saved.selection)] = "malformed"
    elif case == "reference_pin":
        saved.summary["inputs"][str(saved.reference_path)] = "0" * 64
    pairsha = dump(saved.pair_path, saved.pair)
    saved.summary["inputs"][str(saved.pair_path)] = pairsha
    saved.summary["hashes"]["expected_paired_sha256"] = pairsha
    if case == "ambiguous_pair":
        saved.summary["inputs"][str(saved.sources / "other.json")] = pairsha
    saved.repin(grid=True)
    reject(saved)


@pytest.mark.parametrize("which", ["directory", "membership"])
@pytest.mark.parametrize("case", ["row_order", "missing", "identity", "bits"])
def test_full_registry_identity_order_and_census(saved, which, case):
    def change(t):
        if case == "row_order":
            return t.iloc[::-1]
        if case == "missing":
            return t.iloc[:-1]
        if case == "identity":
            t.loc[8999, "event"] = 9001
        else:
            t.loc[8999, "pbtime"] = np.nextafter(t.pbtime.iloc[-1], 0.)
    saved.table(getattr(saved, which), "registry", change)
    reject(saved)


@pytest.mark.parametrize("field", [*G.KEYS, "pbtime", "intensity"])
def test_full_frame_identity_is_not_just_event(saved, field):
    def change(t):
        t.loc[0, field] = "f" * 64 if field == "input_sha256" else t.loc[0, field] + 1
    saved.table(saved.directory, "block0_frames", change)
    reject(saved)


@pytest.mark.parametrize("field", G.KEYS)
def test_cluster_full_source_qualified_identity(saved, field):
    def change(t):
        t.loc[0, field] = "f" * 64 if field == "input_sha256" else t.loc[0, field] + 1
    saved.table(saved.directory, "block0_clusters", change)
    reject(saved, "cluster frame identity")


@pytest.mark.parametrize("case", ["frame_order", "block_order", "frame_record", "verified_frame", "block_count",
    "verified_count", "aggregate_pulses", "negative_census", "cluster_index", "cluster_frame_order", "cluster_census",
    "nonempty_summary", "nonempty_verified"])
def test_frame_cluster_order_and_independent_census(saved, case):
    if case == "frame_order":
        saved.table(saved.directory, "block1_frames", lambda t: t.iloc[::-1])
    elif case == "block_order":
        saved.refit["records"].reverse()
    elif case == "frame_record":
        saved.refit["records"][0]["frames"][0]["pulses"] += 1
    elif case == "verified_frame":
        saved.prior["frames"][0]["pulses"] += 1
    elif case == "block_count":
        saved.refit["records"][0]["counts"]["members"] += 1
    elif case == "verified_count":
        saved.prior["counts"][0]["clusters"] += 1
    elif case in ("aggregate_pulses", "negative_census", "cluster_census"):
        field, value = {"aggregate_pulses": ("pulses", 24), "negative_census": ("members", -1),
                        "cluster_census": ("clusters", 0)}[case]
        saved.frames[0].loc[0, field] = value
        saved.frames[0].to_parquet(saved.directory / "block0_frames.parquet", index=False)
        saved.refit["records"] = saved.records()
        saved.prior["counts"] = [dict(block=r["block"], **r["counts"]) for r in saved.refit["records"]]
        saved.prior["frames"] = pd.concat(saved.frames).to_dict("records")
    elif case == "cluster_index":
        saved.table(saved.directory, "block0_clusters", lambda t: t.assign(idx=t.idx + 1))
    elif case == "cluster_frame_order":
        c = pd.concat([saved.clusters[1], saved.clusters[1]], ignore_index=True)
        c.loc[0, G.KEYS] = saved.frames[1].iloc[1][G.KEYS]
        c.to_parquet(saved.directory / "block1_clusters.parquet", index=False)
        saved.refit["records"][1]["counts"]["clusters"] = 2
        saved.prior["counts"][1]["clusters"] = 2
    else:
        (saved.refit if case == "nonempty_summary" else saved.prior)["nonempty_real_refit_membership"] = False
    saved.repin()
    reject(saved)


def test_all_saved_frames_empty_is_valid_and_zero(saved):
    for i in (0, 1):
        saved.frames[i][["clusters", "members"]] = 0
        saved.clusters[i] = empty(V.CLUSTERS)
        saved.frames[i].to_parquet(saved.directory / f"block{i}_frames.parquet", index=False)
        saved.clusters[i].to_parquet(saved.directory / f"block{i}_clusters.parquet", index=False)
    saved.refit["records"] = saved.records()
    saved.prior["counts"] = [dict(block=r["block"], **r["counts"]) for r in saved.refit["records"]]
    saved.prior["frames"] = pd.concat(saved.frames).to_dict("records")
    saved.refit["nonempty_real_refit_membership"] = saved.prior["nonempty_real_refit_membership"] = False
    saved.repin()
    result = saved.run()
    rows = pd.DataFrame(result["rows"])
    assert result["coincidences"] == result["native_links"] == 0 and len(rows) == 3
    assert rows.track_grid_coverage.eq(0).all() and rows.eligible_window_coverage.eq(0).all()
    assert rows.gate_pass.all() and not rows.event_gate_override.any()


@pytest.mark.parametrize("which,name,rows", [("directory", "registry", 9001), ("membership", "registry", 9001),
    ("directory", "block0_frames", 5), ("directory", "block0_clusters", 2049), ("membership", "grid", 20001)])
def test_parquet_row_caps_before_decode(saved, monkeypatch, which, name, rows):
    root = getattr(saved, which)
    path = root / (name + ".parquet")
    saved.table(root, name, lambda t: pd.concat([t.iloc[:1]] * rows, ignore_index=True))
    read = pd.read_parquet

    def no_decode(p, *args, **kwargs):
        if Path(p) == path:
            pytest.fail("oversized table decoded before metadata admission")
        return read(p, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", no_decode)
    reject(saved, "metadata cap")


@pytest.mark.parametrize("case", ["columns", "dtype", "uncompressed", "groups"])
def test_parquet_physical_admission_before_decode(saved, monkeypatch, case):
    path = saved.directory / "block0_clusters.parquet"
    if case in ("columns", "dtype"):
        saved.table(saved.directory, "block0_clusters", lambda t: t.assign(extra=0) if case == "columns"
                    else t.astype({"avgtime": "float32"}))
    else:
        original = V.pq.ParquetFile

        class Metadata:
            def __init__(self, original_meta):
                self.original = original_meta

            def __getattr__(self, name):
                if case == "groups" and name == "num_row_groups":
                    return 4097
                return getattr(self.original, name)

            def row_group(self, i):
                from types import SimpleNamespace
                return SimpleNamespace(total_byte_size=128 * 1024**2 + 1)

        class Parquet:
            def __init__(self, p, *args, **kwargs):
                self.original = original(p, *args, **kwargs)
                self.metadata = Metadata(self.original.metadata) if Path(p) == path else self.original.metadata
                self.schema_arrow = self.original.schema_arrow

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.original.close()

        monkeypatch.setattr(V.pq, "ParquetFile", Parquet)
    read = pd.read_parquet

    def no_decode(p, *args, **kwargs):
        if Path(p) == path:
            pytest.fail("invalid physical Parquet decoded")
        return read(p, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", no_decode)
    reject(saved, "cap|schema|dtype")


@pytest.mark.parametrize("case", ["file", "transaction", "output", "cells"])
def test_byte_and_computation_caps(saved, monkeypatch, case):
    if case == "file":
        path = saved.verified
        with path.open("wb") as stream:
            stream.truncate(N.CAP + 1)
        original = Path.open

        def no_read(p, *args, **kwargs):
            if Path(p) == path:
                pytest.fail("oversized input read before size cap")
            return original(p, *args, **kwargs)

        monkeypatch.setattr(Path, "open", no_read)
    elif case == "transaction":
        with (saved.directory / "block0_saved_pulses.parquet").open("wb") as stream:
            stream.truncate(N.CAP)
        monkeypatch.setattr(pd, "read_parquet", forbidden)
    elif case == "output":
        monkeypatch.setattr(C, "CAP", 1)
    else:
        monkeypatch.setattr(C, "MAX_CELLS", 8084)
        monkeypatch.setattr(G.deadtime, "frame_coverage", forbidden)
    reject(saved, "cap|max_cells")


@pytest.mark.parametrize("which", ["directory", "membership"])
def test_resealed_payload_without_independent_inherited_pin_is_rejected(saved, which):
    root = getattr(saved, which)
    path = root / ("block0_clusters.parquet" if which == "directory" else "grid.parquet")
    t = pd.read_parquet(path)
    if which == "directory":
        t["avgtime"] += .25
    else:
        t["weight"] = np.roll(t.weight.to_numpy(), 1)
    t.to_parquet(path, index=False)
    reseal(root)
    reject(saved, "changed bound|pinned refit lineage")


@pytest.mark.parametrize("phase", ["products", "evaluation", "publication"])
@pytest.mark.parametrize("target", ["input", "receipt", "reference", "verified", "source", "summary", "unread_payload"])
def test_late_mutation_is_rejected_without_output(saved, monkeypatch, phase, target):
    source = saved.sources / "coverage_source.txt"
    source.write_text("synthetic source snapshot\n")
    paths = C.code_paths()
    monkeypatch.setattr(C, "code_paths", lambda: [*paths, source])
    path = {"input": saved.directory / "block1_clusters.parquet", "receipt": saved.membership / A.PENDING,
            "reference": saved.reference_path, "verified": saved.verified, "source": source,
            "summary": saved.directory / "summary.json", "unread_payload": saved.directory / "block0_saved_pulses.parquet"}[target]

    def mutate():
        with path.open("ab") as stream:
            stream.write(b"\n")

    if phase in ("products", "evaluation"):
        name = "saved_products" if phase == "products" else "evaluate"
        original = getattr(C, name)

        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            mutate()
            return result

        monkeypatch.setattr(C, name, wrapped)
    else:
        original = A._canonical

        def canonical(value):
            result = original(value)
            if value.get("schema") == C.SCHEMA:
                mutate()
            return result

        monkeypatch.setattr(A, "_canonical", canonical)
    reject(saved, "changed bound|source snapshot")


@pytest.mark.parametrize("case", ["existing", "directory", "symlink", "dangling", "inside_refit", "inside_sources",
    "inside_selection", "symlink_parent", "late_created"])
def test_output_exclusive_and_outside_safety(saved, monkeypatch, case):
    out = saved.args["out"]
    victim = saved.root / "victim.txt"
    victim.write_bytes(b"preserve me")
    if case == "existing":
        out.write_bytes(b"existing")
    elif case == "directory":
        out.mkdir()
    elif case in ("symlink", "dangling"):
        out.symlink_to(victim if case == "symlink" else saved.root / "missing")
    elif case.startswith("inside_"):
        root = {"inside_refit": saved.directory, "inside_sources": saved.sources,
                "inside_selection": saved.selection.parent}[case]
        saved.args["out"] = root / "new.json"
    elif case == "symlink_parent":
        alias = saved.root / "alias"
        alias.symlink_to(saved.directory, target_is_directory=True)
        saved.args["out"] = alias / "new.json"
    else:
        original = Path.open

        def race(p, mode="r", *args, **kwargs):
            if Path(p) == out and mode == "xb":
                with original(p, "wb") as stream:
                    stream.write(b"racing writer")
            return original(p, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", race)
    with pytest.raises((ValueError, FileExistsError, IsADirectoryError)):
        saved.run()
    assert victim.read_bytes() == b"preserve me"
    if case in ("existing", "late_created"):
        assert out.read_bytes() == (b"existing" if case == "existing" else b"racing writer")
    elif case in ("symlink", "dangling"):
        assert out.is_symlink()
    elif case != "directory":
        assert not saved.args["out"].exists()


def tiny_tables(n=3, times=(.25,), which=(0,)):
    f = pd.DataFrame(dict(input_sha256=pd.Series(["a" * 64] * n, dtype=object), run=np.ones(n, dtype=np.int64),
        subrun=np.zeros(n, dtype=np.int64), event=np.arange(n, dtype=np.int64), frame_id=np.arange(n, dtype=np.int64),
        pbtime=np.zeros(n), intensity=np.ones(n)))
    c = f.iloc[list(which)][G.KEYS].reset_index(drop=True)
    c = c.assign(idx=c.groupby("frame_id").cumcount().astype("int64"), avgtime=np.asarray(times, dtype=float),
                 start=np.asarray(times, dtype=float) - .125, end=np.asarray(times, dtype=float) + .125,
                 pe=10., nLayers=3, sectortype=1)
    return f, c


@pytest.mark.parametrize("time", [-10., -.5, -.25, 0., .25, .5, 1., 10.,
    np.nextafter(.25, 0.), np.nextafter(.25, 1.)])
def test_tiny_integer_bin_semantics_without_grid_rebuild(time):
    f, c = tiny_tables(times=(time,))
    card = VetoCard(dt_lo=0., dt_hi=.5)
    hist, edges = np.array([.125, .125, .25, .5]), np.arange(5) * .25
    result = C.evaluate(f, c, card, hist, edges)
    lo = np.clip(np.ceil(time / .25), 0, 4)
    hi = np.clip(np.ceil((time + .5) / .25), 0, 4)
    expected = ((np.arange(4) >= lo) & (np.arange(4) < hi)) @ hist
    assert result.track_grid_coverage.iloc[0] == result.eligible_window_coverage.iloc[0] == expected
    assert result.track_grid_coverage.iloc[1:].eq(0).all()
    assert not result.event_gate_override.any()


def test_integer_bins_are_not_strict_continuous_centres():
    f, c = tiny_tables(times=(0.,))
    hist, edges = np.array([1., 0.]), np.array([0., .25, .5])
    card = VetoCard(dt_lo=0., dt_hi=.125)
    result = C.evaluate(f, c, card, hist, edges)
    centres = (edges[1:] + edges[:-1]) / 2
    assert (((centres > card.dt_lo) & (centres < card.dt_hi)) @ hist) == 0.
    assert result.track_grid_coverage.iloc[0] == result.eligible_window_coverage.iloc[0] == 1.


@pytest.mark.parametrize("boundary", ["start", "end"])
def test_genuine_gate_override_does_not_change_eligibility(boundary):
    f, c = tiny_tables(times=(.25, .5), which=(0, 1))
    c.loc[0, "pe"] = 1.
    card = VetoCard(dt_hi=.25, min_pe=10., **({"gate_start_lo": .125} if boundary == "start" else {"gate_end_hi": .375}))
    result = C.evaluate(f, c, card, np.array([.25] * 4), np.arange(5) * .25)
    assert result.event_gate_override.iloc[0] and result.n_veto_eligible.iloc[0] == 0
    assert result.track_grid_coverage.iloc[0] == 1. and result.eligible_window_coverage.iloc[0] == 0.
    assert result.gate_pass.iloc[2] and result.track_grid_coverage.iloc[2] == 0.


def test_overlap_is_union_and_not_sum():
    f, c = tiny_tables(times=(.25, .25, .5), which=(0, 0, 0))
    result = C.evaluate(f, c, VetoCard(dt_hi=.5), np.full(4, .25), np.arange(5) * .25)
    assert result.n_veto_eligible.iloc[0] == 3
    assert result.track_grid_coverage.iloc[0] == result.eligible_window_coverage.iloc[0] == .75


@pytest.mark.parametrize("n", [0, 3])
def test_tiny_empty_frames(n):
    f, c = tiny_tables(n=n, times=(), which=())
    result = C.evaluate(f, c, VetoCard(), np.array([1.]), np.array([0., .25]))
    assert len(result) == n and result.gate_pass.all()
    assert result.track_grid_coverage.eq(0).all() and result.eligible_window_coverage.eq(0).all()
    assert not result.event_gate_override.any()


def test_non_event_gate_refused_before_evaluation(monkeypatch):
    monkeypatch.setattr(G, "evaluate", forbidden)
    with pytest.raises(ValueError, match="event gate"):
        C.evaluate(*tiny_tables(), VetoCard(gate_mode="coinc"), np.array([1.]), np.array([0., .25]))


def test_unrepresentable_window_rejected_before_historical_cast(monkeypatch):
    monkeypatch.setattr(G.deadtime, "frame_coverage", forbidden)
    with pytest.raises(ValueError, match="int64"):
        C.evaluate(*tiny_tables(times=(1e300,)), VetoCard(), np.array([1.]), np.array([0., .25]))


@pytest.mark.parametrize("field,value", [("pe", -1.), ("avgtime", np.inf), ("start", 3.), ("nLayers", -1),
                                        ("sectortype", -1)])
def test_invalid_saved_cluster_scalars(saved, field, value):
    saved.table(saved.directory, "block1_clusters", lambda t: t.assign(**{field: t[field] * 0 + value}))
    reject(saved)


@pytest.mark.parametrize("which", ["directory", "membership"])
def test_original_source_map_order_controls_snapshots(saved, which):
    root = getattr(saved, which)
    first, second = (root / f"code{i}.txt" for i in (0, 1))
    a, b = first.read_bytes(), second.read_bytes()
    assert a != b
    first.write_bytes(b)
    second.write_bytes(a)
    saved.repin(grid=which == "membership")
    reject(saved, "changed bound|conflicting coverage binding")


@pytest.mark.parametrize("case", ["duplicate", "nonfinite", "array"])
def test_pinned_malformed_certificate_json(saved, case):
    payload = {"duplicate": b'{"schema":"a","schema":"b"}', "nonfinite": b'{"schema":NaN}', "array": b'[]'}[case]
    saved.verified.write_bytes(payload)
    saved.args["expected_verified_sha256"] = sha(saved.verified)
    reject(saved, "duplicate JSON key|nonfinite JSON value|JSON object required")


def test_grid_metadata_disagreement(saved):
    saved.mv["grid"]["selected"] -= 1
    saved.repin(grid=True)
    reject(saved, "grid metadata differs")


@pytest.mark.parametrize("case", ["eligibility", "decomposition"])
def test_two_evaluations_must_agree_except_gate_override(monkeypatch, case):
    original, calls = G.evaluate, []

    def evaluate(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(kwargs["max_cells"])
        if len(calls) == 2:
            result.loc[0, "n_veto_eligible" if case == "eligibility" else "track_grid_coverage"] += 1
        return result

    monkeypatch.setattr(G, "evaluate", evaluate)
    with pytest.raises(ValueError, match="eligibility|decomposition"):
        C.evaluate(*tiny_tables(), VetoCard(dt_hi=.25), np.full(4, .25), np.arange(5) * .25)
    assert calls == [C.MAX_CELLS, C.MAX_CELLS]


def test_successful_report_cli_is_compact_saved_only(saved, monkeypatch, capsys):
    saved.reads = []
    with monkeypatch.context() as guard:
        saved.guards(guard)
        argv = [str(saved.directory)]
        for key, value in saved.args.items():
            if key != "directory":
                argv.extend(["--" + key.replace("_", "-"), str(value)])
        C.main(argv)
    result = json.loads(capsys.readouterr().out)
    assert set(result) == {"schema", "rows", "grid", "seconds", "memory", "scope", "F"}
    assert result["F"] is None and len(result["rows"]) == 3
    assert json.loads(saved.args["out"].read_text())["rows"] == result["rows"]


@pytest.mark.parametrize("case", ["json", "parquet"])
def test_byte_caps_precede_decode(saved, monkeypatch, case):
    if case == "json":
        path = saved.verified
        with path.open("wb") as stream:
            stream.truncate(8 * 1024**2 + 1)
        saved.args["expected_verified_sha256"] = sha(path)
        decode = C.json.loads

        def bounded_decode(payload, *args, **kwargs):
            if len(payload) > 8 * 1024**2:
                pytest.fail("oversized JSON decoded")
            return decode(payload, *args, **kwargs)

        monkeypatch.setattr(C.json, "loads", bounded_decode)
    else:
        monkeypatch.setattr(V, "CAP", 1)
        monkeypatch.setattr(V.pq, "ParquetFile", forbidden)
        monkeypatch.setattr(pd, "read_parquet", forbidden)
    reject(saved, "byte cap")


@pytest.mark.parametrize("case", ["columns_order", "missing_value", "cluster_duplicates"])
def test_saved_table_schema_and_duplicate_index(saved, case):
    def change(t):
        if case == "columns_order":
            return t[list(t)[::-1]]
        if case == "missing_value":
            t.loc[0, "input_sha256"] = None
        else:
            return pd.concat([t, t], ignore_index=True)
    saved.table(saved.directory, "block0_clusters", change)
    if case == "cluster_duplicates":
        saved.refit["records"][0]["counts"]["clusters"] = 2
        saved.prior["counts"][0]["clusters"] = 2
        saved.repin()
    reject(saved, "columns|missing|indices/order")


@pytest.mark.parametrize("which", ["directory", "membership"])
def test_inherited_input_symlink_is_not_regular(saved, which):
    path = getattr(saved, which) / ("block0_clusters.parquet" if which == "directory" else "grid.parquet")
    target = saved.sources / "unchanged.parquet"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    reject(saved, "regular")


def test_missing_selected_track_file_is_still_provenance_only(saved):
    saved.selection.unlink()
    result = saved.run()
    assert result["selected_track_provenance"]["path"] == str(saved.selection)
    assert not result["selected_tracks_read"] and not result["selected_tracks_rehashed"]


@pytest.mark.parametrize("n", [0, 1, 2])
def test_each_external_pin_is_required(saved, n):
    key = ["expected_summary_sha256", "expected_verified_sha256", "expected_complete_sha256"][n]
    args = dict(saved.args)
    args.pop(key)
    with pytest.raises(TypeError, match=key):
        C.run(**args)
    assert not saved.args["out"].exists()


def test_outputs_inside_inherited_code_parent_rejected(saved, monkeypatch):
    source_dir = saved.root / "synthetic_code"
    source_dir.mkdir()
    source = source_dir / "coverage_source.txt"
    source.write_text("synthetic code binding\n")
    paths = C.code_paths()
    monkeypatch.setattr(C, "code_paths", lambda: [*paths, source])
    saved.args["out"] = source_dir / "report.json"
    reject(saved, "outside")


@pytest.mark.parametrize("target", ["membership", "selection"])
def test_late_output_parent_symlink_cannot_enter_inherited_evidence(saved, monkeypatch, target):
    safe = saved.root / "safe_output"
    safe.mkdir()
    alias = saved.root / "output_alias"
    alias.symlink_to(safe, target_is_directory=True)
    saved.args["out"] = alias / "report.json"
    original = C.evaluate
    destination = saved.membership if target == "membership" else saved.selection.parent

    def retarget(*args, **kwargs):
        result = original(*args, **kwargs)
        alias.unlink()
        alias.symlink_to(destination, target_is_directory=True)
        return result

    monkeypatch.setattr(C, "evaluate", retarget)
    with pytest.raises(ValueError, match="outside"):
        saved.run()
    assert not (safe / "report.json").exists() and not (destination / "report.json").exists()
