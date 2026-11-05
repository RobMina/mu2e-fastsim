"""Independent synthetic v2 source diagnostics; no persistence or POT certification.

Source rows follow TARGET_SOURCE_SCHEMA.md, not the auditor's builders.
Native-state field names follow the proposal's nativeState serializer. Fixtures
use actual non-dense SimParticle keys, unrelated to assigned G4 track IDs.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json

import pytest

from beamfastsim.tsrecordledger import LedgerError, audit


SCHEMA_V2 = "target-recording-proposal-v2"
GEN_PRODUCT = 83
SIM_PRODUCT = 101
SIM_STAGE = 3
SOURCE_COUNTERS = (
    "source_primary_count", "source_proton_count", "primary_link_count",
    "sim_particle_inserted", "sim_particle_omitted", "sim_particle_final",
)
RECON_COUNTERS = (
    "generator_size", "generator_protons", "primary_links",
    "inserted_particles", "omitted_particles", "final_particles",
)
NATIVE_SCALARS = (
    "global_time_ns", "local_time_ns", "proper_time_ns", "kinetic_energy_MeV",
    "total_energy_MeV", "weight", "track_length_mm", "velocity_mm_per_ns",
    "dynamic_charge_eplus", "dynamic_mass_MeV", "vertex_kinetic_energy_MeV",
)
NATIVE_VECTORS = (
    "momentum_MeV", "momentum_direction", "polarization",
    "vertex_momentum_direction",
)


def payloads(rows, kind):
    return [row["data"] for row in rows if row["kind"] == kind]


def data(rows, kind, index=0):
    return payloads(rows, kind)[index]


def remove(rows, kind, index=0):
    rows.remove([row for row in rows if row["kind"] == kind][index])


def duplicate(rows, kind, index=0):
    row = [row for row in rows if row["kind"] == kind][index]
    rows.insert(rows.index(row), deepcopy(row))


def move_before(rows, kind, before):
    row = next(row for row in rows if row["kind"] == kind)
    rows.remove(row)
    rows.insert(next(i for i, other in enumerate(rows) if other["kind"] == before), row)


def set_path(obj, path, value):
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value


@pytest.fixture
def source_ledger(tmp_path):
    """Write real synthetic bytes, independently deriving counts after mutations."""
    def make(mode="proton", mutate=None, after_recount=None, completion_mutate=None):
        rows = []
        history = [dict(instance_id=1, copy_no=0, physical_name="Target",
                        logical_name="TargetLV", daughter_index=None)]

        def geometry(track=False):
            return dict(instance_id=1, target_member=True,
                        availability=("native_track_touchable_not_vertex_membership"
                                      if track else "native_touchable"),
                        copy_history_root_to_leaf=deepcopy(history))

        def causal(parent=0, token=None):
            return dict(parent_id=parent, child_token=token,
                        birth_association=("observed_secondary_birth_incidence" if parent
                                           else "explicit_primary_root"))

        def native(tid, parent, pdg, step=0, status=0):
            position = dict(g4_world_mm=[1.0, 2.0, 3.0], mu2e_mm=[1.0, 2.0, 3.0])
            return dict(track_id=tid, parent_id=parent, pdg=pdg,
                        particle_name="proton" if pdg == 2212 else "e-",
                        native_status=status, current_step_number=step,
                        position=deepcopy(position), momentum_MeV=[0.0, 0.0, 100.0],
                        momentum_direction=[0.0, 0.0, 1.0], global_time_ns=5.0,
                        local_time_ns=0.0, proper_time_ns=0.0,
                        kinetic_energy_MeV=5.0, total_energy_MeV=943.272,
                        weight=1.0, track_length_mm=0.0, velocity_mm_per_ns=30.0,
                        polarization=[0.0, 0.0, 0.0], dynamic_charge_eplus=1.0,
                        dynamic_mass_MeV=938.272, vertex_position=deepcopy(position),
                        vertex_momentum_direction=[0.0, 0.0, 1.0],
                        vertex_kinetic_energy_MeV=5.0, vertex_logical_volume=None,
                        vertex_placement_membership="unresolved_no_vertex_touchable_claim",
                        creator_process=None, track_geometry=geometry(True))

        def add(kind, **fields):
            rows.append(dict(schema=SCHEMA_V2, kind=kind, data=deepcopy(fields),
                             row=len(rows) + 1, run=1, subrun=2, event=3,
                             library_eligible=False, proposal_only=True,
                             geometry_file="records.jsonl", geometry_hash=None,
                             hash_binding="requires_external_postread_file_hash"))

        def pointer(product, key):
            return dict(product_id=product, key=key)

        def identity(tid, parent, pdg, key, parent_key=None, generator_key=None):
            return dict(track_id=tid, parent_id=parent, pdg=pdg,
                        product_id=SIM_PRODUCT, key=key, sim_stage=SIM_STAGE,
                        creation_code=56 if parent else 1,
                        parent=None if parent_key is None else pointer(SIM_PRODUCT, parent_key),
                        generator=(None if generator_key is None
                                   else pointer(GEN_PRODUCT, generator_key)))

        def stack(tid, parent, pdg, token=None, killed=False, generator_key=None):
            if parent == 0:
                add("primary_link", track_id=tid, generator_product_id=GEN_PRODUCT,
                    generator_key=generator_key, pdg=pdg,
                    classification_native=-9 if killed else 0,
                    native_at_classification=native(tid, parent, pdg),
                    source_pointer_available=True)
            add("stacking", track_id=tid, parent_id=parent, pdg=pdg,
                classification_native=-9 if killed else 0, observation_number=1,
                stacking_cut=killed, common_cut_evaluated=not killed,
                common_cut=None if killed else False, causal=causal(parent, token))

        def begin(ident, token=None):
            add("sim_particle_insert", **ident, insertion_observed=True,
                persistent_output_verified=False)
            add("track_begin", native=native(ident["track_id"], ident["parent_id"], ident["pdg"]),
                causal=causal(ident["parent_id"], token), lifecycle_visit=1,
                stacking_observations=1)

        def end(ident, token=None, steps=0):
            add("track_end", native=native(ident["track_id"], ident["parent_id"],
                                          ident["pdg"], steps, 2),
                causal=causal(ident["parent_id"], token), lifecycle_visit=1)

        generators = [] if mode == "empty" else [2212, 11] if mode == "mixed" else [2212]
        add("source_begin", generator_product_id=GEN_PRODUCT, sim_product_id=SIM_PRODUCT,
            sim_stage=SIM_STAGE, generator_size=len(generators),
            generator_protons=generators.count(2212), configured_primary_tag="generate:beam:SIM",
            process_history_id="0123456789abcdef",
            generator_provenance=dict(branch_name="GenParticles_generate_beam_SIM.",
                                      module_label="generate", product_instance_name="beam",
                                      process_name="SIM", friendly_class_name="GenParticles",
                                      parameter_set_ids=["0123456789abcdef"]),
            max_rows=1000, max_volumes=10, max_bytes=1 << 20,
            persistent_output_verified=False, exposure_certified=False)
        for key, pdg in enumerate(generators):
            add("generated_primary", generator_product_id=GEN_PRODUCT, generator_key=key,
                pdg=pdg, generator_id=-1, position_mu2e_mm=[1.0, 2.0, 3.0],
                four_momentum_MeV=[0.0, 0.0, 100.0, 943.272], global_time_ns=5.0,
                proper_time_ns=0.0, generator_weight=None)
        add("source_end", generator_size=len(generators), generator_protons=generators.count(2212))
        add("event_begin", outcome="begun_not_complete", proposal_only=True,
            target_physical_names=["Target"], target_material="Tungsten",
            max_rows=1000, max_volumes=10, max_bytes=1 << 20)
        add("volume", instance_id=1, parent_id=None, target_member=True,
            physical_name="Target", logical_name="TargetLV", material="Tungsten",
            copy_history_root_to_leaf=history)
        add("inventory_end", volume_count=1)
        finals = []
        if mode != "empty":
            primary_gen_key = 1 if mode == "mixed" else 0
            primary_pdg = generators[primary_gen_key]
            primary = identity(17, 0, primary_pdg, 902, generator_key=primary_gen_key)
            stack(17, 0, primary_pdg, killed=mode == "killed", generator_key=primary_gen_key)
            if mode != "killed":
                begin(primary)
                if mode == "secondary":
                    endpoint = dict(geometry=geometry(), step_status_native=4)
                    step = dict(track_id=17, step_number=1, lifecycle_visit=1,
                                target_step_row_present=True, boundary_observed=False,
                                transition_observation="internal", pre=endpoint,
                                post=deepcopy(endpoint))
                    add("target_step", parent_step=step, target_step_ordinal=1,
                        track_status_native=0, causal=causal())
                    add("secondary_birth", child_token=1, parent_step=step,
                        parent_causal=causal(), raw_child_track_id=0,
                        raw_child_parent_id=17, pdg=11,
                        native_child_at_birth=native(0, 17, 11))
                    add("cuts", track_id=17, parent_id=0, step_number=1,
                        lifecycle_visit=1, stepping_cut=False, common_cut_evaluated=True,
                        common_cut=False, native_status=2, causal=causal())
                end(primary, steps=int(mode == "secondary"))
                daughters = [pointer(SIM_PRODUCT, 11)] if mode == "secondary" else []
                finals.append(dict(primary, daughters=daughters))
            if mode == "secondary":
                add("birth_track_association", child_token=1, track_id=43,
                    parent_id=17, parent_step=1, observed_at="stacking")
                stack(43, 17, 11, token=1)
                child = identity(43, 17, 11, 11, parent_key=902)
                begin(child, token=1)
                end(child, token=1)
                finals.append(dict(child, daughters=[]))
            if mode == "mixed":
                other = identity(43, 0, 2212, 11, generator_key=0)
                stack(43, 0, 2212, generator_key=0)
                begin(other)
                end(other)
                finals.append(dict(other, daughters=[]))
        for final in finals:
            add("sim_particle_final", **final, in_memory_only=True,
                persistent_output_verified=False)
        add("source_reconciliation", in_memory_links_consistent=True,
            persistent_output_verified=False, exposure_certified=False)
        add("event_end", status_native=0, event_aborted_native=False, sim_overflow=False,
            step_kills=0, field_kills=0, unconsumed_birth_tokens=0,
            unresolved_nonprimary_tracks=0, stacked_not_begun_at_end=0,
            suspended_or_postponed_ended_visits=0, zero_exit_or_exposure_claim=False,
            source_reconciled_in_memory=True, persistent_output_verified=False)
        if mutate:
            mutate(rows)
        kinds = Counter(row["kind"] for row in rows)
        generated = payloads(rows, "generated_primary")
        totals = (len(generated), sum(g["pdg"] == 2212 for g in generated),
                  kinds["primary_link"], kinds["sim_particle_insert"],
                  kinds["sim_particle_omitted"], kinds["sim_particle_final"])
        for reconciliation in payloads(rows, "source_reconciliation"):
            reconciliation.update(zip(RECON_COUNTERS, totals))
        for source_end in payloads(rows, "source_end"):
            source_end.update(generator_size=totals[0], generator_protons=totals[1])
        killed = sum(d["classification_native"] == -9 for d in payloads(rows, "stacking"))
        accounting = dict(secondary_births=kinds["secondary_birth"],
                          associated_births=kinds["birth_track_association"],
                          tracks_without_begin=killed, stacking_killed_native_enum=killed,
                          zero_target_steps=kinds["target_step"] == 0,
                          **dict(zip(SOURCE_COUNTERS, totals)))
        for event_end in payloads(rows, "event_end"):
            event_end.update(accounting)
        if after_recount:
            after_recount(rows)
        for number, row in enumerate(rows, 1):
            row["row"] = number
        hooks = dict(prepareEvent=kinds["source_begin"], beginEvent=kinds["event_begin"],
                     endEvent=kinds["event_end"], stacking=kinds["stacking"],
                     beginTrack=kinds["track_begin"], endTrack=kinds["track_end"],
                     step=kinds["cuts"], cuts=kinds["cuts"],
                     simParticleInserted=kinds["sim_particle_insert"],
                     simParticleOmitted=kinds["sim_particle_omitted"],
                     reconcileParticles=kinds["source_reconciliation"])
        encode = lambda row: (json.dumps(row, separators=(",", ":")) + "\n").encode()
        records = b"".join(encode(row) for row in rows)
        completion = deepcopy(data(rows, "event_end"))
        for key in ("tracks_without_begin", "stacking_killed_native_enum", "zero_target_steps",
                    "zero_exit_or_exposure_claim"):
            completion.pop(key)
        kinds["completion"] += 1
        completion.update(outcome="endEvent_and_ledger_flush_complete", library_eligible=False,
                          row_count=len(rows) + 1, attempted_rows=len(rows) + 1,
                          per_kind_counts=dict(kinds), observed_hooks=hooks,
                          volume_count=kinds["volume"], target_step_count=kinds["target_step"],
                          ledger_bytes=len(records), manifest_hash=None,
                          recording_row_overflow=False, recording_byte_overflow=False,
                          recording_volume_overflow=False,
                          byte_cap_scope="records.jsonl_plus_completion.json; filesystem_metadata_excluded",
                          completion_not_physics_or_exposure_certification=True)
        add("completion", **completion)
        if completion_mutate:
            completion_mutate(rows[-1]["data"])
        (tmp_path / "records.jsonl").write_bytes(records)
        (tmp_path / "completion.json").write_bytes(encode(rows[-1]))
        return tmp_path
    return make


@pytest.mark.parametrize("mode", ["empty", "proton", "killed", "secondary", "mixed"])
def test_v2_complete_source_diagnostic_never_certifies(source_ledger, mode):
    directory = source_ledger(mode)
    result = audit(directory)
    assert result["complete_source_only_diagnostic"] is True
    for key in ("library_eligible", "source_valid", "native_valid", "physics_certified",
                "native_certified", "source_native_valid", "geometry_certified", "exposure_certified"):
        assert result[key] is False
    assert result["external_step_coverage"] == result["stock_product_coverage"] == "unverified"
    assert result["per_kind_counts"]["source_begin"] == 1
    assert result["per_kind_counts"]["source_reconciliation"] == 1
    assert result["row_count"] == sum(result["per_kind_counts"].values())
    assert result["observed_hooks"]["prepareEvent"] == 1
    assert result["observed_hooks"]["reconcileParticles"] == 1
    assert result["observed_hooks"]["simParticleOmitted"] == 0
    expected = 0 if mode in ("empty", "killed") else 2 if mode in ("secondary", "mixed") else 1
    assert result["observed_hooks"]["simParticleInserted"] == expected
    assert result["per_kind_counts"].get("sim_particle_final", 0) == expected
    if mode == "killed":
        assert result["per_kind_counts"]["primary_link"] == 1
        assert result["per_kind_counts"].get("track_begin", 0) == 0
    for name in ("records.jsonl", "completion.json"):
        raw = (directory / name).read_bytes()
        assert result["file_bytes"][name] == len(raw)
        assert result["file_sha256"][name] == hashlib.sha256(raw).hexdigest()


def test_v2_actual_sim_keys_and_generator_links_not_track_arithmetic(source_ledger):
    directory = source_ledger("mixed")
    rows = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
    inserts = payloads(rows, "sim_particle_insert")
    assert [(d["track_id"], d["key"], d["generator"]["key"]) for d in inserts] == [
        (17, 902, 1), (43, 11, 0)]
    assert audit(directory)["library_eligible"] is False


def test_v2_primary_classification_geometry_can_be_unavailable(source_ledger):
    def change(rows):
        data(rows, "primary_link")["native_at_classification"]["track_geometry"] = dict(
            instance_id=None, target_member=None,
            availability="unavailable_native_track_touchable", copy_history_root_to_leaf=None)
    assert audit(source_ledger(mutate=change))["exposure_certified"] is False


@pytest.mark.parametrize("kind", [
    "source_begin", "generated_primary", "source_end", "primary_link",
    "sim_particle_insert", "sim_particle_final", "source_reconciliation",
])
def test_v2_missing_source_rows_rejected_with_rebuilt_bookkeeping(source_ledger, kind):
    with pytest.raises(LedgerError):
        audit(source_ledger(mutate=lambda rows: remove(rows, kind)))


@pytest.mark.parametrize("kind", [
    "source_begin", "generated_primary", "source_end", "primary_link",
    "sim_particle_insert", "sim_particle_final", "source_reconciliation",
])
def test_v2_repeated_source_rows_rejected_with_rebuilt_bookkeeping(source_ledger, kind):
    with pytest.raises(LedgerError):
        audit(source_ledger(mutate=lambda rows: duplicate(rows, kind)))


@pytest.mark.parametrize("kind,before", [
    ("event_begin", "source_begin"), ("generated_primary", "inventory_end"),
    ("source_end", "generated_primary"), ("primary_link", "volume"),
    ("sim_particle_insert", "stacking"), ("sim_particle_final", "track_end"),
    ("source_reconciliation", "sim_particle_final"),
])
def test_v2_source_order_and_immediate_hooks(source_ledger, kind, before):
    with pytest.raises(LedgerError):
        audit(source_ledger(mutate=lambda rows: move_before(rows, kind, before)))


@pytest.mark.parametrize("kind,field", [
    ("source_begin", "generator_product_id"), ("source_begin", "sim_product_id"),
    ("source_begin", "sim_stage"), ("source_begin", "generator_size"),
    ("source_begin", "generator_protons"), ("source_begin", "configured_primary_tag"),
    ("source_begin", "process_history_id"), ("source_begin", "generator_provenance"),
    ("source_begin", "persistent_output_verified"), ("source_begin", "exposure_certified"),
    ("generated_primary", "generator_product_id"), ("generated_primary", "generator_key"),
    ("generated_primary", "generator_id"), ("generated_primary", "pdg"),
    ("generated_primary", "position_mu2e_mm"), ("generated_primary", "four_momentum_MeV"),
    ("generated_primary", "global_time_ns"), ("generated_primary", "proper_time_ns"),
    ("generated_primary", "generator_weight"), ("primary_link", "source_pointer_available"),
    ("primary_link", "native_at_classification"), ("sim_particle_insert", "parent"),
    ("sim_particle_insert", "generator"), ("sim_particle_insert", "insertion_observed"),
    ("sim_particle_final", "daughters"), ("sim_particle_final", "in_memory_only"),
    ("source_reconciliation", "in_memory_links_consistent"),
    ("event_end", "source_reconciled_in_memory"),
])
def test_v2_required_source_fields(source_ledger, kind, field):
    with pytest.raises(LedgerError):
        audit(source_ledger(after_recount=lambda rows: data(rows, kind).pop(field)))


@pytest.mark.parametrize("field", [
    "branch_name", "module_label", "product_instance_name", "process_name",
    "friendly_class_name", "parameter_set_ids",
])
def test_v2_missing_generator_provenance_fields(source_ledger, field):
    def change(rows):
        data(rows, "source_begin")["generator_provenance"].pop(field)
    with pytest.raises(LedgerError):
        audit(source_ledger(mutate=change))


@pytest.mark.parametrize("field,value", [
    ("branch_name", ""), ("module_label", ""), ("process_name", ""),
    ("friendly_class_name", ""), ("product_instance_name", None),
    ("parameter_set_ids", []), ("parameter_set_ids", [""]),
    ("parameter_set_ids", "0123456789abcdef"), ("parameter_set_ids", [True]),
])
def test_v2_malformed_generator_provenance(source_ledger, field, value):
    def change(rows):
        data(rows, "source_begin")["generator_provenance"][field] = value
    with pytest.raises(LedgerError):
        audit(source_ledger(mutate=change))


@pytest.mark.parametrize("field,value", [
    ("configured_primary_tag", ""), ("configured_primary_tag", None),
    ("process_history_id", ""), ("process_history_id", 42),
    ("generator_provenance", []), ("generator_product_id", 0),
    ("sim_product_id", 0), ("generator_product_id", True),
    ("sim_stage", True), ("sim_stage", -1), ("generator_product_id", 1 << 64),
])
def test_v2_invalid_source_identity_and_provenance(source_ledger, field, value):
    with pytest.raises(LedgerError):
        audit(source_ledger(mutate=lambda rows: data(rows, "source_begin").update({field: value})))


@pytest.mark.parametrize("kind,field,value", [
    ("generated_primary", "generator_product_id", 84),
    ("generated_primary", "generator_key", 2),
    ("generated_primary", "generator_key", True),
    ("generated_primary", "pdg", True),
    ("generated_primary", "generator_id", True),
    ("generated_primary", "generator_weight", 1.0),
    ("primary_link", "generator_product_id", 84),
    ("primary_link", "generator_key", 9),
    ("primary_link", "track_id", 99),
    ("primary_link", "pdg", 11),
    ("primary_link", "classification_native", -9),
    ("sim_particle_insert", "product_id", 102),
    ("sim_particle_insert", "sim_stage", 4),
    ("sim_particle_insert", "pdg", 11),
    ("sim_particle_insert", "track_id", 99),
    ("sim_particle_insert", "parent_id", 99),
    ("sim_particle_insert", "creation_code", True),
    ("sim_particle_final", "product_id", 102),
    ("sim_particle_final", "sim_stage", 4),
    ("sim_particle_final", "key", 901),
    ("sim_particle_final", "pdg", 11),
    ("sim_particle_final", "creation_code", 2),
])
def test_v2_wrong_source_or_sim_identity(source_ledger, kind, field, value):
    with pytest.raises(LedgerError):
        audit(source_ledger(mutate=lambda rows: data(rows, kind).update({field: value})))


@pytest.mark.parametrize("path,value", [
    (("track_id",), 99), (("parent_id",), 1), (("pdg",), 11),
    (("current_step_number",), True), (("native_status",), True),
])
def test_v2_primary_native_identity_is_cross_checked(source_ledger, path, value):
    def change(rows):
        set_path(data(rows, "primary_link")["native_at_classification"], path, value)
    with pytest.raises(LedgerError):
        audit(source_ledger(mutate=change))


@pytest.mark.parametrize("kind,field", [
    ("source_begin", "persistent_output_verified"), ("source_begin", "exposure_certified"),
    ("sim_particle_insert", "persistent_output_verified"),
    ("sim_particle_final", "persistent_output_verified"),
    ("source_reconciliation", "persistent_output_verified"),
    ("source_reconciliation", "exposure_certified"),
    ("event_end", "persistent_output_verified"),
])
@pytest.mark.parametrize("value", [True, 0, None, "false"])
def test_v2_false_flags_are_strict_and_never_certify(source_ledger, kind, field, value):
    with pytest.raises(LedgerError):
        audit(source_ledger(mutate=lambda rows: data(rows, kind).update({field: value})))


@pytest.mark.parametrize("kind,field", [
    ("primary_link", "source_pointer_available"),
    ("sim_particle_insert", "insertion_observed"), ("sim_particle_final", "in_memory_only"),
    ("source_reconciliation", "in_memory_links_consistent"),
    ("event_end", "source_reconciled_in_memory"),
])
@pytest.mark.parametrize("value", [False, 1, None, "true"])
def test_v2_true_observations_require_real_booleans(source_ledger, kind, field, value):
    with pytest.raises(LedgerError):
        audit(source_ledger(mutate=lambda rows: data(rows, kind).update({field: value})))


@pytest.mark.parametrize("field", ["position_mu2e_mm", "four_momentum_MeV"])
@pytest.mark.parametrize("value", [None, [], [1.0, 2.0], [1.0] * 5, "123", [True] * 3])
def test_v2_generator_vectors_require_exact_numeric_shapes(source_ledger, field, value):
    with pytest.raises(LedgerError):
        audit(source_ledger(mutate=lambda rows: data(rows, "generated_primary").update({field: value})))


@pytest.mark.parametrize("field", ["global_time_ns", "proper_time_ns"])
@pytest.mark.parametrize("value", [True, None, "1", [], float("nan"), float("inf"), -float("inf")])
def test_v2_generator_times_require_finite_numbers(source_ledger, field, value):
    with pytest.raises(LedgerError):
        audit(source_ledger(mutate=lambda rows: data(rows, "generated_primary").update({field: value})))


@pytest.mark.parametrize("path", [
    ("position_mu2e_mm", 0), ("position_mu2e_mm", 1), ("position_mu2e_mm", 2),
    ("four_momentum_MeV", 0), ("four_momentum_MeV", 1),
    ("four_momentum_MeV", 2), ("four_momentum_MeV", 3),
])
@pytest.mark.parametrize("value", [True, None, "1", float("nan"), float("inf")])
def test_v2_every_generator_vector_component_is_finite_numeric(source_ledger, path, value):
    with pytest.raises(LedgerError):
        audit(source_ledger(mutate=lambda rows: set_path(data(rows, "generated_primary"), path, value)))


@pytest.mark.parametrize("field", NATIVE_SCALARS)
@pytest.mark.parametrize("value", [True, None, "1", float("nan"), float("inf")])
def test_v2_every_source_native_scalar_is_finite_numeric(source_ledger, field, value):
    def change(rows):
        data(rows, "primary_link")["native_at_classification"][field] = value
    with pytest.raises(LedgerError):
        audit(source_ledger(mutate=change))


@pytest.mark.parametrize("path", [(field,) for field in NATIVE_VECTORS] + [
    ("position", "g4_world_mm"), ("position", "mu2e_mm"),
    ("vertex_position", "g4_world_mm"), ("vertex_position", "mu2e_mm"),
])
@pytest.mark.parametrize("value", [None, [1.0, 2.0], [1.0] * 4, [True, 0.0, 0.0],
                                   [0.0, "1", 0.0], [0.0, 0.0, float("inf")]])
def test_v2_source_native_vectors_are_strict(source_ledger, path, value):
    def change(rows):
        set_path(data(rows, "primary_link")["native_at_classification"], path, value)
    with pytest.raises(LedgerError):
        audit(source_ledger(mutate=change))


@pytest.mark.parametrize("field", NATIVE_SCALARS + NATIVE_VECTORS + ("position", "vertex_position"))
def test_v2_source_native_payload_is_not_optional(source_ledger, field):
    def change(rows):
        data(rows, "primary_link")["native_at_classification"].pop(field)
    with pytest.raises(LedgerError):
        audit(source_ledger(mutate=change))


@pytest.mark.parametrize("kind,field", [
    ("source_begin", "generator_size"), ("source_begin", "generator_protons"),
    ("source_end", "generator_size"), ("source_end", "generator_protons"),
] + [("source_reconciliation", field) for field in RECON_COUNTERS] + [
    ("event_end", field) for field in SOURCE_COUNTERS
])
@pytest.mark.parametrize("value", [99, True, -1])
def test_v2_source_counters_are_exact_and_typed(source_ledger, kind, field, value):
    with pytest.raises(LedgerError):
        audit(source_ledger(after_recount=lambda rows: data(rows, kind).update({field: value})))


@pytest.mark.parametrize("field", SOURCE_COUNTERS)
@pytest.mark.parametrize("value", [99, True, -1])
def test_v2_completion_source_counters_are_exact(source_ledger, field, value):
    with pytest.raises(LedgerError):
        audit(source_ledger(completion_mutate=lambda d: d.update({field: value})))


@pytest.mark.parametrize("field", SOURCE_COUNTERS + ("source_reconciled_in_memory", "persistent_output_verified"))
def test_v2_completion_source_fields_are_required(source_ledger, field):
    with pytest.raises(LedgerError):
        audit(source_ledger(completion_mutate=lambda d: d.pop(field)))


@pytest.mark.parametrize("field,value", [
    ("source_reconciled_in_memory", False), ("source_reconciled_in_memory", 1),
    ("persistent_output_verified", True), ("persistent_output_verified", 0),
])
def test_v2_completion_cannot_claim_persistence(source_ledger, field, value):
    with pytest.raises(LedgerError):
        audit(source_ledger(completion_mutate=lambda d: d.update({field: value})))


@pytest.mark.parametrize("hook", ["prepareEvent", "simParticleInserted", "simParticleOmitted", "reconcileParticles"])
@pytest.mark.parametrize("value", [99, True, -1, None])
def test_v2_completion_new_hook_counts_are_exact(source_ledger, hook, value):
    with pytest.raises(LedgerError):
        audit(source_ledger(completion_mutate=lambda d: d["observed_hooks"].update({hook: value})))


@pytest.mark.parametrize("hook", ["prepareEvent", "simParticleInserted", "reconcileParticles"])
def test_v2_completion_nonzero_source_hooks_cannot_disappear(source_ledger, hook):
    with pytest.raises(LedgerError):
        audit(source_ledger(completion_mutate=lambda d: d["observed_hooks"].pop(hook)))


@pytest.mark.parametrize("kind", ["generated_primary", "primary_link"])
def test_v2_generator_keys_are_unique_not_just_counted(source_ledger, kind):
    def change(rows):
        data(rows, kind, 1)["generator_key"] = data(rows, kind)["generator_key"]
    with pytest.raises(LedgerError):
        audit(source_ledger("mixed", mutate=change))


@pytest.mark.parametrize("kind", ["sim_particle_insert", "sim_particle_final"])
def test_v2_duplicate_actual_sim_keys_not_just_row_counts(source_ledger, kind):
    def change(rows):
        data(rows, kind, 1)["key"] = data(rows, kind)["key"]
    with pytest.raises(LedgerError):
        audit(source_ledger("secondary", mutate=change))


@pytest.mark.parametrize("kind,field,value", [
    ("sim_particle_insert", "generator", None),
    ("sim_particle_insert", "generator", {"product_id": 84, "key": 0}),
    ("sim_particle_insert", "generator", {"product_id": GEN_PRODUCT, "key": 9}),
    ("sim_particle_insert", "parent", {"product_id": SIM_PRODUCT, "key": 902}),
    ("sim_particle_final", "generator", None),
    ("sim_particle_final", "generator", {"product_id": GEN_PRODUCT, "key": 9}),
    ("sim_particle_final", "parent", {"product_id": SIM_PRODUCT, "key": 902}),
])
def test_v2_primary_sim_generator_and_parent_references(source_ledger, kind, field, value):
    with pytest.raises(LedgerError):
        audit(source_ledger(mutate=lambda rows: data(rows, kind).update({field: value})))


@pytest.mark.parametrize("kind", ["sim_particle_insert", "sim_particle_final"])
@pytest.mark.parametrize("field,value", [
    ("parent", None), ("parent", {"product_id": 102, "key": 902}),
    ("parent", {"product_id": SIM_PRODUCT, "key": 16}),
    ("parent", {"product_id": SIM_PRODUCT, "key": 11}),
    ("generator", {"product_id": GEN_PRODUCT, "key": 0}),
    ("parent_id", 43), ("pdg", 2212),
])
def test_v2_secondary_sim_parent_and_generator_references(source_ledger, kind, field, value):
    def change(rows):
        data(rows, kind, 1)[field] = value
    with pytest.raises(LedgerError):
        audit(source_ledger("secondary", mutate=change))


@pytest.mark.parametrize("value", [
    [], [{"product_id": SIM_PRODUCT, "key": 42}],
    [{"product_id": 102, "key": 11}],
    [{"product_id": SIM_PRODUCT, "key": 902}],
    [{"product_id": SIM_PRODUCT, "key": 11}] * 2,
    [{"product_id": SIM_PRODUCT, "key": True}], None,
])
def test_v2_final_daughter_links_are_exact_and_reciprocal(source_ledger, value):
    def change(rows):
        data(rows, "sim_particle_final")["daughters"] = value
    with pytest.raises(LedgerError):
        audit(source_ledger("secondary", mutate=change))


def test_v2_inserts_and_finals_agreeing_on_wrong_parent_still_fail_native_ancestry(source_ledger):
    def change(rows):
        for kind in ("sim_particle_insert", "sim_particle_final"):
            data(rows, kind, 1)["parent_id"] = 0
            data(rows, kind, 1)["parent"] = None
            data(rows, kind, 1)["generator"] = dict(product_id=GEN_PRODUCT, key=0)
        data(rows, "sim_particle_final")["daughters"] = []
    with pytest.raises(LedgerError):
        audit(source_ledger("secondary", mutate=change))


def test_v2_final_child_cannot_hide_an_extra_reciprocal_edge(source_ledger):
    def change(rows):
        data(rows, "sim_particle_final", 1)["daughters"] = [dict(product_id=SIM_PRODUCT, key=902)]
    with pytest.raises(LedgerError):
        audit(source_ledger("secondary", mutate=change))


def test_v2_consistent_but_wrong_primary_generator_pointer_rejected(source_ledger):
    def change(rows):
        for kind in ("sim_particle_insert", "sim_particle_final"):
            data(rows, kind)["generator"] = dict(product_id=GEN_PRODUCT, key=0)
    with pytest.raises(LedgerError):
        audit(source_ledger("mixed", mutate=change))


def test_v2_stacking_killed_primary_cannot_acquire_a_final_particle(source_ledger):
    def change(rows):
        row = deepcopy(next(row for row in rows if row["kind"] == "primary_link"))
        row["kind"] = "sim_particle_final"
        row["data"] = dict(track_id=17, parent_id=0, pdg=2212, product_id=SIM_PRODUCT,
                           key=902, sim_stage=SIM_STAGE, creation_code=1, parent=None,
                           generator=dict(product_id=GEN_PRODUCT, key=0), daughters=[],
                           in_memory_only=True, persistent_output_verified=False)
        rows.insert(next(i for i, r in enumerate(rows) if r["kind"] == "source_reconciliation"), row)
    with pytest.raises(LedgerError):
        audit(source_ledger("killed", mutate=change))


@pytest.mark.parametrize("keep_final", [False, True])
def test_v2_sim_particle_omission_refuses_source_completeness(source_ledger, keep_final):
    def change(rows):
        row = next(row for row in rows if row["kind"] == "sim_particle_insert")
        row["kind"] = "sim_particle_omitted"
        row["data"] = dict(track_id=17, parent_id=0, pdg=2212,
                           reason="sim_particle_size_limit", persistent_output_verified=False)
        if not keep_final:
            remove(rows, "sim_particle_final")
    with pytest.raises(LedgerError):
        audit(source_ledger(mutate=change))


@pytest.mark.parametrize("kind", ["sim_particle_insert", "sim_particle_final"])
@pytest.mark.parametrize("field,value", [
    ("generator", {}), ("generator", {"product_id": GEN_PRODUCT}),
    ("generator", {"product_id": GEN_PRODUCT, "key": True}),
    ("generator", {"product_id": 0, "key": 0}),
    ("generator", {"product_id": GEN_PRODUCT, "key": 0.0}),
    ("parent", []),
])
def test_v2_sim_reference_shapes_and_types_are_strict(source_ledger, kind, field, value):
    with pytest.raises(LedgerError):
        audit(source_ledger(mutate=lambda rows: data(rows, kind).update({field: value})))


@pytest.mark.parametrize("field,value", [
    ("product_id", 102), ("sim_stage", 4), ("pdg", 11),
])
def test_v2_matching_inserts_and_finals_still_require_source_and_native_identity(source_ledger, field, value):
    def change(rows):
        for kind in ("sim_particle_insert", "sim_particle_final"):
            data(rows, kind)[field] = value
    with pytest.raises(LedgerError):
        audit(source_ledger(mutate=change))


def test_v2_missing_child_final_is_not_excused_by_recounted_reconciliation(source_ledger):
    with pytest.raises(LedgerError):
        audit(source_ledger("secondary", mutate=lambda rows: remove(rows, "sim_particle_final", 1)))


def test_v2_missing_child_insert_is_not_excused_by_recounted_reconciliation(source_ledger):
    with pytest.raises(LedgerError):
        audit(source_ledger("secondary", mutate=lambda rows: remove(rows, "sim_particle_insert", 1)))


def test_v2_missing_killed_primary_link_is_not_excused_by_recounting(source_ledger):
    with pytest.raises(LedgerError):
        audit(source_ledger("killed", mutate=lambda rows: remove(rows, "primary_link")))


def test_v2_recounted_sources_do_not_hide_wrong_proton_denominator(source_ledger):
    def change(rows):
        data(rows, "generated_primary")["pdg"] = 11
        data(rows, "primary_link")["pdg"] = 11
        data(rows, "primary_link")["native_at_classification"]["pdg"] = 11
        for kind in ("stacking", "sim_particle_insert", "sim_particle_final"):
            data(rows, kind)["pdg"] = 11
        for kind in ("track_begin", "track_end"):
            data(rows, kind)["native"]["pdg"] = 11
    with pytest.raises(LedgerError):
        audit(source_ledger(mutate=change))


@pytest.mark.parametrize("field", ["max_rows", "max_volumes", "max_bytes"])
def test_v2_prepared_and_native_caps_must_agree(source_ledger, field):
    def change(rows):
        data(rows, "source_begin")[field] += 1
    with pytest.raises(LedgerError):
        audit(source_ledger(mutate=change))


def test_v2_cannot_silently_downgrade_source_rows_to_v1(source_ledger):
    def change(rows):
        for row in rows:
            if row["kind"] == "generated_primary":
                row["schema"] = "target-recording-proposal-v1"
    with pytest.raises(LedgerError):
        audit(source_ledger(mutate=change))


@pytest.mark.parametrize("history", [[], [dict(instance_id=1)], "unavailable", False])
def test_unavailable_primary_geometry_cannot_claim_a_history(source_ledger, history):
    def change(rows):
        data(rows, "primary_link")["native_at_classification"]["track_geometry"] = dict(
            instance_id=None, target_member=None,
            availability="unavailable_native_track_touchable", copy_history_root_to_leaf=history)
    with pytest.raises(LedgerError, match="unavailable track geometry"):
        audit(source_ledger(mutate=change))


@pytest.mark.parametrize("field,value", [("module_label", "other"), ("product_instance_name", "other"),
                                        ("process_name", "OTHER")])
def test_configured_explicit_tag_matches_resolved_provenance(source_ledger, field, value):
    def change(rows):
        data(rows, "source_begin")["generator_provenance"][field] = value
    with pytest.raises(LedgerError, match="configured tag/resolved provenance"):
        audit(source_ledger(mutate=change))


@pytest.mark.parametrize("tag", ["generate:beam", "generate:beam:"])
def test_unspecified_process_is_resolved_not_invented(source_ledger, tag):
    assert audit(source_ledger(mutate=lambda rows: data(rows, "source_begin").update(
        configured_primary_tag=tag)))["exposure_certified"] is False


def test_no_instance_input_tag_matches_empty_actual_instance(source_ledger):
    def change(rows):
        d = data(rows, "source_begin")
        d["configured_primary_tag"] = "generate"
        d["generator_provenance"]["product_instance_name"] = ""
    assert audit(source_ledger(mutate=change))["source_valid"] is False
