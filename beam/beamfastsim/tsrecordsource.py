"""Streaming v2 source/link checks, not native or persisted-product validation."""
from __future__ import annotations

import math

SCHEMA_V2 = "target-recording-proposal-v2"
SOURCE_KINDS = frozenset("source_begin generated_primary source_end primary_link sim_particle_insert sim_particle_omitted sim_particle_final source_reconciliation".split())
SOURCE_END_FIELDS = frozenset("source_primary_count source_proton_count primary_link_count sim_particle_inserted sim_particle_omitted sim_particle_final source_reconciled_in_memory persistent_output_verified".split())
SOURCE_HOOKS = frozenset(("prepareEvent", "simParticleInserted", "simParticleOmitted", "reconcileParticles"))
V2_ONLY_FIELDS = SOURCE_END_FIELDS | frozenset("generator_product_id sim_product_id generator_key generator_size generator_protons configured_primary_tag process_history_id generator_provenance generator_id position_mu2e_mm four_momentum_MeV global_time_ns proper_time_ns generator_weight native_at_classification source_pointer_available insertion_observed in_memory_only in_memory_links_consistent exposure_certified".split())


class SourceAudit:
    """Keep row-bounded identities while the native auditor consumes its one pass."""

    def __init__(self, need, uint, geometry):
        self.need, self.uint, self.geometry = need, uint, geometry
        self.phase = "start"
        self.header = None
        self.caps = None
        self.generated = []
        self.protons = 0
        self.links = {}
        self.linked_keys = set()
        self.inserted = {}
        self.by_track = {}
        self.finals = set()
        self.expected_children = {}
        self.pending = None
        self.hooks = dict.fromkeys(SOURCE_HOOKS, 0)

    def fields(self, data, names):
        self.need(type(data) is dict and set(data) == set(names.split()), "source payload fields")

    def eq(self, data, key, expected):
        self.need(self.uint(data[key]) == expected, "source counter/identity mismatch: " + key)

    def signed(self, value):
        self.need(type(value) is int, "invalid source signed integer")
        return value

    def number(self, value):
        self.need(type(value) in (int, float), "invalid source number")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        self.need(finite, "nonfinite source number")

    def vector(self, value, size):
        self.need(type(value) is list and len(value) == size, "invalid source vector")
        for item in value:
            self.number(item)

    def text(self, value, nonempty=True):
        self.need(type(value) is str and (bool(value) or not nonempty), "invalid source string")

    def pointer(self, value):
        self.fields(value, "product_id key")
        product, key = self.uint(value["product_id"]), self.uint(value["key"])
        self.need(product > 0, "null source product identity")
        return product, key

    def native(self, data, tid, pdg):
        self.need(type(data) is dict, "missing native classification state")
        self.eq(data, "track_id", tid)
        self.eq(data, "parent_id", 0)
        self.need(self.signed(data["pdg"]) == pdg, "primary native PDG mismatch")
        self.eq(data, "current_step_number", 0)
        self.signed(data["native_status"])
        self.geometry(data["track_geometry"])
        for key in ("position", "vertex_position"):
            self.fields(data[key], "g4_world_mm mu2e_mm")
            for vector in data[key].values():
                self.vector(vector, 3)
        for key in ("momentum_MeV", "momentum_direction", "polarization", "vertex_momentum_direction"):
            self.vector(data[key], 3)
        for key in ("global_time_ns", "local_time_ns", "proper_time_ns", "kinetic_energy_MeV",
                    "total_energy_MeV", "weight", "track_length_mm", "velocity_mm_per_ns",
                    "dynamic_charge_eplus", "dynamic_mass_MeV", "vertex_kinetic_energy_MeV"):
            self.number(data[key])
        self.text(data["particle_name"])
        self.need(data["vertex_logical_volume"] is None or type(data["vertex_logical_volume"]) is str,
                  "invalid native vertex volume")
        self.need(data["vertex_placement_membership"] == "unresolved_no_vertex_touchable_claim",
                  "native vertex membership claim")
        if data["creator_process"] is not None:
            self.fields(data["creator_process"], "name type subtype")
            self.text(data["creator_process"]["name"])
            self.signed(data["creator_process"]["type"])
            self.signed(data["creator_process"]["subtype"])

    def particle(self, data, final=False):
        common = "track_id parent_id pdg product_id key sim_stage creation_code parent generator persistent_output_verified"
        self.fields(data, common + (" daughters in_memory_only" if final else " insertion_observed"))
        tid, parent = self.uint(data["track_id"]), self.uint(data["parent_id"])
        self.need(tid > 0, "zero inserted track")
        self.signed(data["pdg"])
        self.signed(data["creation_code"])
        self.eq(data, "product_id", self.header["sim_product_id"])
        self.eq(data, "sim_stage", self.header["sim_stage"])
        key = (self.uint(data["product_id"]), self.uint(data["key"]))
        parent_key = None if data["parent"] is None else self.pointer(data["parent"])
        generator = None if data["generator"] is None else self.pointer(data["generator"])
        self.need(data["persistent_output_verified"] is False, "source persistence claim")
        if final:
            self.need(data["in_memory_only"] is True, "final not in-memory-only")
        else:
            self.need(data["insertion_observed"] is True, "missing actual insertion observation")
        identity = (tid, parent, data["pdg"], key, data["sim_stage"], data["creation_code"], parent_key, generator)
        return tid, key, parent_key, generator, identity

    def totals(self):
        return dict(source_primary_count=len(self.generated), source_proton_count=self.protons,
                    primary_link_count=len(self.links), sim_particle_inserted=len(self.inserted),
                    sim_particle_omitted=0, sim_particle_final=len(self.finals))

    def end_counters(self, data):
        for key, value in self.totals().items():
            self.eq(data, key, value)
        self.need(data["source_reconciled_in_memory"] is True and data["persistent_output_verified"] is False,
                  "missing source reconciliation or persistence claim")

    def observe(self, kind, data, phase, active, tracks):
        """Validate source rows and native-hook adjacency; return whether consumed."""
        if self.pending is not None:
            expected, tid = self.pending
            native = data.get("native")
            actual_tid = data.get("track_id") if kind == "stacking" else native.get("track_id") if type(native) is dict else None
            self.need(kind == expected and type(actual_tid) is int and actual_tid == tid,
                      "source hook not immediately followed by native callback")
            self.pending = None
        if self.phase in ("finals", "reconciled", "ended", "complete"):
            allowed = {"finals": {"sim_particle_final", "source_reconciliation"},
                       "reconciled": {"event_end"}, "ended": {"completion"}, "complete": set()}
            self.need(kind in allowed[self.phase], "callback after source finalization")

        if kind == "source_begin":
            self.need(self.phase == "start" and phase == "start", "source begin order")
            self.fields(data, "generator_product_id sim_product_id sim_stage generator_size generator_protons configured_primary_tag process_history_id generator_provenance max_rows max_volumes max_bytes persistent_output_verified exposure_certified")
            for key in ("generator_product_id", "sim_product_id"):
                self.need(self.uint(data[key]) > 0, "null source product ID")
            for key in ("sim_stage", "generator_size", "generator_protons"):
                self.uint(data[key])
            self.need(data["generator_protons"] <= data["generator_size"], "source proton count exceeds size")
            self.text(data["configured_primary_tag"])
            self.text(data["process_history_id"])
            provenance = data["generator_provenance"]
            self.fields(provenance, "branch_name module_label product_instance_name process_name friendly_class_name parameter_set_ids")
            for key in ("branch_name", "module_label", "process_name", "friendly_class_name"):
                self.text(provenance[key])
            self.text(provenance["product_instance_name"], nonempty=False)
            tag = data["configured_primary_tag"].split(":")
            self.need(1 <= len(tag) <= 3 and tag[0] == provenance["module_label"]
                      and (tag[1] if len(tag) > 1 else "") == provenance["product_instance_name"]
                      and (len(tag) < 3 or not tag[2] or tag[2] == provenance["process_name"]),
                      "configured tag/resolved provenance mismatch")
            self.need(type(provenance["parameter_set_ids"]) is list and provenance["parameter_set_ids"],
                      "invalid parameter set IDs")
            for value in provenance["parameter_set_ids"]:
                self.text(value)
            self.need(data["persistent_output_verified"] is False and data["exposure_certified"] is False,
                      "source certification claim")
            self.caps = tuple(self.uint(data[key]) for key in ("max_rows", "max_volumes", "max_bytes"))
            self.need(all(self.caps), "zero source cap")
            self.header, self.phase = data, "generating"
            self.hooks["prepareEvent"] += 1
        elif kind == "generated_primary":
            self.need(self.phase == "generating", "generated primary order")
            self.fields(data, "generator_product_id generator_key pdg generator_id position_mu2e_mm four_momentum_MeV global_time_ns proper_time_ns generator_weight")
            self.eq(data, "generator_product_id", self.header["generator_product_id"])
            self.eq(data, "generator_key", len(self.generated))
            self.need(len(self.generated) < self.header["generator_size"], "extra generated primary")
            pdg = self.signed(data["pdg"])
            self.signed(data["generator_id"])
            self.vector(data["position_mu2e_mm"], 3)
            self.vector(data["four_momentum_MeV"], 4)
            self.number(data["global_time_ns"])
            self.number(data["proper_time_ns"])
            self.need(data["generator_weight"] is None, "unsupported source weight")
            self.generated.append(pdg)
            self.protons += pdg == 2212
        elif kind == "source_end":
            self.need(self.phase == "generating", "source end order")
            self.fields(data, "generator_size generator_protons")
            for key, value in (("generator_size", len(self.generated)), ("generator_protons", self.protons)):
                self.eq(data, key, value)
                self.eq(self.header, key, value)
            self.phase = "source_done"
        elif kind == "event_begin":
            self.need(self.phase == "source_done", "event before complete source")
            self.need(tuple(self.uint(data[key]) for key in ("max_rows", "max_volumes", "max_bytes")) == self.caps,
                      "prepare/event cap mismatch")
            self.phase = "inventory"
        elif kind == "volume":
            self.need(self.phase == "inventory", "source inventory order")
        elif kind == "inventory_end":
            self.need(self.phase == "inventory", "source inventory end order")
            self.phase = "callbacks"
        elif kind == "primary_link":
            self.need(self.phase == "callbacks" and phase == "callbacks" and active is None, "primary link order")
            self.fields(data, "track_id generator_product_id generator_key pdg classification_native native_at_classification source_pointer_available")
            tid, key = self.uint(data["track_id"]), self.uint(data["generator_key"])
            self.need(tid > 0 and tid not in tracks and tid not in self.links and key < len(self.generated) and key not in self.linked_keys,
                      "duplicate or unknown primary link")
            self.eq(data, "generator_product_id", self.header["generator_product_id"])
            self.need(self.signed(data["pdg"]) == self.generated[key], "linked generator PDG mismatch")
            self.need(self.signed(data["classification_native"]) in (0, -9), "unsupported primary classification")
            self.need(data["source_pointer_available"] is True, "missing actual source pointer")
            self.native(data["native_at_classification"], tid, data["pdg"])
            self.links[tid] = (key, data["classification_native"])
            self.linked_keys.add(key)
            self.pending = ("stacking", tid)
        elif kind == "stacking":
            self.need(self.phase == "callbacks", "source stacking order")
            tid, parent = self.uint(data["track_id"]), self.uint(data["parent_id"])
            if parent == 0:
                self.need(tid in self.links, "primary stacking without actual generator link")
                key, classification = self.links[tid]
                self.need(self.signed(data["pdg"]) == self.generated[key] and self.signed(data["classification_native"]) == classification,
                          "primary stacking/link mismatch")
            else:
                self.need(tid not in self.links, "secondary has primary link")
        elif kind == "sim_particle_insert":
            self.need(self.phase == "callbacks" and phase == "callbacks" and active is None, "particle insertion order")
            tid, key, parent_key, generator, identity = self.particle(data)
            self.need(tid in tracks and tracks[tid]["life"] == "stacked" and tid not in self.by_track and key not in self.inserted,
                      "duplicate or unaccepted particle insertion")
            track = tracks[tid]
            self.need(data["parent_id"] == track["parent"] and data["pdg"] == track["pdg"], "inserted/native identity mismatch")
            if track["parent"] == 0:
                self.need(tid in self.links and parent_key is None and generator == (self.header["generator_product_id"], self.links[tid][0]),
                          "inserted primary generator mismatch")
            else:
                self.need(generator is None and parent_key is not None and self.by_track.get(track["parent"]) == parent_key,
                          "inserted parent does not match native parent track")
                self.expected_children.setdefault(parent_key, set()).add(key)
            self.inserted[key], self.by_track[tid] = identity, key
            self.pending = ("track_begin", tid)
            self.hooks["simParticleInserted"] += 1
        elif kind == "sim_particle_omitted":
            self.fields(data, "track_id parent_id pdg reason persistent_output_verified")
            self.uint(data["track_id"])
            self.uint(data["parent_id"])
            self.signed(data["pdg"])
            self.need(data["reason"] == "sim_particle_size_limit" and data["persistent_output_verified"] is False,
                      "invalid particle omission")
            self.need(False, "omitted SimParticle prevents complete source-shaped diagnostic")
        elif kind == "track_begin":
            self.need(self.phase == "callbacks" and self.uint(data["native"]["track_id"]) in self.by_track,
                      "track begin without actual insertion")
        elif kind == "sim_particle_final":
            self.need(self.phase in ("callbacks", "finals") and phase == "callbacks" and active is None and
                      (self.phase == "finals" or all(t["life"] in ("killed", "ended") for t in tracks.values())),
                      "particle final before lifecycle end")
            tid, key, parent_key, generator, identity = self.particle(data, final=True)
            self.need(key in self.inserted and key not in self.finals and identity == self.inserted[key], "final/inserted identity mismatch")
            self.need(type(data["daughters"]) is list, "invalid final daughters")
            daughters = [self.pointer(value) for value in data["daughters"]]
            self.need(len(daughters) == len(set(daughters)) and set(daughters) == self.expected_children.get(key, set()),
                      "nonreciprocal, duplicate or dangling final daughter")
            self.finals.add(key)
            self.phase = "finals"
        elif kind == "source_reconciliation":
            self.need(self.phase in ("callbacks", "finals") and phase == "callbacks" and active is None and
                      all(t["life"] in ("killed", "ended") for t in tracks.values()), "source reconciliation order")
            self.fields(data, "generator_size generator_protons primary_links inserted_particles omitted_particles final_particles in_memory_links_consistent persistent_output_verified exposure_certified")
            self.need(len(self.linked_keys) == len(self.generated) and self.finals == set(self.inserted) and
                      set(self.by_track) == {tid for tid, t in tracks.items() if t["life"] == "ended"}, "incomplete source reconciliation")
            values = dict(generator_size=len(self.generated), generator_protons=self.protons, primary_links=len(self.links),
                          inserted_particles=len(self.inserted), omitted_particles=0, final_particles=len(self.finals))
            for key, value in values.items():
                self.eq(data, key, value)
            self.need(data["in_memory_links_consistent"] is True and data["persistent_output_verified"] is False and
                      data["exposure_certified"] is False, "invalid source reconciliation claims")
            self.phase = "reconciled"
            self.hooks["reconcileParticles"] += 1
        elif kind == "event_end":
            self.need(self.phase == "reconciled", "event end without source reconciliation")
            self.end_counters(data)
            self.phase = "ended"
        elif kind == "completion":
            self.need(self.phase == "ended", "source completion order")
            self.end_counters(data)
            self.phase = "complete"
        else:
            self.need(self.phase == "callbacks", "native callback before source/inventory")
        return kind in SOURCE_KINDS

    def result(self):
        self.need(self.phase == "complete" and self.pending is None, "incomplete source stream")
        return dict(source_reconciliation="structurally_consistent_in_memory", persistent_output_verified=False,
                    source_counts=self.totals())
