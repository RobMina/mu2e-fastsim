"""Linked capture audits, energy accounting and the boundary-born fixture; synthetic records, not transport physics."""
import numpy as np
import pytest

from mu2efastsim import capture, capture_accounting, capture_boundary, capture_compare
from test_capture import BIG, tables  # noqa: F401 -- reuse the shared synthetic capture fixture

TEXT = "manifest|1\ng4|synthetic\nphysicsTag|synthetic-not-physics\n"


def make_links(tables):
    """Links aligned with the tracks table: the child of event 2 comes from its primary's second step."""
    t, e = tables["tracks"], tables["events"]
    child = t["parent"] != 0
    links = {field: t[field].copy() for field in (*capture.EVENT, "index", "track", "parent")}
    links.update(creatorStep=np.where(child, 1, -1).astype(np.int64), bornOnSurface=np.zeros(len(child), bool),
                 exitAtBirth=np.zeros(len(child), bool))
    n = len(e["event"])
    manifest = {field: e[field].copy() for field in capture.EVENT}
    manifest.update(linksPresent=np.ones(n, bool), linksIDsMatch=np.ones(n, bool), schema=np.ones(n, np.int32),
                    g4Version=np.full(n, "synthetic", object), manifest=np.full(n, TEXT, object),
                    manifestHash=np.full(n, capture.fnv1a(TEXT), object), nLinks=e["nTracks"].astype(np.int64),
                    nBirthExits=np.zeros(n, np.int64))
    return {"links": links, "manifest": manifest}


def add_birth_exit(tables, links, *, index=2, track=BIG + 3, pdg=None, energy=1.):
    """Append a reason-4 child of event 2's second step that starts on the +z face, outside the component."""
    t, c, e, s = tables["tracks"], tables["classifications"], tables["events"], tables["steps"]
    row = 2
    for name, table in (("tracks", t), ("classifications", c)):
        for field, value in table.items():
            tables[name][field] = np.concatenate([value, value[row:row + 1]])
    t, c = tables["tracks"], tables["classifications"]
    t["index"][-1], t["track"][-1], c["index"][-1], c["track"][-1] = index, track, index, track
    if pdg is not None:
        t["pdg"][-1] = c["pdg"][-1] = pdg
    t["steps"][-1], t["secondaries"][-1], t["reason"][-1], t["status"][-1] = 0, 0, 4, 2
    t["entered"][-1], c["killed"][-1], c["status"][-1] = False, True, 0
    for prefix in ("begin_", "end_"):
        for field, value in (("lx", 0.), ("ly", 0.), ("lz", 1.), ("x", 12.), ("y", 13.), ("z", 17.),
                             ("lpx", 0.), ("lpy", 0.), ("lpz", 1.), ("px", 1.), ("py", 0.), ("pz", 0.),
                             ("energy", energy), ("kinetic", energy), ("mass", 0.), ("charge", 0.)):
            t[prefix + field][-1] = value
        t[prefix + "volume"][-1] = BIG + 2
        t[prefix + "material"][-1] = "G4_Galactic"
    for field in capture.STATE:
        t["end_" + field][-1] = t["begin_" + field][-1]
    for field in capture.FLOATS:
        t["entrance_" + field][-1] = 0. if field != "weight" else 1.
    t["entrance_volume"][-1] = -1
    for field in capture.STATE:
        c["state_" + field][-1] = t["begin_" + field][-1]
    parent = np.flatnonzero((t["event"] == BIG + 2) & (t["parent"] == 0))[0]
    t["secondaries"][parent] += 1
    s["secondaries"][3] += 1
    e["nTracks"][1] += 1
    e["nClassifications"][1] += 1
    lk = links["links"]
    for field, value in lk.items():
        lk[field] = np.concatenate([value, value[row:row + 1]])
    for field in (*capture.EVENT, "index", "track", "parent"):
        lk[field][-1] = t[field][-1]
    lk["creatorStep"][-1], lk["exitAtBirth"][-1], lk["bornOnSurface"][-1] = 1, True, True
    links["manifest"]["nLinks"][1] += 1
    links["manifest"]["nBirthExits"][1] += 1


def test_audit_with_links(tables):
    links = make_links(tables)
    result = capture.audit(tables, 2, links=links)
    assert result["links_checked"] and result["exits_at_birth"] == 0 and result["surface_born_secondaries"] == 0
    assert not capture.audit(tables, 2)["links_checked"]


@pytest.mark.parametrize("fault", ["misaligned", "creator", "primary-creator", "hash", "census", "births", "present"])
def test_bad_links_rejected(tables, fault):
    links = make_links(tables)
    lk, mf = links["links"], links["manifest"]
    child = np.flatnonzero(lk["parent"] != 0)[0]
    if fault == "misaligned":
        lk["track"][child] += 1
    elif fault == "creator":
        lk["creatorStep"][child] = 0
    elif fault == "primary-creator":
        lk["creatorStep"][0] = 0
    elif fault == "hash":
        mf["manifestHash"][0] = "0" * 16
    elif fault == "census":
        mf["nLinks"][0] += 1
    elif fault == "births":
        mf["nBirthExits"][0] = 1
    else:
        mf["linksPresent"][1] = False
    with pytest.raises(ValueError):
        capture.audit(tables, 2, links=links)


def test_exit_at_birth_needs_its_link(tables):
    links = make_links(tables)
    add_birth_exit(tables, links)
    result = capture.audit(tables, 2, links=links)
    assert result["exits_at_birth"] == 1
    with pytest.raises(ValueError):
        capture.audit(tables, 2)
    links["links"]["exitAtBirth"][-1] = False
    with pytest.raises(ValueError):
        capture.audit(tables, 2, links=links)


def test_manifest_must_agree_between_events(tables):
    links = make_links(tables)
    other = TEXT + "extra\n"
    links["manifest"]["manifest"][1] = other
    links["manifest"]["manifestHash"][1] = capture.fnv1a(other)
    with pytest.raises(ValueError, match="manifest differs"):
        capture.audit(tables, 2, links=links)


def test_fnv1a_reference_values():
    assert capture.fnv1a("") == "cbf29ce484222325"
    assert capture.fnv1a("a") == "af63dc4c8601ec8c"


def set_processes(tables, process):
    tables["steps"]["post_process"][:] = process


def balance(tables):
    """Energies that close every step: E_pre - E_post - deposit - E_children = 0, with state continuity kept."""
    s, t, c = tables["steps"], tables["tracks"], tables["classifications"]
    s["pre_energy"][:] = [10., 10., 10., 10., 3.]
    s["post_energy"][:] = [10., 10., 10., 5., 2.]
    t["begin_energy"][:], t["end_energy"][:], t["entrance_energy"][:] = [10., 10., 3.], [10., 5., 2.], [10., 10., 3.]
    c["state_energy"][:] = t["begin_energy"]


def test_accounting_closes_on_a_balanced_capture(tables):
    balance(tables)
    set_processes(tables, "eIoni")
    result = capture_accounting.report(tables, make_links(tables), 2)
    assert result["energy_accounting_passed"] and result["conserving_steps_out_of_tolerance"] == 0
    assert result["nuclear_steps"] == 0 and not result["bank_admissible"]


def test_accounting_flags_unaccounted_energy_but_reports_nuclear_terms(tables):
    balance(tables)
    set_processes(tables, "eIoni")
    tables["steps"]["deposit"][3] += 1e-6
    result = capture_accounting.report(tables, make_links(tables), 2)
    assert result["conserving_steps_out_of_tolerance"] == 1 and not result["energy_accounting_passed"]
    assert result["worst_step"]["index"] == 1 and result["event_closure_out_of_tolerance"] == 0
    set_processes(tables, "hadElastic")
    result = capture_accounting.report(tables, make_links(tables), 2)
    assert result["energy_accounting_passed"] and result["nuclear_steps"] == result["inside_steps"]
    assert result["nuclear_processes"] == ["hadElastic"]
    assert result["totals_MeV"]["nuclear_q"] == pytest.approx(-1e-6, abs=1e-12)


def test_accounting_counts_exits_at_birth(tables):
    balance(tables)
    set_processes(tables, "eIoni")
    links = make_links(tables)
    add_birth_exit(tables, links)
    result = capture_accounting.report(tables, links, 2)
    assert result["exits_at_birth"] == 1 and result["totals_MeV"]["exit_at_birth"] == pytest.approx(1.)


def test_accounting_requires_links(tables):
    with pytest.raises(ValueError):
        capture_accounting.report(tables, None, 2)


def boundary_tables(run=17):
    """Minimal recorded outcome of the five constructed boundary-born cases."""
    events = np.arange(1, 6, dtype=np.int64)
    e = {"run": np.full(5, run), "subrun": np.zeros(5, np.int64), "event": events,
         "capturePresent": np.ones(5, bool), "eventIDsMatch": np.ones(5, bool), "bankAdmissible": np.zeros(5, bool),
         "recordBankAdmissible": np.zeros(5, bool), "errors": np.array(capture_boundary.ERRORS, dtype=object),
         "recordComplete": np.array([True, True, True, False, False])}
    t = {"event": np.repeat(events, 2), "track": np.tile([1, 2], 5), "parent": np.tile([0, 1], 5),
         "index": np.tile([0, 1], 5), "reason": np.array([2, 1, 2, 2, 2, 4, 2, 1, 2, 1]),
         "entered": np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 0], bool), "steps": np.array([3, 0, 3, 1, 3, 0, 3, 0, 3, 0])}
    rows = []
    for number in events:
        entry = number >= 4
        for k in range(3):
            rows.append((number, 1, int(k == (0 if entry else 2)), k != 0, k == 2, 1.))
        if number == 2:
            rows.append((number, 2, 0, True, True, 0.))
    s = {key: np.array(value) for key, value in zip(("event", "track", "secondaries", "preInside", "firstExit", "length"),
                                                    zip(*rows))}
    c = {"event": np.repeat(events, 2), "track": np.tile([1, 2], 5),
         "killed": np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], bool)}
    kid = t["parent"] == 1
    lk = {"event": t["event"], "track": t["track"], "parent": t["parent"], "index": t["index"],
          "creatorStep": np.where(kid, np.where(t["event"] >= 4, 0, 2), -1),
          "bornOnSurface": kid.copy(), "exitAtBirth": kid & (t["event"] == 3)}
    mf = {"linksPresent": np.ones(5, bool), "linksIDsMatch": np.ones(5, bool), "manifest": np.full(5, TEXT, object),
          "manifestHash": np.full(5, capture.fnv1a(TEXT), object)}
    return {"events": e, "tracks": t, "steps": s, "classifications": c}, {"links": lk, "manifest": mf}


def test_boundary_born_cases_pass():
    tables, links = boundary_tables()
    result = capture_boundary.report(tables, links, 17)
    assert result["boundary_born_fixture_passed"] and result["constructed_objects"]
    assert [row["exitAtBirth"] for row in result["cases"]] == [False, False, True, False, False]


@pytest.mark.parametrize("fault", ["errors", "birth", "creator", "kill", "surface", "links"])
def test_boundary_born_cases_are_exact(fault):
    tables, links = boundary_tables()
    if fault == "errors":
        tables["events"]["errors"][0] = "boundary_born_secondary_ownership_unresolved"
    elif fault == "birth":
        links["links"]["exitAtBirth"][5] = False
    elif fault == "creator":
        links["links"]["creatorStep"][1] = 1
    elif fault == "kill":
        tables["classifications"]["killed"][5] = False
    elif fault == "surface":
        links["links"]["bornOnSurface"][3] = False
    else:
        links = None
    with pytest.raises(ValueError):
        capture_boundary.report(tables, links, 17)


def test_compare_links(tables):
    a, b = make_links(tables), make_links(tables)
    assert capture_compare.compare_links(a, b)["links_compared"]
    assert capture_compare.compare_links(None, b) == {"links_compared": False, "reference_links": False,
                                                      "candidate_links": True}
    b["links"]["creatorStep"][2] = 0
    with pytest.raises(ValueError, match="links.creatorStep"):
        capture_compare.compare_links(a, b)


def with_neighbour(tables):
    """Add an unrecorded neighbour volume per event and placement paths; event 1's first exit enters it on +z."""
    from mu2efastsim import capture_partition  # noqa: F401
    v, s, e = tables["volumes"], tables["steps"], tables["events"]
    world_rows = np.flatnonzero(v["name"] == "world")
    for field, value in v.items():
        v[field] = np.concatenate([value, value[world_rows]])
    added = np.arange(len(v["event"]) - len(world_rows), len(v["event"]))
    v["id"][added], v["name"][added], v["index"][added] = BIG + 3, "NeighbourEnvelope", 2
    v["logical"][added] = "NeighbourEnvelopeLogical"
    e["nVolumes"][:] += 1
    world, env, nb = f"/{BIG + 2}:0", f"/{BIG + 2}:0/{BIG + 1}:0", f"/{BIG + 2}:0/{BIG + 3}:3"
    s["pre_placement"][:] = np.where(s["preInside"], env, world)
    s["post_placement"][:] = np.where(s["postInside"], env, world)
    s["post_placement"][1] = nb
    return tables


def test_partition_passes_through_shared_face(tables):
    from mu2efastsim import capture_partition
    balance(tables)
    set_processes(tables, "eIoni")
    with_neighbour(tables)
    result = capture_partition.report(tables, make_links(tables), 2, face="+z")
    assert result["partition_passed"] and result["first_exits_into_neighbour"] == 1
    assert result["steps_starting_in_neighbour"] == 0 and result["accounting"]["energy_accounting_passed"]


@pytest.mark.parametrize("fault", ["pre-in-neighbour", "wrong-face"])
def test_partition_faults(tables, fault):
    from mu2efastsim import capture_partition
    balance(tables)
    set_processes(tables, "eIoni")
    with_neighbour(tables)
    face = "+z"
    if fault == "pre-in-neighbour":
        tables["steps"]["pre_placement"][4] = f"/{BIG + 2}:0/{BIG + 3}:3"
    else:
        face = "+x"
    with pytest.raises(ValueError):
        capture_partition.report(tables, make_links(tables), 2, face=face)


ME = 0.51099895


def muon_decay(tables, links, loss=0.):
    """Event 2: the muon's second step is a decay to its electron child plus two neutrinos exiting at birth."""
    balance(tables)
    set_processes(tables, "eIoni")
    s, t, c = tables["steps"], tables["tracks"], tables["classifications"]
    s["post_process"][3] = "Decay"
    for table, prefixes in ((t, ("begin_", "end_", "entrance_")), (c, ("state_",))):
        for prefix in prefixes:
            table[prefix + "mass"][2] = ME
    s["pre_mass"][4] = s["post_mass"][4] = ME
    share = (5. - ME - loss) / 2.
    add_birth_exit(tables, links, index=2, track=BIG + 3, pdg=12, energy=share)
    add_birth_exit(tables, links, index=3, track=BIG + 4, pdg=-14, energy=share)


def test_declared_muon_decay_term(tables):
    links = make_links(tables)
    muon_decay(tables, links)
    result = capture_accounting.report(tables, links, 2)
    assert result["energy_accounting_passed"] and result["decay_steps"] == 1
    assert result["declared_muon_decay_steps"] == 1 and result["decay_steps_out_of_tolerance"] == 0
    assert result["totals_MeV"]["declared_decay_q"] == pytest.approx(ME)
    assert "G4MuonDecayChannel" in result["declared_terms"]


def test_extra_loss_in_a_muon_decay_fails(tables):
    links = make_links(tables)
    muon_decay(tables, links, loss=1e-6)
    result = capture_accounting.report(tables, links, 2)
    assert not result["energy_accounting_passed"] and result["decay_steps_out_of_tolerance"] == 1
    failure = result["decay_failures"][0]
    assert failure["parent_pdg"] == 13 and failure["children_pdg"] == [-14, 11, 12]
    assert failure["residual_MeV"] == pytest.approx(1e-6, abs=1e-12)


def test_undeclared_decay_channel_fails(tables):
    links = make_links(tables)
    muon_decay(tables, links)
    for table in (tables["tracks"], tables["classifications"]):
        table["pdg"][1] = 211
    result = capture_accounting.report(tables, links, 2)
    assert result["declared_muon_decay_steps"] == 0 and result["decay_steps_out_of_tolerance"] == 1
    assert result["decay_failures"][0]["q_MeV"] == pytest.approx(ME) and not result["energy_accounting_passed"]


@pytest.mark.parametrize("pdg,passes", [(1000010030, True), (11, False)])
def test_decay_without_products_is_declared_only_for_nuclei(tables, pdg, passes):
    balance(tables)
    set_processes(tables, "eIoni")
    tables["steps"]["post_process"][4] = "Decay"
    for table in (tables["tracks"], tables["classifications"]):
        table["pdg"][2] = pdg
    result = capture_accounting.report(tables, make_links(tables), 2)
    assert result["energy_accounting_passed"] is passes
    assert result["declared_nucleus_kill_steps"] == (1 if passes else 0)
    assert result["nucleus_kill_pdgs"] == ([pdg] if passes else [])
    if not passes:
        assert result["decay_failures"][0]["children_pdg"] == [] and result["decay_failures"][0]["parent_pdg"] == 11


def test_em_recoil_nucleus_takes_its_mass_from_the_material(tables):
    balance(tables)
    set_processes(tables, "CoulombScat")
    t, c = tables["tracks"], tables["classifications"]
    mass = 25126.5
    for table, prefixes in ((t, ("begin_", "end_", "entrance_")), (c, ("state_",))):
        for prefix in prefixes:
            table[prefix + "mass"][2] = mass
            table[prefix + "energy"][2] += mass
    t["pdg"][2] = c["pdg"][2] = 1000130270
    for prefix in ("pre_", "post_"):
        tables["steps"][prefix + "energy"][4] += mass
        tables["steps"][prefix + "mass"][4] = mass
    result = capture_accounting.report(tables, make_links(tables), 2)
    assert result["energy_accounting_passed"] and result["em_nuclear_recoils"] == 1
    assert result["totals_MeV"]["medium_electron_mass"] == pytest.approx(mass)


@pytest.mark.parametrize("spread,passes", [(5e-10, True), (2e-8, True), (2e-6, False)])
def test_electron_mass_spread_is_held_to_rounding_level(tables, spread, passes):
    balance(tables)
    set_processes(tables, "eIoni")
    t, c = tables["tracks"], tables["classifications"]
    t["pdg"][0] = c["pdg"][0] = 11
    for row, steps, mass in ((0, [0, 1], ME), (2, [4], ME + spread)):
        for table, prefixes in ((t, ("begin_", "end_", "entrance_")), (c, ("state_",))):
            for prefix in prefixes:
                table[prefix + "mass"][row] = mass
        tables["steps"]["pre_mass"][steps] = tables["steps"]["post_mass"][steps] = mass
    if passes:
        result = capture_accounting.report(tables, make_links(tables), 2)
        assert result["totals_MeV"]["medium_electron_mass"] == pytest.approx(ME + spread / 2, rel=1e-12)
    else:
        with pytest.raises(ValueError, match="inconsistent electron masses"):
            capture_accounting.report(tables, make_links(tables), 2)


def set_track_mass(tables, row, steps, mass):
    """Give one track and its steps a dynamic mass, keeping state continuity."""
    t, c = tables["tracks"], tables["classifications"]
    for table, prefixes in ((t, ("begin_", "end_", "entrance_")), (c, ("state_",))):
        for prefix in prefixes:
            table[prefix + "mass"][row] = mass
    if len(steps):
        tables["steps"]["pre_mass"][steps] = tables["steps"]["post_mass"][steps] = mass


@pytest.mark.parametrize("mass,passes", [(4e-6, True), (1e-3, False)])
def test_converting_photon_leaves_its_rounding_level_dynamic_mass(tables, mass, passes):
    links = make_links(tables)
    balance(tables)
    set_processes(tables, "eIoni")
    tables["steps"]["post_process"][3] = "conv"
    t, c = tables["tracks"], tables["classifications"]
    t["pdg"][1] = c["pdg"][1] = 22
    set_track_mass(tables, 1, [2, 3], mass)
    add_birth_exit(tables, links, pdg=-11, energy=5. - mass)
    result = capture_accounting.report(tables, links, 2)
    assert result["energy_accounting_passed"] is passes
    assert result["declared_photon_mass_steps"] == (1 if passes else 0)
    if passes:
        assert result["totals_MeV"]["declared_photon_mass"] == pytest.approx(mass, rel=1e-12)
        assert "G4PhotonDynamicMass" in result["declared_terms"]
    else:
        assert result["worst_step"]["residual_MeV"] == pytest.approx(mass, rel=1e-9)


@pytest.mark.parametrize("delta,extra,passes", [(5e-8, 0., True), (5e-8, 1e-6, False), (1e-5, 0., False)])
def test_decay_boost_term_needs_a_rounding_level_mass_offset(tables, delta, extra, passes):
    links = make_links(tables)
    balance(tables)
    set_processes(tables, "eIoni")
    tables["steps"]["post_process"][3] = "Decay"
    t, c = tables["tracks"], tables["classifications"]
    nominal, dynamic = 0.5, 0.5 * (1. + delta)
    for row in (0, 1):
        t["pdg"][row] = c["pdg"][row] = 211
    set_track_mass(tables, 0, [0, 1], nominal)
    set_track_mass(tables, 1, [2, 3], dynamic)
    boost = 10. * (1. - dynamic / nominal)
    add_birth_exit(tables, links, index=2, track=BIG + 3, pdg=211, energy=1.)
    t = tables["tracks"]
    set_track_mass(tables, len(t["pdg"]) - 1, [], nominal)
    add_birth_exit(tables, links, index=3, track=BIG + 4, pdg=14, energy=4. - boost - extra)
    result = capture_accounting.report(tables, links, 2)
    assert result["energy_accounting_passed"] is passes
    assert result["declared_decay_boost_steps"] == (1 if passes else 0)
    if passes:
        assert result["totals_MeV"]["declared_decay_q"] == pytest.approx(boost, rel=1e-9)
