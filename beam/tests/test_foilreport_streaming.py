"""Streaming ROOT census parity with the retained eager calculation; no G4 needed."""
import json
from tempfile import TemporaryFile

import numpy as np
import pytest
import uproot

from beamfastsim import foilreport as fr
from beamfastsim.reference_audit import _identity


def sample(dtype=np.float32, pdg=13):
    case = dict(energy_MeV=2., energy_max_MeV=3., thickness_mm=.2,
                cosine=1., cosine_min=.8, pdg=pdg, events=6)
    particles, steps = [], []
    for event in range(1, 7):  # fixture generation only; auditor has no event loop
        trk = event + 10
        particles.append(dict(run=1, evt=event, trk=trk, pdg=pdg, parent_id=-1, code=56,
            pstop=0 if event in (2, 3, 5) else 10, zstop=.01 if event in (2, 3, 5) else 20,
            codestop={2: 32, 3: 0, 5: 14}.get(event, 182), tstop=5., p=25., z=-.101, time=0.))
        states = [(1, -.101, 2., 0.), (2, -.1, 2., .01), (2, -.02, 1.5, .02)]
        if event in (1, 4, 6):
            states += [(1, -.11 if event == 4 else .1, .05 if event == 4 else 1., .03),
                       (2, .01, .5, .04), (1, .1, .2, .05)]  # reentry must not change first exit
        else:
            states += [(2, .01, 0., .04)]
        for sid, z, energy, time in states:
            momentum = np.sqrt(energy * (energy + 2 * fr.MMU))
            cosine = .8 if event == 6 else 1.
            steps.append(dict(run=1, evt=event, trk=trk, pdg=pdg, sid=sid, code=31,
                x=time * .02, y=time * .01, z=z, px=momentum * np.sqrt(1 - cosine**2),
                py=0., pz=momentum * cosine, ke=energy, time=time))
            # Interleaved nonprimary rows can be out of event order, with nonfinite
            # kinematics: only their run/event/track IDs are validated by census.
            steps.append(dict(steps[-1], evt=1, trk=999, pdg=11, x=np.nan, code=29))
        secondaries = [
            (trk, 11, 31, 0., .02, .5),          # accepted direct delta
            (trk, 11, 31, .100002, .030001, .3), # boundary tolerances
            (trk, 22, 30, 0., .02, .2),          # direct brems
            (999, 11, 31, 0., .02, .9),         # descendant, not direct
            (trk, -11, 31, 0., .02, .9),        # wrong PDG
            (trk, 11, 30, 0., .02, .9),         # wrong creation code
            (trk, 11, 31, .100003, .02, .9),     # outside foil
            (trk, 11, 31, 0., .04, .7),         # after exit (but accepted for stoppers)
            (trk, 11, 31, 0., 6., .9),          # after endpoint
        ]
        for parent, child_pdg, code, z, time, momentum in secondaries:
            particles.append(dict(particles[-1], evt=event, trk=200 + len(particles),
                parent_id=parent, pdg=child_pdg, code=code, z=z, time=time, p=momentum))
    rng = np.random.default_rng(71)
    particles = [particles[i] for i in rng.permutation(len(particles))]
    return ({k: np.array([r[k] for r in particles], dtype=dtype) for k in particles[0]},
            {k: np.array([r[k] for r in steps], dtype=dtype) for k in steps[0]}, case)


def write_root(path, particles, steps, cycles=False):
    with uproot.recreate(path) as root:
        if cycles:
            root.mktree("readback/tpart", {k: v[:1] for k, v in particles.items()})
            root.mktree("readback/ntstepper", {k: v[:1] for k, v in steps.items()})
        # Large unused columns must not be read.
        root.mktree("readback/tpart", dict(particles, unused=np.zeros(len(particles["evt"]))))
        root.mktree("readback/ntstepper", dict(steps, unused=np.zeros(len(steps["evt"]))))


def assert_equal(actual, expected):
    ar, ast = actual
    er, est = expected
    assert ast.keys() == est.keys()
    for key in est:
        np.testing.assert_array_equal(ast[key], est[key], err_msg=key)
        assert ast[key].dtype == est[key].dtype, key
    assert ar.keys() == er.keys()
    for key in er:
        if isinstance(er[key], (float, list)):
            np.testing.assert_equal(ar[key], er[key], err_msg=key)
        else:
            assert ar[key] == er[key], key


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 19, 10000, "1 kB"])
@pytest.mark.parametrize("dtype,pdg", [(np.float32, 13), (np.float64, -13)])
def test_chunk_boundaries_match_eager(tmp_path, chunk, dtype, pdg):
    particles, steps, case = sample(dtype, pdg)
    path = tmp_path / "foil.root"
    write_root(path, particles, steps)
    expected = fr.census(particles, steps, case)
    with uproot.open(path) as root:
        actual = fr.census_streaming(root["readback/tpart"], root["readback/ntstepper"], case, step_size=chunk)
    assert_equal(actual, expected)
    assert actual[0]["physical_stops"] == 2
    assert actual[0]["other"] == 1
    assert actual[0]["back_exits"] == actual[0]["subthreshold_exits"] == 1


@pytest.mark.parametrize("mutation,error", [
    ("missing_primary", "one requested primary"), ("wrong_primary", "one requested primary"),
    ("duplicate_primary", "coverage"), ("primary_run", "coverage"), ("creation", "creation code"),
    ("fractional_step_event", "integer IDs"), ("fractional_step_track", "integer IDs"),
    ("nonprimary_run", "outside primary"), ("nonprimary_event", "outside primary"),
    ("wrong_step_pdg", "identity/order"), ("nonfinite", "nonfinite primary"),
    ("order", "identity/order"), ("no_entry", "never enters"),
    ("bad_secondary_event", "secondary event"), ("fractional_secondary", "integer IDs"),
    ("energy", "gun energy"), ("direction", "gun range"), ("msc", "scattering"),
])
def test_streaming_preserves_guards(tmp_path, mutation, error):
    p, s, case = sample(np.float64)
    primary = np.flatnonzero(p["parent_id"] == -1)
    secondary = np.flatnonzero(p["parent_id"] != -1)
    if mutation == "missing_primary":
        p = {k: np.delete(v, primary[0]) for k, v in p.items()}
    elif mutation == "wrong_primary":
        p["pdg"][primary[0]] = 11
    elif mutation == "duplicate_primary":
        p["evt"][primary[0]] = p["evt"][primary[1]]
    elif mutation == "primary_run":
        p["run"][primary[0]] = 2
    elif mutation == "creation":
        p["code"][primary[0]] = 31
    elif mutation == "fractional_step_event":
        s["evt"][1] = 1.5
    elif mutation == "fractional_step_track":
        s["trk"][1] = 1.5
    elif mutation == "nonprimary_run":
        s["run"][1] = 2
    elif mutation == "nonprimary_event":
        s["evt"][1] = 7
    elif mutation == "wrong_step_pdg":
        s["pdg"][0] = 11
    elif mutation == "nonfinite":
        s["ke"][0] = np.nan
    elif mutation == "order":
        # Individually valid primary rows, invalid order across chunk boundaries.
        s = {k: np.roll(v, 12) for k, v in s.items()}
    elif mutation == "no_entry":
        s["sid"][:] = 1
    elif mutation == "bad_secondary_event":
        p["evt"][secondary[0]] = 7
    elif mutation == "fractional_secondary":
        p["evt"][secondary[0]] = 1.5
    elif mutation == "energy":
        case["energy_MeV"] = 2.5
    elif mutation == "direction":
        case["cosine_min"] = .9
    elif mutation == "msc":
        case["no_msc"] = True
        s["code"][0] = 59
    with pytest.raises(ValueError, match=error):
        fr.census(p, s, case)
    path = tmp_path / "foil.root"
    write_root(path, p, s)
    with uproot.open(path) as root, pytest.raises(ValueError, match=error):
        fr.census_streaming(root["readback/tpart"], root["readback/ntstepper"], case, step_size=1)


@pytest.mark.parametrize("size", [0, 1, 2, 3, 16, 65539, 131074])
def test_exact_external_quantiles(size):
    values = np.random.default_rng(38).lognormal(0, 10, size)
    if size > 16:
        values[::13] = 0.
        values[::17] = values[7]  # duplicate radix prefixes / equal order statistics
    with TemporaryFile() as file:
        energies = fr._HardEnergies(file)
        for part in np.array_split(values, 7):
            energies.add(part)
        np.testing.assert_array_equal(energies.quantiles(), np.quantile(values, [0, .5, 1]) if size else [])


def test_nan_hard_quantiles_preserved():
    with TemporaryFile() as file:
        energies = fr._HardEnergies(file)
        energies.add([1., np.nan, 3.])
        np.testing.assert_array_equal(energies.quantiles(), [np.nan] * 3)


def make_receipt(directory, p, s, case):
    path = directory / "foil.root"
    write_root(path, p, s, cycles=True)
    (directory / "job.log").write_text("Art has completed and will exit with status 0.\n")
    receipt = dict(complete=True, artifacts=[_identity(directory / "job.log")],
                   output=_identity(path), case=case, wall_s=1.2)
    # Exercise the copy-back fallback as well as identity validation.
    for item in receipt["artifacts"] + [receipt["output"]]:
        item["path"] = "/missing/worker/directory/" + item["path"].rsplit("/", 1)[-1]
    (directory / "receipt.json").write_text(json.dumps(receipt))


def test_audit_highest_cycle_projection_and_no_eager_reads(tmp_path, monkeypatch):
    p, s, case = sample()
    make_receipt(tmp_path, p, s, case)
    original = fr.highest
    calls = []

    class BoundedTree:
        def __init__(self, tree):
            self.tree = tree

        def iterate(self, expressions, *, step_size, library):
            assert "unused" not in expressions
            assert step_size == 3 and library == "np"
            calls.append(tuple(expressions))
            for rows in self.tree.iterate(expressions, step_size=step_size, library=library):
                assert len(rows["evt"]) <= 3
                yield rows

        def arrays(self, **kwargs):
            pytest.fail("unbounded tree read")

    def highest(root, name, *, arrays=True):
        assert arrays is False
        tree, meta = original(root, name, arrays=False)
        return BoundedTree(tree), meta

    monkeypatch.setattr(fr, "highest", highest)
    report, store, receipt = fr.audit(tmp_path, step_size=3)
    assert len(calls) == 3  # primary pass, step pass, secondary pass
    assert report.pop("trees") == dict(
        particles=dict(key="readback/tpart;2", entries=len(p["evt"]),
                       cycles=["readback/tpart;1", "readback/tpart;2"]),
        steps=dict(key="readback/ntstepper;2", entries=len(s["evt"]),
                   cycles=["readback/ntstepper;1", "readback/ntstepper;2"]))
    assert report.pop("case") == case
    assert report.pop("wall_s") == receipt["wall_s"]
    assert_equal((report, store), fr.census(p, s, case))
    # Receipt identities must still fail before attempting any ROOT reads.
    (tmp_path / "job.log").write_text("changed")
    with pytest.raises(ValueError, match="G4 artifact"):
        fr.audit(tmp_path)


@pytest.mark.parametrize("chunk", [1, 11, 257])
@pytest.mark.parametrize("size", [1, 4097])
def test_one_long_event_and_empty_secondary_population(tmp_path, chunk, size):
    # No event-sized buffer may be necessary; also stress row-order floating
    # accumulation using irregular float32 chords spanning many tiny chunks.
    p, s, case = sample()
    index = np.flatnonzero((p["parent_id"] == -1) & (p["evt"] == 2))[0]
    p = {k: v[index:index + 1].copy() for k, v in p.items()}
    p["evt"][:] = 1
    case["events"] = 1
    s = {k: np.full(size, v[2], dtype=v.dtype) for k, v in s.items()}
    s["trk"][:] = p["trk"][0]
    s["evt"][:] = 1
    s["sid"][:] = 2
    s["z"] = np.random.default_rng(16).uniform(-.1, .1, size).astype(np.float32)
    path = tmp_path / "foil.root"
    write_root(path, p, s)
    with uproot.open(path) as root:
        actual = fr.census_streaming(root["readback/tpart"], root["readback/ntstepper"], case, step_size=chunk)
    assert_equal(actual, fr.census(p, s, case))
    assert actual[0]["hard_energy_MeV_quantiles"] == []
    assert actual[0]["var_exit_MeV2"] is None


def test_audit_empty_steps_and_missing_highest_tree(tmp_path):
    p, s, case = sample()
    path = tmp_path / "empty.root"
    with uproot.recreate(path) as root:
        root.mktree("readback/tpart", p)
        root.mktree("readback/ntstepper", {k: v.dtype for k, v in s.items()})
    with uproot.open(path) as root:
        with pytest.raises(ValueError, match="never enters"):
            fr.census_streaming(root["readback/tpart"], root["readback/ntstepper"], case, step_size=1)
        with pytest.raises(ValueError, match="missing tree"):
            fr.highest(root, "readback/missing", arrays=False)
