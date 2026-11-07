"""Crystal placement remains opt-in, preserves deposits and rejects missing detector cells."""
from types import SimpleNamespace

import numpy as np
import pytest

from calofastsim.showers.crystal_placement import place_crystals
from calofastsim.showers.handoff import from_cechain
from calofastsim.showers.library import frame, sample
from calofastsim.showers.response import place
from test_showers_response import source


def test_identity_and_two_disks():
    library, query, _ = source()
    draw = sample(library, query, np.random.default_rng(0), orientation="disk")
    out = place_crystals(library, query, draw)
    np.testing.assert_allclose(out["position"], draw["position"], atol=1e-12)
    np.testing.assert_array_equal(out["crystal"], [10, 11])
    for name in ("energy_g4", "energy_birks", "time"):
        np.testing.assert_array_equal(out[name], draw[name])


def test_rotation_gap_without_rescaling():
    library, query, _ = source()
    library["step_relative_position"] = np.array([[4., 9., 10.], [4., 9., 110.]]) @ frame(query["direction"])[0]
    query["position"][:] = [5. / np.sqrt(2), 5. / np.sqrt(2), 100.]
    draw = sample(library, query, np.random.default_rng(0), orientation="disk")
    assert not place(library, draw)["mapped"].any()
    out = place_crystals(library, query, draw)
    assert out["mapped"].all()
    np.testing.assert_allclose(out["local_position"][:, :2], [[9., 9.], [9., 9.]], atol=1e-12)
    np.testing.assert_array_equal(out["energy_birks"], draw["energy_birks"])


def test_outside_envelope_not_rescued():
    library, query, _ = source()
    draw = sample(library, query, np.random.default_rng(0), orientation="disk")
    draw["position"][:, 0] = 21.
    out = place_crystals(library, query, draw)
    assert not out["mapped"].any()
    np.testing.assert_array_equal(out["position"], draw["position"])


def test_missing_cell_not_filled():
    library, query, _ = source()
    library["geometry_disk_size_y"][:] = 100.
    query["position"][:, 0] += 30.
    draw = sample(library, query, np.random.default_rng(0), orientation="disk")
    out = place_crystals(library, query, draw)
    assert not out["mapped"].any()
    np.testing.assert_array_equal(out["crystal_loss_reason"], [2, 2])


def test_empty_remap():
    library, query, _ = source()
    query = {k: v[:0] for k, v in query.items()}
    draw = sample(library, query, np.random.default_rng(0), orientation="disk")
    out = place_crystals(library, query, draw)
    assert out["position"].shape == (0, 3) and not out["mapped"].size


def test_handoff_default_and_opt_in_denominators():
    library, _, _ = source()
    result = SimpleNamespace(fate=np.array([1, 2]), x=np.array([[5., 0., 0.], [0., 0., 0.]]),
                             u=np.tile([0., 0., 1.], (2, 1)), T=np.full(2, 100.), t=np.full(2, 1000.))
    default = from_cechain(result, [0, 0, 0], library, np.random.default_rng(1))
    explicit = from_cechain(result, [0, 0, 0], library, np.random.default_rng(1), placement="geometric")
    for key in default["steps"]:
        np.testing.assert_array_equal(default["steps"][key], explicit["steps"][key])
    crystal = from_cechain(result, [0, 0, 0], library, np.random.default_rng(1), placement="crystal")
    assert crystal["generated_events"] == 2 and crystal["disk_front_events"] == 1
    assert crystal["steps"]["mapped"].all()
    with pytest.raises(ValueError, match="placement"):
        from_cechain(result, [0, 0, 0], library, np.random.default_rng(1), placement="snap")
