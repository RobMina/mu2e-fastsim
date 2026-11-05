"""Joint source bins retain shard grouping and sum all deposited ionizing energy."""
import numpy as np
from beamfastsim import flash_ancestry as fa


def test_joint_source_bins_and_subrun_energy_conserve():
    p = {k: np.zeros(4) for k in fa.PARTICLE}
    p.update(run=np.ones(4, int), subrun=np.array([1, 1, 2, 2]), event=np.ones(4, int),
             key=np.array([1, 2, 1, 2]), parent=np.array([-1, 1, -1, 1]), pdg=np.array([22, 11, 22, 11]),
             genidx=np.array([0, -1, 0, -1]), sz=np.array([5., 0., 15., 0.]))
    s = {k: np.zeros(2) for k in fa.STEP}
    s.update(run=np.ones(2, int), subrun=np.array([1, 2]), event=np.ones(2, int), sp=np.array([2, 2]), eion=np.array([2., 3.]))
    g = dict(run=np.ones(2, int), subrun=np.array([1, 2]), event=np.ones(2, int), idx=np.zeros(2, int), pdg=np.array([11, -11]))
    report = fa.summarize(s, fa.trace_steps(s, p, g), [0., 10., 20.], [0., 1.])
    assert report["by_subrun"]["eion_sum_mev"] == [2., 3.]
    joint = report["source_by_subrun"]
    assert joint["photon_z_bin"] == [0., 1.]
    assert joint["linked_root_generator_pdg"] == [11., -11.]
    assert sum(joint["eion_sum_mev"]) == report["eion_sum_mev"] == 5.
