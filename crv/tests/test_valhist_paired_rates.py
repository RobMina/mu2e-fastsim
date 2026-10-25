"""Event-paired rate checks with empty events and nontrivial event ordering."""
import json
import numpy as np
import pytest
import uproot
from valhist.paired_rates import paired_ratio, run


def dump(path, order, clusters):
    keys = np.asarray(order, np.int32)
    counts = np.array([clusters.count(int(k)) for k in keys], dtype=np.int32)
    c = np.array(clusters, np.int32)
    with uproot.recreate(path) as f:
        for name, ids, cols in [('events', keys, dict(ndigi=counts * 4, npulse=counts * 5, ncluster=counts)),
                                ('clusters', c, dict(sectortype=np.ones(len(c), np.int32)))]:
            f.mktree('crvdump/' + name, {k: 'int32' for k in ['run', 'subrun', 'event', *cols]})
            f['crvdump/' + name].extend(dict(run=np.full(len(ids), 1400, np.int32), subrun=np.ones(len(ids), np.int32),
                                             event=ids, **cols))


def test_identical_and_scaled_pair_uncertainty():
    x = np.array([0, 0, 1, 2, 3])
    same = paired_ratio(x, x)
    assert same['relative_se'] == 0 and same['difference_se'] == 0
    scaled = paired_ratio(x, 2 * x)
    assert scaled['relative_difference'] == 1 and scaled['relative_se'] == 0
    swapped = paired_ratio([0, 1, 0, 1], [1, 0, 1, 0])
    assert swapped['relative_difference'] == 0 and swapped['relative_se'] > 0


def test_matching_empty_population_and_no_positional_pairing(tmp_path):
    a, b, out = [tmp_path / n for n in ['a.root', 'b.root', 'out.json']]
    dump(a, [1, 2, 3, 4], [2, 2, 3])
    dump(b, [3, 4, 1, 2], [3, 2, 2])
    run(str(a), str(b), str(out))
    r = json.loads(out.read_text())
    assert r['metrics']['ncluster']['events'] == 4
    assert r['metrics']['ncluster']['total_cand'] == 3
    assert r['metrics']['ncluster']['relative_se'] == 0
    assert r['metrics']['clusters_sector_1']['relative_se'] == 0


@pytest.mark.parametrize('events', [[1, 2, 3], [1, 2, 3, 5], [1, 2, 3, 3]])
def test_bad_event_coverage_rejected(tmp_path, events):
    a, b = tmp_path / 'a.root', tmp_path / 'b.root'
    dump(a, [1, 2, 3, 4], [2])
    dump(b, events, [2])
    with pytest.raises(ValueError):
        run(str(a), str(b), str(tmp_path / 'out.json'))
