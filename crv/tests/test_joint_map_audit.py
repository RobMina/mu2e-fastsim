"""Map comparison is candidate evidence, never historical or physical certification."""
import numpy as np
import pytest

from crvfastsim.objective import joint_map_audit as M


def config():
    return '''Processing file: /unavailable/rec.125026_000001.art
CrvPass1: {
 CrvDigi: { module_type: "CrvDigisFromArtdaqFragmentsFEBII" useROC4asROC2: true }
 crvOrdinal: { useDb: false filePath: "Offline/CRVConditions/data" }
 GeometryService: { inputFile: "Offline/Mu2eG4/geom/geom_common_extracted.txt" }
 DbService: { purpose: "CRV_COMMISSIONING" version: "v1_0" nearestMatch: false }
}
DL0: { online: true }
'''


def test_map_agreement_with_uninstrumented_sentinels():
    a = np.array([[0, 1, 1, 0], [1, 1, 1, 1], [2, -1, -1, -1]])
    r = M.compare_maps(a[::-1], a[:2])
    assert r['exact_mapped_agreement']
    assert r['calibration_mapped'] == 2 and r['calibration_unmapped'] == 1


def test_alternate_map_differs_and_coverage_not_excused():
    a = np.array([[0, 1, 1, 0], [1, 1, 1, 1], [2, 1, 2, 0]])
    b = np.array([[0, 1, 1, 0], [1, 1, 3, 1], [3, 1, 2, 1]])
    r = M.compare_maps(a, b)
    assert not r['exact_mapped_agreement']
    assert r['mismatched_common'] == 1 and r['mismatch_channels'] == [1]
    assert r['calibration_only_channels'] == [2] and r['candidate_only_channels'] == [3]


@pytest.mark.parametrize('bad', [
    [[0, 1, 1, 0], [0, 1, 1, 1]],
    [[0, 1, 1, 0], [1, 1, 1, 0]],
    [[0, 1, -1, -1]], [[0, 1.1, 1, 0]], [[0, -2, -2, -2]]])
def test_invalid_map_fails_closed(bad):
    with pytest.raises(ValueError):
        M.map_rows(bad)


def test_retained_config_values_not_template_defaults():
    r = M.config_evidence(config())
    assert r['database']['nearestMatch'] == 'false'
    assert r['ordinal_use_db'] is False and r['useROC4asROC2'] is True


@pytest.mark.parametrize('before,after', [('useDb: false', 'useDb: true'),
    ('useROC4asROC2: true', 'useROC4asROC2: false'),
    ('CrvPass1:', 'Different:'), ('geom_common_extracted.txt', 'geom_common_extracted_v04.txt')])
def test_different_retained_config_is_not_silently_accepted(before, after):
    with pytest.raises(ValueError):
        M.config_evidence(config().replace(before, after))
