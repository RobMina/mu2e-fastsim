"""The stock-behavior diagnostic is explicit and never mutates registry configurations."""
import copy
import dataclasses

import pytest

from crvfastsim import runconfig
from crvfastsim.response import shared_yield as S


def fcl(tmp_path, response):
    sectors = response.sectors
    p = tmp_path / "resolved.fcl"
    p.write_text("physics.producers.CrvPhotons: {\n" +
                 "CRVSectors: [" + ",".join('"'+n+'"' for n in sectors) + "]\n" +
                 "lookupTableFileNames: [" + ",".join('"LookupTable_'+s.table+'"' for s in sectors.values()) + "]\n" +
                 "scintillationYields: [" + ",".join(str(s.yield_per_mev) for s in sectors.values()) + "]\n" +
                 "reflectors: [" + ",".join(str(s.reflector) for s in sectors.values()) + "]\n" +
                 f"photonYieldScaleFactor: {response.photon_yield_scale}\nphotonYieldVariationScale: 1\n}}\n")
    return p


@pytest.mark.parametrize("name,changed,owner", [("run1a_v01", "T2", "T1"), ("run2_v10", "T5", "T3")])
def test_shared_first_yield_only(name, changed, owner, tmp_path):
    response = runconfig.get(name).response()
    before = copy.deepcopy(response)
    result, evidence = S.emulate(response, fcl(tmp_path, response))
    assert response == before
    assert set(evidence["changes"]) == {changed}
    assert evidence["changes"][changed]["first_sector"] == owner
    assert result.sectors[changed].yield_per_mev == 39794
    assert response.sectors[changed].yield_per_mev == 30482
    assert all(result.sectors[n] == s for n, s in response.sectors.items() if n != changed)
    assert "diagnostic_only" in evidence["scope"]


@pytest.mark.parametrize("field", ["yield", "table", "reflector", "order", "scale", "nochange"])
def test_mismatch_refused(field, tmp_path):
    response = runconfig.get("run1a_v01").response()
    path = fcl(tmp_path, response)
    text = path.read_text()
    if field == "yield":
        text = text.replace("30482.0", "30481.0")
    elif field == "table":
        text = text.replace("LookupTable_6000_0", "LookupTable_6000_1", 1)
    elif field == "reflector":
        text = text.replace("reflectors: [0,0]", "reflectors: [0,1]")
    elif field == "order":
        text = text.replace('"T1","T2"', '"T2","T1"')
    elif field == "scale":
        text = text.replace("photonYieldVariationScale: 1", "photonYieldVariationScale: 2")
    else:
        response = dataclasses.replace(response, sectors={n: dataclasses.replace(s, yield_per_mev=39794.)
                                                         for n, s in response.sectors.items()})
        text = fcl(tmp_path, response).read_text()
    path.write_text(text)
    with pytest.raises(ValueError):
        S.emulate(response, path)
