"""Explicit diagnostic of the stock photon-maker lookup-sharing behavior; never a named default."""
import dataclasses
from pathlib import Path

import numpy as np

from .. import named_conditions as NC


def emulate(response, resolved_fcl):
    """Use the first configured scintillation yield per shared lookup filename, as in Offline v13_39_00."""
    path = Path(resolved_fcl)
    before = NC._sha(path)
    f = NC._fcl_values(path)
    prefix = "physics.producers.CrvPhotons."
    names, files, yields, reflectors = (f[prefix + key] for key in
                                       ("CRVSectors", "lookupTableFileNames", "scintillationYields", "reflectors"))
    if (names != list(response.sectors) or not len(names) == len(files) == len(yields) == len(reflectors)
            or len(set(names)) != len(names)):
        raise ValueError("stock photon sector order/coverage differs from named response")
    yields = np.asarray(yields, float)
    if not np.isfinite(yields).all() or np.any(yields <= 0):
        raise ValueError("invalid stock scintillation yields")
    if (float(f[prefix + "photonYieldScaleFactor"]) != response.photon_yield_scale or
            float(f[prefix + "photonYieldVariationScale"]) != response.photon_yield_variation_scale):
        raise ValueError("stock photon scale differs from named response")
    owners, sectors, changes = {}, {}, {}
    for name, filename, value, reflector in zip(names, files, yields, reflectors):
        sc = response.sectors[name]
        if (Path(filename).name != "LookupTable_" + sc.table or float(reflector) != sc.reflector or
                value != sc.yield_per_mev):
            raise ValueError(f"stock declared lookup/yield/reflector differs for {name}")
        owner, effective = owners.setdefault(filename, (name, float(value)))
        sectors[name] = dataclasses.replace(sc, yield_per_mev=effective)
        if effective != value:
            changes[name] = {"configured_yield_per_mev": float(value), "effective_yield_per_mev": effective,
                             "shared_lookup": filename, "first_sector": owner}
    if not changes:
        raise ValueError("no shared-lookup yield discrepancy to emulate")
    if NC._sha(path) != before:
        raise ValueError("resolved FCL changed during diagnostic setup")
    evidence = {"name": "stock_shared_lookup_first_yield", "scope": "diagnostic_only_not_named_configuration",
                "stock_behavior_source": "Offline v13_39_00 CRVResponse/src/CrvPhotonGenerator_module.cc",
                "resolved_fcl_sha256": before, "changes": changes,
                "warning": "Emulates stock object-sharing behavior, not calibrated yields or a production correction."}
    return dataclasses.replace(response, sectors=sectors), evidence
