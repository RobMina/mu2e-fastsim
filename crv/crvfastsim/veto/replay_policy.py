"""Bounded Run1A yield contracts: intended configuration versus legacy-stock compatibility."""
from __future__ import annotations

import dataclasses
import hashlib
import json

from .fastframes import run_config

INTENDED = "intended_per_sector_v1"
LEGACY_STOCK = "legacy_stock_shared_lut_v1"
POLICIES = (INTENDED, LEGACY_STOCK)
TIMING = "Current fast step folding; pbtime rounded to float32 as in stock event products; no reco smearing."
LUT = "CRVConditions/v6_0/LookupTable_6000_0"
LEGACY_WRITERS = {
    "465bc1ac7a22ddd852f8ff92683ec2eb2a8027a6ab1e3e386bc4cfde87f04ba0": INTENDED,
    "795afb54b83cc475e38254839a88373d9af9e4baec56d7682a6189efd8ee3891": None,
}
LEGACY_FRAMES = "88a01353cf04fe0b2025abd792c923bb6b10316e1c850942adfe792c2fe71022"


def stock_shared_table_yields(response):
    """Copy first-owner yields in supplied sector order; compare literal table strings only."""
    first, sectors = {}, {}
    for name, sector in response.sectors.items():
        value = first.setdefault(sector.table, sector.yield_per_mev)
        sectors[name] = dataclasses.replace(sector, yield_per_mev=value)
    return dataclasses.replace(response, sectors=sectors)


def apply_policy(response, policy):
    """Apply a pinned nominal Run1A contract, rejecting unreviewed order/table/configuration changes."""
    if policy not in POLICIES:
        raise ValueError(f"unknown yield policy: {policy}")
    nominal = {"sectors": {"T1": {"table": "6000_0", "reflector": 0, "yield_per_mev": 39794.0},
                           "T2": {"table": "6000_0", "reflector": 0, "yield_per_mev": 30482.0}},
               "photon_yield_scale": 0.90, "photon_yield_variation_scale": 1.0,
               "speed_ratio": {}, "atten_ratio": {}}
    if list(response.sectors) != ["T1", "T2"] or dataclasses.asdict(response) != nominal:
        raise ValueError("yield policy supports only ordered nominal Run1A response")
    effective = stock_shared_table_yields(response) if policy == LEGACY_STOCK else dataclasses.replace(
        response, sectors={k: dataclasses.replace(v) for k, v in response.sectors.items()})
    contract = {
        "policy": policy, "run_config": "run1a_v01", "sector_order": list(response.sectors),
        "configured_response": dataclasses.asdict(response), "effective_response": dataclasses.asdict(effective),
        "stock_lut_filenames": {k: LUT for k in response.sectors},
        "yield_owner": {k: "T1" if policy == LEGACY_STOCK else k for k in response.sectors},
        "stock_source_scope": {
            "verified_releases": ["v13_33_00", "v13_39_00"],
            "source": "CRVResponse/src/CrvPhotonGenerator_module.cc",
            "source_sha256": "1b5aef320f6f344d607fbf3c18e92b075b9d4daed84532d61ef78938dccd826e",
            "fcl": "CRVResponse/fcl/epilog_run1a_v01.fcl",
            "fcl_sha256": "dc3fc25b201ddfd62950531c6fca5df34429747d515b89572c6182361b946508",
            "scope": "Yield ownership only; not RNG equivalence, physical calibration or full response closure.",
        },
    }
    return effective, contract


def metadata_identity(meta, legacy_policy=None):
    """Validate a new sidecar or explicitly recover a hash-pinned historical replay contract."""
    contract = meta.get("response_contract")
    recovery = None
    if contract is None:
        if legacy_policy not in POLICIES:
            raise ValueError("legacy sidecar requires explicit --legacy-yield-policy")
        code = meta.get("code_sha256", {})
        writer = code.get("replay_fast.py")
        if writer not in LEGACY_WRITERS or code.get("fastframes.py") != LEGACY_FRAMES:
            raise ValueError("unrecognized legacy replay code; cannot infer response contract")
        old = LEGACY_WRITERS[writer]
        if old is None and ("stock_shared_table_yields_diagnostic" not in meta or "effective_yields" not in meta):
            raise ValueError("legacy policy-aware writer is missing yield metadata")
        flag = meta.get("stock_shared_table_yields_diagnostic", False)
        if type(flag) is not bool or (old == INTENDED and flag):
            raise ValueError("invalid legacy diagnostic flag")
        actual = LEGACY_STOCK if flag else INTENDED
        if actual != legacy_policy:
            raise ValueError("legacy yield policy disagrees with recorded diagnostic")
        _, contract = apply_policy(run_config("run1a_v01")[0], actual)
        recovery = "explicit policy plus recognized writer/fastframes hashes; not a full historical code snapshot"
    policy = contract.get("policy")
    _, expected = apply_policy(run_config("run1a_v01")[0], policy)
    if contract != expected:
        raise ValueError("response contract differs from pinned Run1A profile")
    if not recovery and "stock_shared_table_yields_diagnostic" not in meta:
        raise ValueError("response contract is missing diagnostic flag")
    flag = meta.get("stock_shared_table_yields_diagnostic", False)
    if type(flag) is not bool or flag != (policy == LEGACY_STOCK):
        raise ValueError("yield policy and diagnostic flag disagree")
    yields = {k: v["yield_per_mev"] for k, v in contract["effective_response"]["sectors"].items()}
    if meta.get("effective_yields", yields if recovery else None) != yields:
        raise ValueError("effective yields disagree with response contract")
    for key in ("conditions_sha256", "photon_dev_sha256"):
        value = meta.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"invalid {key}")
    if meta.get("timing") != TIMING:
        raise ValueError("unrecognized replay timing contract")
    identity = {"response_contract": contract, "timing": TIMING,
                "conditions_sha256": meta["conditions_sha256"], "photon_dev_sha256": meta["photon_dev_sha256"]}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return identity, digest, recovery
