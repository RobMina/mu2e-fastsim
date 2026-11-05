"""RPC deck bundle keeps measured timing and all event rows."""
import json

import numpy as np
import pytest

from beamfastsim import rpcsummary as S


LOG = """TrigReport Events total = 100 passed = 20 failed = 80
TimeReport CPU = 9.0 Real = 10.0
PrimaryPath:g4run:Mu2eG4 0.01 0.08 0.2 0.07 0.01 100
"""


def test_stock_stage_timing(tmp_path):
    path = tmp_path / "stock.log"
    path.write_text(LOG)
    a = S.stock_timing(path)
    assert a["art_events_s"] == 10 and a["g4_events_s"] == 12.5
    path.write_text(LOG + LOG)
    with pytest.raises(ValueError):
        S.stock_timing(path)


def test_bundle(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    for name in ("paired_dev", "paired_heldout", "generated_dev", "generated_heldout"):
        (results / (name + "_final.json")).write_text(json.dumps({"scope": name}))
    (results / "generated_heldout_final.png").write_bytes(b"png")
    source, plane, starts, log = [tmp_path / p for p in ("source.json", "plane.npz", "starts.npz", "stock.log")]
    source.write_text('{"events": 100}')
    log.write_text(LOG)
    np.savez(plane, idx=np.arange(100), wall=2., fate=np.ones(100, int))
    np.savez(starts, pdg=np.full(100, 11), generator_wall_s=.1)
    out = tmp_path / "incoming"
    r = S.bundle(results, source, plane, starts, log, out)
    assert r["throughput"]["transport_events_s"] == 50
    assert r["throughput"]["rate_ratio_vs_stock_art"] == 5
    assert (out / "numbers.json").is_file() and (out / "rpc_vd13.png").is_file()
    np.savez(plane, idx=np.zeros(100, int), wall=2., fate=np.ones(100, int))
    with pytest.raises(ValueError, match="every"):
        S.bundle(results, source, plane, starts, log, out)
