"""Saved baseline comparison may relocate files but never loosen the output identity gate."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

P = Path(__file__).resolve().parents[1] / "scripts" / "compare_context.py"
spec = importlib.util.spec_from_file_location("context_compare", P)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def report(root, prepared=False):
    return {"status": "complete", "repeat_fingerprints_identical": True, "boundary": "cached chain",
            "context_policy": "prepared_once" if prepared else "reload_each_pass", "context_audits_passed": prepared,
            "device": {"response": "cuda:0"}, "hardware_software": {"versions": {"jax": "test"}, "environment": {}},
            "arguments": {"inputs": [root + "/gen.root"], "library": root + "/lib.root", "calib": "nominal", "seed": 1},
            "provenance": {"input_sha256": {root + "/gen.root": "input"}, "assets_sha256": {root + "/lib.root": "lib"},
                           "source_sha256": {root + "/crvfastsim/chain.py": "physics", root + "/crvfastsim/analyses/runchain.py": str(prepared)},
                           "effective_run_config": {"name": "same"}, "effective_chain_config": {"jax": True}},
            "passes": [{"elapsed_s": 20., "files": [{"fingerprint": "exact"}]},
                       {"elapsed_s": 10. if prepared else 15., "files": [{"fingerprint": "exact"}]}],
            "preparation_elapsed_s": 5., "invocation_elapsed_before_report_s": 40.}


def compare(tmp_path, a, b):
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    return M.compare(pa, pb)


def test_same_physics_relocated_assets_and_new_repeats(tmp_path):
    a, b = report("/old"), report("/new", True)
    b["passes"].append(copy.deepcopy(b["passes"][1]))
    result = compare(tmp_path, a, b)
    assert result["exact_output_identity"] and result["median_warm_speedup"] == 1.5
    assert result["source_changes"] == ["crvfastsim/analyses/runchain.py"]


@pytest.mark.parametrize("change", [
    lambda b: b["passes"][1]["files"][0].update(fingerprint="one ulp different"),
    lambda b: b["provenance"]["source_sha256"].update({"/new/crvfastsim/chain.py": "changed"}),
    lambda b: b["provenance"]["assets_sha256"].update({"/new/lib.root": "different"}),
    lambda b: b["hardware_software"]["environment"].update(CRVFASTSIM_COMPACT_DISPATCH="1"),
    lambda b: b["hardware_software"]["versions"].update(jax="other"),
    lambda b: b["arguments"].update(seed=2),
    lambda b: b["device"].update(response="cpu"),
    lambda b: b.update(context_audits_passed=False),
    lambda b: b.update(repeat_fingerprints_identical=False),
    lambda b: b["provenance"]["effective_chain_config"].update(jax=False),
])
def test_fail_closed(tmp_path, change):
    a, b = report("/old"), report("/new", True)
    change(b)
    with pytest.raises(ValueError):
        compare(tmp_path, a, b)


def test_relative_arguments_use_recorded_bindings_not_current_cwd(tmp_path):
    a, b = report("/old"), report("/new", True)
    for root, item in (("/old", a), ("/new", b)):
        item["arguments"]["library"] = "lib.root"
        item["provenance"]["argument_asset_paths"] = {"library": root + "/lib.root"}
        item["arguments"]["counters"] = root + "/counters.txt"
        item["provenance"]["assets_sha256"][root + "/counters.txt"] = "samecounters"
        item["provenance"]["effective_run_config"]["counters"] = root + "/counters.txt"
    assert compare(tmp_path, a, b)["exact_output_identity"]
    del a["provenance"]["argument_asset_paths"]
    with pytest.raises(ValueError, match="recorded bindings"):
        compare(tmp_path, a, b)


def test_appledouble_removal_requires_explicit_audited_exception(tmp_path):
    a, b = report("/old"), report("/new", True)
    a["provenance"]["source_sha256"]["/old/crvfastsim/._chain.py"] = "metadata-hash"
    with pytest.raises(ValueError, match="non-setup"):
        compare(tmp_path, a, b)
    result = M.compare(tmp_path / "a.json", tmp_path / "b.json", allow_removed_appledouble=True)
    assert result["exact_output_identity"]
    assert result["removed_appledouble_sha256"] == {"crvfastsim/._chain.py": "metadata-hash"}
    assert result["source_gate"] == "explicit removed AppleDouble filename exception"


@pytest.mark.parametrize("kind", ["paired-change", "unpaired", "added", "changed", "output"])
def test_appledouble_exception_never_allows_real_changes(tmp_path, kind):
    a, b = report("/old"), report("/new", True)
    a["provenance"]["source_sha256"]["/old/crvfastsim/._chain.py"] = "metadata-hash"
    if kind == "paired-change":
        b["provenance"]["source_sha256"]["/new/crvfastsim/chain.py"] = "changed"
    elif kind == "unpaired":
        a["provenance"]["source_sha256"]["/old/crvfastsim/._absent.py"] = "unpaired"
    elif kind == "added":
        b["provenance"]["source_sha256"]["/new/crvfastsim/._added.py"] = "new"
    elif kind == "changed":
        b["provenance"]["source_sha256"]["/new/crvfastsim/._chain.py"] = "changed"
    else:
        b["passes"][0]["files"][0]["fingerprint"] = "different"
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    with pytest.raises(ValueError):
        M.compare(pa, pb, allow_removed_appledouble=True)


def test_prefetch_requires_explicit_scheduling_exception_and_exact_outputs(tmp_path):
    a, b = report("/old", True), report("/new", True)
    b["arguments"]["prefetch"] = True
    pa, pb = tmp_path / "serial.json", tmp_path / "prefetch.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    with pytest.raises(ValueError, match="arguments"):
        M.compare(pa, pb)
    assert M.compare(pa, pb, allow_prefetch=True)["prefetch_execution_change"]
    b["passes"][1]["files"][0]["fingerprint"] = "changed"
    pb.write_text(json.dumps(b))
    with pytest.raises(ValueError, match="output differs"):
        M.compare(pa, pb, allow_prefetch=True)


def test_cpu_reco_source_exception_requires_cpu_and_exact_outputs(tmp_path):
    a, b = report('/old', True), report('/new', True)
    for item in (a, b):
        item['device'] = {'response': 'numpy CPU', 'reco': 'numpy CPU'}
    b['provenance']['source_sha256']['/new/crvfastsim/reco/coinc.py'] = 'numpy-attachment'
    pa, pb = tmp_path / 'cpu_a.json', tmp_path / 'cpu_b.json'
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    with pytest.raises(ValueError, match='non-setup'):
        M.compare(pa, pb)
    assert M.compare(pa, pb, allow_cpu_reco=True)['cpu_reco_source_exception']
    b['passes'][1]['files'][0]['fingerprint'] = 'one ulp changed'
    pb.write_text(json.dumps(b))
    with pytest.raises(ValueError, match='output differs'):
        M.compare(pa, pb, allow_cpu_reco=True)
    a['device'] = b['device'] = {'response': 'cuda:0'}
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    with pytest.raises(ValueError, match='NumPy CPU'):
        M.compare(pa, pb, allow_cpu_reco=True)
