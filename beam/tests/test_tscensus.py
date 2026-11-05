"""Read-only catalogue inventory is explicit about limits and lazy query failures."""
import pytest

from beamfastsim.tscensus import census


class Client:
    def __init__(self, fail=False, missing=False):
        self.calls = []
        self.fail, self.missing = fail, missing

    def list_datasets(self, **kwargs):
        self.calls.append(("list_datasets", kwargs))
        return iter([{"namespace": "mu2e", "name": "sim.mu2e.MuBeamCat.MDC2025ac.art"},
                     {"namespace": "mu2e", "name": "dts.mu2e.CeEndpoint.MDC2025ac.art"}])

    def get_dataset(self, **kwargs):
        self.calls.append(("get_dataset", kwargs))
        return {"file_count": 3}

    def query(self, query, **kwargs):
        self.calls.append(("query", query, kwargs))
        if self.fail:
            raise RuntimeError("lazy query failed")
        yield {"namespace": "mu2e", "name": "sim.mu2e.MuBeamCat.MDC2025ac.1.art"}

    def get_file(self, **kwargs):
        self.calls.append(("get_file", kwargs))
        return None if self.missing else {"metadata": {"gen.count": 123}, "size": 456}


def test_census_read_only_and_counts_not_exposure():
    client = Client()
    result = census(client)
    assert len(result["datasets"]) == 1
    assert result["complete"]
    assert result["datasets"][0]["dataset"]["file_count"] == 3
    assert not any(result["claims"].values())
    assert not result["datasets"][0]["contents_verified"]
    assert client.calls[0][1]["with_counts"] is False
    assert client.calls[-1][1]["did"].startswith("mu2e:")


@pytest.mark.parametrize("kwargs", [{"fail": True}, {"missing": True}])
def test_lazy_query_or_missing_file_is_not_zero(kwargs):
    result = census(Client(**kwargs))
    assert not result["complete"]
    assert len(result["errors"]) == 1
    assert "error" in result["datasets"][0]


@pytest.mark.parametrize("samples", [0, -1, True, 1.5])
def test_invalid_limit(samples):
    with pytest.raises(ValueError):
        census(Client(), samples=samples)


def test_empty_discovery_has_no_content_claim():
    result = census(Client(), patterns=["does-not-exist"])
    assert result["datasets"] == []
    assert not any(result["claims"].values())


def test_partial_discovery_failure_is_persisted():
    class Partial(Client):
        def list_datasets(self, **kwargs):
            yield {"namespace": "mu2e", "name": "sim.mu2e.Beam.MDC2025ac.art"}
            raise RuntimeError("discovery interrupted")

    result = census(Partial())
    assert not result["complete"]
    assert len(result["datasets"]) == 1
    assert result["errors"][0]["stage"] == "discovery"


def test_explicit_personal_namespace_and_scope_recorded():
    class Personal(Client):
        def list_datasets(self, **kwargs):
            self.calls.append(("list_datasets", kwargs))
            yield {"namespace": "operator", "name": "sim.operator.TS1Beam.example.art"}

    client = Personal()
    result = census(client, patterns=["sim.*.TS1Beam.*.art"], namespace="operator")
    assert result["namespace_pattern"] == "operator"
    assert client.calls[0][1]["namespace_pattern"] == "operator"
    assert result["datasets"][0]["did"] == "operator:sim.operator.TS1Beam.example.art"
    assert not any(result["claims"].values())


@pytest.mark.parametrize("namespace", [None, "", " ", " mu2e"])
def test_invalid_namespace_pattern(namespace):
    with pytest.raises(ValueError, match="namespace"):
        census(Client(), namespace=namespace)


def test_retired_query_is_explicit_read_only_and_not_replica_evidence():
    class Retired(Client):
        def get_file(self, **kwargs):
            return {"retired": True, "metadata": {"gen.count": 17}, "size": 19}

    default = Client()
    assert not census(default)["include_retired_files"]
    assert "include_retired_files" not in next(c[2] for c in default.calls if c[0] == "query")
    client = Retired()
    result = census(client, include_retired=True)
    assert result["complete"] and result["include_retired_files"]
    assert next(c[2] for c in client.calls if c[0] == "query")["include_retired_files"] is True
    assert result["datasets"][0]["files"][0]["retired"]
    assert not result["datasets"][0]["contents_verified"]
    assert not any(result["claims"].values())


@pytest.mark.parametrize("include_retired", [1, 0, None, "yes"])
def test_retired_switch_requires_bool(include_retired):
    with pytest.raises(ValueError, match="include_retired"):
        census(Client(), include_retired=include_retired)


def test_empty_retired_query_warning_does_not_claim_absence():
    class Empty(Client):
        def query(self, query, **kwargs):
            return iter(())

    result = census(Empty(), include_retired=True)
    assert "including-retired" in result["datasets"][0]["sample_warning"]
    assert not any(result["claims"].values())
