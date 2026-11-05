"""Catalog failures and missing datasets must not masquerade as empty pools."""
import pytest
from beamfastsim.flash_census import census


class Client:
    def __init__(self):
        self.calls = []

    def get_dataset(self, did, exact_file_count):
        self.calls.append(did)
        return None if did == "test:missing" else {"file_count": 3}

    def query(self, query):
        if "broken" in query:
            def failed():
                raise RuntimeError("lazy failure")
                yield
            return failed()
        return iter([{"namespace": "test", "name": "file"}])

    def get_file(self, did, with_metadata):
        return {"did": did, "metadata": {"rse.nevent": 7}}


def test_census_distinguishes_missing_and_lazy_failure():
    rows = census(Client(), ["test:good", "test:missing", "test:broken"])["datasets"]
    assert [r["status"] for r in rows] == ["ok", "missing", "error"]
    assert rows[0]["files"][0]["metadata"]["rse.nevent"] == 7
    assert rows[2]["error_type"] == "RuntimeError"


@pytest.mark.parametrize("did", ["missing_namespace", "test:*", "test:a;drop", "test:a\nb"])
def test_rejects_patterns_and_query_injection(did):
    client = Client()
    with pytest.raises(ValueError):
        census(client, [did])
    assert client.calls == []


def test_rejects_zero_example_limit():
    with pytest.raises(ValueError):
        census(Client(), ["test:good"], examples=0)


def test_discovery_is_narrow_namespace_no_counts_and_local_patterns():
    from beamfastsim.flash_census import discover

    class Listing:
        def list_datasets(self, **kwargs):
            assert kwargs == dict(namespace_pattern="test", with_counts=False)
            return iter([dict(namespace="test", name="mcs.owner.EleBeam.test.art"),
                         dict(namespace="test", name="dts.owner.EleBeam.test.art"),
                         dict(namespace="other", name="mcs.owner.EleBeam.test.art")])
    result = discover(Listing(), "test", ["mcs.*EleBeam*", "nts.*elebeam*"])
    assert result["status"] == "ok"
    assert result["scanned_datasets"] == 3
    assert result["selected_datasets"] == ["test:mcs.owner.EleBeam.test.art"]


def test_lazy_discovery_failure_is_not_empty_success():
    from beamfastsim.flash_census import discover

    class Broken:
        def list_datasets(self, **kwargs):
            raise RuntimeError("unavailable")
            yield
    result = discover(Broken(), "test", ["*"])
    assert result["status"] == "error"
    assert result["error_type"] == "RuntimeError"


@pytest.mark.parametrize("namespace,patterns", [("*", ["*"]), ("test", []), ("test", ["x;y"])])
def test_invalid_discovery_fails_before_query(namespace, patterns):
    from beamfastsim.flash_census import discover
    with pytest.raises(ValueError):
        discover(None, namespace, patterns)
