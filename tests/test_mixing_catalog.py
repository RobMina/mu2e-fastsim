"""Catalog errors stay errors; cached counts do not replace file membership."""
import pytest

from mixing.catalog import census


class Catalog:
    def get_dataset(self, **kwargs):
        return {"file_count": 2000}

    def query(self, query):
        return iter(())

    def get_dataset_files(self, **kwargs):
        return iter(())


def test_empty_membership_not_cached_count():
    result = census(Catalog(), ["mu2e:example"], 2)[0]
    assert result["estimated_files"] == 2000
    assert result["query_examples"] == result["membership_examples"] == []


def test_lazy_failure_is_not_empty():
    class Broken(Catalog):
        def query(self, query):
            yield from ()
            raise RuntimeError("catalog unavailable")
    with pytest.raises(RuntimeError, match="unavailable"):
        census(Broken(), ["mu2e:example"])


def test_limit_and_file_details():
    class WithFiles(Catalog):
        def get_dataset_files(self, **kwargs):
            return iter([{"namespace": "mu2e", "name": "example.art"}] * 3)

        def get_file(self, **kwargs):
            assert kwargs["did"] == "mu2e:example.art"
            return {"namespace": "mu2e", "name": "example.art", "retired": False, "size": 20}
    result = census(WithFiles(), ["mu2e:example"], 1)[0]
    assert len(result["membership_examples"]) == 1
    assert result["membership_examples"][0]["size"] == 20
    with pytest.raises(ValueError):
        census(Catalog(), ["mu2e:example"], 0)
