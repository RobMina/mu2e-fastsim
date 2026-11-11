"""runchain's resident-response default: on with --jax on an accelerator unless --fused-response, off on the CPU
backend; --resident-response / --no-resident-response choose explicitly."""
import pytest

from crvfastsim.analyses.runchain import resident_default
from crvfastsim.cli import build_parser


@pytest.mark.parametrize("flag, jax, fused, expected", [
    (None, True, False, True), (None, True, True, False), (None, False, False, False),
    (False, True, False, False), (True, True, False, True), (True, False, False, True)])
def test_resident_default(flag, jax, fused, expected):
    assert resident_default(flag, jax, fused, backend="gpu") is expected


@pytest.mark.parametrize("flag, expected", [(None, False), (True, True), (False, False)])
def test_resident_default_cpu_backend(flag, expected):
    assert resident_default(flag, True, False, backend="cpu") is expected


def test_resident_default_looks_up_the_backend():
    jax = pytest.importorskip("jax")
    assert resident_default(None, True, False) is (jax.default_backend() != "cpu")


@pytest.mark.parametrize("argv, expected", [([], None), (["--resident-response"], True),
                                            (["--no-resident-response"], False)])
def test_runchain_cli_flag(argv, expected):
    a = build_parser().parse_args(["runchain", "gen.root", "--calib", "nominal", *argv])
    assert a.resident_response is expected
