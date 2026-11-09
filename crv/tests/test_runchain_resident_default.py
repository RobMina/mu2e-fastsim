"""runchain's resident-response default: on with --jax unless --fused-response; --no-resident-response opts out."""
import pytest

from crvfastsim.analyses.runchain import resident_default
from crvfastsim.cli import build_parser


@pytest.mark.parametrize("flag, jax, fused, expected", [
    (None, True, False, True), (None, True, True, False), (None, False, False, False),
    (False, True, False, False), (True, True, False, True), (True, False, False, True)])
def test_resident_default(flag, jax, fused, expected):
    assert resident_default(flag, jax, fused) is expected


@pytest.mark.parametrize("argv, expected", [([], None), (["--resident-response"], True),
                                            (["--no-resident-response"], False)])
def test_runchain_cli_flag(argv, expected):
    a = build_parser().parse_args(["runchain", "gen.root", "--calib", "nominal", *argv])
    assert a.resident_response is expected
