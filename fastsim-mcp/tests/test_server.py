import asyncio

import pytest

from fastsim_mcp_server import cli


def test_tool_modules_from_pyproject():
    from fastsim_mcp_server import server
    assert server.configured_tool_module_names() == ["fastsim_tools"]
    assert [n for n, _ in server.tool_functions()] == ["list_workflows", "describe_approximations",
                                                      "run_design_point", "job_status", "fetch_outputs",
                                                      "cancel_design_point"]


def test_server_registers_every_tool():
    pytest.importorskip("mcp")
    from fastsim_mcp_server import server
    mcp = server.create_server()
    tools = asyncio.run(mcp.list_tools())
    by_name = {t.name: t for t in tools}
    assert set(by_name) == {n for n, _ in server.tool_functions()}
    props = by_name["run_design_point"].inputSchema["properties"]
    assert {"name", "target", "geometry", "stage", "electrons", "dtypes"} <= set(props)
    assert "Asynchronous" in by_name["run_design_point"].description


def test_load_site_env(tmp_path):
    f = tmp_path / "site.env"
    f.write_text("# comment\n\nexport A=/x/y\nB=\"$A/z\"\nC='q'  # trailing\nPYTHONPATH=/ignored\n")
    env = {"C": "kept"}
    assert cli.load_site_env(f, env) == ["A", "B"]
    assert env == {"A": "/x/y", "B": "/x/y/z", "C": "kept"}


@pytest.mark.parametrize("line", ["source other.sh", "A=$(id)x y", "A=one two"])
def test_load_site_env_refuses_scripts(tmp_path, line):
    f = tmp_path / "site.env"
    f.write_text(line + "\n")
    with pytest.raises(ValueError):
        cli.load_site_env(f, {})
