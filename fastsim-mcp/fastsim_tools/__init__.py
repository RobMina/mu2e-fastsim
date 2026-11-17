"""Science tool package of the fastsim MCP server. Only the names in __all__ become MCP tools.

Layout:
  catalog.py            the workflows (boundaries, packages, documents, kit tasks)
  approximations_doc.py the approximations of each workflow, read from docs/approximations.md
  design_point.py       a stopping-target design point: overlay, kit requests per dtype, result summary
  backend.py            the fastsim kit (mu2efastsim.endpoints.kit) and the point records
  tools.py              the tool functions
No MCP import here: fastsim_mcp_server publishes these functions.
"""

from .tools import (ToolResult, cancel_design_point, describe_approximations, fetch_outputs, job_status,
                    list_workflows, run_design_point)

__all__ = ["list_workflows", "describe_approximations", "run_design_point", "job_status", "fetch_outputs",
           "cancel_design_point"]
