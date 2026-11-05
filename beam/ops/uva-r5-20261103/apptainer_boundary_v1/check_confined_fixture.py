"""Read only closed no-G4 outputs; keep analytic results in the counted log pipe."""

import json
import sys
import types


package = types.ModuleType("beamfastsim")
package.__path__ = ["/opt/r5/checks/beamfastsim"]
sys.modules["beamfastsim"] = package
from beamfastsim.flash_reference_fixture import check_fixture
from beamfastsim.flash_reference_compare import compare_exports


def main():
    first = "/mnt/work/fixture/export.root"
    reread = "/mnt/work/reread/export.root"
    reports = {"fixture": check_fixture(first), "reread": check_fixture(reread),
               "persistence": compare_exports(first, reread)}
    print(json.dumps({"noG4_fixture_checks": reports}, sort_keys=True), flush=True)
    if not reports["fixture"]["fixture_values_valid"] or not reports["reread"]["fixture_values_valid"] or not reports["persistence"]["identical"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
