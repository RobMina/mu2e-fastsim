"""Aluminium uncollided-photon survival; energies MeV, lengths mm, mu/rho cm²/g.

This is not scattered/secondary photon transport. See docs/flash-attenuation.md.
Regenerate the bundled table with --xcom-data MDATX3.013 --out flash_al_xcom.csv.

The table is a library of the shared core (mu2efastsim.library, core/docs/libraries.md): its sidecar manifest
data/flash_al_xcom.manifest.json pins the content hash of the arrays read from the CSV, and _table() checks it
(store.verify). Energies outside the table and inside the K-edge gap give NaN (axis policy "nan"), never a clamp.
"""
from functools import lru_cache
from pathlib import Path
import argparse

import numpy as np

from mu2efastsim.library import manifest as LM
from mu2efastsim.library import store as LS
from mu2efastsim.library.axes import Axis

LIBRARY = "beam.flash_attenuation.al_xcom"
_DATA = Path(__file__).resolve().parent / "data" / "flash_al_xcom.csv"
_BARN_TO_CM2_G = 0.60221367 / 26.981541
_EDGE_LOW, _EDGE_HIGH = 0.0015595, 0.0015596


def _arrays(table):
    return {"energy_mev": table[0], "mu_rho_cm2_g": table[1]}


def manifest_fields(table, path=_DATA):
    """Manifest fields of an XCOM table (mu2efastsim.library.manifest), from its arrays and the CSV bytes."""
    from mu2efastsim.library.hashing import file_sha256
    return dict(
        name=LIBRARY, subsystem="beam", version=1, kind="interpolation",
        description="Total aluminium mass attenuation coefficient mu/rho (coherent scattering included) on the NIST "
                    "XCOM energy grid, interpolated log-log; the uncollided survival weight of a photon is "
                    "exp(-mu/rho * density * path).",
        axes=[Axis("energy", "MeV", table[0], frame="photon energy", array="energy_mev", scale="irregular",
                   policy="nan", description="XCOM grid with both K-edge endpoints; the open interval "
                                             "(0.0015595, 0.0015596) MeV is unsupported").describe()],
        outputs=[dict(name="mu_rho", unit="cm2/g", dtype="float64", description="NaN outside the domain"),
                 dict(name="weight", unit="1", dtype="float64",
                      description="uncollided survival; NaN with supported False when unsupported")],
        source=dict(description="NIST XCOM v3.1 data file MDATX3.013 (Al): coherent, incoherent, photoelectric, "
                                "nuclear and electron pair cross sections, summed", samples=[],
                    files=[dict(file=Path(path).name, sha256=file_sha256(path))]),
        generator=dict(name="NIST XCOM", release="v3.1, MDATX3.013", physics_list=None,
                       configuration="barns/atom to cm2/g: 0.60221367 / 26.981541"),
        release=None, seeds={}, geometry=dict(tag="none (aluminium material data)", hash=None), config_hash=None,
        split=dict(training=None, held_out=None, rule="not trained: NIST data converted unchanged"),
        dtype=dict(storage={"energy_mev": "float64", "mu_rho_cm2_g": "float64"}, compute=["float32", "float64"],
                   notes="inputs are cast to float64 (float32 energies give the float64 result of their values)"),
        rng=dict(host="none", device="none"),
        domain=dict(coverage="0.001 to 100000 MeV, except the open K-edge interval; nonfinite energies "
                             "unsupported", fallbacks=[]),
        validation=dict(status="exact", record="beam/tests/test_flash_attenuation.py: NIST public values, every "
                                               "table node, log-log interpolation, the K edge"),
        unrecorded=dict(release="written by write_xcom_csv before the manifest; the commit is not recorded",
                        config_hash="no build configuration beyond the data file"),
        legacy=dict(schema=0, format="CSV with a comment header, no manifest"),
    )


def write_manifest(path=_DATA):
    """(Re)write the sidecar manifest of a CSV table; the CSV is not touched."""
    table = np.loadtxt(path, delimiter=",", comments="#", unpack=True)
    m = LS.finalize(_arrays(table), manifest_fields(table, path))
    LM.write(LS.sidecar_path(path), m)
    return m


def load_table(path=_DATA):
    """The (energy, mu/rho) table of a CSV, checked against its sidecar manifest (store.verify)."""
    table = np.loadtxt(path, delimiter=",", comments="#", unpack=True)
    side = LS.sidecar_path(path)
    if not side.exists():
        raise LS.LibraryError(path, [f"no sidecar manifest {side.name}; a library without a manifest is refused"])
    LS.verify(LM.read(side), _arrays(table), str(path), expect={"name": LIBRARY})
    return table


@lru_cache(maxsize=1)
def _table():
    """Load the checked-in table directly from the module's filesystem path (checked against its manifest)."""
    return load_table(_DATA)


def al_mu_rho(energy):
    """Total Al mass attenuation in cm²/g for energy (MeV), as an ndarray.

    Log-log interpolation includes both K-edge endpoints, but excludes the
    open interval (0.0015595, 0.0015596) MeV: no artificial edge smoothing.
    Outside [0.001, 100000] MeV, or for nonfinite input, return NaN, not a clamp.
    Scalars produce zero-dimensional arrays; input array shape is preserved.
    """
    energy = np.asarray(energy, dtype=float)
    grid, mu = _table()
    supported = (np.isfinite(energy) & (energy >= grid[0]) & (energy <= grid[-1])
                 & ~((energy > _EDGE_LOW) & (energy < _EDGE_HIGH)))
    safe = np.where(supported, energy, grid[0])
    result = np.exp(np.interp(np.log(safe), np.log(grid), np.log(mu)))
    return np.where(supported, result, np.nan)


def uncollided_weight(energy, distance_mm, density_g_cm3=2.699):
    """Return {'weight': ndarray, 'supported': bool ndarray}, broadcasting inputs.

    weight = exp[-al_mu_rho(energy) * density_g_cm3 * distance_mm / 10].
    Finite nonnegative lengths/densities are valid (zero gives unity). Invalid
    inputs or unsupported energies give NaN/False, even at zero path length.
    """
    mu, distance, density = np.broadcast_arrays(
        al_mu_rho(energy), np.asarray(distance_mm, dtype=float),
        np.asarray(density_g_cm3, dtype=float))
    supported = (np.isfinite(mu) & np.isfinite(distance) & (distance >= 0)
                 & np.isfinite(density) & (density >= 0))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        optical_depth = np.where((distance == 0) | (density == 0), 0.0,
                                 mu * density * (distance / 10.0))
        weight = np.exp(-optical_depth)
    return {"weight": np.where(supported, weight, np.nan),
            "supported": supported}


def write_xcom_csv(xcom_data, out):
    """Parse NIST MDATX3.013's five barns/atom blocks; write deterministic CSV.

    XCOM.f lines 144–158: Z/atomic weight, edge/energy counts, edge indices,
    labels, energies, then E (eV), coherent, incoherent, photoelectric, nuclear
    pair and electron pair arrays. Trailing subshell interpolation data are not
    used. This parser deliberately accepts only the aluminium K-edge format.
    """
    raw = Path(xcom_data).read_bytes()
    tokens = raw.decode("ascii").split()
    if len(tokens) < 7:
        raise ValueError("Incomplete XCOM aluminium header")
    z, atomic_weight = int(tokens[0]), float(tokens[1])
    edges, count = int(tokens[2]), int(tokens[3])
    if (z != 13 or atomic_weight != 26.981541 or edges != 1 or count < 4
            or tokens[4:6] != ["4", "K"] or float(tokens[6]) != 1559.6):
        raise ValueError("Expected NIST aluminium MDATX3.013 K-edge format")
    if len(tokens) < 7 + 6 * count:
        raise ValueError("Incomplete XCOM energy/cross-section blocks")
    blocks = np.asarray(tokens[7:7 + 6 * count], dtype=float).reshape(6, count)
    energy_ev, cross_sections = blocks[0], blocks[1:]
    if (not np.all(np.isfinite(blocks)) or np.any(cross_sections < 0)
            or np.any(np.diff(energy_ev) <= 0)
            or energy_ev[0] != 1000 or energy_ev[-1] != 1e11
            or energy_ev[2] != 1559.5 or energy_ev[3] != 1559.6):
        raise ValueError("Invalid XCOM aluminium energies or cross sections")
    mu = cross_sections.sum(axis=0) * _BARN_TO_CM2_G
    if np.any(mu <= 0):
        raise ValueError("Total cross sections must be positive")
    lines = [
        "# NIST XCOM v3.1 MDATX3.013 (Al); total including coherent scattering",
        "# https://physics.nist.gov/PhysRefData/Xcom/XCOM.tar.gz",
        "# barns/atom to cm2/g: 0.60221367 / 26.981541; eV to MeV: / 1e6",
        "# energy_mev,mu_rho_cm2_g",
    ]
    lines.extend(f"{e:.12g},{m:.15g}" for e, m in zip(energy_ev / 1e6, mu))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="ascii")


def main():
    """Convert the supplied local NIST data; no network or compilation needed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xcom-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_xcom_csv(args.xcom_data, args.out)


if __name__ == "__main__":
    main()
