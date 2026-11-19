"""Optical materials and surfaces of the stock CRV counter simulation, with Geant4 10.7 property-vector semantics.

Source: Mu2e/CRVStandalone (Apache-2.0), wls/src/WLSMaterials.cc and wls/src/WLSDetectorConstruction.cc. Every table
below is copied from there verbatim (energies in eV, lengths in mm after the unit factors the C++ applies).

Geant4 semantics reproduced here (numpy, used to build the device tables and as the reference in the tests):

- ``G4MaterialPropertyVector`` is a G4PhysicsOrderedFreeVector filled by InsertValues, so the energies are sorted on
  insertion (a point is inserted before an existing equal energy). The PS+PPO absorption table has an out-of-order
  energy (2.32 eV where 3.32 eV is meant); Geant4 moves it, and so does this module.
- ``Value(E)`` is G4PhysicsVector::Value with a fresh index: the edges clamp, the bin is std::lower_bound - 1, and the
  interpolation is linear with the fraction clamped to [0, 1].
- ``GetEnergy(v)`` is G4PhysicsOrderedFreeVector::GetEnergy (the inverse of a cumulative emission integral).
- GROUPVEL is G4MaterialPropertiesTable::CalculateGROUPVEL, computed from RINDEX as Geant4 does whenever RINDEX is
  added; optical photons move at the group velocity of the volume they traverse.
"""
import bisect
from dataclasses import dataclass

import numpy as np

C_LIGHT = 299.792458  # mm/ns
_E_RINDEX_PS = (2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85,
    2.9, 2.95, 3.0, 3.05, 3.1, 3.15, 3.2, 3.25, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.5, 5.0, 5.25, 5.5, 5.75,
    6.0, 6.25, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0,
    15.5, 15.75)
_RINDEX_PS = (1.589, 1.59, 1.591, 1.593, 1.594, 1.596, 1.597, 1.599, 1.601, 1.602, 1.604, 1.606, 1.608, 1.609, 1.611,
    1.613, 1.615, 1.617, 1.62, 1.622, 1.624, 1.626, 1.629, 1.631, 1.633, 1.636, 1.639, 1.644, 1.649, 1.655, 1.661,
    1.668, 1.675, 1.682, 1.72, 1.81, 1.95, 2.1, 2.1, 2.2, 1.7, 1.35, 1.24, 1.4, 1.55, 1.62, 1.64, 1.62, 1.6, 1.57,
    1.54, 1.5, 1.44, 1.38, 1.32, 1.26, 1.2, 1.14, 1.1, 1.04, 1.0)
_E_RINDEX_PMMA = (2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
    4.0, 4.5, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5,
    13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 15.75)
_RINDEX_PMMA_BASE = (1.489, 1.49, 1.492, 1.493, 1.495, 1.496, 1.498, 1.5, 1.501, 1.503, 1.505, 1.507, 1.509, 1.512,
    1.514, 1.516, 1.519, 1.521, 1.524, 1.527, 1.53, 1.72, 1.81, 1.95, 2.1, 2.1, 2.2, 1.7, 1.35, 1.24, 1.4, 1.55,
    1.62, 1.64, 1.62, 1.6, 1.57, 1.54, 1.5, 1.44, 1.38, 1.32, 1.26, 1.2, 1.14, 1.1, 1.04, 1.0)
_E_ABS_Y11 = (2.0, 2.002, 2.003, 2.005, 2.006, 2.008, 2.009, 2.011, 2.013, 2.014, 2.016, 2.017, 2.019, 2.02, 2.022,
    2.024, 2.025, 2.027, 2.028, 2.03, 2.031, 2.033, 2.035, 2.036, 2.038, 2.039, 2.041, 2.043, 2.044, 2.046, 2.047,
    2.049, 2.051, 2.052, 2.054, 2.056, 2.057, 2.059, 2.06, 2.062, 2.064, 2.065, 2.067, 2.069, 2.07, 2.072, 2.074,
    2.075, 2.077, 2.079, 2.08, 2.082, 2.084, 2.085, 2.087, 2.089, 2.09, 2.092, 2.094, 2.095, 2.097, 2.099, 2.1,
    2.102, 2.104, 2.105, 2.107, 2.109, 2.11, 2.112, 2.114, 2.116, 2.117, 2.119, 2.121, 2.123, 2.124, 2.126, 2.128,
    2.129, 2.131, 2.133, 2.135, 2.136, 2.138, 2.14, 2.142, 2.143, 2.145, 2.147, 2.149, 2.15, 2.152, 2.154, 2.156,
    2.158, 2.159, 2.161, 2.163, 2.165, 2.167, 2.168, 2.17, 2.172, 2.174, 2.176, 2.177, 2.179, 2.181, 2.183, 2.185,
    2.186, 2.188, 2.19, 2.192, 2.194, 2.196, 2.197, 2.199, 2.201, 2.203, 2.205, 2.207, 2.209, 2.21, 2.212, 2.214,
    2.216, 2.218, 2.22, 2.222, 2.224, 2.225, 2.227, 2.229, 2.231, 2.233, 2.235, 2.237, 2.239, 2.241, 2.243, 2.244,
    2.246, 2.248, 2.25, 2.252, 2.254, 2.256, 2.258, 2.26, 2.262, 2.264, 2.266, 2.268, 2.27, 2.272, 2.274, 2.276,
    2.278, 2.279, 2.281, 2.283, 2.285, 2.287, 2.289, 2.291, 2.293, 2.295, 2.297, 2.299, 2.301, 2.303, 2.305, 2.307,
    2.31, 2.312, 2.314, 2.316, 2.318, 2.32, 2.322, 2.324, 2.326, 2.328, 2.33, 2.332, 2.334, 2.336, 2.338, 2.34,
    2.342, 2.344, 2.347, 2.349, 2.351, 2.353, 2.355, 2.357, 2.359, 2.361, 2.363, 2.365, 2.368, 2.37, 2.372, 2.374,
    2.376, 2.378, 2.38, 2.383, 2.385, 2.387, 2.389, 2.391, 2.393, 2.396, 2.398, 2.4, 2.402, 2.404, 2.406, 2.409,
    2.411, 2.413, 2.415, 2.417, 2.42, 2.422, 2.424, 2.426, 2.429, 2.431, 2.433, 2.435, 2.437, 2.44, 2.442, 2.444,
    2.446, 2.449, 2.451, 2.453, 2.456, 2.458, 2.46, 2.462, 2.465, 2.467, 2.469, 2.472, 2.474, 2.476, 2.479, 2.481,
    2.483, 2.486, 2.488, 2.49, 2.493, 2.495, 2.497, 2.5, 2.502, 2.504, 2.507, 2.509, 2.511, 2.514, 2.516, 2.519,
    2.521, 2.523, 2.56, 2.58, 2.6, 2.62, 2.64, 2.66, 2.68, 2.7, 2.72, 2.74, 2.76, 2.78, 2.8, 2.82, 2.84, 2.86, 2.88,
    2.9, 2.92, 2.94, 2.96, 2.98, 3.0, 3.02, 3.04, 3.06, 3.08, 3.1, 3.12, 3.14, 3.16, 3.18, 3.2, 3.22, 3.24, 3.26,
    3.28, 3.3, 3.32, 3.34, 3.36, 3.38, 3.4, 3.42, 3.44, 3.46, 3.48, 3.5, 3.52, 3.54, 3.5401, 15.75)
_ABS_Y11_RAW = (22.3, 21.6, 20.2, 20.0, 19.2, 19.4, 17.8, 17.0, 15.9, 15.2, 14.6, 13.6, 12.7, 11.5, 10.5, 10.3, 9.4,
    8.7, 8.5, 7.8, 7.4, 7.5, 5.7, 6.8, 6.8, 6.8, 7.3, 7.0, 7.1, 7.6, 7.4, 7.9, 8.4, 8.7, 9.0, 9.5, 10.5, 10.3, 11.0,
    11.5, 12.1, 12.9, 12.8, 13.4, 13.7, 14.5, 15.1, 15.3, 15.9, 16.9, 17.0, 17.3, 18.1, 18.8, 19.6, 19.7, 19.6, 20.3,
    20.5, 21.2, 21.3, 21.5, 22.1, 22.1, 22.4, 22.7, 22.8, 23.9, 23.2, 23.5, 23.5, 23.4, 23.6, 23.7, 23.9, 24.1, 24.0,
    24.3, 24.5, 24.4, 24.2, 24.4, 24.1, 24.4, 24.4, 24.7, 24.4, 24.1, 24.1, 23.6, 24.1, 23.6, 23.3, 23.3, 23.1, 22.9,
    23.0, 22.5, 22.4, 22.2, 22.2, 21.8, 21.8, 21.8, 21.5, 21.5, 21.6, 21.3, 21.4, 21.7, 22.0, 21.3, 21.2, 21.2, 21.3,
    21.3, 21.2, 21.1, 21.2, 21.6, 21.3, 21.2, 21.1, 21.2, 21.2, 21.3, 21.0, 21.0, 20.8, 20.8, 21.1, 20.8, 20.6, 20.6,
    20.6, 20.5, 20.4, 20.3, 20.3, 20.1, 20.1, 20.0, 19.8, 19.8, 19.7, 19.5, 19.4, 19.4, 19.1, 19.0, 18.8, 19.0, 18.4,
    18.3, 18.3, 18.1, 17.9, 17.7, 17.5, 17.3, 17.1, 16.9, 16.6, 16.5, 16.2, 16.0, 15.8, 15.5, 15.2, 14.9, 14.6, 14.2,
    13.9, 13.5, 13.1, 12.8, 12.4, 12.0, 11.7, 11.4, 11.2, 10.9, 10.7, 10.5, 10.4, 10.3, 10.2, 10.1, 10.1, 10.1, 10.1,
    10.0, 10.0, 10.0, 10.0, 10.0, 9.9, 9.9, 9.9, 9.8, 9.8, 9.7, 9.6, 9.5, 9.4, 9.4, 9.3, 9.1, 9.0, 8.9, 8.7, 8.6,
    8.4, 8.3, 8.1, 8.0, 7.8, 7.7, 7.4, 7.3, 7.1, 6.9, 6.7, 6.5, 6.3, 6.1, 5.9, 5.7, 5.5, 5.3, 5.1, 4.9, 4.7, 4.5,
    4.3, 4.1, 3.9, 3.7, 3.5, 3.3, 3.13, 2.96, 2.77, 2.61, 2.46, 2.31, 2.16, 2.03, 1.9, 1.78, 1.67, 1.56, 1.45, 1.36,
    1.27, 1.19, 1.11, 1.03, 0.97, 0.9, 0.84, 0.78, 0.73, 0.684, 0.644, 0.596, 0.561, 0.526, 0.494, 0.467, 0.44,
    0.411, 0.399, 38.9, 12.97, 4.86, 2.16, 1.11, 0.75, 0.56, 0.47, 0.44, 0.45, 0.47, 0.5, 0.5, 0.46, 0.42, 0.4, 0.39,
    0.41, 0.45, 0.5, 0.55, 0.6, 0.63, 0.64, 0.68, 0.72, 0.78, 0.85, 0.95, 1.05, 1.18, 1.34, 1.56, 1.85, 2.16, 2.59,
    2.99, 3.54, 4.32, 5.19, 5.98, 6.27, 6.82, 7.48, 7.78, 8.64, 9.26, 10.51, 11.11, 12.16, 0.0, 0.0)
_E_QY_Y11 = (2.0, 2.55999, 2.56, 3.54, 3.5401, 15.75)
_QY_Y11 = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
_E_EM_Y11 = (2.0, 2.02, 2.04, 2.06, 2.08, 2.1, 2.12, 2.14, 2.16, 2.18, 2.2, 2.22, 2.24, 2.26, 2.28, 2.3, 2.32, 2.34,
    2.36, 2.38, 2.4, 2.42, 2.44, 2.46, 2.48, 2.5, 2.52, 2.54, 2.56, 2.58, 2.6, 2.62, 2.64, 2.66, 2.68, 2.7, 2.72,
    2.74, 2.76, 2.78, 2.8, 2.82, 15.75)
_EM_Y11 = (1.0, 2.0, 4.0, 5.0, 5.0, 6.0, 8.0, 10.0, 13.0, 17.0, 23.0, 28.0, 34.0, 39.0, 44.0, 50.0, 54.0, 60.0, 75.0,
    94.0, 108.0, 123.0, 129.0, 126.0, 117.0, 110.0, 112.0, 122.0, 134.0, 147.0, 148.0, 131.0, 100.0, 71.0, 42.0,
    23.0, 14.0, 7.0, 2.0, 1.0, 1.0, 0.0, 0.0)
_E_EM_PPO = (2.0, 2.48, 2.5, 2.52, 2.54, 2.56, 2.58, 2.6, 2.62, 2.64, 2.66, 2.68, 2.7, 2.72, 2.74, 2.76, 2.78, 2.8,
    2.82, 2.84, 2.86, 2.88, 2.9, 2.92, 2.94, 2.96, 2.98, 3.0, 3.02, 3.04, 3.06, 3.08, 3.1, 3.12, 3.14, 3.16, 3.18,
    3.2, 3.22, 3.24, 3.26, 3.28, 3.3, 3.32, 3.34, 3.36, 3.38, 3.4, 3.42, 3.44, 3.46, 3.48, 3.5, 3.52, 3.54, 3.56,
    3.58, 3.6, 3.62, 3.64, 3.66, 3.68, 3.7, 3.72, 3.74, 3.76, 3.78, 3.8, 3.82, 15.75)
_EM_PPO = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0,
    4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 13.0, 16.0, 19.0, 21.0, 24.0, 27.0, 29.0, 31.0, 33.0, 36.0, 42.0, 48.0,
    55.0, 59.0, 63.0, 64.0, 64.0, 59.0, 62.0, 70.0, 80.0, 92.0, 100.0, 96.0, 89.0, 77.0, 66.0, 55.0, 51.0, 60.0,
    73.0, 85.0, 82.0, 62.0, 40.0, 20.0, 11.0, 6.0, 5.0, 0.0, 0.0)
_E_QY_PSPPO = (2.0, 3.39, 3.4, 4.34, 4.35, 4.43, 4.59, 4.77, 4.96, 5.17, 5.39, 5.64, 5.9, 6.2, 6.53, 6.89, 7.29,
    7.75, 8.27, 8.86, 9.54, 10.33, 11.27, 12.4, 13.78, 15.5, 17.71)
_QY_PSPPO = (0.0, 0.0, 0.94, 0.94, 0.78, 0.74, 0.67, 0.63, 0.6, 0.65, 0.68, 0.57, 0.48, 0.39, 0.38, 0.48, 0.53, 0.58,
    0.56, 0.49, 0.43, 0.4, 0.38, 0.35, 0.35, 0.38, 0.43)
_E_ABS_PSPPO = (2.0, 2.04, 2.08, 2.12, 2.16, 2.2, 2.24, 2.28, 2.32, 2.36, 2.4, 2.44, 2.48, 2.52, 2.56, 2.6, 2.64,
    2.68, 2.72, 2.76, 2.8, 2.84, 2.88, 2.92, 2.96, 3.0, 3.04, 3.08, 3.12, 3.16, 3.2, 3.24, 3.28, 2.32, 3.36, 3.4,
    3.44, 3.48, 3.52, 3.56, 3.6, 3.64, 3.68, 3.72, 15.75)
_ABS_PSPPO_CM = (449.0, 361.0, 449.0, 449.0, 439.0, 430.0, 421.0, 421.0, 396.0, 404.0, 396.0, 389.0, 381.0, 367.0,
    355.0, 342.0, 331.0, 321.0, 311.0, 297.0, 285.0, 273.0, 253.0, 241.0, 230.0, 220.0, 202.0, 180.0, 135.0, 92.0,
    72.0, 56.0, 39.0, 29.0, 23.0, 8.43, 2.05, 0.637, 0.191, 0.11, 0.0287, 0.00382, 0.00106, 0.001, 0.001)
_E_EM_POPOP = (2.0, 2.24, 2.26, 2.28, 2.3, 2.32, 2.34, 2.36, 2.38, 2.4, 2.42, 2.44, 2.46, 2.48, 2.5, 2.52, 2.54,
    2.56, 2.58, 2.6, 2.62, 2.64, 2.66, 2.68, 2.7, 2.72, 2.74, 2.76, 2.78, 2.8, 2.82, 2.84, 2.86, 2.88, 2.9, 2.92,
    2.94, 2.96, 2.98, 3.0, 3.02, 3.04, 3.06, 3.08, 3.1, 3.12, 3.14, 3.16, 3.18, 3.2, 3.22, 3.24, 3.26, 3.28, 3.3,
    3.32, 3.34, 3.36, 3.38, 3.4, 3.42, 3.44, 15.75)
_EM_POPOP = (0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 9.0, 10.0, 13.0,
    15.0, 19.0, 21.0, 23.0, 24.0, 23.0, 26.0, 28.0, 35.0, 43.0, 52.0, 56.0, 58.0, 58.0, 54.0, 49.0, 50.0, 57.0, 74.0,
    89.0, 100.0, 97.0, 85.0, 73.0, 56.0, 46.0, 44.0, 56.0, 78.0, 94.0, 88.0, 60.0, 36.0, 18.0, 9.0, 4.0, 2.0, 1.0,
    1.0, 1.0, 1.0, 0.0, 0.0)
_E_ABS_POPOP = (2.0, 2.6, 2.64, 2.68, 2.72, 2.76, 2.8, 2.84, 2.88, 2.92, 2.96, 3.0, 3.04, 3.08, 3.12, 3.16, 3.2,
    3.24, 3.28, 3.32, 3.36, 3.4, 3.44, 3.48, 3.52, 15.75)
_ABS_POPOP_CM = (2000.0, 2000.0, 1658.0, 1324.0, 946.0, 880.0, 661.0, 529.0, 358.0, 149.0, 42.6, 14.7, 4.95, 1.79,
    0.517, 0.192, 0.084, 0.056, 0.045, 0.038, 0.034, 0.031, 0.03, 0.029, 0.03, 0.03)


class G4Vector:
    """A G4PhysicsOrderedFreeVector: (energy, value) pairs inserted in order with std::lower_bound."""

    def __init__(self, energies, values):
        e, v = [], []
        for x, y in zip(energies, values):
            i = bisect.bisect_left(e, float(x))
            e.insert(i, float(x))
            v.insert(i, float(y))
        self.e = np.array(e, np.float64)
        self.v = np.array(v, np.float64)

    def __len__(self):
        return len(self.e)

    def bin_of(self, x):
        """G4PhysicsVector::FindBin(x, 0) for x strictly inside (edgeMin, edgeMax)."""
        e, n = self.e, len(self.e)
        x = np.asarray(x, np.float64)
        loc = np.minimum(np.searchsorted(e, x, side="left") - 1, n - 2)
        loc = np.where((x > e[1]) | (x < e[0]), loc, 0)   # idx 0 is kept when e[0] <= x <= e[1]
        loc = np.where(x < e[1], 0, loc)
        return np.where(x >= e[n - 2], n - 2, loc)

    def value(self, x):
        e, v = self.e, self.v
        x = np.asarray(x, np.float64)
        i = np.clip(self.bin_of(x), 0, len(e) - 2)
        dl = e[i + 1] - e[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            b = np.where(dl > 0, np.clip((x - e[i]) / np.where(dl > 0, dl, 1.0), 0.0, 1.0), 1.0)
        y = v[i] + b * (v[i + 1] - v[i])
        y = np.where(x <= e[0], v[0], y)
        return np.where(x >= e[-1], v[-1], y)

    def get_energy(self, a):
        """G4PhysicsOrderedFreeVector::GetEnergy: the inverse of a non-decreasing vector."""
        e, v, n = self.e, self.v, len(self.e)
        a = np.asarray(a, np.float64)
        i = np.minimum(np.searchsorted(v, a, side="left") - 1, n - 2)
        i = np.clip(i, 0, n - 2)
        d = v[i + 1] - v[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(d > 0, e[i] + (a - v[i]) * (e[i + 1] - e[i]) / np.where(d > 0, d, 1.0), e[i])
        r = np.where(a <= v[0], e[0], r)
        return np.where(a >= v[-1], e[-1], r)


def integral(vec):
    """Cumulative trapezoid integral as G4Scintillation / G4OpWLS build it (the emission integral)."""
    c = [0.0]
    for i in range(1, len(vec)):
        c.append(c[-1] + (vec.e[i] - vec.e[i - 1]) * (0.5 * (vec.v[i - 1] + vec.v[i])))
    out = G4Vector([], [])
    out.e, out.v = vec.e.copy(), np.array(c)
    return out


def group_velocity(rindex):
    """G4MaterialPropertiesTable::CalculateGROUPVEL (Geant4 10.7) of a RINDEX vector, in mm/ns."""
    E, n = rindex.e, rindex.v
    es, vs = [], []
    E0, n0, E1, n1 = E[0], n[0], E[1], n[1]
    vg = C_LIGHT / (n0 + (n1 - n0) / np.log(E1 / E0))
    if vg < 0 or vg > C_LIGHT / n0:
        vg = C_LIGHT / n0
    es.append(E0), vs.append(vg)
    for i in range(2, len(E)):
        vg = C_LIGHT / (0.5 * (n0 + n1) + (n1 - n0) / np.log(E1 / E0))
        if vg < 0 or vg > C_LIGHT / (0.5 * (n0 + n1)):
            vg = C_LIGHT / (0.5 * (n0 + n1))
        es.append(0.5 * (E0 + E1)), vs.append(vg)
        E0, n0, E1, n1 = E1, n1, E[i], n[i]
    vg = C_LIGHT / (n1 + (n1 - n0) / np.log(E1 / E0))
    if vg < 0 or vg > C_LIGHT / n1:
        vg = C_LIGHT / n1
    es.append(E1), vs.append(vg)
    return G4Vector(es, vs)


# --- the 2019 tables that differ (Offline CRVResponse/standalone, January 2019) ---
_V6_E_EM_PPO = (2.0, 2.479, 2.48, 2.49, 2.5, 2.51, 2.52, 2.53, 2.541, 2.551, 2.562, 2.572, 2.583, 2.594, 2.605,
    2.616, 2.627, 2.638, 2.649, 2.661, 2.672, 2.684, 2.695, 2.707, 2.719, 2.731, 2.743, 2.755, 2.768, 2.78, 2.792,
    2.805, 2.818, 2.831, 2.844, 2.857, 2.87, 2.883, 2.897, 2.91, 2.924, 2.938, 2.952, 2.966, 2.98, 2.995, 3.009,
    3.024, 3.039, 3.054, 3.069, 3.084, 3.1, 3.115, 3.131, 3.147, 3.163, 3.179, 3.195, 3.212, 3.229, 3.246, 3.263,
    3.28, 3.297, 3.315, 3.333, 3.351, 3.369, 3.388, 3.406, 3.425, 3.444, 3.463, 3.483, 3.502, 3.522, 3.542, 3.563,
    3.583, 3.604, 3.625, 3.647, 3.668, 3.69, 3.712, 3.734, 3.757, 3.78, 3.803, 3.827, 3.85, 3.875, 3.899, 3.924,
    3.949, 3.974, 3.999, 4.0, 15.75)
_V6_EM_PPO = (0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0, 14.0, 16.0, 17.0, 19.0, 21.0, 23.0, 26.0, 28.0, 31.0, 33.0, 37.0, 41.0, 46.0, 52.0, 59.0,
    66.0, 73.0, 80.0, 88.0, 95.0, 104.0, 112.0, 121.0, 131.0, 144.0, 162.0, 181.0, 204.0, 227.0, 250.0, 270.0, 290.0,
    306.0, 320.0, 329.0, 339.0, 359.0, 395.0, 443.0, 495.0, 537.0, 568.0, 586.0, 594.0, 590.0, 570.0, 548.0, 550.0,
    598.0, 683.0, 767.0, 797.0, 754.0, 676.0, 596.0, 501.0, 418.0, 388.0, 434.0, 528.0, 574.0, 482.0, 301.0, 163.0,
    101.0, 77.0, 65.0, 57.0, 49.0, 40.0, 30.0, 24.0, 23.0, 23.0, 0.0, 0.0)
_V6_E_EM_POPOP = (2.0, 2.479, 2.48, 2.492, 2.505, 2.517, 2.53, 2.543, 2.556, 2.57, 2.583, 2.597, 2.61, 2.624, 2.638,
    2.652, 2.666, 2.681, 2.695, 2.71, 2.725, 2.74, 2.755, 2.771, 2.786, 2.802, 2.818, 2.834, 2.85, 2.867, 2.883, 2.9,
    2.917, 2.935, 2.952, 2.97, 2.988, 3.006, 3.024, 3.043, 3.061, 3.08, 3.1, 3.119, 3.139, 3.159, 3.179, 3.2, 3.22,
    3.241, 3.263, 3.284, 3.306, 3.328, 3.351, 3.374, 3.397, 15.75)
_V6_EM_POPOP = (0.0, 0.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0, 16.0, 19.0, 21.0, 24.0, 26.0, 28.0,
    30.0, 31.0, 31.0, 32.0, 33.0, 35.0, 40.0, 46.0, 53.0, 60.0, 66.0, 70.0, 71.0, 69.0, 65.0, 60.0, 57.0, 60.0, 71.0,
    85.0, 101.0, 108.0, 102.0, 92.0, 77.0, 62.0, 49.0, 45.0, 53.0, 69.0, 85.0, 79.0, 53.0, 29.0, 13.0, 6.0, 2.0, 1.0,
    0.0, 0.0)
_V6_QY_PSPPO = (0.0, 0.0, 0.83, 0.83, 0.78, 0.74, 0.67, 0.63, 0.6, 0.65, 0.68, 0.57, 0.48, 0.39, 0.38, 0.48, 0.53,
    0.58, 0.56, 0.49, 0.43, 0.4, 0.38, 0.35, 0.35, 0.38, 0.43)
_V6_QY_POPOP = (0.86, 0.86)


@dataclass(frozen=True)
class Model:
    """One version of the stock optical model: material tables, surfaces and the end readout.

    ``v6``: Offline CRVResponse/standalone of January 2019 (Geant4 10.3), which made the released CRVConditions v6
    tables. ``current``: Mu2e/CRVStandalone after the July 2021 update (SiPM windows, 1 um air gap, rough fiber-end
    surfaces, lower TiO2 absorption). Both share the RINDEX, absorption and Y11 emission tables.
    """
    name: str
    rindex_ps: G4Vector
    rindex_pmma: G4Vector
    rindex_fpe: G4Vector
    rindex_epoxy: G4Vector
    abs_y11: G4Vector
    qy_y11: G4Vector
    em_y11: G4Vector
    tau_y11: float
    em_ppo: G4Vector
    qy_psppo: G4Vector
    abs_psppo: G4Vector
    tau_psppo: float
    em_popop: G4Vector
    qy_popop: G4Vector
    abs_popop: G4Vector
    tau_popop: float
    scint_decay: float
    scint_rise: float
    refl_tio2: G4Vector
    eff_mppc: G4Vector
    refl_mppc: float
    refl_reflector: float
    fiber_end: str            # "contact": SiPM / reflector cylinders on the fiber end; "gap": 1 um air, epoxy window
    rough_fiber_air: bool     # fiber/air interfaces are unified ground (sigma 0.02) surfaces (current model)
    writer: str               # "v6": probability over all arrivals; "current": arrivals inside the histograms

    @property
    def vg_ps(self):
        return group_velocity(self.rindex_ps)

    @property
    def vg_pmma(self):
        return group_velocity(self.rindex_pmma)

    @property
    def vg_fpe(self):
        return group_velocity(self.rindex_fpe)

    @property
    def vg_epoxy(self):
        return group_velocity(self.rindex_epoxy)


RINDEX_AIR = G4Vector((2.0, 15.75), (1.0, 1.0))
VG_AIR = group_velocity(RINDEX_AIR)
BIRKS = 0.126                            # mm/MeV, written to the table header
SIGMA_TIO2 = 1.5        # rad, unified/ground dielectric_metal, specular lobe 1
SIGMA_FIBER_END = 0.02  # rad, unified/ground dielectric_dielectric between the fiber layers and the air (current)
_E_TIO2 = (2.00, 2.75, 2.88, 2.95, 3.02, 3.10, 3.18, 3.26, 3.35, 3.44, 15.75)
_R_TIO2 = (0.91, 0.91, 0.90, 0.85, 0.69, 0.44, 0.27, 0.13, 0.08, 0.07, 0.07)
_E_MPPC = tuple(2.0 + 0.1 * i for i in range(20)) + (15.75,)
_EFF_MPPC = (0.602, 0.697, 0.784, 0.858, 0.933, 0.970, 0.990, 1.000, 0.978, 0.958, 0.920, 0.871, 0.803, 0.709, 0.622,
             0.547, 0.398, 0.249, 0.100, 0.000, 0.000)


def _model(name):
    v6 = name == "v6"
    if name not in ("v6", "current"):
        raise ValueError(f"unknown optical model {name!r} (v6, current)")
    pmma_uv = 0.1 if v6 else 0.15                       # PMMA = PS - this above 4.5 eV
    pmma = G4Vector(_E_RINDEX_PMMA, [r - pmma_uv if i >= 21 else r for i, r in enumerate(_RINDEX_PMMA_BASE)])
    y11_scale = 0.80 if v6 else 0.85                    # measured Y11 attenuation (m) scaled to the test beam
    tio2_scale = 0.9 if v6 else 0.75                    # R = 1 - scale * (1 - R_measured)
    return Model(
        name=name,
        rindex_ps=G4Vector(_E_RINDEX_PS, _RINDEX_PS),
        rindex_pmma=pmma,
        rindex_fpe=G4Vector(pmma.e, pmma.v - 0.07),
        rindex_epoxy=G4Vector(_E_RINDEX_PS, [r - 0.04 for r in _RINDEX_PS]),
        abs_y11=G4Vector(_E_ABS_Y11, [a * (y11_scale * 1000.0 if i < 273 else 1.0) for i, a in enumerate(_ABS_Y11_RAW)]),
        qy_y11=G4Vector(_E_QY_Y11, _QY_Y11),
        em_y11=G4Vector(_E_EM_Y11, _EM_Y11),
        tau_y11=8.7 if v6 else 10.0,
        em_ppo=G4Vector(_V6_E_EM_PPO, _V6_EM_PPO) if v6 else G4Vector(_E_EM_PPO, _EM_PPO),
        qy_psppo=G4Vector(_E_QY_PSPPO, _V6_QY_PSPPO if v6 else _QY_PSPPO),
        abs_psppo=G4Vector(_E_ABS_PSPPO, [a * 10.0 for a in _ABS_PSPPO_CM]),
        tau_psppo=1.68 if v6 else 1.36,
        em_popop=G4Vector(_V6_E_EM_POPOP, _V6_EM_POPOP) if v6 else G4Vector(_E_EM_POPOP, _EM_POPOP),
        qy_popop=G4Vector((2.0, 15.75), _V6_QY_POPOP if v6 else (0.97, 0.97)),
        abs_popop=G4Vector(_E_ABS_POPOP, [a * 10.0 for a in _ABS_POPOP_CM]),
        tau_popop=1.5 if v6 else 1.12,
        scint_decay=1.68 if v6 else 1.36,
        scint_rise=1.0,
        refl_tio2=G4Vector(_E_TIO2, [1.0 - tio2_scale * (1.0 - r) for r in _R_TIO2]),
        eff_mppc=G4Vector(_E_MPPC, _EFF_MPPC),
        refl_mppc=0.30 if v6 else 0.20,
        refl_reflector=0.90 if v6 else 0.83,
        fiber_end="contact" if v6 else "gap",
        rough_fiber_air=not v6,
        writer="v6" if v6 else "current",
    )


MODELS = {k: _model(k) for k in ("v6", "current")}
DEFAULT_MODEL = "v6"

# module-level aliases of the current model (tests and documentation)
_C = MODELS["current"]
RINDEX_PS, RINDEX_PMMA, RINDEX_FPE, RINDEX_EPOXY = _C.rindex_ps, _C.rindex_pmma, _C.rindex_fpe, _C.rindex_epoxy
ABS_Y11, QY_Y11, EM_Y11, EM_PPO, QY_PSPPO, ABS_PSPPO = _C.abs_y11, _C.qy_y11, _C.em_y11, _C.em_ppo, _C.qy_psppo, \
    _C.abs_psppo
EM_POPOP, QY_POPOP, ABS_POPOP, REFL_TIO2, EFF_MPPC = _C.em_popop, _C.qy_popop, _C.abs_popop, _C.refl_tio2, \
    _C.eff_mppc
VG_PS = group_velocity(RINDEX_PS)


def facet_alpha_icdf(sigma, n=8192):
    """Inverse CDF of the facet tilt alpha drawn by G4OpBoundaryProcess::GetFacetNormal (unified model).

    Geant4 draws alpha ~ N(0, sigma) and keeps it when U * f_max <= sin(alpha) and alpha < pi/2, with
    f_max = min(1, 4 sigma): the kept density is gauss(alpha) * min(1, sin(alpha) / f_max) on (0, pi/2). Returns the
    alpha values at n + 1 equally spaced CDF points (linear interpolation in between)."""
    fmax = min(1.0, 4.0 * sigma)
    a = np.linspace(0.0, min(np.pi / 2, 12.0 * sigma), 200001)
    pdf = np.exp(-0.5 * (a / sigma) ** 2) * np.minimum(1.0, np.sin(a) / fmax)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(a))])
    cdf /= cdf[-1]
    return np.interp(np.linspace(0.0, 1.0, n + 1), cdf, a)


# --- Cerenkov photons per mm (G4CerenkovNew, written to the table header) ---------------------------------------
def _energy_intervals(rindex, beta_inv):
    """G4CerenkovNew::GetEnergyIntervals: the energy ranges where RINDEX > 1/beta."""
    E, n = rindex.e, rindex.v
    out, active, start = [], False, None
    if n[0] > beta_inv:
        active, start = True, E[0]
    for i in range(1, len(E)):
        f = (beta_inv - n[i - 1]) / (n[i] - n[i - 1]) if n[i] != n[i - 1] else 0.0
        if active and n[i] <= beta_inv:
            active = False
            out.append((start, E[i - 1] + (E[i] - E[i - 1]) * f))
        elif not active and n[i] > beta_inv:
            active, start = True, E[i - 1] + (E[i] - E[i - 1]) * f
    if active:
        out.append((start, E[-1]))
    return out


def cerenkov_photons_per_mm(beta, rindex=RINDEX_PS):
    """G4CerenkovNew::GetAverageNumberOfPhotons for a unit charge (Geant4 internal units: MeV, mm)."""
    eV = 1.0e-6
    if beta <= 0:
        return 0.0
    bi = 1.0 / beta
    ev = G4Vector(rindex.e * eV, rindex.v)
    if ev.v[0] <= 1.0:
        return 0.0
    cai = [0.0]
    for i in range(1, len(ev)):
        cai.append(cai[-1] + (ev.e[i] - ev.e[i - 1]) * (0.5 * (1.0 / (ev.v[i - 1] * ev.v[i - 1]) + 1.0 / (ev.v[i] * ev.v[i]))))
    CAI = G4Vector([], [])
    CAI.e, CAI.v = ev.e.copy(), np.array(cai)
    nmin, nmax = ev.v.min(), ev.v.max()
    if nmax < bi:
        dp = ge = 0.0
    elif nmin > bi:
        dp, ge = ev.e[-1] - ev.e[0], CAI.v[-1]
    else:
        dp = ge = 0.0
        for a, b in _energy_intervals(ev, bi):
            dp += b - a
            ge += float(CAI.value(b)) - float(CAI.value(a))
    n = 369.81 / (eV * 10.0) * (dp - ge * bi * bi)
    return max(n, 0.0)


def cerenkov_map(rindex=RINDEX_PS):
    """The (beta, photons/mm) map of the table header: beta = 1, 1/1.01, ... until no photons are made."""
    out, beta = [], 1.0
    while True:
        n = cerenkov_photons_per_mm(beta, rindex)
        out.append((beta, n))
        if n == 0:
            break
        beta /= 1.01
    return np.array(sorted(out))
