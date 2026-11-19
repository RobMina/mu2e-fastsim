"""Geometry of one scintillator strip with two wavelength-shifting fibers (dimensions in mm).

The strip is an extrusion of rectangular cross-section (``thickness`` along x, ``width`` along y, ``length`` along z,
origin at its centre) with a reflective coating of thickness ``coating`` and rounded outer corners, two elliptical
fiber holes (semi-axes ``hole_rx``, ``hole_ry``) at y = +-fiber_separation/2, double-clad fibers (each cladding
``clad_fraction`` of the diameter) running the full length plus a fiber guide bar of length ``guide_bar`` at each
end, and at each fiber end a photodetector (or, at +z, a reflector or black tape) behind an air gap. Every dimension is
a parameter; an experiment adapter supplies its own counters.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Counter:
    """One strip. ``reflector``: 0 photodetectors at both ends, 1 reflector at +z, 2 black tape at +z."""
    length: float
    thickness: float
    width: float
    fiber_separation: float
    fiber_diameter: float
    clad_fraction: float                 # each cladding layer, as a fraction of the fiber diameter
    hole_rx: float                       # fiber hole semi-axis along x (thickness)
    hole_ry: float                       # fiber hole semi-axis along y (width)
    coating: float
    corner_radius: float                 # outer (extrusion) corner radius
    guide_bar: float
    # end readout of the "gap" optical model; the "contact" model puts a photodetector / reflector disc of the fiber
    # radius directly on the fiber end
    sipm_width: float
    sipm_window: float                   # G4Box half-length of the epoxy window
    sipm_length: float
    air_gap: float
    reflector: int = 0

    def __post_init__(self):
        if self.reflector not in (0, 1, 2):
            raise ValueError("reflector must be 0 (none), 1 (reflector at +z) or 2 (black tape at +z)")
        if self.r_clad2 >= self.hole_ry or self.r_clad2 >= self.hole_rx:
            raise ValueError("the fiber must fit inside its hole")
        if self.fiber_separation / 2 + self.hole_ry >= self.hy or self.hole_rx >= self.hx:
            raise ValueError("the fiber holes must lie inside the scintillator")

    # derived dimensions (the expressions of the Geant4 geometry, same floating-point order)
    @property
    def hx(self):
        return self.thickness / 2.0 - self.coating

    @property
    def hy(self):
        return self.width / 2.0 - self.coating

    @property
    def half_length(self):
        return self.length / 2.0

    @property
    def rc(self):
        return self.corner_radius - self.coating

    @property
    def r_clad2(self):
        return self.fiber_diameter / 2.0

    @property
    def r_clad1(self):
        return self.fiber_diameter / 2.0 - self.clad_fraction * self.fiber_diameter

    @property
    def r_core(self):
        c = self.clad_fraction * self.fiber_diameter
        return self.fiber_diameter / 2.0 - c - c


def with_length(c, length, reflector):
    return replace(c, length=float(length), reflector=int(reflector))
