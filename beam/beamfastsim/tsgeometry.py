"""Opt-in Run1B TS development geometry, not a general GDML navigator.

Coordinates and lengths are Mu2e mm. The stock v40 world is already Mu2e;
HallAir's placement MUST be composed with its daughters (not subtracted twice).
Only TS1..5 vacuum envelopes and their material daughters are selected. The
upstream TS1 entrance window, cryostats, coils and DS are deliberately excluded.
``inside`` includes material within the vacuum envelopes: it is a transport-domain
mask, not a free-space test. ``columns`` returns exact straight-chord material
lengths, including thin absorbers, not ordered interactions or curved-path lengths.
Supported selected solids: tubes, constant-radius cones, boxes, unscaled xtru,
and their booleans; tori are supported for the vacuum envelopes only. No default
inputs, automatic activation, geometry certification or general overlap handling.
"""
from dataclasses import dataclass, field
from functools import cached_property
import xml.etree.ElementTree as ET

import numpy as np

from mu2efastsim.materials import Material, N_A
from mu2efastsim.world import _rot, _strip, signed_local


@dataclass(frozen=True)
class GDMLMaterial(Material):
    """Core-compatible material with local isotope-derived Z,A (no global edits)."""
    elements: dict = field(default_factory=dict)

    @cached_property
    def zeff(self):
        return sum(w * self.elements[e][0] for e, w in self.fractions.items())

    @cached_property
    def electron_density_per_mm3(self):
        return self.rho * N_A * 1e-3 * sum(
            w * self.elements[e][0] / self.elements[e][1] for e, w in self.fractions.items())

    @cached_property
    def x0_mm(self):
        """Same Dahl/PDG mixture law as core eloss, using the complete composition."""
        inv = 0.0
        light = {1: (5.31, 6.144), 2: (4.79, 5.621), 3: (4.74, 5.805), 4: (4.71, 5.924)}
        for e, w in self.fractions.items():
            z, a = self.elements[e]
            lr, lp = light[z] if z <= 4 else (np.log(184.15 * z ** (-1/3)), np.log(1194 * z ** (-2/3)))
            az = (z / 137.035999) ** 2
            f = az * (1/(1+az) + .20206 - .0369*az + .0083*az**2 - .002*az**3)
            inv += w * (z*z*(lr-f) + z*lp) / (716.408*a)
        return 10 / (self.rho * inv)


def _material(ref, nodes):
    def atom(e):
        a = e.find('atom')
        if a is not None:
            if a.get('unit', 'g/mole') not in ('g/mole', 'g/mol'):
                raise ValueError('unsupported GDML atomic mass unit')
            return float(e.get('Z')), float(a.get('value'))
        fs = e.findall('fraction')
        za = [atom(nodes[f.get('ref')]) for f in fs]
        w = np.array([float(f.get('n')) for f in fs])
        if not za or not np.isclose(w.sum(), 1) or len({z for z, _ in za}) != 1:
            raise ValueError('invalid GDML isotope fractions')
        return za[0][0], sum(q * a for q, (_, a) in zip(w, za))

    fractions, elements = {}, {}

    def composition(name, weight):
        e = nodes[name]
        if e.tag in ('element', 'isotope'):
            key = _strip(name)
            elements[key] = atom(e)
            fractions[key] = fractions.get(key, 0) + weight
        else:
            fs = e.findall('fraction')
            if not fs or not np.isclose(sum(float(f.get('n')) for f in fs), 1):
                raise ValueError(f'unsupported GDML composition: {name}')
            for f in fs:
                composition(f.get('ref'), weight * float(f.get('n')))

    e = nodes[ref]
    composition(ref, 1.0)
    d, mee = e.find('D'), e.find('MEE')
    if d is None or mee is None or d.get('unit') != 'g/cm3' or mee.get('unit') != 'eV':
        raise ValueError(f'missing/unsupported GDML density or MEE: {ref}')
    rho, excitation = float(d.get('value')), float(mee.get('value'))
    values = [rho, excitation, *fractions.values(), *np.asarray(list(elements.values())).ravel()]
    if not np.all(np.isfinite(values)) or min(values) <= 0:
        raise ValueError(f'invalid GDML material: {ref}')
    return GDMLMaterial(_strip(ref), rho, excitation, fractions,
                        state=e.get('state', 'solid'), elements=elements)


def _placement(e):
    if any(e.find(tag) is not None for tag in ('positionref', 'rotationref', 'scale', 'scaleref')):
        raise ValueError('unsupported GDML referenced/scaled placement')
    p, r = e.find('position'), e.find('rotation')
    if p is not None and p.get('unit', 'mm') != 'mm':
        raise ValueError('TS geometry requires mm placements')
    if r is not None and r.get('unit', 'deg') not in ('deg', 'rad'):
        raise ValueError('unsupported GDML angle unit')
    t = np.zeros(3) if p is None else np.array([float(p.get(a, 0)) for a in 'xyz'])
    return _rot(r), t


def _ratio(num, den):
    num, den = np.broadcast_arrays(num, den)
    return np.divide(num, den, out=np.full(num.shape, np.inf), where=abs(den) > 1e-15)


class _Solid:
    """Contains and analytic surface intersections for the selected primitive set."""
    def __init__(self, e, solids, vacuum=False):
        self.tag, self.vacuum = e.tag, vacuum
        fail = f'unsupported selected solid {e.get("name")}: {e.tag}'
        if e.get('lunit', 'mm') != 'mm' or e.get('aunit', 'deg') not in ('deg', 'rad'):
            raise ValueError(fail + ' units')
        if self.tag in ('subtraction', 'intersection', 'union'):
            self.a = _Solid(solids[e.find('first').get('ref')], solids, vacuum)
            self.b = _Solid(solids[e.find('second').get('ref')], solids, vacuum)
            self.R, self.t = _placement(e)
        elif self.tag in ('box', 'xtru'):
            if self.tag == 'xtru' and any(float(s.get(k, default)) != float(default)
                    for s in e.findall('section') for k, default in
                    (('scalingFactor', '1'), ('xOffset', '0'), ('yOffset', '0'))):
                raise ValueError(fail + ' scaled/offset sections')
            pieces = signed_local(e, solids)
            if not pieces:
                raise ValueError(fail)
            self.planes = [p for _, p, _, _ in pieces]
            self.box = (np.min([lo for _, _, lo, _ in pieces], axis=0),
                        np.max([hi for _, _, _, hi in pieces], axis=0))
        elif self.tag in ('tube', 'cone', 'torus'):
            if self.tag == 'cone':
                if any(e.get(k+'1') != e.get(k+'2') for k in ('rmin', 'rmax')):
                    raise ValueError(fail + ' tapered cone')
                self.tag = 'tube'
                self.rmin, self.rmax = (float(e.get(k+'1')) for k in ('rmin', 'rmax'))
            else:
                self.rmin, self.rmax = float(e.get('rmin', 0)), float(e.get('rmax'))
            if self.tag == 'torus' and not vacuum:
                raise ValueError(fail + ' material torus')
            self.extent = float(e.get('rtor')) if self.tag == 'torus' else float(e.get('z')) / 2
            unit = np.pi / 180 if e.get('aunit', 'deg') == 'deg' else 1
            self.phi = float(e.get('startphi', 0)) * unit
            self.width = float(e.get('deltaphi', 360 if unit != 1 else 2*np.pi)) * unit
            if not (0 <= self.rmin < self.rmax and self.extent > 0 and 0 < self.width <= 2*np.pi+1e-12):
                raise ValueError(fail + ' dimensions')
        else:
            raise ValueError(fail)

    def bounds(self):
        """Conservative local box used only to skip impossible chord intersections."""
        if self.tag in ('subtraction', 'intersection', 'union'):
            lo, hi = self.a.bounds()
            if self.tag != 'union':
                return lo, hi
            bl, bh = self.b.bounds()
            corners = np.array([[x, y, z] for x in (bl[0], bh[0]) for y in (bl[1], bh[1])
                                for z in (bl[2], bh[2])]) @ self.R.T + self.t
            return np.minimum(lo, corners.min(0)), np.maximum(hi, corners.max(0))
        if self.tag in ('box', 'xtru'):
            return self.box
        if self.tag == 'tube':
            hi = np.array([self.rmax, self.rmax, self.extent])
            return -hi, hi
        raise ValueError('no material-torus bounds')

    def contains(self, p):
        if self.tag in ('subtraction', 'intersection', 'union'):
            a, b = self.a.contains(p), self.b.contains((p-self.t) @ self.R)
            return a & ~b if self.tag == 'subtraction' else (a & b if self.tag == 'intersection' else a | b)
        if self.tag in ('box', 'xtru'):
            return np.logical_or.reduce([np.all(p @ q[:, :3].T <= q[:, 3]+1e-10, axis=1) for q in self.planes])
        radial = np.hypot(p[:, 0], p[:, 1])
        r = np.hypot(radial-self.extent, p[:, 2]) if self.tag == 'torus' else radial
        # Half-open radial shells give adjacent COL1 layers a single owner.
        # Vacuum envelopes are closed so seams remain inside the transport domain.
        ok = ((r >= self.rmin-1e-10) & (r <= self.rmax+1e-10) if self.vacuum
              else (r >= self.rmin) & (r < self.rmax))
        if self.tag == 'tube':
            ok &= abs(p[:, 2]) <= self.extent+1e-10
        if self.width < 2*np.pi-1e-12:
            phi = (np.arctan2(p[:, 1], p[:, 0])-self.phi) % (2*np.pi)
            ok &= (phi <= self.width+1e-12) | (phi >= 2*np.pi-1e-12)
        return ok

    def safety(self, p):
        """Lower bound on distance to leaving one closed tube/torus envelope."""
        if not self.vacuum or self.tag not in ('tube', 'torus'):
            return np.zeros(len(p))
        radial = np.hypot(p[:, 0], p[:, 1])
        r = np.hypot(radial-self.extent, p[:, 2]) if self.tag == 'torus' else radial
        distance = self.rmax-r
        if self.rmin > 0:
            distance = np.minimum(distance, r-self.rmin)
        if self.tag == 'tube':
            distance = np.minimum(distance, self.extent-abs(p[:, 2]))
        if self.width < 2*np.pi-1e-12:
            if self.width > np.pi:
                return np.zeros(len(p))
            start = np.array([-np.sin(self.phi), np.cos(self.phi)])
            end = np.array([np.sin(self.phi+self.width), -np.cos(self.phi+self.width)])
            distance = np.minimum(distance, np.minimum(p[:, :2] @ start, p[:, :2] @ end))
        return np.maximum(distance-1e-8, 0)

    def cuts(self, o, d):
        if self.tag in ('subtraction', 'intersection', 'union'):
            return np.concatenate([self.a.cuts(o, d), self.b.cuts((o-self.t) @ self.R, d @ self.R)], axis=1)
        if self.tag in ('box', 'xtru'):
            q = np.concatenate(self.planes)
            return _ratio(q[:, 3]-o @ q[:, :3].T, d @ q[:, :3].T)
        cuts = [_ratio(-self.extent-o[:, 2], d[:, 2]), _ratio(self.extent-o[:, 2], d[:, 2])]
        a = np.sum(d[:, :2]**2, axis=1)
        b = np.sum(o[:, :2]*d[:, :2], axis=1)
        for r in (self.rmin, self.rmax):
            disc = b*b - a*(np.sum(o[:, :2]**2, axis=1)-r*r)
            root = np.sqrt(np.maximum(disc, 0))
            cuts += [np.where(disc >= 0, _ratio(-b+s*root, a), np.inf) for s in (-1, 1)]
        if self.width < 2*np.pi-1e-12:
            for phi in (self.phi, self.phi+self.width):
                n = np.array([-np.sin(phi), np.cos(phi), 0])
                cuts.append(_ratio(-o @ n, d @ n))
        return np.stack(cuts, axis=1)


def _points(x):
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.all(np.isfinite(x)):
        raise ValueError('positions must be finite (N,3) arrays in Mu2e mm')
    return x


class Geometry:
    """Explicit ``Geometry(gdml_path)``; no bundled/default input or global registration.

    ``materials`` maps column keys to core-compatible GDMLMaterial instances. Use
    their ``x0_mm`` rather than core's incomplete element lookup. Geometry assumes
    disjoint material siblings, as in Run1B. Nested material daughters are rejected.
    """
    def __init__(self, gdml_path):
        root = ET.parse(gdml_path).getroot()
        vols = {v.get('name'): v for v in root.findall('structure/volume')}
        solids = {s.get('name'): s for s in root.find('solids')}
        nodes = {m.get('name'): m for m in root.find('materials')}
        wanted = {f'TS{i}Vacuum' for i in range(1, 6)}
        found = []
        self._vacuum, self._matter, self.materials = [], [], {}

        def walk(name, R, t, selected=False, material_parent=False):
            v = vols[name]
            is_envelope = _strip(name) in wanted
            selected |= is_envelope
            m = v.find('materialref').get('ref')
            nonvac = not any(s in _strip(m).lower() for s in ('vacuum', 'galactic', 'air'))
            if selected and (is_envelope or nonvac):
                if material_parent:
                    raise ValueError('nested selected material volumes are unsupported')
                solid = _Solid(solids[v.find('solidref').get('ref')], solids, is_envelope)
                if is_envelope:
                    found.append(_strip(name))
                    self._vacuum.append((solid, R, t))
                else:
                    key = _strip(m)
                    if key not in self.materials:
                        self.materials[key] = _material(m, nodes)
                    self._matter.append((solid, R, t, key))
            if selected and nonvac and v.findall('physvol'):
                raise ValueError('nested selected material volumes are unsupported')
            for pv in v.findall('physvol'):
                r, p = _placement(pv)
                walk(pv.find('volumeref').get('ref'), R @ r, t+R @ p, selected, selected and nonvac)

        walk(root.find('setup/world').get('ref'), np.eye(3), np.zeros(3))
        if sorted(found) != sorted(wanted):
            raise ValueError('expected exactly one each of TS1Vacuum through TS5Vacuum')
        ts1, ts5 = (self._vacuum[found.index(name)] for name in ('TS1Vacuum', 'TS5Vacuum'))
        self.z_start = float(ts1[2][2] - ts1[0].extent)
        self.z_end = float(ts5[2][2] + ts5[0].extent)

    def inside(self, x):
        """Boolean (N,) membership in the exact cylinder/quarter-torus union."""
        x = _points(x)
        return np.logical_or.reduce([s.contains((x-t) @ R) for s, R, t in self._vacuum])

    def safety(self, x):
        """Radius of a ball contained in at least one envelope; seams may return zero."""
        x = _points(x)
        return np.maximum.reduce([s.safety((x-t) @ R) for s, R, t in self._vacuum])

    def columns(self, x0, x1):
        """Dict material -> (N,) mm on straight chords; exact planar/circular cuts.

        No midpoint sampling: split at EVERY surface root, then classify each open
        interval. Disk/wedge axial thickness is retained even for long chords.
        Zero-length and empty batches return zeros; reversing a chord is symmetric.
        """
        x0, x1 = _points(x0), _points(x1)
        if x0.shape != x1.shape:
            raise ValueError('chord endpoint shapes differ')
        out = {m: np.zeros(len(x0)) for m in self.materials}
        for start in range(0, len(x0), 2048):
            sl = slice(start, start+2048)
            delta = x1[sl]-x0[sl]
            length = np.linalg.norm(delta, axis=1)
            for s, R, t, key in self._matter:
                o, d = (x0[sl]-t) @ R, delta @ R
                lo, hi = s.bounds()
                candidate = np.all((np.maximum(o, o+d) >= lo-1e-9) & (np.minimum(o, o+d) <= hi+1e-9), axis=1)
                if not candidate.any():
                    continue
                o, d = o[candidate], d[candidate]
                cuts = np.sort(np.clip(np.c_[np.zeros(len(o)), s.cuts(o, d), np.ones(len(o))], 0, 1), axis=1)
                mid = (cuts[:, 1:]+cuts[:, :-1]) / 2
                p = o[:, None, :]+mid[:, :, None]*d[:, None, :]
                mask = s.contains(p.reshape(-1, 3)).reshape(mid.shape)
                out[key][sl][candidate] += length[candidate] * np.sum(np.diff(cuts, axis=1)*mask, axis=1)
        return out
