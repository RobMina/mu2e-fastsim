"""Synthetic Run1B-shaped tests; real GDML check opts in with TS1_GDML.

No stock input is bundled. These are geometry/column tests, not G4 certification.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from beamfastsim.tsgeometry import Geometry
from mu2efastsim import eloss, materials


def _fixture():
    root = ET.Element('gdml')
    mats = ET.SubElement(root, 'materials')
    # Atomic isotope abundances (not mass fractions); Cu exercises averaging.
    za = {'C': (6, 12), 'H': (1, 1), 'Al': (13, 27), 'Fe': (26, 56),
          'Cu': (29, 63), 'As': (33, 75), 'V': (23, 51), 'Ti': (22, 48),
          'Mo': (42, 96), 'P': (15, 31), 'S': (16, 32)}
    for name, (z, a) in za.items():
        isotope = ET.SubElement(mats, 'isotope', name=name+'Iso', Z=str(z), N=str(a))
        ET.SubElement(isotope, 'atom', unit='g/mole', value=str(a))
        element = ET.SubElement(mats, 'element', name=name+'0xabc')
        ET.SubElement(element, 'fraction', n='0.75' if name == 'Cu' else '1', ref=name+'Iso')
        if name == 'Cu':
            isotope = ET.SubElement(mats, 'isotope', name='Cu65', Z='29', N='65')
            ET.SubElement(isotope, 'atom', unit='g/mole', value='65')
            ET.SubElement(element, 'fraction', n='0.25', ref='Cu65')
    recipes = {
        'Vacuum': (1e-20, 20, {'H': 1}),
        'G4_GRAPHITE_POROUS': (1.7, 81, {'C': 1}),
        'BronzeC608': (8.17, 310.5, {'Cu': .99, 'As': .01}),
        'StainlessSteel': (8.02, 280.9, {'Fe': 1}),
        'StainlessSteel316': (8, 284.2, {'Fe': .94, 'Mo': .04, 'P': .01, 'S': .01}),
        'Ti6Al4V': (4.43, 228.4, {'Ti': .9, 'Al': .06, 'V': .04}),
        'A1100': (2.71, 166.4, {'Al': 1}),
        'COL5Poly': (.94, 56.5, {'C': .856, 'H': .144}),
    }
    for name, (rho, excitation, fs) in recipes.items():
        m = ET.SubElement(mats, 'material', name=name+'0xdef', state='solid')
        ET.SubElement(m, 'D', unit='g/cm3', value=str(rho))
        ET.SubElement(m, 'MEE', unit='eV', value=str(excitation))
        for e, w in fs.items():
            ET.SubElement(m, 'fraction', n=str(w), ref=e+'0xabc')
    solids, structure = ET.SubElement(root, 'solids'), ET.SubElement(root, 'structure')

    def solid(tag, name, **kw):
        return ET.SubElement(solids, tag, name=name, **{k: str(v) for k, v in kw.items()})

    def tube(name, rmin, rmax, length, **kw):
        return solid('tube', name, rmin=rmin, rmax=rmax, z=length, lunit='mm', aunit='deg', **kw)

    def volume(name, material, shape):
        v = ET.SubElement(structure, 'volume', name=name+'0x123')
        ET.SubElement(v, 'materialref', ref=material+'0xdef')
        ET.SubElement(v, 'solidref', ref=shape)
        return v

    def place(parent, name, pos=(0, 0, 0), rot=None):
        p = ET.SubElement(parent, 'physvol')
        ET.SubElement(p, 'volumeref', ref=name+'0x123')
        ET.SubElement(p, 'position', unit='mm', **dict(zip('xyz', map(str, pos))))
        if rot is not None:
            ET.SubElement(p, 'rotation', unit='deg', **dict(zip('xyz', map(str, rot))))

    def daughter(parent, name, mat, pos=(0, 0, 0), rot=None):
        volume(name, mat, name)
        place(parent, name, pos, rot)

    solid('box', 'world', x=1e5, y=1e5, z=1e5)
    world, hall = volume('World', 'Vacuum', 'world'), volume('HallAir', 'Vacuum', 'world')
    offset = np.array([913.5, 5917.6, 5701.795])
    place(world, 'HallAir', offset)
    vac = {}
    specs = [(1, (3904, 0, -3466.5), None), (2, (975, 0, -2929), (90, 0, 0)),
             (3, (0, 0, 0), (0, 90, 0)), (4, (-975, 0, 2929), (90, 0, 0)),
             (5, (-3904, 0, 3554), None)]
    for i, centre, rotation in specs:
        name = f'TS{i}Vacuum'
        if i in (2, 4):
            solid('torus', name, rmin=0, rmax=250, rtor=2929, startphi=270 if i == 2 else 90,
                  deltaphi=90, aunit='deg', lunit='mm')
        else:
            tube(name, 0, 248 if i == 5 else 250, {1: 1075, 3: 1950, 5: 1250}[i])
        vac[i] = volume(name, 'Vacuum', name)
        place(hall, name, np.array(centre)-offset, rotation)
    solid('cone', 'Coll11', rmin1=150, rmin2=150, rmax1=165.5, rmax2=165.5, z=999.96)
    daughter(vac[1], 'Coll11', 'G4_GRAPHITE_POROUS', (0, 0, -37.5))
    for name, rlo, rhi in [('Coll12', 167, 240), ('Coll13', 240, 243)]:
        tube(name, rlo, rhi, 999.96)
        daughter(vac[1], name, 'BronzeC608', (0, 0, -37.5))
    tube('PbarAbsTS1Out', 120, 150, 200, startphi=210, deltaphi=120)
    daughter(vac[1], 'PbarAbsTS1Out', 'G4_GRAPHITE_POROUS', (0, 0, 312.5))
    tube('holeCircle', 0, 150, 804)
    solid('box', 'holeBox', x=310, y=200, z=802)
    shape = solid('intersection', 'hole')
    ET.SubElement(shape, 'first', ref='holeBox')
    ET.SubElement(shape, 'second', ref='holeCircle')
    for name, z in [('Coll31', -452), ('Coll32', 452)]:
        tube(name+'Mother', 0, 243, 799.96)
        shape = solid('subtraction', name)
        ET.SubElement(shape, 'first', ref=name+'Mother')
        ET.SubElement(shape, 'second', ref='hole')
        ET.SubElement(shape, 'position', unit='mm', x='0', y='90', z='0')
        daughter(vac[3], name, 'BronzeC608', (0, 0, z))
    for name, lo, hi, dz, mat in [('pBarAbsSupport', 225, 243, 10, 'StainlessSteel'),
                                 ('PbarAbsDisk', 0, 225, .15, 'Ti6Al4V')]:
        tube(name, lo, hi, dz)
        daughter(vac[3], name, mat)
    wedge = solid('xtru', 'PbarAbsWedge', lunit='mm')
    for x, y in [(.254, 0), (.254, -140), (.127, -140), (.127, -90), (0, -90), (0, 0)]:
        ET.SubElement(wedge, 'twoDimVertex', x=str(x), y=str(y))
    for z in [-152.3, 152.3]:
        ET.SubElement(wedge, 'section', zPosition=str(z), scalingFactor='1', xOffset='0', yOffset='0')
    daughter(vac[3], 'PbarAbsWedge', 'A1100', (0, 190, 5.1), (0, 90, 0))
    for name, lo, hi, dz, pos, rot, parent, mat in [
        ('Coll51', 128, 232.48, 999.96, (0, 0, -124), None, 5, 'COL5Poly'),
        ('Coll52', 235, 243, 1249.98, (0, 0, -.01), None, 5, 'StainlessSteel316'),
        ('Coll52InTS4', 235, 243, 74.98, (-2929, 37.49, 0), (90, 0, 0), 4, 'StainlessSteel316'),
        ('Coll53InTS5', 243.01, 248, 18.96, (0, 0, 615.5), None, 5, 'StainlessSteel316')]:
        tube(name, lo, hi, dz)
        daughter(vac[parent], name, mat, pos, rot)
    # Unselected unsupported solid and upstream window must neither fail nor count.
    solid('sphere', 'unrelated', rmax=20)
    volume('unrelated', 'A1100', 'unrelated')
    place(hall, 'unrelated', -offset)
    tube('upstreamWindow', 0, 160, .25)
    daughter(hall, 'upstreamWindow', 'A1100', np.array([3904, 0, -4144])-offset)
    setup = ET.SubElement(root, 'setup')
    ET.SubElement(setup, 'world', ref='World0x123')
    return root


@pytest.fixture
def geometry(tmp_path):
    path = tmp_path / 'synthetic.gdml'
    ET.ElementTree(_fixture()).write(path)
    return Geometry(path)


def _chord(g, a, b, key):
    return g.columns(np.atleast_2d(a), np.atleast_2d(b))[key]


def test_selection_and_complete_compositions(geometry):
    g = geometry
    assert len(g._vacuum) == 5 and len(g._matter) == 13 and len(g.materials) == 7
    assert not {'As', 'V', 'Mo', 'P', 'S'} & materials.ELEMENTS.keys()
    for m in g.materials.values():
        assert isinstance(m, materials.Material)
        assert sum(m.fractions.values()) == pytest.approx(1)
        assert m.x0_mm > 0 and m.electron_density_per_mm3 > 0
        assert np.isfinite(m.zeff) and np.all(np.isfinite(m.density_params()))
        assert np.isfinite(eloss.dedx_restricted(m, np.array([100.]), 1e9)).all()
    bronze = g.materials['BronzeC608']
    assert bronze.elements['Cu'] == (29, 63.5)
    assert bronze.zeff == pytest.approx(.99*29+.01*33)
    assert bronze.electron_density_per_mm3 == pytest.approx(
        8.17 * materials.N_A * 1e-3 * (.99*29/63.5+.01*33/75))


def test_vacuum_frames_and_endpoints(geometry):
    yes = [[3904, 0, -4004], [3904, 0, -2929], [975, 0, 0], [-975, 0, 0],
           [-3904, 0, 2929], [-3904, 0, 4179], [3904, 180, -3504]]
    no = [[3904, 0, -4004.01], [3904, 250.01, -3500], [0, 250.01, 0],
          [-3904, 248.01, 3500], [-3904, 0, 4179.01], [0, 0, 1000]]
    assert geometry.inside(yes).all()
    assert not geometry.inside(no).any()


@pytest.mark.parametrize('ts', [2, 4])
def test_exact_quarter_torus(geometry, ts):
    angles = np.linspace(.01, np.pi/2-.01, 23)
    sign, cx, cz = (1, 975, -2929) if ts == 2 else (-1, -975, 2929)
    xyz = np.c_[cx+sign*2929*np.cos(angles), np.zeros(len(angles)), cz+sign*2929*np.sin(angles)]
    assert geometry.inside(xyz).all()
    xyz[:, 1] = 249.999
    assert geometry.inside(xyz).all()
    xyz[:, 1] = 250.001
    assert not geometry.inside(xyz).any()
    # Points on the other half of the full torus are outside the TS union.
    wrong = [cx-sign*2929/np.sqrt(2), 0, cz-sign*2929/np.sqrt(2)]
    assert not geometry.inside([wrong])[0]


@pytest.mark.parametrize('y,z,disk,wedge', [(0, 0, .15, 0), (75, 0, .15, .127),
    (150, 0, .15, .254), (150, 153, .15, 0), (49, 0, .15, 0), (191, 0, .15, 0)])
def test_thin_disk_and_stepped_wedge_no_midpoint_sampling(geometry, y, z, disk, wedge):
    a, b = np.array([[10., y, z]]), np.array([[-10., y, z]])
    c = geometry.columns(a, b)
    assert c['Ti6Al4V'][0] == pytest.approx(disk, abs=2e-11)
    assert c['A1100'][0] == pytest.approx(wedge, abs=2e-11)
    back = geometry.columns(b, a)
    for k in c:
        np.testing.assert_allclose(c[k], back[k], atol=2e-11)


def test_disk_oblique_partial_tangent_and_zero(geometry):
    # 45-degree axial chord, then starts in disk, then tangent to the circular rim.
    a = [[1, 1, 0], [0, 0, 0], [0, -300, 225], [0, 0, 0]]
    b = [[-1, -1, 0], [1, 0, 0], [0, 300, 225], [0, 0, 0]]
    np.testing.assert_allclose(geometry.columns(a, b)['Ti6Al4V'], [.15*np.sqrt(2), .075, 0, 0], atol=1e-9)
    assert _chord(geometry, [-10, 230, 0], [10, 230, 0], 'StainlessSteel')[0] == pytest.approx(10)
    assert _chord(geometry, [3904, 0, -4200], [3904, 0, -4100], 'A1100')[0] == 0


def test_collimator_shells_collar_and_col5_extension(geometry):
    for radius, key, expected in [(149, 'G4_GRAPHITE_POROUS', 0), (160, 'G4_GRAPHITE_POROUS', 999.96),
                                  (166, 'BronzeC608', 0), (180, 'BronzeC608', 999.96),
                                  (240, 'BronzeC608', 999.96), (241, 'BronzeC608', 999.96)]:
        assert _chord(geometry, [3904, radius, -4010], [3904, radius, -2929], key)[0] == pytest.approx(expected)
    assert _chord(geometry, [3904, -135, -3300], [3904, -135, -3000], 'G4_GRAPHITE_POROUS')[0] == pytest.approx(200)
    assert _chord(geometry, [3904, 135, -3300], [3904, 135, -3000], 'G4_GRAPHITE_POROUS')[0] == 0
    assert _chord(geometry, [-3904, 180, 2900], [-3904, 180, 4179], 'COL5Poly')[0] == pytest.approx(999.96)
    assert _chord(geometry, [-3904, 239, 2800], [-3904, 239, 4200], 'StainlessSteel316')[0] == pytest.approx(1324.96)
    assert _chord(geometry, [-3904, 246, 4100], [-3904, 246, 4200], 'StainlessSteel316')[0] == pytest.approx(18.96)


@pytest.mark.parametrize('y,z,expected', [(90, 0, 0), (-9.99, 0, 0), (-10.01, 0, 1599.92),
                                        (190.01, 0, 1599.92), (90, 149, 0), (90, 151, 1599.92)])
def test_col3_boolean_clipped_offset_hole(geometry, y, z, expected):
    assert _chord(geometry, [-900, y, z], [900, y, z], 'BronzeC608')[0] == pytest.approx(expected, abs=1e-8)


def test_split_invariance_and_batch_shape(geometry):
    rng = np.random.default_rng(42)
    a = np.c_[np.full(80, 900.), rng.uniform(-240, 240, 80), rng.uniform(-150, 150, 80)]
    b = a.copy()
    b[:, 0] = -900
    mid = a + .371*(b-a)
    whole, first, second = geometry.columns(a, b), geometry.columns(a, mid), geometry.columns(mid, b)
    for k in whole:
        np.testing.assert_allclose(whole[k], first[k]+second[k], atol=1e-8)
        assert np.all(whole[k] >= 0)
    assert geometry.inside(np.empty((0, 3))).shape == (0,)
    assert all(v.shape == (0,) for v in geometry.columns(np.empty((0, 3)), np.empty((0, 3))).values())


@pytest.mark.parametrize('bad', [[1, 2, 3], [[np.nan, 0, 0]], [[0, np.inf, 0]], np.zeros((2, 4))])
def test_invalid_points_rejected(geometry, bad):
    with pytest.raises(ValueError, match='positions'):
        geometry.inside(bad)
    with pytest.raises(ValueError, match='positions'):
        geometry.columns(bad, [[0, 0, 0]])


@pytest.mark.parametrize('mutation,match', [('sphere', 'unsupported selected solid'),
    ('cone', 'tapered cone'), ('xtru', 'scaled/offset'), ('missing', 'expected exactly'),
    ('units', 'units'), ('composition', 'composition')])
def test_unsupported_selected_geometry_fails(tmp_path, mutation, match):
    root = _fixture()
    if mutation == 'sphere':
        root.find("solids/tube[@name='PbarAbsDisk']").tag = 'sphere'
    elif mutation == 'cone':
        root.find("solids/cone[@name='Coll11']").set('rmax2', '166')
    elif mutation == 'xtru':
        root.find('solids/xtru/section').set('xOffset', '1')
    elif mutation == 'missing':
        for p in root.find("structure/volume[@name='HallAir0x123']").findall('physvol'):
            if p.find('volumeref').get('ref') == 'TS2Vacuum0x123':
                root.find("structure/volume[@name='HallAir0x123']").remove(p)
    elif mutation == 'units':
        root.find("solids/tube[@name='PbarAbsDisk']").set('lunit', 'cm')
    else:
        root.find("materials/material[@name='Ti6Al4V0xdef']/fraction").set('n', '.8')
    path = tmp_path / 'unsupported.gdml'
    ET.ElementTree(root).write(path)
    with pytest.raises(ValueError, match=match):
        Geometry(path)


@pytest.mark.skipif(not os.environ.get('TS1_GDML'), reason='opt-in saved Run1B GDML')
def test_saved_run1b_gdml():
    g = Geometry(os.environ['TS1_GDML'])
    assert (len(g._vacuum), len(g._matter), len(g.materials)) == (5, 13, 7)
    assert g.inside([[3904, 0, -4004], [0, 0, 0], [-3904, 0, 4179]]).all()
    c = g.columns([[10, 75, 0], [10, 150, 0]], [[-10, 75, 0], [-10, 150, 0]])
    np.testing.assert_allclose(c['Ti6Al4V'], [.15, .15], atol=2e-11)
    np.testing.assert_allclose(c['A1100'], [.127, .254], atol=2e-11)
    assert g.materials['Ti6Al4V'].rho == 4.43
    assert g.materials['Ti6Al4V'].fractions['V'] == .04
    assert g.materials['BronzeC608'].I_eV == pytest.approx(310.513212304619)
    assert g.materials['StainlessSteel316'].fractions['Mo'] == .025
    assert g.materials['Ti6Al4V'].x0_mm == pytest.approx(37.2006, rel=1e-6)


@pytest.mark.skipif(not os.environ.get('TS1_RUN1A_GDML'), reason='opt-in saved Run1A GDML')
def test_saved_run1a_boundaries():
    g = Geometry(os.environ['TS1_RUN1A_GDML'])
    assert g.z_start == -4004
    assert g.z_end == 4179
    assert len(g._vacuum) == 5
    assert g.inside([[3904, 0, -4004], [0, 0, 0], [-3904, 0, 4179]]).all()
    assert 'BronzeC608' in g.materials
    assert 'G4_GRAPHITE_POROUS' in g.materials


def test_envelope_safety_balls_are_contained(geometry):
    rng = np.random.default_rng(521)
    points = rng.uniform([-4200, -300, -4200], [4200, 300, 4400], (5000, 3))
    points = np.r_[points, [[3904, 0, -3500], [0, 0, 0], [-3904, 0, 3500],
                            [3904, 0, -4004], [-3904, 0, 4179]]]
    safety = geometry.safety(points)
    assert np.all(safety >= 0)
    assert np.all(safety[~geometry.inside(points)] == 0)
    assert np.count_nonzero(safety > 0) > 100
    assert np.all(safety[-2:] == 0)
    for _ in range(30):
        direction = rng.normal(size=points.shape)
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        displaced = points + (0.999 * safety)[:, None] * direction
        assert geometry.inside(displaced[safety > 0]).all()
    assert geometry.safety(np.empty((0, 3))).shape == (0,)


def test_tube_and_torus_safety_handles_inner_shell_and_phi():
    from beamfastsim.tsgeometry import _Solid
    tube = _Solid(ET.Element('tube', name='tube', rmin='2', rmax='5', z='10'), {}, vacuum=True)
    np.testing.assert_allclose(tube.safety(np.array([[3., 0, 0], [4., 0, 4], [1., 0, 0],
                                                    [6., 0, 0], [3., 0, 6]])), [1, 1, 0, 0, 0], atol=2e-8)
    element = ET.Element('torus', name='torus', rmin='0', rmax='2', rtor='10',
                         startphi='270', deltaphi='90')
    torus = _Solid(element, {}, vacuum=True)
    assert torus.safety(np.array([[10., 0, 0], [0, -10., 0], [-10., 0, 0]])).max() == 0
    p = np.array([[10 / np.sqrt(2), -10 / np.sqrt(2), 0]])
    np.testing.assert_allclose(torus.safety(p), [2], atol=2e-8)
    element.set('deltaphi', '270')
    assert _Solid(element, {}, vacuum=True).safety(p)[0] == 0
