"""
synthetic_points.py

Builds a small synthetic REG dataset whose correct answer is known exactly.

Each "atom" here is a cluster of point charges at radius rho around its nucleus,
so three things are true at once and none of them are fudged:

  * the atom's spherical multipole moments can be written down exactly, from the
    solid harmonics of its own point charges;
  * the exact classical interaction of two atoms is a finite Coulomb sum;
  * the multipole series between them converges if and only if R_AB > rho_A +
    rho_B, which is the real convergence-sphere condition the gates exist to
    detect.

The geometry is a water-dimer-like approach: one unit is held still and the
other walks in along the donor H, so the H-bond pair starts well outside the
overlap condition and ends inside it.  A pair that converges at one end of the
path and not at the other is the case path-wide admission exists for.

The solid harmonics here are built from scipy's spherical harmonics, not from
the recursion in multipole_utils, so the fixture is an independent check of that
recursion rather than a restatement of it.

Files written per point mimic the parts of AIMAll and Gaussian output that
REG.py actually reads: a .wfn (atom list and total energy), a Gaussian output
(Standard orientation block), a .sum (IQA pair table) and one .int per atom
(moments, net charge, beta-sphere radius).
"""

import os

import numpy as np
from scipy.special import sph_harm

BOHR_PER_ANGSTROM = 1.8897261254578281

# Element, nuclear-charge-like Z, cluster radius rho (Angstrom), net charge.
ATOM_TYPES = {
    'O': {'Z': 8.0, 'rho': 1.20, 'q': -0.60},
    'H': {'Z': 1.0, 'rho': 0.80, 'q': 0.30},
}

# Fragment A is the acceptor, its hydrogens pointing away from the approach.
# Fragment B walks in along x with H5 leading, giving an O1...H5 contact.
FRAGMENT_A = [('O', (0.00, 0.00, 0.00)),
              ('H', (-0.24, 0.93, 0.00)),
              ('H', (-0.24, -0.93, 0.00))]
FRAGMENT_B = [('O', (0.00, 0.00, 0.00)),
              ('H', (-0.96, 0.00, 0.00)),
              ('H', (0.24, 0.93, 0.00))]

# Separations of the two oxygens, in Angstrom.  R(O1...H5) = d - 0.96, so it
# runs 3.64 -> 1.84 while rho_O + rho_H = 2.00: the contact pair crosses the
# convergence condition partway along the path.
DEFAULT_SEPARATIONS = (4.60, 4.20, 3.80, 3.40, 3.00, 2.80)

_TETRAHEDRON = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0],
                         [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]) / np.sqrt(3.0)


def n_components(l_max):
    return (l_max + 1) ** 2


def component_index(l, k):
    if k == 0:
        return l * l
    return l * l + (2 * abs(k) - 1 if k > 0 else 2 * abs(k))


def solid_harmonics(xyz, l_max):
    """Stone's real regular solid harmonics C_lm, as AIMAll's Q[l,m] are defined.

    Condon-Shortley phase included, the sqrt((2l+1)/4pi) normalisation left out.
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    safe_r = np.where(r > 0, r, 1.0)
    theta = np.arccos(np.clip(np.where(r > 0, z / safe_r, 1.0), -1.0, 1.0))
    phi = np.arctan2(y, x)
    out = np.zeros((xyz.shape[0], n_components(l_max)))
    for l in range(l_max + 1):
        prefactor = np.sqrt(4 * np.pi / (2 * l + 1)) * r ** l
        out[:, component_index(l, 0)] = (prefactor * sph_harm(0, l, phi, theta)).real
        for m in range(1, l + 1):
            harmonic = prefactor * sph_harm(m, l, phi, theta)
            out[:, component_index(l, m)] = ((-1) ** m) * np.sqrt(2) * harmonic.real
            out[:, component_index(l, -m)] = ((-1) ** m) * np.sqrt(2) * harmonic.imag
    return out


def _random_rotation(rng):
    matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return matrix * np.sign(np.linalg.det(matrix))


def build_system(separations=DEFAULT_SEPARATIONS, l_max=5, seed=7):
    """Return everything about the synthetic system, geometry by geometry.

    The internal charge clouds are rigid, so an atom's moments are the same at
    every geometry and only the separation changes — which keeps the exact
    reference smooth and the test focused on the gates.
    """
    rng = np.random.default_rng(seed)

    atoms = []
    for f_i, fragment in enumerate((FRAGMENT_A, FRAGMENT_B)):
        for symbol, offset in fragment:
            spec = ATOM_TYPES[symbol]
            weights = rng.uniform(0.5, 1.5, size=4)
            charges = spec['q'] * weights / weights.sum()
            cloud = _random_rotation(rng) @ (_TETRAHEDRON * spec['rho']).T
            atoms.append({
                'symbol': symbol, 'Z': spec['Z'], 'rho': spec['rho'],
                'fragment': f_i, 'offset': np.array(offset),
                'charges': charges, 'cloud': cloud.T,      # (4, 3) Angstrom
                'net_charge': float(charges.sum()),
            })

    for atom in atoms:
        atom['moments'] = (atom['charges'][:, None] *
                           solid_harmonics(atom['cloud'] * BOHR_PER_ANGSTROM, l_max)).sum(axis=0)

    n_atoms = len(atoms)
    pairs = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)]

    points = []
    for separation in separations:
        coords = np.array([atom['offset'] + (np.array([separation, 0.0, 0.0])
                                             if atom['fragment'] == 1 else 0.0)
                           for atom in atoms])
        exact = {}
        for (i, j) in pairs:
            pos_i = (coords[i] + atoms[i]['cloud']) * BOHR_PER_ANGSTROM
            pos_j = (coords[j] + atoms[j]['cloud']) * BOHR_PER_ANGSTROM
            distance = np.linalg.norm(pos_i[:, None, :] - pos_j[None, :, :], axis=-1)
            exact[(i, j)] = float(np.sum(np.outer(atoms[i]['charges'], atoms[j]['charges'])
                                         / distance))
        points.append({'separation': separation, 'coords': coords, 'exact': exact})

    # A stand-in total energy that genuinely tracks the system: the sum of every
    # interfragment pair interaction, offset to look like an electronic energy.
    for point in points:
        inter = sum(v for (i, j), v in point['exact'].items()
                    if atoms[i]['fragment'] != atoms[j]['fragment'])
        point['wfn_energy'] = -152.9 + inter

    labels = [atom['symbol'].lower() + str(i + 1) for i, atom in enumerate(atoms)]
    return {'atoms': atoms, 'labels': labels, 'pairs': pairs, 'points': points,
            'l_max': l_max,
            'fragments': {'A': [i + 1 for i, a in enumerate(atoms) if a['fragment'] == 0],
                          'B': [i + 1 for i, a in enumerate(atoms) if a['fragment'] == 1]}}


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def _write_wfn(path, system, point):
    lines = ['SYNTHETIC',
             'GAUSSIAN              6 MOL ORBITALS     60 PRIMITIVES        '
             + str(len(system['atoms'])) + ' NUCLEI   B3LYP']
    for i, atom in enumerate(system['atoms']):
        x, y, z = point['coords'][i] * BOHR_PER_ANGSTROM
        lines.append('{s:<2s}{n:>7d}    (CENTRE{n:>3d}) {x:>12.8f}{y:>12.8f}{z:>12.8f}  '
                     'CHARGE = {z_:.1f}'.format(s=atom['symbol'], n=i + 1, x=x, y=y, z=z,
                                                z_=atom['Z']))
    lines.append('END DATA')
    lines.append(' TOTAL ENERGY = {e:>22.12f} THE VIRIAL(-V/T)=   2.00000000'
                 .format(e=point['wfn_energy']))
    open(path, 'w').write('\n'.join(lines) + '\n')


def _write_gaussian_output(path, system, point):
    elements = {'H': 1, 'O': 8}
    lines = [' Some synthetic single point output', '',
             '                         Standard orientation:',
             ' ---------------------------------------------------------------------',
             ' Center     Atomic      Atomic             Coordinates (Angstroms)',
             ' Number     Number       Type             X           Y           Z',
             ' ---------------------------------------------------------------------']
    for i, atom in enumerate(system['atoms']):
        x, y, z = point['coords'][i]
        lines.append('{c:>7d}{z:>11d}{t:>12d}{x:>16.6f}{y:>12.6f}{zc:>12.6f}'
                     .format(c=i + 1, z=elements[atom['symbol']], t=0, x=x, y=y, zc=z))
    lines.append(' ---------------------------------------------------------------------')
    lines.append(' SCF Done:  E(RB3LYP) = {e:.9f}'.format(e=point['wfn_energy']))
    open(path, 'w').write('\n'.join(lines) + '\n')


def _write_sum(path, system, point):
    labels = system['labels']
    lines = [' Synthetic AIMAll summary', '',
             'Atom A     E_IQA_Intra(A)          T(A)          q(A)          L(A)',
             '-------------------------------------------------------------------']
    for i, atom in enumerate(system['atoms']):
        lines.append('{a:<10s}{e:>20.10E}{t:>20.10E}{q:>20.10E}{l:>20.10E}'.format(
            a=labels[i].upper(), e=-0.5 * atom['Z'], t=0.5 * atom['Z'],
            q=atom['net_charge'], l=1.0e-6))
    lines.append('')
    lines.append('Atom A    Atom B     E_IQA_Inter(A,B)/2     Vne(A,B)/2     Ven(A,B)/2'
                 '     Vee(A,B)/2     Vnn(A,B)/2    VeeC(A,B)/2    VeeX(A,B)/2')
    lines.append('-' * 130)
    for (i, j) in system['pairs']:
        exchange = -1.0e-3                       # VeeX(A,B)/2
        half_inter = 0.5 * point['exact'][(i, j)] + exchange
        columns = [half_inter, 0.0, 0.0, 0.0, 0.0, half_inter - exchange, exchange]
        lines.append('{a:<10s}{b:<10s}'.format(a=labels[i].upper(), b=labels[j].upper())
                     + ''.join('{v:>20.10E}'.format(v=v) for v in columns))
    lines.append('')
    open(path, 'w').write('\n'.join(lines) + '\n')


def _write_int(path, system, atom_index):
    atom = system['atoms'][atom_index]
    l_max = system['l_max']
    lines = [' Synthetic AIMAll atomic integration',
             ' The radius of the Beta sphere is  {r:.10E}'
             .format(r=0.7 * atom['rho'] * BOHR_PER_ANGSTROM),
             '',
             ' Results of the basin integration:',
             '          N =  {n:.10E}   q = {q:.10E} = Net Charge'
             .format(n=atom['Z'] - atom['net_charge'], q=atom['net_charge']),
             '          L =  1.0000000000E-06',
             '',
             ' Real Spherical Harmonic Moments Q[l,|m|,?] of the Electronic Charge Density '
             'Distribution (Nuclear Origin):',
             '   Condon-Shortly phase, (-1)**|m|, included',
             '   Normalization factor, SqRt((2*l+1)/(4*pi)), not included',
             ' ' + '-' * 100]
    for l in range(l_max + 1):
        # As AIMAll does it: Q[0,0] is the electronic population alone, and only
        # the "Net Charge" line above carries the total.  Whether a reader picks
        # the right one up is part of what this fixture tests.
        value = (atom['net_charge'] - atom['Z'] if l == 0
                 else atom['moments'][component_index(l, 0)])
        lines.append(' Q[{l},0]         = {v:>19.10E}'.format(l=l, v=value))
        for m in range(1, l + 1):
            lines.append(' Q[{l},{m},c]       = {v:>19.10E}'
                         .format(l=l, m=m, v=atom['moments'][component_index(l, m)]))
            lines.append(' Q[{l},{m},s]       = {v:>19.10E}'
                         .format(l=l, m=m, v=atom['moments'][component_index(l, -m)]))
    lines.append('')
    lines.append('Molecular Orbital (MO) Data:')
    lines.append('---------------------------')
    open(path, 'w').write('\n'.join(lines) + '\n')


def write_dataset(target_dir, system=None, write_config=True):
    """Write the whole synthetic dataset under *target_dir*; returns the system."""
    system = system or build_system()
    os.makedirs(target_dir, exist_ok=True)

    for p_i, point in enumerate(system['points']):
        folder = os.path.join(target_dir, str(p_i + 1))
        atomic = os.path.join(folder, 'synth_atomicfiles')
        os.makedirs(atomic, exist_ok=True)
        _write_wfn(os.path.join(folder, 'synth.wfn'), system, point)
        _write_gaussian_output(os.path.join(folder, 'synth.out'), system, point)
        _write_sum(os.path.join(folder, 'synth.sum'), system, point)
        for a_i, label in enumerate(system['labels']):
            _write_int(os.path.join(atomic, label + '.int'), system, a_i)

    if write_config:
        with open(os.path.join(target_dir, 'auto_reg.config'), 'w') as handle:
            handle.write('FRAG ID 1 <Acceptor>\nFRAG ATOMS [{a}]\n\n'
                         'FRAG ID 2 <Donor>\nFRAG ATOMS [{b}]\n'
                         .format(a=','.join(str(n) for n in system['fragments']['A']),
                                 b=','.join(str(n) for n in system['fragments']['B'])))
    return system


if __name__ == '__main__':
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else 'synthetic_reg_points'
    built = write_dataset(target)
    print('wrote {n} points to {t}'.format(n=len(built['points']), t=target))
    for point in built['points']:
        print('  d(O-O) = {d:.2f} A   R(O1-H5) = {r:.2f} A   E = {e:.6f}'.format(
            d=point['separation'],
            r=float(np.linalg.norm(point['coords'][0] - point['coords'][4])),
            e=point['wfn_energy']))
