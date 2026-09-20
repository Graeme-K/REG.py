"""
multipole_utils.py

Atomic multipole moments, the multipole interaction tensor, and the admission
gates that decide which atom pairs a rank-resolved electrostatic analysis may
legitimately describe.

The physics.  The classical electrostatic interaction between two QTAIM basins
A and B can be written as a sum over ranks of their nucleus-centred spherical
multipole moments,

    V_cl(A,B) = sum_{la,ka} sum_{lb,kb} Q[la,ka](A) T[la,ka;lb,kb](R) Q[lb,kb](B)

which is what turns one number, V_cl(A,B), into a chemically readable series:
charge-charge, charge-dipole, dipole-dipole and so on.  The series is asymptotic,
not convergent: it is exact only while the two basins' convergence spheres do not
overlap, and for close pairs it can look settled at low rank and then diverge.
Deciding which pairs may be described this way is therefore part of the method,
not a detail — see :func:`admit_pairs`.

Moments are read from AIMAll .int files ("Real Spherical Harmonic Moments",
nuclear origin, Condon-Shortley phase included, normalisation factor excluded).
The l=0 moment is replaced by the atom's *net* charge, since the printed Q[0,0]
is the electronic population alone and the nuclear charge contributes only at
l=0 for a nucleus-centred expansion.

Conventions
-----------
* Real moments are indexed by a signed k: k = +|m| is the cosine component
  (Q[l,|m|,c]) and k = -|m| the sine component (Q[l,|m|,s]), matching the order
  AIMAll prints them in.  k = +1, -1, 0 are therefore x, y, z respectively.
* Distances enter in bohr; moments are in atomic units, so energies come out in
  hartree.
* The interaction tensor recursion follows Popelier's implementation of Stone's
  formulation and is vectorised here: every tensor element is an array over a
  batch of (pair, geometry) instances rather than a float, which is what makes a
  whole path affordable to compute.

coded for the REG.py package
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BOHR_PER_ANGSTROM = 1.8897261254578281
HA_TO_KJ = 2625.5

# Covalent radii in Angstrom (Cordero et al., Dalton Trans., 2008, 2832).
# Used only by the topology pre-filter, which is a cheap rejection of pairs the
# numerical test would reject anyway — so a missing element falls back to a
# generous default rather than failing.
_COVALENT_RADII = {
    'h': 0.31, 'he': 0.28, 'li': 1.28, 'be': 0.96, 'b': 0.84, 'c': 0.76,
    'n': 0.71, 'o': 0.66, 'f': 0.57, 'ne': 0.58, 'na': 1.66, 'mg': 1.41,
    'al': 1.21, 'si': 1.11, 'p': 1.07, 's': 1.05, 'cl': 1.02, 'ar': 1.06,
    'k': 2.03, 'ca': 1.76, 'sc': 1.70, 'ti': 1.60, 'v': 1.53, 'cr': 1.39,
    'mn': 1.39, 'fe': 1.32, 'co': 1.26, 'ni': 1.24, 'cu': 1.32, 'zn': 1.22,
    'ga': 1.22, 'ge': 1.20, 'as': 1.19, 'se': 1.20, 'br': 1.20, 'kr': 1.16,
    'rb': 2.20, 'sr': 1.95, 'y': 1.90, 'zr': 1.75, 'nb': 1.64, 'mo': 1.54,
    'tc': 1.47, 'ru': 1.46, 'rh': 1.42, 'pd': 1.39, 'ag': 1.45, 'cd': 1.44,
    'in': 1.42, 'sn': 1.39, 'sb': 1.39, 'te': 1.38, 'i': 1.39, 'xe': 1.40,
}
_DEFAULT_COVALENT_RADIUS = 1.50

# Van der Waals radii in Angstrom (Bondi 1964, with Truhlar's later additions).
# A free atom's 0.001 au isodensity envelope sits close to its vdW radius, so
# these stand in for the basin's outer radial extent when nothing better is
# available.  See basin_radii() for why that matters.
_VDW_RADII = {
    'h': 1.10, 'he': 1.40, 'li': 1.81, 'be': 1.53, 'b': 1.92, 'c': 1.70,
    'n': 1.55, 'o': 1.52, 'f': 1.47, 'ne': 1.54, 'na': 2.27, 'mg': 1.73,
    'al': 1.84, 'si': 2.10, 'p': 1.80, 's': 1.80, 'cl': 1.75, 'ar': 1.88,
    'k': 2.75, 'ca': 2.31, 'fe': 2.00, 'co': 2.00, 'ni': 1.63, 'cu': 1.40,
    'zn': 1.39, 'ga': 1.87, 'ge': 2.11, 'as': 1.85, 'se': 1.90, 'br': 1.85,
    'kr': 2.02, 'i': 1.98, 'xe': 2.16,
}
_DEFAULT_VDW_RADIUS = 2.00


def element_of(atom_label):
    """'c26' -> 'c'.  Atom labels are element symbol + index throughout REG.py."""
    return re.sub(r'\d+$', '', str(atom_label)).lower()


# ---------------------------------------------------------------------------
# Component indexing
# ---------------------------------------------------------------------------

def n_components(l_max):
    """Number of real spherical components up to and including rank *l_max*."""
    return (l_max + 1) ** 2


def component_index(l, k):
    """Flat index of the real component (l, k) in AIMAll's printing order.

    AIMAll prints Q[l,0], Q[l,1,c], Q[l,1,s], Q[l,2,c], Q[l,2,s], ... so within
    the rank-l block the cosine component of |m| sits at 2|m|-1 and the sine
    component at 2|m|.
    """
    if k == 0:
        return l * l
    return l * l + (2 * abs(k) - 1 if k > 0 else 2 * abs(k))


def component_labels(l_max):
    """Human-readable labels for every component up to *l_max*, in flat order."""
    labels = [None] * n_components(l_max)
    for l in range(l_max + 1):
        labels[component_index(l, 0)] = 'Q[{l},0]'.format(l=l)
        for m in range(1, l + 1):
            labels[component_index(l, m)] = 'Q[{l},{m},c]'.format(l=l, m=m)
            labels[component_index(l, -m)] = 'Q[{l},{m},s]'.format(l=l, m=m)
    return labels


# ---------------------------------------------------------------------------
# Reading moments out of AIMAll .int files
# ---------------------------------------------------------------------------

_MOMENT_RE = re.compile(r'^\s*Q\[(\d+)(?:,(\d+))?(?:,([cs]))?\]\s*=\s*(-?\d+\.\d+[EeDd][+-]?\d+)')
_NET_CHARGE_RE = re.compile(r'q\s*=\s*(-?\d+\.\d+[EeDd][+-]?\d+)\s*=\s*Net Charge')
_BETA_RE = re.compile(r'radius of the Beta sphere is\s+(-?\d+\.\d+[EeDd][+-]?\d+)')

_MOMENT_HEADER = 'Real Spherical Harmonic Moments'
_MOMENT_END = 'Molecular Orbital (MO) Data'


def read_atomic_multipoles(int_path, l_max):
    """Read one atom's spherical multipole moments from its .int file.

    The file is streamed and abandoned as soon as the moment section closes.
    The sections that follow — molecular orbital data and the atomic overlap
    matrix — are by far the largest part of a .int file and are never needed
    here, so not reading them is what keeps a whole path affordable over NFS.

    Returns a dict with:
        moments        : ndarray, length n_components(l_max); NaN where the file
                         does not go that high
        l_available    : highest rank the file actually contains
        net_charge     : q(A), which replaces the printed Q[0,0]
        beta_radius    : beta-sphere radius in bohr, or None
    or None when the file cannot be read.
    """
    values = {}
    net_charge = None
    beta_radius = None
    l_available = -1

    try:
        with open(int_path, 'r', errors='ignore') as handle:
            in_moments = False
            for line in handle:
                if not in_moments:
                    if beta_radius is None:
                        beta_match = _BETA_RE.search(line)
                        if beta_match:
                            beta_radius = float(beta_match.group(1).replace('D', 'E'))
                            continue
                    if net_charge is None:
                        charge_match = _NET_CHARGE_RE.search(line)
                        if charge_match:
                            net_charge = float(charge_match.group(1).replace('D', 'E'))
                            continue
                    if _MOMENT_HEADER in line:
                        in_moments = True
                    continue

                if _MOMENT_END in line:
                    break
                moment_match = _MOMENT_RE.match(line)
                if moment_match:
                    l = int(moment_match.group(1))
                    m = int(moment_match.group(2) or 0)
                    sign = moment_match.group(3)
                    k = 0 if m == 0 else (-m if sign == 's' else m)
                    values[(l, k)] = float(moment_match.group(4).replace('D', 'E'))
                    l_available = max(l_available, l)
    except OSError:
        return None

    if not values:
        return None

    moments = np.full(n_components(l_max), np.nan)
    for (l, k), value in values.items():
        if l <= l_max:
            moments[component_index(l, k)] = value

    # The printed Q[0,0] is the electronic population (negative).  Only rank 0
    # carries a nuclear contribution for a nucleus-centred expansion, so this is
    # the one component that has to be swapped for the total.
    if net_charge is not None:
        moments[0] = net_charge

    return {
        'moments': moments,
        'l_available': l_available,
        'net_charge': net_charge,
        'beta_radius': beta_radius,
    }


def get_atomic_multipoles(atomic_files, atoms, l_max, max_workers=16):
    """Read multipole moments for every atom at every geometry point.

    Parameters
    ----------
    atomic_files : list of _atomicfiles folder paths, one per geometry point.
    atoms        : atom labels, e.g. ['o1', 'h2', ...].
    l_max        : highest rank to keep.

    Returns
    -------
    moments      : ndarray (n_points, n_atoms, n_components)
    beta_radii   : ndarray (n_points, n_atoms), bohr; NaN where unavailable
    info         : dict with 'l_available' (lowest rank ceiling seen anywhere),
                   'missing' (unreadable .int paths) and 'net_charges'
    """
    n_points = len(atomic_files)
    n_atoms = len(atoms)
    moments = np.full((n_points, n_atoms, n_components(l_max)), np.nan)
    beta_radii = np.full((n_points, n_atoms), np.nan)
    charges = np.full((n_points, n_atoms), np.nan)
    missing = []
    l_seen = []

    jobs = [(p_i, a_i, os.path.join(atomic_files[p_i], str(atoms[a_i]).lower() + '.int'))
            for p_i in range(n_points) for a_i in range(n_atoms)]

    def _work(job):
        p_i, a_i, path = job
        return p_i, a_i, path, read_atomic_multipoles(path, l_max)

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(jobs)))) as executor:
        for p_i, a_i, path, result in executor.map(_work, jobs):
            if result is None:
                missing.append(path)
                continue
            moments[p_i, a_i] = result['moments']
            l_seen.append(result['l_available'])
            if result['beta_radius'] is not None:
                beta_radii[p_i, a_i] = result['beta_radius']
            if result['net_charge'] is not None:
                charges[p_i, a_i] = result['net_charge']

    return moments, beta_radii, {
        'l_available': min(l_seen) if l_seen else -1,
        'missing': missing,
        'net_charges': charges,
    }


# ---------------------------------------------------------------------------
# The interaction tensor
# ---------------------------------------------------------------------------

def norm_factor(l, k):
    """Stone's normalisation N(l,k) = prod_i sqrt(l(l+1) - i(i-1)), i = 1..|k|."""
    factor = 1.0
    for i in range(1, abs(k) + 1):
        factor *= (l * (l + 1) - i * (i - 1)) ** 0.5
    return factor


def _eta_m(mu, k):
    """Coefficient M and shifted index eta for Cartesian direction *mu*.

    Encodes how the real solid harmonic of rank l+1 and order k is built from
    the rank-l harmonics multiplied by the x (mu=0), y (mu=1) or z (mu=2)
    component of the unit vector.  M = 0 marks a term that does not contribute.
    """
    if mu == 0:      # x
        if k == 0 or k == -1:
            return 0.0, 0
        if k == 1:
            return 2.0 ** 0.5, 0
        if k >= 2:
            return float(k), k - 1
        return float(abs(k)), 1 - abs(k)
    if mu == 1:      # y
        if k == 0 or k == 1:
            return 0.0, 0
        if k == -1:
            return 2.0 ** 0.5, 0
        if k >= 2:
            return -float(k), 1 - k
        return float(abs(k)), abs(k) - 1
    return 1.0, k    # z


def interaction_tensor(coords_a, coords_b, l_a_max, l_b_max):
    """Multipole interaction tensor for a batch of A-B pairs.

    Parameters
    ----------
    coords_a, coords_b : ndarray (batch, 3), nuclear positions in **bohr**.
    l_a_max, l_b_max   : highest rank on each site.

    Returns
    -------
    dict {(la, lb): ndarray (batch, 2*la+1, 2*lb+1)} indexed [., ka+la, kb+lb].

    The recursion is Stone's, as implemented in Popelier's multipole codes, with
    every element carried as an array over the batch.  Local axes are the global
    frame, so the A->B unit vector is the only geometry the tensor needs.
    """
    coords_a = np.asarray(coords_a, dtype=float)
    coords_b = np.asarray(coords_b, dtype=float)
    batch = coords_a.shape[0]

    r_vector = coords_b - coords_a
    distance = np.sqrt(np.sum(r_vector ** 2, axis=1))
    unit = r_vector / distance[:, None]

    # Local axes are the global Cartesian frame: rA points A->B, rB points B->A,
    # and the axis-overlap matrix cAB is the identity.
    r_a = unit
    r_b = -unit
    c_ab = np.zeros((batch, 3, 3))
    c_ab[:, 0, 0] = c_ab[:, 1, 1] = c_ab[:, 2, 2] = 1.0

    zero = np.zeros(batch)
    t = {}          # (la, lb, ka, kb) -> ndarray(batch)

    def have(la, lb, ka, kb):
        return (la, lb, ka, kb) in t

    t[(0, 0, 0, 0)] = np.ones(batch)

    if l_a_max == 0 and l_b_max == 0:
        return {(0, 0): (1.0 / distance).reshape(batch, 1, 1)}

    # ---- column lb = 0 ----------------------------------------------------
    if l_a_max > 0:
        t[(1, 0, 0, 0)] = r_a[:, 2]
        t[(1, 0, -1, 0)] = r_a[:, 1]
        t[(1, 0, 1, 0)] = r_a[:, 0]

        for la in range(0, 2):
            for ka in range(-1, 2):
                if abs(ka) > la:
                    continue
                t[(la, 0, ka, 0)] = t[(la, 0, ka, 0)] * norm_factor(la, ka)

        for la in range(1, l_a_max):
            if have(la + 1, 0, 0, 0):
                continue
            t_sum = zero.copy()
            for xyz in range(3):
                m_coef, eta = _eta_m(xyz, 0)
                if abs(m_coef) < 1e-10 or abs(eta) > la:
                    continue
                t_sum = t_sum + m_coef * r_a[:, xyz] * t[(la, 0, eta, 0)]
            t[(la + 1, 0, 0, 0)] = ((2 * la + 1) * t_sum - la * t[(la - 1, 0, 0, 0)]) / (la + 1)

        for ka in range(-1, -(l_a_max + 2), -1):
            for la in range(1, l_a_max):
                if -ka > la + 1 or have(la + 1, 0, ka, 0) or have(la + 1, 0, -ka, 0):
                    continue
                for signed_k in (ka, -ka):
                    t_sum = zero.copy()
                    for xyz in range(3):
                        m_coef, eta = _eta_m(xyz, signed_k)
                        if abs(m_coef) < 1e-10 or abs(eta) > la:
                            continue
                        t_sum = t_sum + m_coef * r_a[:, xyz] * t[(la, 0, eta, 0)]
                    if -ka > la - 1:
                        t[(la + 1, 0, signed_k, 0)] = ((2 * la + 1) * t_sum) / (la + 1)
                    else:
                        t[(la + 1, 0, signed_k, 0)] = (((2 * la + 1) * t_sum
                                                        - la * t[(la - 1, 0, signed_k, 0)]) / (la + 1))

    # ---- row la = 0 -------------------------------------------------------
    if l_b_max > 0:
        t[(0, 1, 0, 0)] = r_b[:, 2]
        t[(0, 1, 0, -1)] = r_b[:, 1]
        t[(0, 1, 0, 1)] = r_b[:, 0]

        for lb in range(0, 2):
            for kb in range(-1, 2):
                if abs(kb) > lb:
                    continue
                t[(0, lb, 0, kb)] = t[(0, lb, 0, kb)] * norm_factor(lb, kb)

        for lb in range(1, l_b_max):
            if have(0, lb + 1, 0, 0):
                continue
            t_sum = zero.copy()
            for xyz in range(3):
                m_coef, eta = _eta_m(xyz, 0)
                if abs(m_coef) < 1e-10 or abs(eta) > lb:
                    continue
                t_sum = t_sum + m_coef * r_b[:, xyz] * t[(0, lb, 0, eta)]
            t[(0, lb + 1, 0, 0)] = ((2 * lb + 1) * t_sum - lb * t[(0, lb - 1, 0, 0)]) / (lb + 1)

        for kb in range(-1, -(l_b_max + 2), -1):
            for lb in range(1, l_b_max):
                if -kb > lb + 1 or have(0, lb + 1, 0, kb) or have(0, lb + 1, 0, -kb):
                    continue
                for signed_k in (kb, -kb):
                    t_sum = zero.copy()
                    for xyz in range(3):
                        m_coef, eta = _eta_m(xyz, signed_k)
                        if abs(m_coef) < 1e-10 or abs(eta) > lb:
                            continue
                        t_sum = t_sum + m_coef * r_b[:, xyz] * t[(0, lb, 0, eta)]
                    if -kb > lb - 1:
                        t[(0, lb + 1, 0, signed_k)] = ((2 * lb + 1) * t_sum) / (lb + 1)
                    else:
                        t[(0, lb + 1, 0, signed_k)] = (((2 * lb + 1) * t_sum
                                                        - lb * t[(0, lb - 1, 0, signed_k)]) / (lb + 1))

    # ---- seed la = lb = 1 --------------------------------------------------
    if l_a_max > 0 and l_b_max > 0:
        axis_of = {0: 2, -1: 1, 1: 0}   # k -> Cartesian component
        for ka in (-1, 0, 1):
            for kb in (-1, 0, 1):
                i, j = axis_of[ka], axis_of[kb]
                t[(1, 1, ka, kb)] = (3.0 * r_a[:, i] * r_b[:, j] + c_ab[:, i, j]) \
                    * (norm_factor(1, ka) * norm_factor(1, kb))

        # ---- general recursion in lb ---------------------------------------
        for ka_seed in range(0, -(l_a_max + 1), -1):
            for la in range(0, l_a_max + 1):
                if -ka_seed > la:
                    continue
                for ka in ((ka_seed,) if ka_seed == 0 else (ka_seed, -ka_seed)):
                    for kb_seed in range(0, -(l_b_max + 2), -1):
                        for lb in range(0, l_b_max):
                            if -kb_seed > lb + 1:
                                continue
                            for kb in ((kb_seed,) if kb_seed == 0 else (kb_seed, -kb_seed)):
                                if have(la, lb + 1, ka, kb):
                                    continue
                                if la < 2 or -ka_seed > la - 2 or -kb_seed > lb + 1:
                                    t_sum = zero.copy()
                                else:
                                    t_sum = t[(la - 2, lb + 1, ka, kb)].copy()

                                if not (-ka_seed > la or lb == 0 or -kb_seed > lb - 1):
                                    t_sum = t_sum - ((2 * la + lb) / (lb + 1)) * t[(la, lb - 1, ka, kb)]

                                t_sum1 = zero.copy()
                                for xyz_i in range(3):
                                    m_coef, eta = _eta_m(xyz_i, kb)
                                    if -ka_seed > la or abs(eta) > lb:
                                        continue
                                    t_sum1 = t_sum1 + m_coef * r_b[:, xyz_i] * t[(la, lb, ka, eta)]
                                t_sum = t_sum + t_sum1 * ((2 * la + 2 * lb + 1) / (lb + 1))

                                t_sum1 = zero.copy()
                                for xyz_i in range(3):
                                    m_coef, eta = _eta_m(xyz_i, ka)
                                    if abs(eta) > la - 1:
                                        continue
                                    for xyz_j in range(3):
                                        m_coef1, eta1 = _eta_m(xyz_j, kb)
                                        if abs(eta1) > lb:
                                            continue
                                        t_sum1 = t_sum1 + (m_coef * m_coef1 * c_ab[:, xyz_i, xyz_j]
                                                           * t[(la - 1, lb, eta, eta1)])

                                t[(la, lb + 1, ka, kb)] = t_sum + t_sum1 * ((2 * la - 1) / (lb + 1))

    # ---- strip normalisation and apply the R^-(la+lb+1) factor -------------
    blocks = {}
    for la in range(l_a_max + 1):
        for lb in range(l_b_max + 1):
            block = np.empty((batch, 2 * la + 1, 2 * lb + 1))
            power = distance ** (-(la + lb + 1))
            for ka in range(-la, la + 1):
                n_a = norm_factor(la, ka)
                for kb in range(-lb, lb + 1):
                    key = (la, lb, ka, kb)
                    if key not in t:
                        raise RuntimeError(
                            'interaction tensor element {} was never computed — '
                            'the recursion did not cover the requested ranks'.format(key))
                    block[:, ka + la, kb + lb] = t[key] * power / (n_a * norm_factor(lb, kb))
            blocks[(la, lb)] = block
    return blocks


def chunk_for_budget(l_max, memory_budget_mb=200.0, minimum=256):
    """Batch size whose interaction tensor fits in *memory_budget_mb*.

    The tensor holds (l_max+1)**4 arrays of the batch length, so the memory is
    8 * (l_max+1)**4 * chunk bytes: 10.4 kB per batch element at l_max = 5.
    Throughput rises with chunk size until the per-element Python overhead is
    amortised — measured on this code, 23k pair-geometries/s at chunk 2048
    against 34k at 16384, and flat above that — so the default budget sits just
    past that knee.
    """
    per_element = 8.0 * (l_max + 1) ** 4
    return max(minimum, int(memory_budget_mb * 1024 * 1024 / per_element))


def pair_rank_energies(moments_a, moments_b, coords_a, coords_b, l_max,
                       chunk=None, memory_budget_mb=200.0):
    """Rank-resolved electrostatic energies for a batch of A-B pairs.

    Parameters
    ----------
    moments_a, moments_b : ndarray (batch, n_components(l_max))
    coords_a, coords_b   : ndarray (batch, 3) in **Angstrom**
    l_max                : highest rank on each site
    chunk                : batch elements per tensor build; None sizes it from
                           *memory_budget_mb*

    Returns
    -------
    ndarray (batch, l_max+1, l_max+1) — energy in hartree summed over k for each
    (la, lb) rank pair, so each entry is rotation-invariant.  Individual k
    components are frame-dependent and are deliberately not the unit of output.

    Performance.  The whole batch of (pair, geometry) instances goes through the
    recursion at once, every tensor element being an array rather than a float.
    Which elements exist and which branches are taken depend only on the (l, k)
    indices, never on the values, so one pass of Python control flow serves the
    entire batch.  That is the difference between this and a scalar
    implementation: about 32 tensor builds/s one at a time against 34,000
    pair-geometries/s batched, at l_max = 5.  A 133-atom path over 20 geometries
    (175,560 pair-geometries) is then ~6 s rather than ~1.5 h.
    """
    moments_a = np.asarray(moments_a, dtype=float)
    moments_b = np.asarray(moments_b, dtype=float)
    coords_a = np.asarray(coords_a, dtype=float) * BOHR_PER_ANGSTROM
    coords_b = np.asarray(coords_b, dtype=float) * BOHR_PER_ANGSTROM

    batch = moments_a.shape[0]
    out = np.zeros((batch, l_max + 1, l_max + 1))
    chunk = chunk or chunk_for_budget(l_max, memory_budget_mb)

    for start in range(0, batch, chunk):
        stop = min(start + chunk, batch)
        blocks = interaction_tensor(coords_a[start:stop], coords_b[start:stop], l_max, l_max)
        for la in range(l_max + 1):
            idx_a = [component_index(la, ka) for ka in range(-la, la + 1)]
            q_a = moments_a[start:stop][:, idx_a]
            for lb in range(l_max + 1):
                idx_b = [component_index(lb, kb) for kb in range(-lb, lb + 1)]
                q_b = moments_b[start:stop][:, idx_b]
                # sum_{ka,kb} Q_a[ka] T[ka,kb] Q_b[kb]
                out[start:stop, la, lb] = np.einsum('ni,nij,nj->n', q_a, blocks[(la, lb)], q_b)
    return out


def partial_sums_by_total_rank(rank_energies):
    """Cumulative partial sums of the series, ordered by l_tot = la + lb.

    Parameters
    ----------
    rank_energies : ndarray (..., l_max+1, l_max+1)

    Returns
    -------
    increments : ndarray (..., 2*l_max+1) — the contribution of each l_tot shell
    partial    : ndarray (..., 2*l_max+1) — sum over all shells up to that l_tot
    """
    rank_energies = np.asarray(rank_energies, dtype=float)
    l_max = rank_energies.shape[-1] - 1
    shape = rank_energies.shape[:-2]
    increments = np.zeros(shape + (2 * l_max + 1,))
    for la in range(l_max + 1):
        for lb in range(l_max + 1):
            increments[..., la + lb] += rank_energies[..., la, lb]
    return increments, np.cumsum(increments, axis=-1)


# ---------------------------------------------------------------------------
# Gate 1: topology
# ---------------------------------------------------------------------------

def bond_graph(atoms, coordinates, tolerance=1.2):
    """Covalent bond graph, taken as the union over every geometry on the path.

    A pair counts as bonded when its distance falls below *tolerance* times the
    sum of covalent radii at **any** geometry.  Taking the union rather than a
    per-geometry graph is what keeps the admitted set fixed along the path: a
    bond that forms halfway would otherwise reclassify a pair mid-analysis and
    put a step in the term it produces.

    Covalent radii — not AIM bond critical points — define "bonded" here on
    purpose.  A hydrogen bond has a BCP but its donor and acceptor basins are
    two separate convergence spheres, and those pairs are the ones a fragment
    analysis exists to describe; letting AIM connectivity reject them would
    throw away the signal.

    Parameters
    ----------
    atoms       : atom labels
    coordinates : ndarray (n_points, n_atoms, 3) in Angstrom

    Returns
    -------
    set of frozenset({i, j}) index pairs
    """
    coordinates = np.asarray(coordinates, dtype=float)
    radii = np.array([_COVALENT_RADII.get(element_of(a), _DEFAULT_COVALENT_RADIUS) for a in atoms])
    n_atoms = len(atoms)

    # Minimum separation over the path, so a bond present at any point counts.
    diff = coordinates[:, :, None, :] - coordinates[:, None, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1)).min(axis=0)
    cutoff = tolerance * (radii[:, None] + radii[None, :])

    bonds = set()
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if distances[i, j] < cutoff[i, j]:
                bonds.add(frozenset((i, j)))
    return bonds


def topology_separation(n_atoms, bonds, max_depth=3):
    """Bond-path separation for every atom pair, capped at *max_depth*.

    Returns {(i, j): n} where n = 2 for a 1,2 pair (bonded), 3 for 1,3 and so on;
    pairs further apart than *max_depth* bonds, or in different connected
    components, are reported as None (meaning "at least 1,4 — admissible").
    """
    adjacency = {i: set() for i in range(n_atoms)}
    for bond in bonds:
        i, j = tuple(bond)
        adjacency[i].add(j)
        adjacency[j].add(i)

    separation = {}
    for start in range(n_atoms):
        frontier = {start}
        seen = {start}
        for depth in range(1, max_depth):
            frontier = {n for node in frontier for n in adjacency[node]} - seen
            if not frontier:
                break
            seen |= frontier
            for other in frontier:
                if other > start:
                    separation[(start, other)] = depth + 1
                elif start > other:
                    separation.setdefault((other, start), depth + 1)
    return separation


# ---------------------------------------------------------------------------
# Gate 2: geometry
# ---------------------------------------------------------------------------

def basin_radii(atoms, beta_radii, source='beta'):
    """Per-atom radial extent R_A used by the geometric gate, in Angstrom.

    Two sources, and the difference between them matters:

    'beta'  the beta-sphere radius AIMAll reports in each .int file.  The beta
            sphere is inscribed *inside* the basin, so R_A is a strict lower
            bound and R_AB >= R_A + R_B is then a **necessary but not
            sufficient** condition — it will pass pairs whose convergence
            spheres genuinely overlap.
    'vdw'   tabulated van der Waals radii, which approximate the 0.001 au
            envelope of a free atom and so stand in for the basin's outer
            extent.  Stricter, and closer to the condition actually wanted, but
            an estimate rather than a measurement of this molecule's basins.

    Either way gate 2 is only a cheap pre-filter; the numerical convergence test
    is what decides.

    Parameters
    ----------
    beta_radii : ndarray (n_points, n_atoms) in bohr, NaN where unavailable

    Returns
    -------
    ndarray (n_points, n_atoms) in Angstrom, plus a bool saying whether the
    values are a lower bound (True) or an outer estimate (False).
    """
    beta_radii = np.asarray(beta_radii, dtype=float)
    if source == 'vdw':
        radii = np.array([_VDW_RADII.get(element_of(a), _DEFAULT_VDW_RADIUS) for a in atoms])
        return np.tile(radii, (beta_radii.shape[0], 1)), False
    return beta_radii / BOHR_PER_ANGSTROM, True


# ---------------------------------------------------------------------------
# Gates 3 and 4: numerical convergence, applied path-wide
# ---------------------------------------------------------------------------

def convergence_verdict(partial, increments, exact, segments=None,
                        residual_tolerance=0.05, increment_ranks=None,
                        absolute_floor_kj=0.05):
    """Decide whether a pair's multipole series may be trusted along the path.

    Parameters
    ----------
    partial    : ndarray (n_points, n_shells) — partial sums by l_tot
    increments : ndarray (n_points, n_shells) — each shell's own contribution
    exact      : ndarray (n_points) — the exact IQA V_cl(A,B), in hartree
    segments   : list of (start, stop) inclusive index ranges the REG will fit
                 over; the residual test is applied within each, since that is
                 the interval a REG gradient is taken over.  Defaults to the
                 whole path as one segment.
    residual_tolerance : allowed peak-to-peak variation of the truncation
                 residual as a fraction of the pair's own peak-to-peak V_cl.
    increment_ranks : how many of the top *complete* l_tot shells are examined
                 for growth, or None for all of them above l_tot = 1.  This is
                 the condition that catches an asymptotic series turning around
                 after looking settled at low rank.  Two things keep it honest:

                 Only shells with l_tot <= l_max are tested.  A shell above that
                 is missing the blocks its rank would need (l_tot = 2*l_max has
                 only the single (l_max, l_max) block), so it is small for a
                 bookkeeping reason rather than a physical one, and growth
                 cannot be judged there.  This is why the rank AIMAll actually
                 wrote matters: at a low ceiling there is little series left to
                 test, and the run says so.

                 A shell only counts as growth if it is larger than the one
                 before it *and* big enough to matter — see the resolution scale
                 below — so a converged pair is not rejected for a wobble in its
                 ninth significant figure.
    absolute_floor_kj : a pair whose V_cl varies by less than this along a
                 segment has no gradient worth resolving, so the relative test
                 would divide by noise; such a segment is judged on the absolute
                 residual variation against this floor instead.  It also sets
                 the resolution scale: max(floor, tolerance * smallest segment
                 span of this pair V_cl), the size an error has to reach before
                 it could change any conclusion the REG draws.

    Returns
    -------
    dict with 'admitted' (bool) and the diagnostics behind it.  A pair is
    admitted only if every geometry passes the increment test and every segment
    passes the residual test — a pair that converges at the reactant and fails
    near a transition state produces a kink that would dominate the REG
    gradient and read as chemistry.
    """
    partial = np.asarray(partial, dtype=float)
    increments = np.asarray(increments, dtype=float)
    exact = np.asarray(exact, dtype=float)
    n_points, n_shells = partial.shape

    residual = exact - partial[:, -1]

    if not segments:
        segments = [(0, n_points - 1)]

    # The scale an error has to reach before it could change a conclusion: the
    # same one the residual test uses, taken over the segment that demands the
    # most.  Anything below this is below the resolution of the analysis.
    floor_ha = absolute_floor_kj / HA_TO_KJ
    segment_spans = [float(np.nanmax(exact[a:b + 1]) - np.nanmin(exact[a:b + 1]))
                     if np.any(np.isfinite(exact[a:b + 1])) else 0.0
                     for a, b in segments]
    resolution = max(floor_ha, residual_tolerance * min(segment_spans or [0.0]))

    # --- increment test, per geometry ------------------------------------
    # The series fails here when one of its shells is *larger* than the shell
    # before it and large enough to matter.  Requiring plain monotone decrease
    # instead would reject well-converged pairs whose last shells wobble at the
    # 1e-5 kJ/mol level, which says nothing about whether the series diverges.
    #
    # The window starts at l_tot = 2: the first shells are the leading
    # behaviour, where a charge-dipole term legitimately exceeds its
    # charge-charge term, and it ends at l_max because the shells above that are
    # incomplete (see increment_ranks).
    l_max = (n_shells - 1) // 2
    complete = np.abs(increments[:, :l_max + 1])
    first = 2 if increment_ranks is None else max(1, l_max + 1 - int(increment_ranks))
    window = complete[:, first:]
    increment_testable = window.shape[1] >= 2
    if increment_testable:
        growing = np.diff(window, axis=1) > 0
        matters = window[:, 1:] > resolution
        increment_pass = ~np.any(growing & matters, axis=1)
    else:
        increment_pass = np.ones(n_points, bool)

    # --- residual test, per segment --------------------------------------
    segment_reports = []
    residual_pass = True
    for seg_i, (start, stop) in enumerate(segments):
        sl = slice(start, stop + 1)
        exact_span = float(np.nanmax(exact[sl]) - np.nanmin(exact[sl]))
        residual_span = float(np.nanmax(residual[sl]) - np.nanmin(residual[sl]))
        if exact_span > floor_ha:
            ratio = residual_span / exact_span
            passed = ratio <= residual_tolerance
            basis = 'relative'
        else:
            # Flat term: judge the residual's own wander against the floor, so a
            # large well-converged constant is not rejected for having no slope.
            ratio = residual_span / floor_ha
            passed = residual_span <= floor_ha
            basis = 'absolute'
        residual_pass = residual_pass and passed
        segment_reports.append({
            'segment': seg_i,
            'steps': [int(start), int(stop)],
            'vcl_span_kj_mol': exact_span * HA_TO_KJ,
            'residual_span_kj_mol': residual_span * HA_TO_KJ,
            'ratio': ratio,
            'basis': basis,
            'passed': bool(passed),
        })

    finite = np.isfinite(residual)
    return {
        'admitted': bool(residual_pass and np.all(increment_pass) and finite.all()),
        'increment_pass': increment_pass,
        'increment_failures': [int(i) for i in np.flatnonzero(~increment_pass)],
        'residual_pass': bool(residual_pass),
        'residual_ha': residual,
        'mean_abs_residual_kj_mol': float(np.nanmean(np.abs(residual)) * HA_TO_KJ),
        'max_abs_residual_kj_mol': float(np.nanmax(np.abs(residual)) * HA_TO_KJ) if finite.any() else float('nan'),
        'segments': segment_reports,
        'resolution_kj_mol': resolution * HA_TO_KJ,
        'increment_window': [int(first), int(l_max)],
        'increment_testable': bool(increment_testable),
        'values_readable': bool(finite.all()),
    }


# ---------------------------------------------------------------------------
# Grouped-atom (fragment-centred) moments
# ---------------------------------------------------------------------------
#
# Everything above expands each atom about its own nucleus and sums at the
# *energy* level: a fragment-pair rank term is the sum of atom-pair terms of that
# rank.  The functions below do the other thing — sum at the *moment* level, by
# translating every atom's moments onto one fragment centre — which is what gives
# "the dipole of this fragment against the dipole of that one".
#
# The two are not interchangeable and must not share a table.  They converge to
# the same energy where both converge, but they attribute it to different ranks:
# for two neutral fragments the fragment-centred monopole and charge-dipole terms
# vanish identically, while the atom-centred series puts large charge-charge
# terms against each other that mostly cancel.  Which is the point of having
# both — a net dipole-dipole interaction that drives one reaction can be a set of
# atomic contributions that cancel in another.
#
# The price is the convergence sphere.  A fragment-centred expansion needs its
# sphere to enclose the whole fragment, so the separation it demands grows with
# fragment size; see fragment_extents().


def complex_solid_harmonics(xyz, l_max):
    """Racah-normalised complex regular solid harmonics R_l^m of *xyz*.

    R_l^m = sqrt(4*pi/(2l+1)) r^l Y_l^m, with the Condon-Shortley phase, which is
    the normalisation AIMAll's printed moments are built on.

    Parameters
    ----------
    xyz : ndarray (batch, 3), in bohr

    Returns
    -------
    complex ndarray (batch, l_max+1, 2*l_max+1), indexed [., l, m + l_max]

    Built by the standard recursion rather than from scipy's spherical
    harmonics, so it does not depend on which spelling of sph_harm the installed
    scipy has.
    """
    xyz = np.asarray(xyz, dtype=float)
    batch = xyz.shape[0]
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r_squared = x ** 2 + y ** 2 + z ** 2

    out = np.zeros((batch, l_max + 1, 2 * l_max + 1), dtype=complex)
    out[:, 0, l_max] = 1.0
    for l in range(1, l_max + 1):
        # R_l^l  = -(x + iy) sqrt((2l-1)/(2l)) R_{l-1}^{l-1}
        out[:, l, l_max + l] = (-(x + 1j * y) * np.sqrt((2.0 * l - 1.0) / (2.0 * l))
                                * out[:, l - 1, l_max + l - 1])
        # R_l^{l-1} = z sqrt(2l-1) R_{l-1}^{l-1}
        out[:, l, l_max + l - 1] = z * np.sqrt(2.0 * l - 1.0) * out[:, l - 1, l_max + l - 1]
        for m in range(-(l - 2), l - 1):
            # R_l^m = [(2l-1) z R_{l-1}^m - sqrt((l-1+m)(l-1-m)) r^2 R_{l-2}^m]
            #         / sqrt((l+m)(l-m))
            out[:, l, l_max + m] = (((2 * l - 1) * z * out[:, l - 1, l_max + m]
                                     - np.sqrt(float((l - 1 + m) * (l - 1 - m)))
                                     * r_squared * out[:, l - 2, l_max + m])
                                    / np.sqrt(float((l + m) * (l - m))))
        # R_l^{-m} = (-1)^m conj(R_l^m) for a real vector
        for m in range(1, l + 1):
            out[:, l, l_max - m] = ((-1) ** m) * np.conj(out[:, l, l_max + m])
    return out


def real_to_complex_moments(moments, l_max):
    """AIMAll's real Q[l,k] -> complex Q_l^m, shape (batch, l_max+1, 2*l_max+1).

    Q_l^0 = Q[l,0] and, for m > 0, Q_l^m = (-1)^m (Q[l,m,c] + i Q[l,m,s]) / sqrt(2),
    with Q_l^-m = (-1)^m conj(Q_l^m) because the charge distribution is real.
    """
    moments = np.asarray(moments, dtype=float)
    batch = moments.shape[0]
    out = np.zeros((batch, l_max + 1, 2 * l_max + 1), dtype=complex)
    for l in range(l_max + 1):
        out[:, l, l_max] = moments[:, component_index(l, 0)]
        for m in range(1, l + 1):
            cosine = moments[:, component_index(l, m)]
            sine = moments[:, component_index(l, -m)]
            value = ((-1) ** m) * (cosine + 1j * sine) / np.sqrt(2.0)
            out[:, l, l_max + m] = value
            out[:, l, l_max - m] = ((-1) ** m) * np.conj(value)
    return out


def complex_to_real_moments(moments, l_max):
    """Inverse of :func:`real_to_complex_moments`."""
    moments = np.asarray(moments)
    batch = moments.shape[0]
    out = np.zeros((batch, n_components(l_max)))
    for l in range(l_max + 1):
        out[:, component_index(l, 0)] = moments[:, l, l_max].real
        for m in range(1, l + 1):
            value = ((-1) ** m) * np.sqrt(2.0) * moments[:, l, l_max + m]
            out[:, component_index(l, m)] = value.real
            out[:, component_index(l, -m)] = value.imag
    return out


def translate_moments(moments, displacement, l_max):
    """Re-express multipole moments about a shifted origin.

    Parameters
    ----------
    moments      : ndarray (batch, n_components(l_max)), real, about the old origin
    displacement : ndarray (batch, 3), in **bohr** — the vector from the new
                   origin to the old one, so a site at the old origin sits at
                   *displacement* in the new frame
    l_max        : rank kept.  Translation mixes ranks downward only, so every
                   moment of the result is exact given the input up to l_max:
                   nothing is lost, but nothing above l_max can be created either.

    Returns
    -------
    ndarray (batch, n_components(l_max)), real, about the new origin

    Uses the addition theorem for regular solid harmonics,
    R_l^m(r + d) = sum_{l',m'} sqrt(C(l+m, l'+m') C(l-m, l'-m'))
                              R_l'^m'(d) R_{l-l'}^{m-m'}(r),
    so Q_l^m(new) = sum_{l',m'} (same coefficient) R_l'^m'(d) Q_{l-l'}^{m-m'}(old).
    """
    from math import comb

    complex_moments = real_to_complex_moments(moments, l_max)
    harmonics = complex_solid_harmonics(np.asarray(displacement, dtype=float), l_max)
    batch = complex_moments.shape[0]
    out = np.zeros((batch, l_max + 1, 2 * l_max + 1), dtype=complex)

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            total = np.zeros(batch, dtype=complex)
            for l_shift in range(l + 1):
                l_rest = l - l_shift
                for m_shift in range(-l_shift, l_shift + 1):
                    m_rest = m - m_shift
                    if abs(m_rest) > l_rest:
                        continue
                    upper = l_shift + m_shift
                    lower = l_shift - m_shift
                    if not (0 <= upper <= l + m) or not (0 <= lower <= l - m):
                        continue
                    coefficient = np.sqrt(float(comb(l + m, upper) * comb(l - m, lower)))
                    total = total + (coefficient
                                     * harmonics[:, l_shift, l_max + m_shift]
                                     * complex_moments[:, l_rest, l_max + m_rest])
            out[:, l, l_max + m] = total
    return complex_to_real_moments(out, l_max)


def fragment_moments(atom_moments, atom_coords, centre, l_max):
    """Multipole moments of a group of atoms about one centre.

    Parameters
    ----------
    atom_moments : ndarray (n_atoms_in_group, n_components(l_max)), nucleus-centred
    atom_coords  : ndarray (n_atoms_in_group, 3), in **Angstrom**
    centre       : ndarray (3,), in **Angstrom**

    Returns
    -------
    ndarray (n_components(l_max),) — the group's moments about *centre*

    Each atom's moments are translated onto the centre and summed.  Rank 0 comes
    out as the group's net charge, and for a neutral group the rank-1 result is
    the familiar origin-independent dipole.
    """
    displacement = (np.asarray(atom_coords, dtype=float)
                    - np.asarray(centre, dtype=float)) * BOHR_PER_ANGSTROM
    return translate_moments(np.asarray(atom_moments, dtype=float), displacement, l_max).sum(axis=0)


def fragment_centre(atom_coords, weights=None):
    """Expansion centre for a group of atoms, in Angstrom.

    *weights* None gives the plain centroid, which is close to the centre that
    minimises the group's radial extent — and the radial extent is what limits
    whether a fragment-centred expansion converges at all.  Pass nuclear charges
    or masses for the other usual conventions.
    """
    atom_coords = np.asarray(atom_coords, dtype=float)
    if weights is None:
        return atom_coords.mean(axis=0)
    weights = np.asarray(weights, dtype=float)
    return (weights[:, None] * atom_coords).sum(axis=0) / weights.sum()


def fragment_extents(atom_coords, centre, basin_radii_ang):
    """Radius of the sphere about *centre* that encloses the group's basins.

    max over atoms of (|r_A - centre| + R_A).  This is what the geometric gate
    has to clear for a fragment-centred expansion, and it is why the demand grows
    with fragment size: a big fragment needs its partner far away before its own
    expansion converges, however innocuous the individual atom pairs look.
    """
    offsets = np.linalg.norm(np.asarray(atom_coords, dtype=float)
                             - np.asarray(centre, dtype=float), axis=-1)
    return float(np.nanmax(offsets + np.asarray(basin_radii_ang, dtype=float)))


# Atomic number and mass for the elements a fragment centre might be weighted by.
# Only used to place an expansion centre, so an unlisted element falls back to
# the plain centroid contribution rather than failing the run.
_ELEMENT_DATA = {
    'h': (1, 1.008), 'he': (2, 4.003), 'li': (3, 6.94), 'be': (4, 9.012),
    'b': (5, 10.81), 'c': (6, 12.011), 'n': (7, 14.007), 'o': (8, 15.999),
    'f': (9, 18.998), 'ne': (10, 20.180), 'na': (11, 22.990), 'mg': (12, 24.305),
    'al': (13, 26.982), 'si': (14, 28.085), 'p': (15, 30.974), 's': (16, 32.06),
    'cl': (17, 35.45), 'ar': (18, 39.948), 'k': (19, 39.098), 'ca': (20, 40.078),
    'fe': (26, 55.845), 'co': (27, 58.933), 'ni': (28, 58.693), 'cu': (29, 63.546),
    'zn': (30, 65.38), 'br': (35, 79.904), 'i': (53, 126.904),
}


def centre_weights(atom_labels, scheme):
    """Weights for :func:`fragment_centre`; None for the plain centroid."""
    if scheme in (None, 'centroid'):
        return None
    column = 0 if scheme in ('nuclear-charge', 'charge', 'z') else 1
    return np.array([_ELEMENT_DATA.get(element_of(a), (6, 12.011))[column]
                     for a in atom_labels], dtype=float)


def rank_magnitude(moments, l):
    """Rotation-invariant size of rank *l*: sqrt(sum_k Q[l,k]^2).

    The sum of squares over the real components of one rank is the same in any
    frame, which makes this the honest way to ask "how big is this fragment's
    dipole" — and, compared against the same quantity summed over the fragment's
    atoms, how much those atomic moments cancel.
    """
    moments = np.asarray(moments, dtype=float)
    block = [component_index(l, k) for k in range(-l, l + 1)]
    return np.sqrt(np.sum(moments[..., block] ** 2, axis=-1))


# Names of the multipole each rank corresponds to, for reporting.  A term is
# easier to argue about as "dipole-quadrupole" than as "V[1,2]", so the reports
# lead with the name and keep the rank indices beside it.
_MULTIPOLE_NAMES = {
    0: 'charge', 1: 'dipole', 2: 'quadrupole', 3: 'octupole',
    4: 'hexadecapole', 5: 'triacontadipole',
}


def multipole_name(l):
    """'dipole' for l = 1, and so on; a 2^l-pole label beyond the named ranks."""
    return _MULTIPOLE_NAMES.get(l, '2^{l}-pole'.format(l=l))


def rank_pair_name(l_a, l_b):
    """Name of the interaction between rank *l_a* on A and rank *l_b* on B.

    Order matters and is preserved: 'charge-dipole' is A's charge with B's
    dipole, which is a different term from 'dipole-charge'.
    """
    return multipole_name(l_a) + '-' + multipole_name(l_b)
