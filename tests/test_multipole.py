"""
test_multipole.py

Checks for the REG_Multi machinery.  Run it directly:

    python3 tests/test_multipole.py

Three kinds of check, in order of how much they prove:

  * the interaction tensor against closed-form charge-charge, charge-dipole and
    dipole-dipole energies;
  * the whole series against the exact Coulomb energy of two point-charge
    clusters, which exercises every rank up to l_tot = 10 and uses solid
    harmonics built independently of the recursion under test;
  * the admission gates against a synthetic path where the convergence condition
    R_AB > rho_A + rho_B is known atom by atom, so "which pairs should be
    admitted" has a right answer rather than a plausible one.

The last of these also runs the driver end to end and checks that the published
decomposition closes: rank terms + residual + unresolved = the exact IQA V_cl.
"""

import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), 'src')
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

import multipole_utils as mp          # noqa: E402
import synthetic_points as synth      # noqa: E402

FAILURES = []


def check(condition, message):
    if condition:
        print('  ok   ' + message)
    else:
        print('  FAIL ' + message)
        FAILURES.append(message)


def close(a, b, tol, message):
    difference = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    check(difference <= tol, '{m} (max deviation {d:.3e}, tol {t:.1e})'
          .format(m=message, d=difference, t=tol))


# ---------------------------------------------------------------------------

def test_tensor_against_closed_forms():
    print('\ninteraction tensor against closed-form energies')
    rng = np.random.default_rng(0)
    batch, l_max = 8, 2
    pos_a = rng.normal(size=(batch, 3)) * 2
    pos_b = pos_a + rng.normal(size=(batch, 3)) * 3
    r_vector = (pos_b - pos_a) * mp.BOHR_PER_ANGSTROM
    distance = np.linalg.norm(r_vector, axis=1)
    unit = r_vector / distance[:, None]

    def moments(charge=0.0, dipole=None):
        out = np.zeros((batch, mp.n_components(l_max)))
        out[:, mp.component_index(0, 0)] = charge
        if dipole is not None:
            out[:, mp.component_index(1, 1)] = dipole[:, 0]    # x is k = +1
            out[:, mp.component_index(1, -1)] = dipole[:, 1]   # y is k = -1
            out[:, mp.component_index(1, 0)] = dipole[:, 2]    # z is k =  0
        return out

    q_a, q_b = 1.3, -0.7
    mu_a, mu_b = rng.normal(size=(batch, 3)), rng.normal(size=(batch, 3))

    energies = mp.pair_rank_energies(moments(charge=q_a), moments(charge=q_b),
                                     pos_a, pos_b, l_max)
    close(energies[:, 0, 0], q_a * q_b / distance, 1e-12, 'charge-charge = qA qB / R')

    energies = mp.pair_rank_energies(moments(charge=q_a), moments(dipole=mu_b),
                                     pos_a, pos_b, l_max)
    close(energies[:, 0, 1], -q_a * np.sum(mu_b * unit, axis=1) / distance ** 2, 1e-12,
          'charge-dipole = -qA (muB.Rhat) / R^2')

    energies = mp.pair_rank_energies(moments(dipole=mu_a), moments(charge=q_b),
                                     pos_a, pos_b, l_max)
    close(energies[:, 1, 0], q_b * np.sum(mu_a * unit, axis=1) / distance ** 2, 1e-12,
          'dipole-charge = +qB (muA.Rhat) / R^2')

    energies = mp.pair_rank_energies(moments(dipole=mu_a), moments(dipole=mu_b),
                                     pos_a, pos_b, l_max)
    expected = (np.sum(mu_a * mu_b, axis=1)
                - 3 * np.sum(mu_a * unit, axis=1) * np.sum(mu_b * unit, axis=1)) / distance ** 3
    close(energies[:, 1, 1], expected, 1e-12,
          'dipole-dipole = (muA.muB - 3 (muA.Rhat)(muB.Rhat)) / R^3')


def test_series_against_exact_coulomb():
    print('\nfull series against the exact Coulomb energy of two charge clusters')
    rng = np.random.default_rng(3)
    l_max = 5
    for separation, tolerance_kj in ((6.0, 0.01), (4.0, 0.1), (3.0, 0.5)):
        q_a = rng.normal(size=4) + 0.3
        q_b = rng.normal(size=4) - 0.2
        cloud_a = rng.normal(size=(4, 3)) * 0.35
        cloud_b = rng.normal(size=(4, 3)) * 0.35
        centre_a = np.zeros(3)
        centre_b = np.array([separation, 0.0, 0.0])

        positions_a = (centre_a + cloud_a) * mp.BOHR_PER_ANGSTROM
        positions_b = (centre_b + cloud_b) * mp.BOHR_PER_ANGSTROM
        distance = np.linalg.norm(positions_a[:, None, :] - positions_b[None, :, :], axis=-1)
        exact = float(np.sum(np.outer(q_a, q_b) / distance))

        moments_a = (q_a[:, None] * synth.solid_harmonics(
            cloud_a * mp.BOHR_PER_ANGSTROM, l_max)).sum(axis=0)
        moments_b = (q_b[:, None] * synth.solid_harmonics(
            cloud_b * mp.BOHR_PER_ANGSTROM, l_max)).sum(axis=0)

        ranks = mp.pair_rank_energies(moments_a[None, :], moments_b[None, :],
                                      centre_a[None, :], centre_b[None, :], l_max)
        _, partial = mp.partial_sums_by_total_rank(ranks)
        error_kj = abs(exact - float(partial[0, -1])) * mp.HA_TO_KJ
        check(error_kj < tolerance_kj,
              'clusters {s:.1f} A apart converge to the exact energy '
              '(error {e:.2e} kJ/mol < {t} kJ/mol)'.format(s=separation, e=error_kj,
                                                           t=tolerance_kj))


def test_moment_reading(dataset_dir, system):
    print('\nreading moments out of AIMAll .int files')
    atomic_files = [os.path.join(dataset_dir, str(i + 1), 'synth_atomicfiles')
                    for i in range(len(system['points']))]
    moments, beta_radii, info = mp.get_atomic_multipoles(atomic_files, system['labels'], 5)

    check(not info['missing'], 'every .int file was read')
    check(info['l_available'] == 5, 'highest available rank reported as 5')

    expected_charges = np.array([atom['net_charge'] for atom in system['atoms']])
    close(moments[0, :, 0], expected_charges, 1e-9,
          'Q[0,0] is replaced by the net charge, not the electronic population')

    expected = np.array([atom['moments'] for atom in system['atoms']])
    close(moments[0, :, 1:], expected[:, 1:], 1e-9,
          'every higher moment round-trips through the file')

    expected_beta = np.array([0.7 * atom['rho'] * mp.BOHR_PER_ANGSTROM
                              for atom in system['atoms']])
    close(beta_radii[0], expected_beta, 1e-8, 'beta-sphere radii are read in bohr')


def test_gates(dataset_dir, system):
    print('\nadmission gates against a known convergence condition')
    import reg_multipole as rm

    atomic_files = [os.path.join(dataset_dir, str(i + 1), 'synth_atomicfiles')
                    for i in range(len(system['points']))]
    labels = system['labels']
    moments, beta_radii, _ = mp.get_atomic_multipoles(atomic_files, labels, 5)
    coords = np.array([point['coords'] for point in system['points']])

    pair_index = system['pairs']
    exact = np.array([[point['exact'][pair] for point in system['points']]
                      for pair in pair_index])

    flat_pairs = np.repeat(np.arange(len(pair_index)), len(system['points']))
    flat_steps = np.tile(np.arange(len(system['points'])), len(pair_index))
    idx_a = np.array([pair_index[p][0] for p in flat_pairs])
    idx_b = np.array([pair_index[p][1] for p in flat_pairs])
    ranks = np.zeros((len(pair_index), len(system['points']), 6, 6))
    ranks[flat_pairs, flat_steps] = mp.pair_rank_energies(
        moments[flat_steps, idx_a], moments[flat_steps, idx_b],
        coords[flat_steps, idx_a], coords[flat_steps, idx_b], 5)

    settings = {'topology': True, 'radii': 'beta', 'tolerance': 0.05, 'floor': 0.05,
                'increment_ranks': None, 'skip_convergence': False, 'bond_tolerance': 1.2}
    segments = [(0, len(system['points']) - 1)]
    verdicts, _ = rm.admit_pairs(mp, labels, coords, pair_index, exact, ranks,
                                 beta_radii, segments, settings)

    # Ground truth: the series converges exactly when the charge clouds do not
    # overlap.  Intramolecular pairs are 1,2 or 1,3 and never reach the test.
    radius = {i: atom['rho'] for i, atom in enumerate(system['atoms'])}
    for verdict, (i, j) in zip(verdicts, pair_index):
        same_fragment = system['atoms'][i]['fragment'] == system['atoms'][j]['fragment']
        overlapping = verdict['min_distance_ang'] < radius[i] + radius[j]
        if same_fragment:
            check(verdict['rejected_by'] == 'topology',
                  '{a}-{b} (same fragment) rejected by the topology pre-filter'
                  .format(a=labels[i], b=labels[j]))
        elif overlapping:
            check(not verdict['admitted'],
                  '{a}-{b} rejected: clouds overlap at R = {r:.2f} A < {s:.2f} A'
                  .format(a=labels[i], b=labels[j], r=verdict['min_distance_ang'],
                          s=radius[i] + radius[j]))
        else:
            check(verdict['admitted'],
                  '{a}-{b} admitted: R = {r:.2f} A clears {s:.2f} A, residual {e:.3f} kJ/mol'
                  .format(a=labels[i], b=labels[j], r=verdict['min_distance_ang'],
                          s=radius[i] + radius[j],
                          e=verdict.get('max_abs_residual_kj_mol', float('nan'))))

    # The increment condition has to stand on its own: with the residual test
    # turned off, the overlapping pair must still be caught by its series
    # turning around.  That is the case the residual test alone can miss.
    contact = pair_index.index((0, 4))
    increments, partial = mp.partial_sums_by_total_rank(ranks[contact])
    report = mp.convergence_verdict(partial, increments, exact[contact], segments=segments,
                                    absolute_floor_kj=0.05)
    check(not all(report['increment_pass']),
          'o1-h5 is caught by the increment condition on its own, not only by the residual')
    check(report['increment_testable'],
          '... and the window it was tested over is wide enough to mean something')

    # A pair that is genuinely converged must survive an increment test that is
    # not scaled away: its top shells are small *and* not growing.
    far = pair_index.index((2, 5))
    increments, partial = mp.partial_sums_by_total_rank(ranks[far])
    report = mp.convergence_verdict(partial, increments, exact[far], segments=segments,
                                    residual_tolerance=0.05, absolute_floor_kj=0.05)
    check(report['admitted'] and all(report['increment_pass']),
          'the most distant pair passes both conditions')


def test_moment_translation(system):
    print('\nmoment translation onto a fragment centre')
    l_max = 5
    coords = system['points'][0]['coords']
    atom_moments = np.array([atom['moments'] for atom in system['atoms']])

    # Translating by nothing must change nothing.
    unmoved = mp.translate_moments(atom_moments, np.zeros((len(atom_moments), 3)), l_max)
    close(unmoved, atom_moments, 1e-10, 'a zero translation is the identity')

    # The fixture's atoms are explicit point-charge clouds, so the true moments of
    # a group about any centre can be written down directly — an independent
    # reference for the addition theorem, not a restatement of it.
    for members, name in (((0, 1, 2), 'Acceptor'), ((3, 4, 5), 'Donor')):
        centre = mp.fragment_centre(coords[list(members)])
        translated = mp.fragment_moments(atom_moments[list(members)],
                                         coords[list(members)], centre, l_max)
        charges, positions = [], []
        for i in members:
            charges += list(system['atoms'][i]['charges'])
            positions += list(coords[i] + system['atoms'][i]['cloud'] - centre)
        reference = (np.array(charges)[:, None] * synth.solid_harmonics(
            np.array(positions) * mp.BOHR_PER_ANGSTROM, l_max)).sum(axis=0)
        close(translated, reference, 1e-9,
              '{n} moments about its centre match the exact group moments'.format(n=name))
        check(abs(translated[0] - sum(system['atoms'][i]['net_charge'] for i in members)) < 1e-9,
              '... and rank 0 is the group net charge')


def test_fragment_view(dataset_dir):
    print('\nfragment-centred analysis')
    import reg_multipole as rm

    # A well-separated version of the same system, where a fragment-centred
    # expansion is expected to converge.
    far = synth.build_system(separations=(9.0, 8.0, 7.0))
    l_max = 5
    coords = np.array([point['coords'] for point in far['points']])
    moments = np.array([[atom['moments'] for atom in far['atoms']]] * len(far['points']))
    labels = far['labels']
    pair_index = far['pairs']
    exact = np.array([[point['exact'][pair] for point in far['points']] for pair in pair_index])
    radii = np.array([[atom['rho'] for atom in far['atoms']]] * len(far['points'])) * 0.7

    groups, _ = rm.build_groups(labels, ['Acceptor', 'Donor'],
                                [far['fragments']['A'], far['fragments']['B']], 'inter', None)
    options = {'tolerance': 0.05, 'floor': 0.05, 'increment_ranks': None,
               'fragment_centre': 'centroid'}
    view = rm.fragment_moment_analysis(
        mp, labels, coords, moments, ['Acceptor', 'Donor'],
        [far['fragments']['A'], far['fragments']['B']], groups, pair_index, exact,
        radii, [(0, len(far['points']) - 1)], l_max, options)

    entry = view['pairs'][0]
    check(entry['admitted'],
          'a well-separated fragment pair is admitted (margin {m:.2f} A)'
          .format(m=float(entry['extent_margin_ang'].min())))
    close(entry['total'], entry['exact'], 0.5 / mp.HA_TO_KJ,
          'the fragment-centred series reproduces the exact interfragment V_cl')

    # Neutral fragments: the monopole and every charge-multipole term must vanish
    # identically, which is the structural difference from the atom-centred view.
    monopole_terms = np.abs(entry['ranks'][:, 0, :]).max() + np.abs(entry['ranks'][:, :, 0]).max()
    check(monopole_terms < 1e-12,
          'for neutral fragments every monopole term is identically zero')

    for name, rows in view['cancellation'].items():
        ratios = np.concatenate([row['ratio'][np.isfinite(row['ratio'])] for row in rows])
        check(np.all(ratios <= 1.0 + 1e-9) and np.all(ratios >= -1e-9),
              '{n}: cancellation ratios stay within [0, 1]'.format(n=name))
        check(rows[0]['fragment'].max() < 1e-9,
              '{n}: rank 0 cancels completely, as a neutral fragment requires'.format(n=name))

    # The close system: the fragment spheres overlap even though most of its atom
    # pairs are fine, which is the whole reason the two views gate differently.
    near = synth.build_system()
    coords_near = np.array([point['coords'] for point in near['points']])
    moments_near = np.array([[a['moments'] for a in near['atoms']]] * len(near['points']))
    exact_near = np.array([[p['exact'][pair] for p in near['points']] for pair in near['pairs']])
    radii_near = np.array([[a['rho'] for a in near['atoms']]] * len(near['points'])) * 0.7
    view_near = rm.fragment_moment_analysis(
        mp, near['labels'], coords_near, moments_near, ['Acceptor', 'Donor'],
        [near['fragments']['A'], near['fragments']['B']], groups, near['pairs'], exact_near,
        radii_near, [(0, len(near['points']) - 1)], l_max, options)
    near_entry = view_near['pairs'][0]
    check(not near_entry['admitted'] and near_entry['rejected_by'] == 'fragment_extent',
          'the close fragment pair is rejected on overlapping expansion spheres, '
          'though most of its atom pairs are admissible')


def test_grouping(system):
    print('\npair selection and grouping')
    import reg_multipole as rm
    labels = system['labels']
    names = ['Acceptor', 'Donor']
    frags = [system['fragments']['A'], system['fragments']['B']]

    groups, pairs = rm.build_groups(labels, names, frags, 'inter', None)
    check(len(groups) == 1 and len(pairs) == 9,
          'inter scope gives one fragment pair and its 9 atom pairs')

    groups, pairs = rm.build_groups(labels, names, frags, 'intra', None)
    check(len(groups) == 2 and len(pairs) == 6,
          'intra scope gives one group per fragment, 3 atom pairs each')

    groups, pairs = rm.build_groups(labels, names, frags, 'intra', ['Donor'])
    check(len(groups) == 1 and groups[0]['label'] == 'Donor(intra)' and len(pairs) == 3,
          'analysing within one named fragment takes only that fragment pairs')

    groups, pairs = rm.build_groups(labels, names, frags, 'all', None)
    check(len(groups) == 3 and len(pairs) == 15,
          'all scope covers every pair in the system, grouped by fragment')

    groups, pairs = rm.build_groups(labels, names, frags, 'pairs', None)
    check(len(groups) == 15 and len(pairs) == 15
          and all(len(g['pairs']) == 1 for g in groups)
          and groups[0]['label'] == labels[0] + '-' + labels[1],
          'pairs scope gives one group per atom pair, ignoring the fragments')

    groups, pairs = rm.build_groups(labels, None, None, 'all', None)
    check(len(groups) == 1 and groups[0]['label'] == 'System' and len(pairs) == 15,
          'with no fragments defined, every atom pair in the system is one group')

    try:
        rm.build_groups(labels, names, frags, 'intra', ['Nonexistent'])
        check(False, 'an unknown fragment name is rejected')
    except ValueError:
        check(True, 'an unknown fragment name is rejected')

    # --ignore-fragments makes the run behave as though the config had none, so
    # which pairs get described is left entirely to the admission gates.
    ignored_groups, ignored_pairs = rm.build_groups(labels, None, None, 'all', None)
    fragment_groups, fragment_pairs = rm.build_groups(labels, names, frags, 'all', None)
    check(len(ignored_groups) == 1 and len(ignored_pairs) == len(fragment_pairs),
          'ignoring the fragments keeps every pair but drops the partition')


def test_end_to_end(dataset_dir):
    print('\nend-to-end run of the driver')
    import json
    import reg_multipole as rm

    # The plain run first — no fragment-centred analysis — because that path
    # produces fewer tables, and a table a run does not produce must not be able
    # to take the workbook down with it.  A 0-byte REG_Multi.xlsx is how that
    # failure shows up, so the size is checked rather than just the existence.
    cwd = os.getcwd()
    try:
        os.chdir(dataset_dir)
        rm.main(argv=['-d', SRC, '-o', 'REG_Multi_plain_results'])
    finally:
        os.chdir(cwd)
    plain_workbook = os.path.join(dataset_dir, 'REG_Multi_plain_results', 'REG_Multi.xlsx')
    check(os.path.exists(plain_workbook) and os.path.getsize(plain_workbook) > 0,
          'a run without --fragment-moments still writes a complete workbook')
    check(os.path.exists(os.path.join(dataset_dir, 'REG_Multi_plain_results',
                                      'atomic_moments.csv')),
          'atomic moment coefficients are reported without any flag')
    check(os.path.exists(os.path.join(dataset_dir, 'REG_Multi_plain_results',
                                      'fragment_moments.csv')),
          'fragment moment coefficients are reported whenever fragments are defined')
    check(os.path.exists(os.path.join(dataset_dir, 'REG_Multi_plain_results',
                                      'REG_Multi_plain_model_transfer.json')),
          'the bundle is named after the results directory')

    # Asked to work inside a fragment, every pair here is 1,2 or 1,3 and none can
    # carry a nucleus-centred expansion.  "Nothing admitted" is a real answer and
    # has to be reported, not raised: there are no terms to regress, no REG to
    # run, and the admission report still has to come out.
    cwd = os.getcwd()
    try:
        os.chdir(dataset_dir)
        rm.main(argv=['-d', SRC, '--scope', 'intra', '-o', 'REG_Multi_empty_results'])
    finally:
        os.chdir(cwd)
    empty_dir = os.path.join(dataset_dir, 'REG_Multi_empty_results')
    check(os.path.exists(os.path.join(empty_dir, 'multipole_admission.txt')),
          'a run that admits nothing still reports why')
    empty_workbook = os.path.join(empty_dir, 'REG_Multi.xlsx')
    check(os.path.exists(empty_workbook) and os.path.getsize(empty_workbook) > 0,
          '... and still writes a valid workbook')
    with open(os.path.join(empty_dir, 'REG_Multi_empty_model_transfer.json')) as handle:
        empty_bundle = json.load(handle)
    check(empty_bundle['reg']['rank'] is None,
          '... and records that there was no ranking to make')

    cwd = os.getcwd()
    try:
        os.chdir(dataset_dir)
        rm.main(argv=['-d', SRC, '--fragment-moments'])
    finally:
        os.chdir(cwd)

    results = os.path.join(dataset_dir, 'REG_Multi_results')
    workbook = os.path.join(results, 'REG_Multi.xlsx')
    check(os.path.getsize(workbook) > 0, 'REG_Multi.xlsx is not empty')
    for name in ('multipole_admission.txt', 'multipole_admission.csv', 'multipole_terms.csv',
                 'excluded_accounting.csv', 'REG_Multi_model_transfer.json', 'REG_Multi.xlsx',
                 'fragment_moments.csv', 'fragment_moment_cancellation.csv',
                 'fragment_moment_comparison.csv', 'REG_Multi_energy.csv',
                 'REG_Multi_recovery.csv', 'energy_recovery.csv', 'energy_by_type.csv',
                 'energy_by_pair.csv'):
        check(os.path.exists(os.path.join(results, name)), 'wrote ' + name)

    with open(os.path.join(results, 'REG_Multi_model_transfer.json')) as handle:
        bundle = json.load(handle)

    headers = bundle['terms']['rank']['headers']
    values = np.array(bundle['terms']['rank']['values'])
    shell_headers = bundle['terms']['shell']['headers']
    shell_values = np.array(bundle['terms']['shell']['values'])
    for group in bundle['groups']:
        rows = [i for i, h in enumerate(headers) if h.startswith(group['label'] + ' ')]
        if not group['n_admitted']:
            check(not rows,
                  'a group with nothing admitted contributes no rank terms: ' + group['label'])
            continue
        close(values[rows].sum(axis=0), group['multipolar_total'], 1e-9,
              'the rank terms sum to the multipole energy for ' + group['label'])
        # The balance is on the energies, not in the term table: the table holds
        # multipole terms only, and what they leave out is accounted separately.
        close(np.array(group['multipolar_total']) + np.array(group['residual'])
              + np.array(group['unresolved']), group['exact_vcl'], 1e-9,
              'multipole + residual + unresolved = exact V_cl for ' + group['label'])
        rows = [i for i, h in enumerate(shell_headers) if h.startswith(group['label'] + ' ')]
        close(shell_values[rows].sum(axis=0), group['multipolar_total'], 1e-9,
              'the l_tot shell view sums to the same multipole energy for ' + group['label'])

    check(not any('residual' in h or 'unresolved' in h for h in headers + shell_headers),
          'the REG term tables hold multipole terms only — no accounting rows')

    view = bundle.get('fragment_view')
    check(view is not None, 'the bundle carries the fragment-centred view')
    if view:
        for entry in view['pairs']:
            if entry['admitted']:
                close(np.array(entry['multipolar_total']) + np.array(entry['residual']),
                      entry['exact_vcl'], 1e-9,
                      'fragment ranks + residual reproduce the exact V_cl for ' + entry['label'])
        check(all(len(entry['separation_ang']) == len(bundle['control_coordinates'])
                  for entry in view['pairs']),
              'fragment separations are recorded at every step')
        atom_headers = set(bundle['terms']['rank']['headers'])
        frag_block = bundle['terms'].get('fragment_rank')
        if frag_block:
            check(not (atom_headers & set(frag_block['headers'])),
                  'atom-centred and fragment-centred terms share no header, so they cannot be '
                  'ranked in one table by accident')

    # The energy files are what a reader plots.  They are written one row per
    # series so that a large system does not run past Excel's column limit, and
    # the balance still has to close at every step.
    def read_series(name):
        import csv as _csv
        with open(os.path.join(results, name)) as handle:
            reader = _csv.reader(handle)
            next(reader)
            return {row[0]: np.array([float(v) for v in row[1:]]) for row in reader if row}

    balance = read_series('REG_Multi_energy.csv')
    n_steps = len(bundle['control_coordinates'])
    check(all(len(v) == n_steps for v in balance.values()),
          'REG_Multi_energy.csv is one row per series, one column per step')
    close(balance['TOTAL Vcl_multipole'] + balance['TOTAL Vcl_residual (included pairs)'],
          balance['TOTAL Vcl_IQA (included pairs)'], 1e-12,
          'the balance closes over the included pairs: multipole + residual = V_cl(IQA)')
    check('TOTAL Vxc_IQA (all pairs)' in balance,
          'V_xc is reported beside the classical energy')

    # The recovery file must compare like with like: the same pairs on both
    # curves, and no residual or excluded channel mixed in.
    recovery = read_series('energy_recovery.csv')
    close(recovery['TOTAL Vcl_multipole (included pairs)'], balance['TOTAL Vcl_multipole'],
          1e-12, 'energy_recovery.csv carries the multipole sum over the included pairs')
    close(recovery['TOTAL Vcl_IQA (included pairs)'],
          balance['TOTAL Vcl_IQA (included pairs)'], 1e-12,
          '... against the exact V_cl of those same pairs')
    check(not any('residual' in k or 'excluded' in k or 'unresolved' in k for k in recovery),
          'energy_recovery.csv holds only the two curves being compared')

    # The by-type file sums the atom-pair terms into fragment channels whatever
    # the scope was, and each channel's types must add to its total.
    by_type = read_series('energy_by_type.csv')
    for key in [k for k in by_type if k.endswith('total (included pairs)')]:
        label = key[:-len(' total (included pairs)')]
        parts = [v for k, v in by_type.items()
                 if k.startswith(label + ' ') and not k.endswith('(included pairs)')]
        check(bool(parts), 'energy_by_type.csv breaks ' + label + ' down by multipole type')
        close(np.sum(parts, axis=0), by_type[key], 1e-12,
              '... and those types sum to its total')

    by_pair = read_series('energy_by_pair.csv')
    check(len(by_pair) == sum(g['n_admitted'] for g in bundle['groups']),
          'energy_by_pair.csv has one row per admitted atom pair')

    for group in bundle['groups']:
        if group['n_admitted']:
            recovered = group['recovered_fraction']
            check(recovered is not None and recovered > 0.5,
                  '{g}: the series accounts for most of V_cl ({r:.1%} recovered)'
                  .format(g=group['label'], r=recovered))

    # Terms must say which multipoles they couple, not only which ranks.
    named = [h for h in headers if 'dipole-dipole [1,1]' in h]
    check(bool(named), 'rank terms are labelled by the multipoles they couple')

    check(bundle['admission']['convergence_gate_applied'],
          'the bundle records that the convergence gate was applied')
    check(bundle['settings']['lmax_used'] == 5, 'the bundle records the L_max actually used')
    n_reg = len(bundle['reg']['rank']['reg'][0])
    check(n_reg == len(headers), 'every term carries a REG value')


def main():
    dataset_dir = tempfile.mkdtemp(prefix='reg_multi_test_')
    try:
        system = synth.write_dataset(dataset_dir)
        test_tensor_against_closed_forms()
        test_series_against_exact_coulomb()
        test_moment_reading(dataset_dir, system)
        test_gates(dataset_dir, system)
        test_moment_translation(system)
        test_fragment_view(dataset_dir)
        test_grouping(system)
        test_end_to_end(dataset_dir)
    finally:
        shutil.rmtree(dataset_dir, ignore_errors=True)

    print('\n' + '=' * 70)
    if FAILURES:
        print('{n} check(s) FAILED:'.format(n=len(FAILURES)))
        for failure in FAILURES:
            print('  - ' + failure)
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
