"""
test_batch.py

Checks for running REG over a folder of systems.  Run it directly:

    python3 tests/test_batch.py

Everything here is answered from a directory tree alone — no wavefunctions are
read — so the checks are fast and say exactly one thing each:

  * which directories of a tree are systems, and what each one is called, over
    the layouts that occur in practice: the config beside a REG-IQA/ folder of
    numbered points, numbered points directly in the system folder, and systems
    nested a level or two below the root;
  * that two systems sharing a folder name are told apart by the folder above;
  * that geometry points are paired with the Gaussian output in their own
    folder, which is what stops a stray reg.log in the tree from shifting every
    point onto the wrong output;
  * that naming a finished results directory after its system renames everything
    inside it, once, and leaves what is already named alone;
  * that a sweep collects every system's bundle and config into one folder under
    the system's own name, and that the overview it writes says what each system
    is and what its analysis found;
  * that a single run does the same for itself, flat, into a folder beside its
    results that can be downloaded whole.
"""

import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), 'src')
sys.path.insert(0, SRC)

import reg_batch                      # noqa: E402
import reg_setup                      # noqa: E402

FAILURES = []


def check(condition, message):
    if condition:
        print('  ok   ' + message)
    else:
        print('  FAIL ' + message)
        FAILURES.append(message)


def equal(got, expected, message):
    check(got == expected, '{m} (got {g!r}, expected {e!r})'.format(
        m=message, g=got, e=expected))


def make_point(path, wavefunction='frame.wfx', output='SP_WFN.out'):
    """A geometry point: a folder with a wavefunction and its Gaussian output."""
    os.makedirs(path, exist_ok=True)
    for name in (wavefunction, output):
        if name:
            open(os.path.join(path, name), 'w').close()
    os.makedirs(os.path.join(path, wavefunction.rsplit('.', 1)[0] + '_atomicfiles'),
                exist_ok=True)


def make_system(path, n_points=3, config=True, points_in=None):
    """A system: numbered points, optionally under a folder, optionally with a config."""
    os.makedirs(path, exist_ok=True)
    if config:
        with open(os.path.join(path, 'auto_reg.config'), 'w') as handle:
            handle.write('FRAG ID 1 <A>\nFRAG ATOMS [1,2]\n')
    holder = os.path.join(path, points_in) if points_in else path
    for step in range(1, n_points + 1):
        make_point(os.path.join(holder, str(step)))
    return path


# ---------------------------------------------------------------------------

def test_discovery(root):
    print('\nwhich directories of a tree are systems')

    # The layout in the wild: the config sits with the system, the points sit a
    # level down in REG-IQA/, and the systems are grouped by ion.
    make_system(os.path.join(root, 'Chlorine', 'CLOBEN'), points_in='REG-IQA')
    make_system(os.path.join(root, 'Chlorine', 'CLOPY'), points_in='REG-IQA')
    # Points directly in the system folder, no config: still a system.
    make_system(os.path.join(root, 'Sodium', 'NABEN'), config=False)
    # A folder that is not a system: one point is not a path.
    make_point(os.path.join(root, 'stray', '1'))

    systems = reg_setup.find_reg_systems(root)
    names = sorted(system['name'] for system in systems)
    equal(names, ['CLOBEN', 'CLOPY', 'NABEN'], 'the three systems are found')
    check(all(system['steps'] == 3 for system in systems),
          'each is found with its three geometry points')
    check(not any(os.path.basename(system['path']) == '1' for system in systems),
          'a geometry point is never mistaken for a system')

    # A claimed system is not searched again from the inside, so its own
    # numbered folders cannot come back as systems of their own.
    claimed = [system['path'] for system in systems]
    equal(len(claimed), len(set(claimed)), 'no system is reported twice')

    by_name = {system['name']: system for system in systems}
    equal(os.path.basename(by_name['CLOBEN']['path']), 'CLOBEN',
          'a system with its points in REG-IQA/ is named after the system folder')
    check(by_name['CLOBEN']['config'] and not by_name['NABEN']['config'],
          'whether a system has an auto_reg.config is reported')


def test_generic_holder_naming(root):
    print('\nnaming a system whose points sit in a folder that describes them')
    # No config to mark the system folder, and the points are in REG-IQA/: the
    # name has to come from the folder above, or every system in a sweep would
    # be called REG-IQA.
    make_system(os.path.join(root, 'SYSX'), config=False, points_in='REG-IQA')
    systems = reg_setup.find_reg_systems(root)
    equal([system['name'] for system in systems], ['SYSX'],
          'the system is named SYSX, not REG-IQA')
    equal([os.path.basename(system['path']) for system in systems], ['REG-IQA'],
          'but it is still analysed in the folder that holds the points')


def test_disambiguation(root):
    print('\ntwo systems that share a folder name')
    make_system(os.path.join(root, 'Chlorine', 'BEN'), points_in='REG-IQA')
    make_system(os.path.join(root, 'Fluorine', 'BEN'), points_in='REG-IQA')
    systems = reg_batch.disambiguate(reg_setup.find_reg_systems(root), root)
    names = sorted(system['name'] for system in systems)
    equal(names, ['Chlorine_BEN', 'Fluorine_BEN'],
          'the folder above tells them apart')

    # Names that cannot be told apart by any folder still have to be unique, or
    # one system would write over the other's results.
    forced = [{'path': os.path.join(root, 'a'), 'name': 'SAME'},
              {'path': os.path.join(root, 'b'), 'name': 'SAME'}]
    reg_batch.disambiguate(forced, root)
    equal(len({system['name'] for system in forced}), 2,
          'a name that cannot be resolved by folder is made unique anyway')


def test_point_pairing(root):
    print('\npairing each geometry point with its own Gaussian output')
    system = make_system(os.path.join(root, 'CLOBEN'), n_points=3, points_in='REG-IQA')
    # The log of an earlier run, sitting where a sweep or a shell redirect leaves
    # it.  It is not in a point folder, so it must not be read as an output.
    with open(os.path.join(system, 'CLOBEN_reg.log'), 'w') as handle:
        handle.write('not a gaussian output\n')

    points = reg_setup.discover_reg_points(system)
    equal(points['reg_folders'], ['1', '2', '3'], 'the three points are in path order')
    check(all(os.path.dirname(wf) == os.path.dirname(g16)
              for wf, g16 in zip(points['wf_files'], points['g16_files'])),
          'every wavefunction is paired with the output in its own folder')
    check(not any(path.endswith('CLOBEN_reg.log') for path in points['g16_files']),
          'a stray log outside the point folders is not taken for an output')

    # A point with no output at all is an error that names the folder, rather
    # than a silent shift of every later point onto the wrong output.
    os.remove(os.path.join(system, 'REG-IQA', '2', 'SP_WFN.out'))
    try:
        reg_setup.discover_reg_points(system)
        check(False, 'a point with no Gaussian output raises')
    except FileNotFoundError as error:
        check('2' in str(error), 'a point with no Gaussian output raises, naming the folder')


def test_output_prefix(root):
    print('\nnaming a finished results directory after its system')
    results = os.path.join(root, 'CLOBEN_REG_IQA_results')
    os.makedirs(results)
    for name in ('REG.xlsx', 'Energy.xlsx', 'REG_final_analysis.csv',
                 'CLOBEN_REG_IQA_model_transfer.json'):
        open(os.path.join(results, name), 'w').close()

    renamed = reg_setup.apply_output_prefix(results, 'CLOBEN')
    listing = sorted(os.listdir(results))
    equal(listing, ['CLOBEN_Energy.xlsx', 'CLOBEN_REG.xlsx',
                    'CLOBEN_REG_IQA_model_transfer.json',
                    'CLOBEN_REG_final_analysis.csv'],
          'every file in the directory carries the system name')
    equal(len(renamed), 3, 'the file already named after the system is left alone')

    # Running the sweep again must not give CLOBEN_CLOBEN_REG.xlsx.
    reg_setup.apply_output_prefix(results, 'CLOBEN')
    equal(sorted(os.listdir(results)), listing, 'naming the directory twice changes nothing')

    equal(reg_setup.sanitise_prefix('Cl pi/run 2'), 'Cl_pi_run_2',
          'a name with spaces and slashes is made safe for a file name')


def write_bundle(path, system, elements, wfn, top_term):
    """A model-transfer bundle with the keys the overview reads, and no more."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bundle = {
        'schema_version': 3,
        'analysis_kind': 'REG_IQA',
        'system': system,
        'primary_level': 'REG_IQA',
        'units': {'energy_to_kj_mol': 2625.5},
        'settings': {'r_threshold': 0.0},
        'availability': {'dispersion': True},
        'control_coordinate_label': 'Control Coordinate [REG step]',
        'control_coordinates': [float(i + 1) for i in range(len(wfn))],
        'atoms': ['a{i}'.format(i=i + 1) for i in range(len(elements))],
        'elements': elements,
        'fragments': {'definitions': [{'name': 'F1', 'atom_numbers': [1, 2]}]},
        'energies': {'wfn': wfn, 'iqa': wfn, 'rmse_kj_mol': 0.02},
        'quality_report': {'missing_files': [], 'bad_lagrangians': []},
        'levels': {'REG_IQA': {
            'entity_kind': 'atom',
            'entity_count': len(elements),
            'segments': {'energy': {'count': 1, 'step_ranges': [[0, len(wfn) - 1]],
                                    'critical_point_indices': []}},
            'reg': {
                'intra': {'headers': ['E_IQA_Intra(A)-a1', top_term],
                          'reg': [[1.0, 9.0]], 'pearson': [[0.5, 0.99]]},
                'inter': {'headers': ['VC_IQA(A,B)-a1_a2'],
                          'reg': [[2.0]], 'pearson': [[0.7]]},
            },
        }},
    }
    with open(path, 'w') as handle:
        json.dump(bundle, handle)


def test_collection(root):
    print('\ncollecting a sweep into one folder with an overview')
    systems = []
    for name, elements, span in (('CLOBEN', ['c'] * 6 + ['h'] * 6 + ['cl'], 0.001),
                                 ('LIBEN', ['c'] * 6 + ['h'] * 6 + ['li'], 0.05)):
        path = make_system(os.path.join(root, name), points_in='REG-IQA')
        write_bundle(os.path.join(path, name + '_REG_IQA_results',
                                  name + '_REG_IQA_model_transfer.json'),
                     name + '_REG_IQA', elements,
                     [-100.0, -100.0 + span / 2, -100.0 + span],
                     'E_IQA_Intra(A)-' + name)
        # An earlier run of the same analysis, from before the outputs were named
        # after the system. It is the same kind of bundle, so the newer one above
        # stands in for it rather than both being collected.
        older = os.path.join(path, 'REG_IQA_results', 'REG_IQA_model_transfer.json')
        write_bundle(older, 'REG_IQA', elements, [-100.0, -100.0, -100.0], 'old')
        os.utime(older, (1, 1))
        systems.append({'path': path, 'name': name, 'steps': 3, 'config': True})

    results = [{'system': system, 'returncode': 0, 'minutes': 1.5, 'log': None}
               for system in systems]
    overview_path = reg_batch.collect_sweep(results, root, level='REG_IQA')
    collection = os.path.dirname(overview_path)

    bundles = sorted(os.listdir(os.path.join(collection, 'bundles')))
    equal(bundles, ['CLOBEN_REG_IQA_model_transfer.json',
                    'LIBEN_REG_IQA_model_transfer.json'],
          'the newest bundle of each kind is collected, one per system')
    equal(sorted(os.listdir(os.path.join(collection, 'configs'))),
          ['CLOBEN_auto_reg.config', 'LIBEN_auto_reg.config'],
          'each system config is collected under the system name')

    overview = json.load(open(overview_path))
    equal(overview['counts'], {'systems': 2, 'analysed': 2, 'failed': 0, 'not_run': 0,
                               'with_bundle': 2}, 'the overview counts the sweep')
    by_name = {entry['name']: entry for entry in overview['systems']}
    equal(by_name['CLOBEN']['primary_bundle'],
          'bundles/CLOBEN_REG_IQA_model_transfer.json',
          "the run's own bundle is the primary one, not an older one")
    superseded = by_name['CLOBEN']['superseded_bundles']
    equal([entry['kind'] for entry in superseded], ['REG_IQA'],
          'the older bundle of the same kind is recorded as passed over')
    check(superseded[0]['path'].endswith('REG_IQA_results/REG_IQA_model_transfer.json')
          and 'written' in superseded[0],
          'and the overview says where it is and when it was written')
    check(all(os.path.isfile(os.path.join(collection, entry['primary_bundle']))
              for entry in overview['systems']),
          'every path in the overview resolves from the collection folder')

    summary = by_name['LIBEN']['summary']
    equal(summary['formula'], 'C6H6Li', 'the formula is read from the bundle')
    equal(summary['n_atoms'], 13, 'the atom count is read from the bundle')
    check(abs(summary['energy']['wfn_span_kj_mol'] - 0.05 * 2625.5) < 1e-6,
          'the energy span is reported in kJ/mol')
    equal(summary['top_reg_terms'][0]['term'], 'E_IQA_Intra(A)-LIBEN',
          'the leading REG term of the segment is shortlisted')
    equal(summary['segments']['count'], 1, 'the segmentation is carried over')

    with open(os.path.join(collection, reg_batch.OVERVIEW_NAME + '.csv')) as handle:
        rows = handle.read().strip().split('\n')
    equal(len(rows), 3, 'the overview CSV has a header and one row per system')
    check(rows[0].startswith('system,status,path,level,n_atoms,formula'),
          'the CSV columns are the scalar fields of the overview')

    # A bundle from an older schema, or one whose run produced less than usual,
    # must give a thinner summary rather than stopping the collection.
    thin = reg_batch.summarise_reg_bundle({'schema_version': 2, 'atoms': ['a1']})
    check(isinstance(thin, dict) and thin['n_atoms'] == 1 and thin['formula'] is None,
          'a bundle missing most of its keys summarises to what it does have')
    check(reg_batch.summarise_reg_bundle(None) is None,
          'no bundle at all summarises to nothing')


def test_single_run_collection(root):
    print('\ncollecting a single run into one folder to take away')
    system = make_system(os.path.join(root, 'CLOPY'), points_in='REG-IQA')
    elements = ['c'] * 5 + ['h'] * 5 + ['n', 'cl']
    for results_dir, bundle_name in (('REG_IQA_results', 'REG_IQA_model_transfer.json'),
                                     ('REG_IQF_results', 'REG_IQF_model_transfer.json'),
                                     ('REG_Multi_IQA_results',
                                      'REG_Multi_IQA_model_transfer.json')):
        write_bundle(os.path.join(system, results_dir, bundle_name),
                     bundle_name.replace('_model_transfer.json', ''), elements,
                     [-100.0, -99.9, -99.8], 'E_IQA_Intra(A)-cl12')

    # The pre-rename spelling of the same analysis, older than the bundles above:
    # a level-specific bundle stands in for it.
    legacy = os.path.join(system, 'REG_results', 'REG_model_transfer.json')
    write_bundle(legacy, 'REG', elements, [-100.0, -99.9, -99.8], 'legacy')
    os.utime(legacy, (1, 1))

    overview_path = reg_batch.collect_run(system, level='REG_IQF')
    collection = os.path.dirname(overview_path)
    equal(os.path.basename(collection), 'REG_collection',
          'an unprefixed run collects into REG_collection beside its results')
    equal(sorted(os.listdir(collection)),
          ['CLOPY_REG_IQA_model_transfer.json',
           'CLOPY_REG_IQF_model_transfer.json',
           'CLOPY_REG_Multi_IQA_model_transfer.json',
           'CLOPY_auto_reg.config',
           'REG_collection_overview.csv',
           'REG_collection_overview.json'],
          'every bundle and the config sit flat in it, named after the system')

    overview = json.load(open(overview_path))
    equal(overview['scope'], 'run', 'the overview says this is one run, not a sweep')
    entry = overview['systems'][0]
    equal(entry['primary_bundle'], 'CLOPY_REG_IQF_model_transfer.json',
          'the level the run was asked for is the primary bundle')
    equal(entry['path'], 'CLOPY', 'the system is named by its folder, not as "."')
    check(entry['multipole'] is not None and entry['summary'] is not None,
          'both the REG and the multipole sides are summarised')
    equal([item['kind'] for item in entry['superseded_bundles']], ['REG'],
          'a bundle from the level-agnostic naming is passed over for a newer level')

    # Running something else later rebuilds the folder rather than leaving it
    # stale, and a prefixed run keeps its prefix on the folder too.
    write_bundle(os.path.join(system, 'REG_Multi_results',
                              'REG_Multi_model_transfer.json'),
                 'REG_Multi', elements, [-100.0, -99.9, -99.8], 'later')
    reg_batch.collect_run(system, level='REG_IQF')
    check('CLOPY_REG_Multi_model_transfer.json' in os.listdir(collection),
          'a later run refreshes the folder instead of leaving it out of date')

    prefixed_path = reg_batch.collect_run(system, name='CLOPY', dir_prefix='CLOPY')
    equal(os.path.basename(os.path.dirname(prefixed_path)), 'CLOPY_REG_collection',
          'a run named after its system names the folder too')


# ---------------------------------------------------------------------------

def main():
    print('=' * 70)
    print('REG batch-mode checks')
    print('=' * 70)

    for test in (test_discovery, test_generic_holder_naming, test_disambiguation,
                 test_point_pairing, test_output_prefix, test_collection,
                 test_single_run_collection):
        root = tempfile.mkdtemp(prefix='reg_batch_test_')
        try:
            test(root)
        finally:
            shutil.rmtree(root, ignore_errors=True)

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
