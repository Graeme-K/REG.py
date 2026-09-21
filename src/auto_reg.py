"""
auto_reg.py v0.1
F. Falcioni, P. L. A. Popelier

Library with function to run a REG analysis
Check for updates at github.com/FabioFalcioni
For details about the method, please see XXXXXXX

Please, report bugs and issues to fabio.falcioni@manchester.ac.uk
coded by F.Falcioni

NOTE: The automatic analysis works if this file is run with python3 inside a folder containing all the REG points (
saved in numbered folders)

It can also be pointed at a folder of such folders with -R, which runs a full
analysis on every system below it and names each run's output after its system.
"""

# IMPORT LIBRARIES
import json
import os
import re
import shlex
import sys
import time
from optparse import OptionParser

import numpy as np


# Numbers in the transfer bundle are rounded to this many decimal places.  The
# bundle is written in hartree, so 1e-10 Ha is 2.6e-7 kJ/mol — far below any
# quantity the analysis reports, while cutting roughly a third off the file size
# of the per-pair term arrays.
_BUNDLE_DECIMALS = 10

HA_TO_KJ = 2625.5


def _json_ready(obj, decimals=_BUNDLE_DECIMALS):
    """Convert numpy/python data into something json.dump can write literally.

    NaN and +/-Inf become null.  Python's json module happily writes bare NaN
    and Infinity tokens, which are not valid JSON and make JSON.parse throw in
    a browser — so anything that reads the bundle from a web page would fail on
    a single missing integration.  Floats are rounded to *decimals* places.
    """
    if isinstance(obj, dict):
        return {str(k): _json_ready(v, decimals) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v, decimals) for v in obj]
    if hasattr(obj, 'tolist'):
        return _json_ready(obj.tolist(), decimals)
    if isinstance(obj, bool):
        return obj
    if hasattr(obj, 'item') and not isinstance(obj, (str, bytes)):
        try:
            return _json_ready(obj.item(), decimals)
        except Exception:
            pass
    if isinstance(obj, float):
        if obj != obj or obj in (float('inf'), float('-inf')):  # NaN / +-Inf
            return None
        return round(obj, decimals)
    return obj


def _read_fragment_config(path):
    """Read the fragment definitions in auto_reg.config.

    Delegates to reg_setup so that an IQF run and a REG_Multi run cannot drift
    apart on what a fragment is.  The import is deferred to call time because
    this script is meant to be copied into the data directory and pointed at its
    installation with -d, so the package is only on sys.path once main() has run.
    """
    import reg_setup  # type: ignore
    return reg_setup.read_fragment_config(path)


def _term_block(headers, values, properties=None, extra=None):
    """Package a term table (one row per header, one column per step)."""
    if values is None or headers is None:
        return None
    block = {
        'headers': _json_ready(headers),
        'values': _json_ready(values),
    }
    if properties is not None:
        block['properties'] = list(properties)
    if extra:
        block.update(extra)
    return block


def _reg_block(reg_result, headers):
    """Package a reg.reg() result together with the labels of its rows.

    reg.reg returns (REG_values, pearson_values) indexed [segment][term], with
    no record of which term each column belongs to.  Pairing them with the
    headers here is what lets a reader rank pairs by name instead of having to
    reverse-engineer the pair-block layout from the array length.
    """
    if reg_result is None or headers is None:
        return None
    return {
        'headers': _json_ready(headers),
        'reg': _json_ready(reg_result[0]),
        'pearson': _json_ready(reg_result[1]),
    }


def _pair_list(atoms):
    """Canonical atom-pair ordering used by aimall_utils for inter-atomic terms."""
    return [(i, j) for i in range(len(atoms)) for j in range(i + 1, len(atoms))]


def _select_pairs_within_budget(pair_term_groups, n_pairs, budget_ha):
    """Drop the least active atom pairs while bounding what dropping them loses.

    *pair_term_groups* is a list of (n_props * n_pairs, n_steps) arrays whose rows
    run property-major over the canonical pair ordering, i.e. exactly the layout
    get_iqa_properties returns.

    The obvious rule — drop every pair whose own peak-to-peak variation is below
    some threshold — is wrong at scale.  Thousands of pairs that each move by less
    than the threshold still sum to a real drift along the path: on a 238-atom
    system, 8105 pairs each under 0.01 kJ/mol together drifted by 2.71 kJ/mol,
    about a sixth of the total energy change.  A per-pair test bounds each row and
    says nothing about their sum, which is the quantity that actually matters.

    So pairs are dropped least-active-first while accumulating the residual, and
    the cut is made before the residual's own peak-to-peak variation would exceed
    *budget_ha* in any channel.  What the bundle leaves out is then bounded by the
    budget directly.  A pair with unreadable values has infinite activity and is
    never dropped, so gaps stay visible.

    Returns (kept_pair_indices, residual_drift_ha).  budget_ha <= 0 keeps all.
    """
    blocks = []
    for group in pair_term_groups:
        if group is None:
            continue
        arr = np.asarray(group, dtype=float)
        if arr.ndim != 2 or n_pairs == 0 or arr.shape[0] % n_pairs != 0 or arr.shape[1] == 0:
            continue
        n_props = arr.shape[0] // n_pairs
        for p_i in range(n_props):
            blocks.append(arr[p_i * n_pairs:(p_i + 1) * n_pairs])

    if not blocks or budget_ha is None or budget_ha <= 0:
        return list(range(n_pairs)), 0.0

    # Activity per pair: the largest peak-to-peak variation it shows in any channel.
    activity = np.zeros(n_pairs)
    for block in blocks:
        spread = np.nanmax(block, axis=1) - np.nanmin(block, axis=1)
        activity = np.maximum(activity, np.nan_to_num(spread, nan=np.inf, posinf=np.inf))

    order = np.argsort(activity, kind='stable')          # least active first
    n_droppable = int(np.count_nonzero(np.isfinite(activity)))

    # Cumulative residual after dropping the first k pairs of *order*, vectorised:
    # cums[k] is the summed curve of those k+1 pairs, and drift[k] its spread.
    drift = np.zeros(n_pairs)
    for block in blocks:
        cums = np.cumsum(np.nan_to_num(block[order]), axis=0)
        drift = np.maximum(drift, cums.max(axis=1) - cums.min(axis=1))

    over = drift > budget_ha
    # Cut at the first violation rather than the last prefix that happens to fit, so
    # the kept set never depends on a dip in a non-monotonic residual.
    n_drop = int(np.argmax(over)) if over.any() else n_pairs
    n_drop = min(n_drop, n_droppable)

    if n_drop <= 0:
        return list(range(n_pairs)), 0.0

    dropped = set(order[:n_drop].tolist())
    kept = [i for i in range(n_pairs) if i not in dropped]
    residual_drift = float(drift[n_drop - 1])
    return kept, residual_drift


def _prune_pair_terms(values, headers, n_pairs, kept_pairs):
    """Restrict a property-major pair term table to *kept_pairs*.

    The rows that are dropped are not simply discarded: their per-step sum is
    returned per property as a residual, so a reader can still reconstruct the
    exact total of every property from what the bundle contains.
    """
    arr = np.asarray(values, dtype=float)
    headers = list(headers)
    if arr.ndim != 2 or n_pairs == 0 or arr.shape[0] % n_pairs != 0:
        return list(values), headers, {}, []
    if len(kept_pairs) == n_pairs:
        return list(values), headers, {}, list(range(n_pairs))

    n_props = arr.shape[0] // n_pairs
    dropped = sorted(set(range(n_pairs)) - set(kept_pairs))

    kept_rows = []
    kept_headers = []
    residuals = {}
    for p_i in range(n_props):
        block = arr[p_i * n_pairs:(p_i + 1) * n_pairs]
        for pair_i in kept_pairs:
            kept_rows.append(block[pair_i])
            kept_headers.append(headers[p_i * n_pairs + pair_i])
        prop_label = headers[p_i * n_pairs].rsplit('-', 1)[0] if headers else 'prop_' + str(p_i)
        residuals[prop_label] = np.nansum(block[dropped], axis=0) if dropped else np.zeros(arr.shape[1])

    return kept_rows, kept_headers, residuals, list(kept_pairs)


def _residual_by_fragment_pair(values, headers, n_pairs, dropped_pairs, pair_index,
                               frag_lists, frag_names):
    """Break the dropped-pair residual down by which fragment pair it belongs to.

    Pruning costs attribution, not energy: the residual keeps every total exact, but
    a single lumped curve cannot say *where* the omitted drift sits.  Splitting it
    over fragment pairs restores that at fragment resolution for a few hundred
    numbers, so the omitted part can still be placed on the map even though the
    individual atom pairs behind it were not written.

    Returns {property: {"F|G": [per-step sum]}} or None when no fragments are known.
    """
    if not frag_lists or not dropped_pairs:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or n_pairs == 0 or arr.shape[0] % n_pairs != 0:
        return None

    atom_frag = {}
    for f_i, frag_atoms in enumerate(frag_lists):
        for a in frag_atoms:
            atom_frag[a - 1] = f_i

    groups = {}
    for k in dropped_pairs:
        i, j = pair_index[k]
        f_i, f_j = atom_frag.get(i), atom_frag.get(j)
        if f_i is None or f_j is None:
            key = 'unassigned'
        else:
            lo, hi = (f_i, f_j) if f_i <= f_j else (f_j, f_i)
            key = str(frag_names[lo]) + '|' + str(frag_names[hi])
        groups.setdefault(key, []).append(k)

    headers = list(headers)
    n_props = arr.shape[0] // n_pairs
    out = {}
    for p_i in range(n_props):
        block = arr[p_i * n_pairs:(p_i + 1) * n_pairs]
        prop_label = headers[p_i * n_pairs].rsplit('-', 1)[0] if headers else 'prop_' + str(p_i)
        out[prop_label] = {key: np.nansum(block[rows], axis=0) for key, rows in groups.items()}
    return out


def _pair_distances(structures, pair_index):
    """Per-step A-B distances (Angstrom) for the given pairs.

    Lets a reader plot how far each term reaches without re-deriving the
    geometry; returns None when the XYZ files were unavailable.
    """
    if not structures or not pair_index:
        return None
    coords = []
    for structure in structures:
        atom_xyz = structure.get('atoms') or []
        coords.append(np.array([[a['x'], a['y'], a['z']] for a in atom_xyz], dtype=float))
    if not coords or any(c.size == 0 for c in coords):
        return None

    n_atoms = min(c.shape[0] for c in coords)
    distances = []
    for (i, j) in pair_index:
        if i >= n_atoms or j >= n_atoms:
            distances.append([None] * len(coords))
            continue
        distances.append([float(np.linalg.norm(c[i] - c[j])) for c in coords])
    return distances


def _load_xyz_structure(path):
    """Read one XYZ file; see reg_setup.load_xyz_structure."""
    import reg_setup  # type: ignore
    return reg_setup.load_xyz_structure(path)

def sum_into_fragments(fragment_names,fragment_atom_list,atoms,int_prop_skp=True,inter_terms=[],inter_prop=[],inp_iqf_intra=[],intra_terms=[],intra_prop=[]):
    """
    ###########################################################################################################
    FUNCTION: sum_iqa_into_fragments
              Adds intra and inter terms together into fragment energies.

    INPUT: fragment_names,fragment_atom_list,inter_terms,inter_headers,intra_terms,intra_headers
        fragment_names      : List of fragment labels
        fragment_atom_list  : List of atoms (number only) in each fragment
        inter_terms         : Interatomic terms as numpy array
        inter_prop          : List of interatomic properties
        intra_terms         : Intraatomic terms as numpy array
        intra_prop          : List of intraatomic properties
        int_prop_skp        : True, skips the last inter prop when summing inter properties, required if you have
                              [Vxc,Vcl,E_inter], adding all energies together results in double counting

    OUTPUT: [iqf_intra,iqf_intra_headers,iqf_inter,iqf_inter_headers]
        iqf_inter           : Inter-fragment terms as numpy array
        iqf_inter_headers   : Headers for inter-fragment terms as numpy array
        iqf_inter_comps     : The terms that make up the total inter-fragment term
        iqf_inter_comp_head : Headers for terms making up total inter-fragment term
        iqf_intra           : Intra-fragment terms as numpy array
        iqf_intra_headers   : Headers for intra-fragment terms as numpy array
        iqf_intra_comps     : The terms that make up the total intra-fragment term
        iqf_intra_comp_head : Headers for terms making up total intra-fragment term


    ERROR:

    ###########################################################################################################
    """
    # Obtain number of interatomic properties
    n_prop = len(inter_prop)
    # Work out number of inter-atomic terms per property
    no_inter =int(len(inter_terms) / n_prop)
    # The number of fragments that the atomic properties will be summed into
    N_frag = int(len(fragment_atom_list))
    # Number of frag - frag interactions
    N_FF_int = int((N_frag*(N_frag - 1)) / 2)

    # Summing all intra terms into fragments
    iqf_intra = []
    iqf_intra_header = []
    iqf_intra_comps = [ [] for frag in fragment_atom_list]
    iqf_intra_comp_head = [ [] for frag in fragment_atom_list]

    f_indx = 0
    if len(intra_prop) > 0:
        iqf_intra = []
        iqf_intra_header = []
        for fragment,frag_nam in zip(fragment_atom_list,fragment_names):
            fragment = np.sort(fragment)
            frag_e = intra_terms[(int(fragment[0]) - 1)].copy()
            iqf_intra_comps[f_indx].append(intra_terms[(int(fragment[0]) - 1)].copy())
            iqf_intra_comp_head[f_indx].append(str(atoms[fragment[0]-1]))
            for atom_ind in fragment[1:]:
                frag_e += intra_terms[(int(atom_ind) - 1)].copy()
                iqf_intra_comps[f_indx].append(intra_terms[(int(atom_ind) - 1)].copy())
                iqf_intra_comp_head[f_indx].append(str(atoms[atom_ind - 1]))
            iqf_intra.append(frag_e)
            iqf_intra_header.append(str(intra_prop[0]) + "_" + str(frag_nam))
            f_indx += 1

    if (len(inp_iqf_intra) > 0) and (len(intra_prop) == 0):
        iqf_intra = inp_iqf_intra[0]
        iqf_intra_comps = inp_iqf_intra[1]
        iqf_intra_comp_head = inp_iqf_intra[2]
    else:
        iqf_intra = np.array(iqf_intra)

    # Summing all inter terms into intra-fragment terms and inter-fragment terms
    # Setting the index to skip when summing intra terms
    if int_prop_skp:
        prop_skp_no = int(n_prop-1)
    else:
        prop_skp_no = int(-1)
    
    iqf_inter = np.zeros((int(N_FF_int * n_prop),int(len(inter_terms[0]))),dtype=float)
    iqf_inter_comps = [[] for _ in range(int(N_FF_int * n_prop))]
    iqf_inter_comp_head = [[] for _ in range(int(N_FF_int * n_prop))]
    iqf_inter_header = []

    # Iterating through fragments to obtain final intra and inter terms
    for f1_indx in range(len(fragment_atom_list)):
        for f2_indx in range(len(fragment_atom_list)):
            for atom1 in fragment_atom_list[f1_indx]:
                for atom2 in fragment_atom_list[f2_indx]:
                    for prop_indx in range(n_prop):
                        if (f1_indx == f2_indx) and (atom1 < atom2) and (prop_indx != prop_skp_no):
                            iqf_intra[f1_indx] += inter_terms[int((prop_indx*no_inter)+((atom1-1)*(2*len(atoms)-atom1))/2+(atom2-atom1-1))]
                            iqf_intra_comps[f1_indx].append(inter_terms[int((prop_indx*no_inter)+((atom1-1)*(2*len(atoms)-atom1))/2+(atom2-atom1-1))])
                            iqf_intra_comp_head[f1_indx].append(str(atoms[atom1 - 1]) + "-" + str(atoms[atom2 - 1]) + "_" + str(inter_prop[prop_indx]))
                        elif (f1_indx != f2_indx) and (atom1 < atom2):
                            F1_ID = f1_indx + 1
                            F2_ID = f2_indx + 1
                            if F1_ID > F2_ID:
                                F1_ID, F2_ID = F2_ID, F1_ID
                            iqf_inter[int((prop_indx*N_FF_int)+((F1_ID-1)*(2*N_frag-F1_ID))/2+(F2_ID-F1_ID-1))] += inter_terms[int((prop_indx*no_inter)+((atom1-1)*(2*len(atoms)-atom1))/2+(atom2-atom1-1))]
                            iqf_inter_comps[int((prop_indx*N_FF_int)+((F1_ID-1)*(2*N_frag-F1_ID))/2+(F2_ID-F1_ID-1))].append(inter_terms[int((prop_indx*no_inter)+((atom1-1)*(2*len(atoms)-atom1))/2+(atom2-atom1-1))])
                            iqf_inter_comp_head[int((prop_indx*N_FF_int)+((F1_ID-1)*(2*N_frag-F1_ID))/2+(F2_ID-F1_ID-1))].append(str(atoms[atom1 - 1]) + "-" + str(atoms[atom2 - 1]) + "_" + str(inter_prop[prop_indx]))

    # Creating list of iqf inter headers
    for prop in inter_prop:
        for f1_indx in range(len(fragment_atom_list)):
            for f2_indx in range((f1_indx + 1),len(fragment_atom_list)):   
                iqf_inter_header.append(str(prop) + "_" + str(fragment_names[f1_indx] + "_" + fragment_names[f2_indx]))

    return iqf_inter, iqf_inter_header, iqf_inter_comps, iqf_inter_comp_head, iqf_intra, iqf_intra_header, iqf_intra_comps, iqf_intra_comp_head

def compute_entity_totals(entities, iqa_intra, iqa_intra_header, iqa_inter, iqa_inter_header,
                           intra_prop_total, inter_prop_total, prop_sep):
    """
    ###########################################################################################################
    FUNCTION: compute_entity_totals
              Computes, per atom or fragment, the IQA total energy E_intra(A) + sum_B (1/2 * E_inter(A,B))
              across all geometry points, for use in the REG_IQA ranking output.

    INPUT: entities, iqa_intra, iqa_intra_header, iqa_inter, iqa_inter_header, intra_prop_total,
           inter_prop_total, prop_sep
        entities          : list of atom or fragment labels to rank
        iqa_intra         : intra-atomic/fragment terms as numpy array (rows = TERM, cols = geometry points)
        iqa_intra_header  : headers matching iqa_intra rows
        iqa_inter         : inter-atomic/fragment terms as numpy array (rows = TERM, cols = geometry points)
        iqa_inter_header  : headers matching iqa_inter rows
        intra_prop_total  : name of the intra-atomic property representing the total (e.g. 'E_IQA_Intra(A)')
        inter_prop_total  : name of the inter-atomic property representing the total (e.g. 'E_IQA_Inter(A,B)')
        prop_sep          : separator used between the property name and the entity label(s) in the headers
                             ('-' for atoms, '_' for fragments)

    OUTPUT: entity_totals
        entity_totals : numpy array, one row per entity, E_intra(A) + sum_B (1/2 * E_inter(A,B))

    ERROR:

    ###########################################################################################################
    """
    intra_index = {h: idx for idx, h in enumerate(iqa_intra_header)}
    inter_index = {h: idx for idx, h in enumerate(iqa_inter_header)}
    n_entities = len(entities)
    n_steps = iqa_intra.shape[1]
    entity_totals = np.zeros((n_entities, n_steps))
    for i, ent in enumerate(entities):
        intra_h = intra_prop_total + prop_sep + str(ent)
        if intra_h in intra_index:
            entity_totals[i] = iqa_intra[intra_index[intra_h]].astype(float)
    for i in range(n_entities):
        for j in range(i + 1, n_entities):
            inter_h = inter_prop_total + prop_sep + str(entities[i]) + '_' + str(entities[j])
            if inter_h in inter_index:
                pair_val = iqa_inter[inter_index[inter_h]].astype(float)
                entity_totals[i] += 0.5 * pair_val
                entity_totals[j] += 0.5 * pair_val
    return entity_totals


def _safe_sheet_name(name, used_names, max_len=31):
    """Return an Excel-safe sheet name (<= 31 chars, unique within used_names)."""
    if len(name) <= max_len and name not in used_names:
        used_names.add(name)
        return name
    base = name[:max_len - 3]
    for idx in range(1, 10000):
        candidate = base + '~' + str(idx)
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
    raise RuntimeError("Could not generate a unique sheet name for: " + name)


def _single_system_argv(option, reg_dir, prefix):
    """The command line one system of a -R sweep is analysed with.

    Rebuilt from the parsed options rather than by editing the sweep's own argv,
    so that what each system runs is explicit and can be read straight off the
    log.  A new option that should reach the systems belongs here as well as in
    the parser.
    """
    argv = ['-d', reg_dir]
    if option.IQF_TF:
        argv += ['-f', option.IQF_TF]
    if option.use_zeros:
        argv.append('-z')
    if option.table_rows:
        argv += ['-n', str(option.table_rows)]
    if option.multipole_visualization:
        argv.append('--multipole-visualization')
    if option.bundle_pair_budget is not None:
        argv += ['--bundle-pair-budget', repr(float(option.bundle_pair_budget))]
    if option.bundle_indent:
        argv += ['--bundle-indent', str(option.bundle_indent)]
    if option.reg_multi:
        argv.append('-m')
    if option.ignore_fragments:
        argv.append('-i')
    if option.multi_options:
        argv += ['--multi-options', option.multi_options]
    if prefix:
        argv += ['--prefix', prefix]
    # The sweep collects every system centrally at its root, so a system does not
    # also copy its own bundles into a folder of its own.
    argv.append('--no-collect')
    return argv


def _run_recursive(option, args, reg_dir):
    """Run a full REG analysis on every system below a root directory.

    Returns a process exit code: 0 when every system was analysed, 1 when any
    of them failed.
    """
    import reg_batch  # type: ignore
    import reg_setup  # type: ignore

    root = os.path.abspath(args[0]) if args else os.getcwd()
    if not os.path.isdir(root):
        raise NotADirectoryError('Not a directory: ' + root)

    systems = reg_batch.disambiguate(reg_setup.find_reg_systems(root), root)
    level = 'REG_IQF' if option.IQF_TF == 'T' else 'REG_IQA'

    def expected_dirs(system):
        """The results directories a complete run on this system would leave behind."""
        prefix = '' if option.no_prefix else system['name']
        names = [reg_setup.prefixed(level + '_results', prefix)]
        if option.reg_multi:
            names.append(reg_setup.prefixed(
                'REG_Multi_' + level.replace('REG_', '') + '_results', prefix))
        return names

    def results_dirs(system):
        """Of those, the ones this system already holds."""
        return [name for name in expected_dirs(system)
                if os.path.isdir(os.path.join(system['path'], name))]

    def is_complete(system):
        """Skipping needs every directory the run would produce, not just one of
        them: a system whose IQA finished but whose REG_Multi did not is not done."""
        return len(results_dirs(system)) == len(expected_dirs(system))

    reg_batch.print_plan(systems, root, results_names=results_dirs)
    if option.list_systems:
        return 0
    if not systems:
        return 1

    if option.skip_existing:
        keep = [system for system in systems if not is_complete(system)]
        if len(keep) != len(systems):
            print('  --skip-existing: {n} system(s) already analysed and left alone'.format(
                n=len(systems) - len(keep)))
            print('')
        systems = keep
        if not systems:
            print('  Nothing left to do.')
            return 0

    def collect(results):
        if option.no_collect:
            return
        reg_batch.collect_sweep(results, root,
                                dir_name=option.collect_dir or reg_batch.COLLECTION_DIR,
                                level=level, command=[sys.argv[0]] + list(sys.argv[1:]))

    if option.collect_only:
        # Nothing is analysed: the systems are taken as they stand, so an overview
        # can be built over a tree that was swept days ago.
        collect([{'system': system, 'returncode': None, 'minutes': None, 'log': None}
                 for system in systems])
        return 0

    script = os.path.abspath(__file__)
    logs = reg_batch.log_directory(root)
    print('  Logs: ' + logs)
    print('')

    def command_for(system):
        prefix = '' if option.no_prefix else system['name']
        return [sys.executable, script] + _single_system_argv(option, reg_dir, prefix)

    def log_path_for(system):
        return os.path.join(logs, system['name'] + '_reg.log')

    results = reg_batch.run_batch(systems, command_for, log_path_for, jobs=option.jobs)
    collect(results)
    return reg_batch.exit_code(results)


def main(argv=None):
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-d", "--directory", action='store', type='string', dest='reg_dir',
                        help="PLEASE INSERT THE PATH OF REG.py folder installation")
    parser.add_option("-f", "--IQF", action='store', type='string', dest='IQF_TF',
                        help="Select T or F based on if you want to run IQF or not")
    parser.add_option("-z", "--zeros", action='store_true', dest='use_zeros', default=False,
                        help="Continue REG analysis using zeros for missing or poor-quality atoms instead of aborting")
    parser.add_option("-n", "--table-rows", action='store', type='int', dest='table_rows', default=None,
                        help="Smallest acceptable size for the REG_final table, overriding MIN_TABLE_ROWS in "
                             "default_settings.py. The |REG| cut is loosened until at least this many terms "
                             "qualify, so raising it grows the table. Growth is stepped, not smooth: the cut "
                             "moves in rungs, so nearby values can give the same table. The row cap is lifted "
                             "to 3x this value so it cannot truncate the result. Per-property tables are "
                             "unaffected; they keep PROPERTY_MIN_TABLE_ROWS")
    parser.add_option("--multipole-visualization", action='store_true', dest='multipole_visualization', default=False,
                        help="Generate direct charge / dipole / quadrupole diagnostics from AIMAll .sum tables")
    parser.add_option("--bundle-pair-budget", action='store', type='float', dest='bundle_pair_budget',
                        default=0.05,
                        help="How much drift (kJ/mol, peak-to-peak along the path) may be left out of the "
                             "model-transfer bundle's per-pair energy curves in total. The least active atom "
                             "pairs are dropped until this budget is reached and their combined per-step sum is "
                             "stored as a residual. Use 0 to write every pair (default: 0.05)")
    parser.add_option("--bundle-indent", action='store', type='int', dest='bundle_indent', default=0,
                        help="Indentation for the model-transfer JSON bundle; 0 writes it compact (default: 0)")
    parser.add_option("-m", "--reg-multi", action='store_true', dest='reg_multi', default=False,
                        help="Also run the REG_Multi analysis: REG over the ranks of the multipole "
                             "expansion of V_cl (charge-charge, charge-dipole, dipole-dipole, ...) "
                             "between the fragments defined in auto_reg.config. Results go to "
                             "REG_Multi_results/ beside this run's. Needs AIMAll to have been run "
                             "with IQA, which this analysis needs anyway. It follows the level of "
                             "the run: -f T sums atom pairs into fragment channels the way IQF sums "
                             "the IQA terms, -f F reports one set of terms per atom pair. See "
                             "REG_MULTI.md")
    parser.add_option("-i", "--ignore-fragments", action='store_true', dest='ignore_fragments',
                        default=False,
                        help="REG_Multi only: run it as if auto_reg.config defined no fragments, so "
                             "which atom pairs get a multipole description is decided by the "
                             "admission gates alone rather than by the partition. Has no effect on "
                             "the IQA/IQF analysis itself")
    parser.add_option("--multi-options", action='store', type='string', dest='multi_options',
                        default='',
                        help="Extra flags passed straight to reg_multipole, e.g. "
                             "--multi-options='--scope all --lmax 4'. Per-system settings are "
                             "better put in auto_reg.config as MULTI lines, so that the same "
                             "command reproduces the same analysis from the directory alone")
    parser.add_option("-R", "--recursive", action='store_true', dest='recursive', default=False,
                        help="Run a full REG analysis on every system in a folder of systems, "
                             "each one exactly as if you had cd'd into it and run this command "
                             "there. The root folder is given as the argument, or is the current "
                             "directory if none is given. A system is a directory holding an "
                             "auto_reg.config with numbered geometry folders beneath it, or one "
                             "holding at least two numbered folders that each contain a "
                             ".wfn/.wfx. Every folder and file the run creates is named after the "
                             "system, so the results of the whole sweep can be collected in one "
                             "place. All the other options apply to each system in turn")
    parser.add_option("--list-systems", action='store_true', dest='list_systems', default=False,
                        help="With -R: list the systems that would be analysed, and the name "
                             "each one's output would carry, then stop without analysing anything")
    parser.add_option("--skip-existing", action='store_true', dest='skip_existing', default=False,
                        help="With -R: leave alone any system that already holds every results "
                             "directory this run would produce, so an interrupted sweep can be "
                             "restarted without redoing the systems it finished. A system whose "
                             "IQA finished but whose REG_Multi did not is redone")
    parser.add_option("-j", "--jobs", action='store', type='int', dest='jobs', default=1,
                        help="With -R: how many systems to analyse at the same time (default: 1). "
                             "Each runs in its own process; with more than one, each system's "
                             "output goes to its own log file instead of the terminal")
    parser.add_option("--prefix", action='store', type='string', dest='prefix', default=None,
                        help="Put NAME in front of every folder and file this run creates: "
                             "NAME_REG_IQA_results/, NAME_REG.xlsx, NAME_REG_IQA_model_transfer.json "
                             "and so on. -R sets this to the system's folder name for each system, "
                             "so it is only needed for a single run analysed on its own")
    parser.add_option("--no-prefix", action='store_true', dest='no_prefix', default=False,
                        help="With -R: keep the usual unprefixed output names in each system's "
                             "own directory instead of naming them after the system")
    parser.add_option("--collect-dir", action='store', type='string', dest='collect_dir',
                        default=None,
                        help="Name the folder this run's model-transfer bundles and "
                             "auto_reg.config are copied into, together with an overview file "
                             "describing them (default: REG_collection in the system's own "
                             "directory, or REG_sweep_collection at the root of a -R sweep). An "
                             "absolute path puts it anywhere")
    parser.add_option("--no-collect", action='store_true', dest='no_collect', default=False,
                        help="Do not gather the bundles and config into a folder of their own; "
                             "leave the results where they are")
    parser.add_option("--collect-only", action='store_true', dest='collect_only', default=False,
                        help="Analyse nothing, and gather what is already on disk into the "
                             "collection folder: with -R every system below the root, otherwise "
                             "the one directory given as the argument or the current one. Builds "
                             "the folder and its overview for work that was run earlier")
    # Note for anyone adding options here: an option a -R sweep should hand to each
    # system belongs in _single_system_argv() as well as in this parser.

    (option, args) = parser.parse_args(args=argv)

    reg_dir = option.reg_dir or os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(1, reg_dir)  # PLEASE INSERT THE PATH OF REG.py folder installation using -d option

    import pandas as pd # type: ignore

    import aimall_utils as aim_u # type: ignore
    import default_settings # type: ignore
    import dftd3_utils as disp_u # type: ignore
    import gaussian_utils as gauss_u # type: ignore
    import iqa_diagnostics # type: ignore
    import reg
    import reg_setup # type: ignore
    import reg_vis as rv # type: ignore

    ###############################################################################
    #                                                                             #
    #                      A FOLDER OF SYSTEMS (-R / --recursive)                 #
    #                                                                             #
    ###############################################################################
    # Handled before anything else reads the current directory: in this mode this
    # process analyses nothing itself, it finds the systems and runs one copy of
    # this same command inside each of them.
    if option.recursive:
        return _run_recursive(option, args, reg_dir)

    # Collecting one directory without analysing it: the folder is rebuilt from
    # the bundles already there, which is how a directory worked in over several
    # days is turned into something to download without re-running anything.
    if option.collect_only:
        import reg_batch  # type: ignore
        target = os.path.abspath(args[0]) if args else os.getcwd()
        if not os.path.isdir(target):
            raise NotADirectoryError('Not a directory: ' + target)
        prefix = reg_setup.sanitise_prefix(option.prefix) if option.prefix else ''
        reg_batch.collect_run(target, name=prefix or None, dir_prefix=prefix,
                              dir_name=option.collect_dir,
                              command=[sys.argv[0]] + list(sys.argv[1:]))
        return 0

    ### STARTING TIMER ###
    start_time = time.time()
    ##############################    VARIABLES    ##################################

    SYS = 'REG_IQA'  # output prefix; an IQF run overrides this to 'REG_IQF' below.
    # Named for the analysis level rather than just 'REG' so that running IQA and then
    # IQF on the same system leaves REG_IQA_results/ and REG_IQF_results/ side by side,
    # each saying which level produced it.  This also matches the bundle path the
    # iqa-explorer tooling already expects, REG_IQA_results/REG_IQA_model_transfer.json.

    ### PES Critical points options ###
    POINTS = default_settings.POINTS  # number of points for "find_critical" function
    AUTO = default_settings.AUTO  # Search for critical points
    turning_points = default_settings.turning_points  # manually put critical points in the PES if necessary

    # DEFINE THE DESIRED TERMS:
    intra_prop = default_settings.intra_prop  # chose the AIMAll intra atomic properties to analyse
    intra_prop_names = default_settings.intra_prop_names  # names of the properties shown in the output
    inter_prop = default_settings.inter_prop  # chose the AIMAll inter atomic properties to analyse
    inter_prop_names = default_settings.inter_prop_names  # names of the properties shown in the output

    REVERSE = default_settings.REVERSE  # Reverse the REG points

    INFLEX = default_settings.INFLEX

    ### CONTROL COORDINATE OPTIONS ###
    CONTROL_COORDINATE_TYPE = default_settings.CONTROL_COORDINATE_TYPE  # 'Scan' or 'IRC'. If empty ('') then default will be used
    Scan_Atoms = default_settings.Scan_Atoms  # list of the atoms used for PES Scan (i.e. ModRedundant option in Gaussian)
    IRC_output = default_settings.IRC_output  # insert the g16 output file path if using IRC as control coordinate

    CHARGE_TRANSFER_POLARISATION = default_settings.CHARGE_TRANSFER_POLARISATION  # Split the classical electrostatic term into polarisation and monopolar charge-transfer

    MULTIPOLE_VISUALIZATION = option.multipole_visualization or default_settings.MULTIPOLE_VISUALIZATION  # Directly reconstruct charge, dipole and quadrupole diagnostics from AIMAll .sum tables

    REG_MULTI = option.reg_multi or getattr(default_settings, 'REG_MULTI', False)  # Rank-resolved multipolar electrostatics between fragments (REG_MULTI.md)

    DISPERSION = default_settings.DISPERSION  # Run DFT-D3 program to consider dispersion

    ### DISPERSION OPTIONS ###
    DFT_D3_PATH = default_settings.DFT_D3_PATH  # insert path of DFT-D3 program
    DISP_FUNCTIONAL = default_settings.DISP_FUNCTIONAL  # insert the functional used for D3 correction
    BJ_DAMPING = default_settings.BJ_DAMPING  # Becke-Johnson Damping

    WRITE = default_settings.WRITE  # write csv files for energy values and REG analysis
    SAVE_FIG = default_settings.SAVE_FIG  # save figures
    ANNOTATE = default_settings.ANNOTATE  # annotate figures
    DETAILED_ANALYSIS = default_settings.DETAILED_ANALYSIS
    LABELS = default_settings.LABELS  # label figures
    n_terms = default_settings.n_terms  # number of terms to rank in figures
    MIN_TABLE_ROWS = option.table_rows or default_settings.MIN_TABLE_ROWS  # smallest useful REG_final table
    MAX_TABLE_ROWS = default_settings.MAX_TABLE_ROWS  # largest useful REG_final table
    if option.table_rows:
        # The ladder normally overshoots the minimum by about 1.6x, so the default cap
        # would silently truncate a table asked for from the command line — and a rank
        # truncation is the "no separation found" fallback, a weaker claim than the cut
        # the ladder would have made.  Give it room rather than let it fight the flag.
        MAX_TABLE_ROWS = max(MAX_TABLE_ROWS, 3 * option.table_rows)
    PROPERTY_MIN_TABLE_ROWS = default_settings.PROPERTY_MIN_TABLE_ROWS  # smallest useful per-property table
    PROPERTY_MAX_TABLE_ROWS = default_settings.PROPERTY_MAX_TABLE_ROWS  # largest useful per-property table
    R_THRESHOLD = default_settings.R_THRESHOLD  # weak |R| guard applied before the significance ranking
    ERR_R_THRESHOLD = default_settings.ERR_R_THRESHOLD  # |R| filter for error REG output

    ###### REG-IQF
    print(option.IQF_TF)
    if option.IQF_TF == 'T':
        IQF = True
    else:
        IQF = False
    # Fragment definitions are read whether or not IQF is requested: IQF needs them
    # to build fragment terms, and the model-transfer bundle carries them either way
    # so that anything reading the bundle can group atoms without the config file.
    FRAG_CONFIG_PATH = 'auto_reg.config'
    Frag_names, List_of_frags = _read_fragment_config(FRAG_CONFIG_PATH)

    if IQF:
        SYS = 'REG_IQF'
        if not List_of_frags:
            raise FileNotFoundError(
                "Error: no fragment definitions found — auto_reg.config is missing from the directory "
                "or contains no 'FRAG ID ... FRAG ATOMS [...]' entries")

    # A sweep over a folder of systems (-R) gives every run the name of its own
    # system, so forty analyses that would all have written REG_IQA_results/ and
    # REG.xlsx can be collected in one place.  Carrying it in SYS puts the name on
    # the results directory, the transfer bundle and the figure titles at once;
    # everything else the run writes is renamed to match when the run is done.
    PREFIX = reg_setup.sanitise_prefix(option.prefix) if option.prefix else ''
    if PREFIX:
        SYS = PREFIX + '_' + SYS
        print('Output of this run is named after the system: {p}_...'.format(p=PREFIX))

    #List_of_frags = [[1,3,5,7,9,11],[2,4,6,8,10,12],[13]]
    #Frag_names = ["C(pi)","H(pi)","F-"]

    ##################################################################################

    ###############################################################################
    #                                                                             #
    #                           AUTOMATIC FILES SETUP                             #
    #                                                                             #
    ###############################################################################

    # DEFINE PATHS AND FILES AUTOMATICALLY:
    cwd = str(os.getcwd())

    # Finding file paths and folders, ordered along the control coordinate.
    # reg_multipole.py starts from the same layout, so the walk, the numeric
    # folder ordering and the dihedral roll all live in reg_setup.
    _points = reg_setup.discover_reg_points('.')
    reg_folders = _points['reg_folders']
    reg_root_list = _points['reg_roots']
    wf_files = _points['wf_files']
    g16_out_files = _points['g16_files']
    WFX = _points['wfx']

    #if REVERSE:
    #    reg_folders = reg_folders[::-1]

    os.chdir(cwd)  # working directory
    # Create results directory
    access_rights = 0o755
    try:
        os.mkdir(SYS + "_results", access_rights)
    except OSError:
        print("Creation of the directory {a}/{b}_results failed or has already been created".format(a=cwd,b=SYS))
    else:
        print("Successfully created the directory {a}/{b}_results".format(a=cwd,b=SYS))

    # GET ATOM LIST FROM ANY .WFN FILE:
    if WFX:
        atoms = aim_u.get_atom_list_wfx(wf_files[0])
    else:
        atoms = aim_u.get_atom_list(wf_files[0])

    # Arrange files and folders in lists
    wfn_files = wf_files # Need to edit
    atomic_files = [wf[:-4] + '_atomicfiles' for wf in wf_files]
    g16_files = g16_out_files
    xyz_files = [gauss_u.get_xyz_file(file) for file in g16_out_files]

    # Get control coordinate list
    if CONTROL_COORDINATE_TYPE == 'Scan':
        cc = gauss_u.get_control_coordinates_PES_Scan(g16_files, Scan_Atoms)
        X_LABEL = r"Control Coordinate [$\AA$]"
    elif CONTROL_COORDINATE_TYPE == 'IRC':
        cc = gauss_u.get_control_coordinates_IRC_g16(IRC_output)
        X_LABEL = r"Control Coordinate r[$\AA$]"
    else:
        #cc = [int(reg_folders[i]) for i in range(0, len(reg_folders))]
        cc = [float(re.search(r'-?\d+(?:\.\d+)?', reg_folders[i]).group()) for i in range(0, len(reg_folders)) ]
        X_LABEL = "Control Coordinate [REG step]"
    cc = np.array(cc)
    if REVERSE:
        cc = -cc

    ### INTRA AND INTER ENERGY TERMS ###

    # GET TOTAL ENERGY FROM THE .WFN or .WFX FILES:
    if WFX:
        total_energy_wfn = aim_u.get_aimall_wfx_energies(wfn_files)
        total_energy_wfn = np.array(total_energy_wfn)
    else:
        total_energy_wfn = aim_u.get_aimall_wfn_energies(wfn_files)
        total_energy_wfn = np.array(total_energy_wfn)



    # GET INTRA AND INTER IQA TERMS (single .sum parse per geometry point):
    iqa_intra, iqa_intra_header, iqa_inter, iqa_inter_header, missing_files = aim_u.get_iqa_properties(
        atomic_files, intra_prop, inter_prop, atoms)

    # READ LAGRANGIANS — done before REG so missing-file and |L| problems are
    # reported together, letting the user identify everything to resubmit at once.
    lagrangians = aim_u.get_lagrangians(atomic_files, atoms)
    _L_THRESHOLD_H     = 1e-4
    _L_THRESHOLD_HEAVY = 1e-3
    bad_L = []
    for i, folder_L in enumerate(lagrangians):
        for atom, L_val in folder_L.items():
            threshold = _L_THRESHOLD_H if atom.startswith('h') else _L_THRESHOLD_HEAVY
            if L_val is not None and abs(L_val) > threshold:
                bad_L.append((reg_folders[i], atom, L_val, threshold, atomic_files[i]))

    # BUILD COMBINED QUALITY REPORT (missing files + Lagrangian check)
    sep_wide  = '=' * 90
    sep_inner = '-' * 90
    q_lines = []
    q_lines.append(sep_wide)
    q_lines.append('  MISSING / INCOMPLETE FILE REPORT')
    q_lines.append(sep_wide)
    q_lines.append('')
    if not missing_files:
        q_lines.append('  All expected files present and readable.')
    else:
        q_lines.append('  The following files are missing or incomplete:')
        for f in missing_files:
            q_lines.append('    ' + f)
    q_lines.append('')
    q_lines.append(sep_wide)
    q_lines.append('  LAGRANGIAN QUALITY REPORT  (H: |L(A)| < {:.0e}, heavy: |L(A)| < {:.0e})'.format(
        _L_THRESHOLD_H, _L_THRESHOLD_HEAVY))
    q_lines.append(sep_wide)
    q_lines.append('')
    if not bad_L:
        q_lines.append('  All atoms within threshold at every geometry point.')
    else:
        q_lines.append('  Atoms exceeding threshold (*** = poor integration):')
        q_lines.append('  {:<25s}  {:>6s}  {:>16s}  {:>10s}'.format('Step', 'Atom', 'L(A)', 'Threshold'))
        q_lines.append('  ' + '-' * 65)
        for step_lbl, atom, L_val, threshold, _ in bad_L:
            q_lines.append('  {:<25s}  {:>6s}  {:>+16.6e}  {:>10.0e}  ***'.format(
                step_lbl, atom, L_val, threshold))
    q_lines.append('')
    q_lines.append(sep_wide)
    q_lines.append('  FILES REQUIRING RESUBMISSION')
    q_lines.append(sep_wide)
    q_lines.append('')
    resubmit = list(missing_files)
    seen = set(missing_files)
    for _, atom, _, _, atomic_file in bad_L:
        path = atomic_file + '/' + atom + '.int'
        if path not in seen:
            seen.add(path)
            resubmit.append(path)
    if not resubmit:
        q_lines.append('  None.')
    else:
        for path in resubmit:
            q_lines.append('  ' + path)
    q_lines.append('')

    quality_report_text = '\n'.join(q_lines)
    print(quality_report_text)
    quality_report_path = cwd + '/' + SYS + '_results/quality_report.txt'
    with open(quality_report_path, 'w') as _qf:
        _qf.write(quality_report_text + '\n')

    # ── EXTENDED IQA DIAGNOSTICS ─────────────────────────────────────────────
    # Run before the missing-file abort so the report is available even when
    # files need resubmitting.  T(A), q(A) and E_IQA(A) are fetched via the
    # already-cached .sum parser (no extra NFS I/O).  Missing values become NaN
    # and are handled gracefully inside iqa_diagnostics.  Any failure here is
    # non-fatal — a warning is printed and the main REG analysis continues.
    q_raw = None
    E_raw = None
    T_raw = None
    direct_multipoles = None
    try:
        _T_raw,   _, _, _, _ = aim_u.get_iqa_properties(atomic_files, ['T(A)'],     [], atoms)
        _q_raw,   _, _, _, _ = aim_u.get_iqa_properties(atomic_files, ['q(A)'],     [], atoms)
        _E_raw,   _, _, _, _ = aim_u.get_iqa_properties(atomic_files, ['E_IQA(A)'], [], atoms)
        q_raw = np.array(_q_raw)
        E_raw = np.array(_E_raw)
        T_raw = np.array(_T_raw)

        # Multipoles are harvested on every run, not only under
        # MULTIPOLE_VISUALIZATION.  get_sum_intra_properties reads them out of the
        # already-parsed .sum tables and never falls back to per-atom .int reads,
        # so collecting them is free; the flag now only controls whether the
        # multipole report and its plots are produced.  Doing it unconditionally
        # is what keeps direct_multipoles out of the bundle's "needs a rerun" list.
        _multipole_props = [
            ('Mu_X(A)', 'mu_x'),
            ('Mu_Y(A)', 'mu_y'),
            ('Mu_Z(A)', 'mu_z'),
            ('Q_XX(A)', 'q_xx'),
            ('Q_XY(A)', 'q_xy'),
            ('Q_XZ(A)', 'q_xz'),
            ('Q_YY(A)', 'q_yy'),
            ('Q_YZ(A)', 'q_yz'),
            ('Q_ZZ(A)', 'q_zz'),
        ]
        _multi_sum = aim_u.get_sum_intra_properties(
            atomic_files, [prop for prop, _ in _multipole_props], atoms)
        direct_multipoles = {key: np.array(_multi_sum[prop])
                             for prop, key in _multipole_props
                             if _multi_sum.get(prop) is not None}
        if not direct_multipoles:
            direct_multipoles = None
            if MULTIPOLE_VISUALIZATION:
                print('WARNING: no multipole columns found in the .sum files — '
                      'multipole diagnostics unavailable.')

        iqa_diagnostics.run(
            lagrangians=lagrangians,
            T_vals=np.array(_T_raw),
            q_vals=q_raw,
            iqa_atom_total=E_raw,
            atoms=atoms,
            cc=cc,
            total_energy_wfn=total_energy_wfn,
            reg_folders=reg_folders,
            results_dir=cwd + '/' + SYS + '_results',
            direct_multipoles=direct_multipoles if MULTIPOLE_VISUALIZATION else None,
        )
    except Exception as _diag_err:
        print('WARNING: Extended IQA diagnostics skipped — ' + str(_diag_err))
    # ─────────────────────────────────────────────────────────────────────────

    if missing_files:
        if option.use_zeros:
            print('WARNING: Missing or incomplete files detected — continuing with zeros (--zeros flag active).\n'
                  'Full report written to: ' + quality_report_path)
        else:
            raise ValueError(
                'Missing or incomplete files detected — REG analysis aborted.\n'
                'Full report written to: ' + quality_report_path)

    iqa_intra = np.array(iqa_intra)
    iqa_intra_header = np.array(iqa_intra_header)
    iqa_inter = np.array(iqa_inter)
    iqa_inter_header = np.array(iqa_inter_header)

    if option.use_zeros:
        iqa_intra = np.nan_to_num(iqa_intra)
        iqa_inter = np.nan_to_num(iqa_inter)
        if bad_L:
            print('WARNING: Zeroing contributions from atoms with poor Lagrangian integration.')
            reg_folders_list = list(reg_folders)
            for step_lbl, atom, _, _, _ in bad_L:
                if step_lbl not in reg_folders_list:
                    continue
                step_idx = reg_folders_list.index(step_lbl)
                atom_upper = atom.upper()
                for row_idx, hdr in enumerate(iqa_intra_header):
                    if atom_upper in hdr.upper():
                        iqa_intra[row_idx, step_idx] = 0.0
                for row_idx, hdr in enumerate(iqa_inter_header):
                    if atom_upper in hdr.upper():
                        iqa_inter[row_idx, step_idx] = 0.0

    # Preserve the atom-level IQA arrays before IQF remaps them to fragment terms.
    # Taken after the --zeros clean-up so that these are exactly the numbers the
    # REG analysis consumes, and so the per-pair terms written to the transfer
    # bundle agree with the REG values written beside them.
    iqa_intra_iqa = np.array(iqa_intra)
    iqa_intra_header_iqa = np.array(iqa_intra_header)
    iqa_inter_iqa = np.array(iqa_inter)
    iqa_inter_header_iqa = np.array(iqa_inter_header)

    if IQF:
        iqf_inter, iqf_inter_header, iqf_inter_comps, iqf_inter_comp_head, iqf_intra, iqf_intra_header, iqf_intra_comps, iqf_intra_comp_head = sum_into_fragments(Frag_names,List_of_frags,atoms,True,iqa_inter,inter_prop,[],iqa_intra,intra_prop)
        # Keep the pre-dispersion fragment intra terms.  The dispersion block below
        # folds each fragment's own-pair E_Disp into iqf_intra, which leaves
        # "intra" meaning something different at the fragment level than at the atom
        # level, and different again from the fragment totals (computed just below,
        # before that fold).  The bundle publishes both forms explicitly rather than
        # leaving the difference for a reader to discover.
        iqf_intra_no_disp = np.array(iqf_intra, dtype=float)
        iqa_intra = iqf_intra
        iqa_intra_header = np.array(iqf_intra_header)
        iqa_inter = iqf_inter
        iqa_inter_header = np.array(iqf_inter_header)

    # PER-ENTITY (ATOM OR FRAGMENT) IQA TOTAL: E_intra(A) + sum_B (1/2 * E_inter(A,B))
    # Ranked further down against the control coordinate to produce the REG_IQA ranking output.
    entities = list(Frag_names) if IQF else [str(a) for a in atoms]
    entity_prop_sep = '_' if IQF else '-'
    entity_totals = compute_entity_totals(entities, iqa_intra, iqa_intra_header, iqa_inter, iqa_inter_header,
                                           intra_prop[0], inter_prop[-1], entity_prop_sep)
    entity_headers = np.array(['E_IQA_Total(' + ent + ')' for ent in entities])

    ###############################################################################
    #                                                                             #
    #                               REG ANALYSIS                                  #
    #                                                                             #
    ###############################################################################

    # FINDING CRITICAL POINT (Maybe remove lower part)
    if AUTO:
        critical_points = reg.find_critical(total_energy_wfn, cc, min_points=POINTS, use_inflex=INFLEX)
    else:
        critical_points = turning_points

    # GET CT and PL TERMS:
    if CHARGE_TRANSFER_POLARISATION:
        iqa_charge_transfer_terms, iqa_charge_transfer_headers, iqa_polarisation_terms, iqa_polarisation_headers = aim_u.charge_transfer_and_polarisation_from_int_file(
            atomic_files, atoms, iqa_inter, xyz_files)
        iqa_charge_transfer_headers = np.array(iqa_charge_transfer_headers)
        iqa_polarisation_headers = np.array(iqa_polarisation_headers)
        iqa_polarisation_terms = np.array(iqa_polarisation_terms)
        iqa_charge_transfer_terms = np.array(iqa_charge_transfer_terms)
        # CHARGE TRANSFER CONTRIBUTION
        reg_ct = reg.reg(total_energy_wfn, cc, iqa_charge_transfer_terms, np=POINTS, critical=AUTO, inflex=INFLEX,
                        critical_index=turning_points)
        # POLARISATION CONTRIBUTION
        reg_pl = reg.reg(total_energy_wfn, cc, iqa_polarisation_terms, np=POINTS, critical=AUTO, inflex=INFLEX,
                        critical_index=turning_points)

    ### DISPERSION ANALYSIS ###
    iqa_disp_atomic = None
    iqa_disp_header_atomic = None
    reg_disp = None
    if DISPERSION:
        for i in range(0, len(reg_folders)):
            xyz_file = xyz_files[i]
            disp_u.run_DFT_D3(DFT_D3_PATH, reg_root_list[i], xyz_file, DISP_FUNCTIONAL,BJ_DAMPING)
        # DFT-D3 drops a histo.dat in whatever directory it was started from, which
        # is this system's own directory rather than the results directory.  It is
        # named after the system too, so a sweep leaves one per system instead of
        # each system overwriting a shared one.
        if PREFIX and os.path.isfile(os.path.join(cwd, 'histo.dat')):
            try:
                os.replace(os.path.join(cwd, 'histo.dat'),
                           os.path.join(cwd, PREFIX + '_histo.dat'))
            except OSError as histo_error:
                print('WARNING: could not rename histo.dat — ' + str(histo_error))
        folders_disp = [reg_root_list[i] + '/dft-d3.log' for i in range(0, len(reg_folders))]
        # GET INTER-ATOMIC DISPERSION TERMS:
        iqa_disp, iqa_disp_header = disp_u.disp_property_from_dftd3_file(folders_disp, atoms)
        iqa_disp_header = np.array(iqa_disp_header)  # used for reference
        iqa_disp = np.array(iqa_disp)
        # Total D3 must be captured here — the IQF block below absorbs intra-fragment
        # pairs into iqf_intra, so summing iqa_disp afterwards gives inter-fragment only.
        total_energy_dispersion = sum(iqa_disp)
        # Snapshot the atom-pair dispersion table for the same reason as above:
        # the IQF branch replaces iqa_disp with fragment-pair terms.
        iqa_disp_atomic = np.array(iqa_disp)
        iqa_disp_header_atomic = np.array(iqa_disp_header)
        if IQF:
            iqa_disp, iqa_disp_header, iqa_disp_comps, _, iqf_intra, _, iqf_intra_comps, iqf_intra_comp_hea = sum_into_fragments(Frag_names,List_of_frags,atoms,False,iqa_disp,['E_Disp(A,B)'],[iqf_intra,iqf_intra_comps,iqf_intra_comp_head])  #### To remove
            iqa_intra = iqf_intra
        # REG
        reg_disp = reg.reg(total_energy_wfn, cc, iqa_disp, np=POINTS, critical=AUTO, inflex=INFLEX,
                        critical_index=turning_points)

    if IQF:
        reg_intra = reg.reg(total_energy_wfn, cc, iqf_intra, np=POINTS, critical=AUTO, inflex=INFLEX,
                            critical_index=turning_points)

        reg_inter = reg.reg(total_energy_wfn, cc, iqf_inter, np=POINTS, critical=AUTO, inflex=INFLEX,
                            critical_index=turning_points)

    else:
        # INTRA ATOMIC CONTRIBUTION
        reg_intra = reg.reg(total_energy_wfn, cc, iqa_intra, np=POINTS, critical=AUTO, inflex=INFLEX,
                            critical_index=turning_points)
        # INTER ATOMIC CONTRIBUTION
        reg_inter = reg.reg(total_energy_wfn, cc, iqa_inter, np=POINTS, critical=AUTO, inflex=INFLEX,
                            critical_index=turning_points)

    # REG_IQA RANKING: per-atom/fragment E_IQA total against the control coordinate
    reg_entity_totals = reg.reg(total_energy_wfn, cc, entity_totals, np=POINTS, critical=AUTO, inflex=INFLEX,
                                critical_index=turning_points)

    # ATOM-LEVEL REG — always computed, even in IQF mode where reg_intra/reg_inter
    # above describe fragments.  Without this an IQF run carries no per-pair REG at
    # all, so a reader cannot rank atom pairs or locate the reaction centre.
    # The regression is vectorised over terms, so the extra pass is cheap.
    if IQF:
        reg_atom_intra = reg.reg(total_energy_wfn, cc, iqa_intra_iqa, np=POINTS, critical=AUTO, inflex=INFLEX,
                                 critical_index=turning_points)
        reg_atom_inter = reg.reg(total_energy_wfn, cc, iqa_inter_iqa, np=POINTS, critical=AUTO, inflex=INFLEX,
                                 critical_index=turning_points)
        atom_entity_totals = compute_entity_totals(
            [str(a) for a in atoms], iqa_intra_iqa, iqa_intra_header_iqa,
            iqa_inter_iqa, iqa_inter_header_iqa,
            intra_prop[0], inter_prop[-1], '-')
        reg_atom_totals = reg.reg(total_energy_wfn, cc, atom_entity_totals, np=POINTS, critical=AUTO,
                                  inflex=INFLEX, critical_index=turning_points)
        reg_atom_disp = None
        if DISPERSION and iqa_disp_atomic is not None:
            reg_atom_disp = reg.reg(total_energy_wfn, cc, iqa_disp_atomic, np=POINTS, critical=AUTO,
                                    inflex=INFLEX, critical_index=turning_points)
    else:
        reg_atom_intra = reg_intra
        reg_atom_inter = reg_inter
        atom_entity_totals = entity_totals
        reg_atom_totals = reg_entity_totals
        reg_atom_disp = reg_disp
    atom_entity_headers = np.array(['E_IQA_Total(' + str(a) + ')' for a in atoms])

    ### REG breakdown IQF ######

    if IQF:
        iqf_intra_comp_list = []
        iqf_intra_prop_list = []   # REG results for property-grouped sub-totals
        iqf_intra_prop_heads = []  # group labels per fragment
        for i in range(len(iqa_intra)):
            iqf_val = iqa_intra[i]
            iqf_comp_np = np.array(iqf_intra_comps[i])
            reg_int = reg.reg(iqf_val,cc,iqf_comp_np,np=POINTS, critical=False, inflex=INFLEX,
                            critical_index=critical_points)
            iqf_intra_comp_list.append(reg_int)

            # Group components by property (E_IQA_Intra atomic, VC_IQA, VX_IQA, ...)
            grouped = {}
            for comp, head in zip(iqf_intra_comps[i], iqf_intra_comp_head[i]):
                prop_label = head.split('_', 1)[1] if '_' in head else intra_prop[0]
                if prop_label not in grouped:
                    grouped[prop_label] = np.array(comp, dtype=float)
                else:
                    grouped[prop_label] = grouped[prop_label] + np.array(comp, dtype=float)
            grp_labels = list(grouped.keys())
            grp_terms = np.array(list(grouped.values()))
            reg_grp = reg.reg(iqf_val, cc, grp_terms, np=POINTS, critical=False,
                              inflex=INFLEX, critical_index=critical_points)
            iqf_intra_prop_list.append(reg_grp)
            iqf_intra_prop_heads.append(grp_labels)

        # Same breakdown for inter-fragment terms: each fragment-pair interaction is
        # regressed against its constituent atom-pair energies to show which pairs drive it.
        iqf_inter_comp_list = []
        for i in range(len(iqf_inter)):
            inter_comp_np = np.array(iqf_inter_comps[i])
            if inter_comp_np.ndim == 2 and inter_comp_np.shape[0] > 0:
                reg_int = reg.reg(iqf_inter[i], cc, inter_comp_np, np=POINTS, critical=False,
                                  inflex=INFLEX, critical_index=critical_points)
            else:
                n_segs = len(critical_points) + 1
                reg_int = ([[]] * n_segs, [[]] * n_segs)
            iqf_inter_comp_list.append(reg_int)

    # CALCULATE TOTAL ENERGIES
    # Sum the per-atom E_IQA(A) from the .sum file intra table directly.  This is the
    # authoritative total (E_IQA_Intra + half-sum of inter pairs as AIMAll accounts for
    # the "A'=Mol-A" correction), and avoids the ~3 kJ/mol error that arises from summing
    # individual pair E_IQA_Inter(A,B)/2 values whose sum ("SumB") differs from the
    # corrected per-atom inter contribution ("A'=Mol-A") stored in the .sum file.
    _iqa_atom_total, _, _, _, _ = aim_u.get_iqa_properties(atomic_files, ['E_IQA(A)'], [], atoms)
    _iqa_atom_total = np.array(_iqa_atom_total)
    total_energy_iqa = sum(_iqa_atom_total[:len(atoms)])

    # CALCULATE CLOSURE ERROR
    # Sum(E_IQA_Intra(A)) + Sum_{A<B} E_IQA_Inter(A,B)/2 versus the per-atom
    # E_IQA(A) that AIMAll reports.  This is a *different* quantity from the
    # recovery error below: it measures whether the term decomposition adds back
    # up to the reported IQA energy, while the recovery error measures how far the
    # IQA energy sits from the WFN energy.  Neither substitutes for the other, so
    # the bundle carries both.
    closure_error_ha = np.asarray(atom_entity_totals, dtype=float).sum(axis=0) - np.asarray(total_energy_iqa,
                                                                                            dtype=float)
    closure_error_kj = [HA_TO_KJ * e for e in closure_error_ha]

    # CALCULATE RECOVERY ERROR
    iqa_for_error = total_energy_iqa + total_energy_dispersion if DISPERSION else total_energy_iqa
    per_step_errors_ha, rmse_kj = reg.integration_error(total_energy_wfn, iqa_for_error)
    per_step_errors_kj = [2625.5 * e for e in per_step_errors_ha]

    # REG AGAINST RECOVERY ERROR — which IQA terms track E_WFN − E_IQA?
    # NOTE: with AUTO on, reg.reg finds critical points on the surface it is handed,
    # which here is the error surface — so these segments need not match the energy
    # ones.  The bundle reports the two segmentations separately for that reason.
    _err_ha = np.array(per_step_errors_ha)
    if IQF:
        reg_intra_err = reg.reg(_err_ha, cc, iqf_intra, np=POINTS, critical=AUTO, inflex=INFLEX,
                                critical_index=turning_points)
        reg_inter_err = reg.reg(_err_ha, cc, iqf_inter, np=POINTS, critical=AUTO, inflex=INFLEX,
                                critical_index=turning_points)
    else:
        reg_intra_err = reg.reg(_err_ha, cc, iqa_intra, np=POINTS, critical=AUTO, inflex=INFLEX,
                                critical_index=turning_points)
        reg_inter_err = reg.reg(_err_ha, cc, iqa_inter, np=POINTS, critical=AUTO, inflex=INFLEX,
                                critical_index=turning_points)

    # ATOM-LEVEL ERROR REG — as with the energy REG above, reg_intra_err/reg_inter_err
    # describe fragments on an IQF run.  Recomputing them per atom and per pair keeps
    # the atom-level view of the bundle as complete as the fragment-level one, so a
    # reader can split the recovery error by channel either way.
    if IQF:
        reg_atom_intra_err = reg.reg(_err_ha, cc, iqa_intra_iqa, np=POINTS, critical=AUTO, inflex=INFLEX,
                                     critical_index=turning_points)
        reg_atom_inter_err = reg.reg(_err_ha, cc, iqa_inter_iqa, np=POINTS, critical=AUTO, inflex=INFLEX,
                                     critical_index=turning_points)
    else:
        reg_atom_intra_err = reg_intra_err
        reg_atom_inter_err = reg_inter_err

    # REG of per-atom E_IQA(A) totals against the recovery error.
    # E_IQA(A) already includes the correct intra + inter correction from the .sum file,
    # so this ranks atoms by how much their total energy tracks the integration gap.
    _atom_err_headers = np.array([str(a) for a in atoms])
    reg_atom_err = reg.reg(_err_ha, cc, _iqa_atom_total[:len(atoms)], np=POINTS, critical=AUTO,
                           inflex=INFLEX, critical_index=turning_points)

    ###############################################################################
    #                                                                             #
    #                         MODEL-TRANSFER JSON BUNDLE                          #
    #                                                                             #
    ###############################################################################
    # The bundle is meant to be a self-contained substitute for the whole results
    # directory: everything a downstream visualisation needs, without also needing
    # Energy.xlsx, REG.xlsx, auto_reg.config or the XYZ files alongside it.  That
    # means per-step term tables (not just their REG summaries), the headers that
    # say which term each row is, the fragment definitions, and the per-atom
    # integration quality numbers.

    pair_index = _pair_list(atoms)
    n_pairs = len(pair_index)

    # Choose which atom pairs get per-step energies written out.  For anything past
    # a few dozen atoms the pair tables dominate the file size and most rows barely
    # move along the path — but "barely" summed over thousands of rows is not
    # nothing, so the cut is made against a total drift budget rather than a
    # per-pair threshold.  See _select_pairs_within_budget.
    pair_budget_kj = option.bundle_pair_budget
    pair_budget_ha = (pair_budget_kj / HA_TO_KJ) if pair_budget_kj else 0.0
    kept_pairs, residual_drift_ha = _select_pairs_within_budget(
        [iqa_inter_iqa, iqa_disp_atomic], n_pairs, pair_budget_ha)

    inter_values, inter_headers_kept, inter_residual, kept_pairs = _prune_pair_terms(
        iqa_inter_iqa, iqa_inter_header_iqa, n_pairs, kept_pairs)

    atom_inter_block = _term_block(
        inter_headers_kept, inter_values, properties=inter_prop,
        extra={
            'property_names': list(inter_prop_names),
            'pair_index': [list(pair_index[k]) for k in kept_pairs],
            'pair_labels': [[atoms[pair_index[k][0]], atoms[pair_index[k][1]]] for k in kept_pairs],
            'layout': 'property-major: row = property_i * n_kept_pairs + pair_i',
            'definition': 'per-atom-pair interaction energies, not halved',
            'includes_dispersion': False,
            'n_pairs_total': n_pairs,
            'n_pairs_written': len(kept_pairs),
            'pruning': {
                'budget_kj_mol': pair_budget_kj,
                'criterion': ('least active pairs dropped until their combined drift would exceed the budget; '
                              'the bound is on the summed residual, not on any single pair'),
                'residual_drift_kj_mol': residual_drift_ha * HA_TO_KJ,
                'residual_per_property': _json_ready(inter_residual),
                'residual_note': 'per-step sum of the pairs left out, so each property total still closes',
                'residual_by_fragment_pair': _json_ready(_residual_by_fragment_pair(
                    iqa_inter_iqa, iqa_inter_header_iqa, n_pairs,
                    sorted(set(range(n_pairs)) - set(kept_pairs)),
                    pair_index, List_of_frags, Frag_names)),
                'residual_by_fragment_pair_note': ('the same residual split over fragment pairs, so the '
                                                   'omitted drift can still be placed even though the '
                                                   'individual atom pairs were not written'),
                'warning': ('add residual_per_property back before using any total. Summing only the written '
                            'pairs omits a large constant offset as well as residual_drift_kj_mol of variation, '
                            'so fragment energies must be read from the fragment_* terms and never rebuilt '
                            'from these atom pairs'),
            },
        })

    atom_disp_block = None
    if DISPERSION and iqa_disp_atomic is not None:
        disp_values, disp_headers_kept, disp_residual, _ = _prune_pair_terms(
            iqa_disp_atomic, iqa_disp_header_atomic, n_pairs, kept_pairs)
        atom_disp_block = _term_block(
            disp_headers_kept, disp_values, properties=['E_Disp(A,B)'],
            extra={
                'property_names': ['Vdisp'],
                'pair_index': [list(pair_index[k]) for k in kept_pairs],
                'pair_labels': [[atoms[pair_index[k][0]], atoms[pair_index[k][1]]] for k in kept_pairs],
                'n_pairs_total': n_pairs,
                'n_pairs_written': len(kept_pairs),
                'pruning': {
                    'budget_kj_mol': pair_budget_kj,
                    'residual_per_property': _json_ready(disp_residual),
                    'residual_note': 'per-step sum of the pairs left out, so the dispersion total still closes',
                },
            })

    structure_payload = []
    for idx, xyz_path in enumerate(xyz_files):
        structure = _load_xyz_structure(xyz_path)
        if structure is None:
            continue
        structure['step'] = reg_folders[idx] if idx < len(reg_folders) else str(idx)
        structure['control_coordinate'] = float(cc[idx]) if idx < len(cc) else None
        structure_payload.append(structure)

    kept_pair_tuples = [pair_index[k] for k in kept_pairs]
    pair_distances = _pair_distances(structure_payload, kept_pair_tuples)

    # Own-fragment dispersion, summed straight from the atom-pair table so it can be
    # checked against the fragment definitions rather than inferred from a difference.
    # This is the piece that sum_into_fragments folds into iqf_intra: publishing it
    # separately lets a reader rebuild either convention, and makes explicit that
    # fragment_intra and fragment_totals do not treat dispersion the same way.
    fragment_own_dispersion = None
    if IQF and DISPERSION and iqa_disp_atomic is not None:
        _disp_arr = np.asarray(iqa_disp_atomic, dtype=float)[:n_pairs]
        fragment_own_dispersion = np.zeros((len(List_of_frags), _disp_arr.shape[1]))
        for f_i, frag_atoms in enumerate(List_of_frags):
            members = set(a - 1 for a in frag_atoms)
            rows = [k for k, (i, j) in enumerate(pair_index) if i in members and j in members]
            if rows:
                fragment_own_dispersion[f_i] = np.nansum(_disp_arr[rows], axis=0)

    fragment_definitions = None
    if List_of_frags:
        fragment_definitions = [
            {
                'name': Frag_names[f_i],
                'atom_numbers': List_of_frags[f_i],
                'atom_indices': [n - 1 for n in List_of_frags[f_i]],
                'atom_labels': [atoms[n - 1] for n in List_of_frags[f_i] if 0 < n <= len(atoms)],
            }
            for f_i in range(len(List_of_frags))
        ]

    # Segment boundaries: REG arrays are indexed [segment][term], so without the
    # critical-point indices a reader cannot say which steps a segment covers.
    # reg.split_segm makes consecutive segments share their critical point — a
    # segment runs [previous critical point .. next critical point] inclusive at
    # both ends — so the ranges below overlap by one step by design.
    def _segment_block(crit_points, surface_label):
        crit_sorted = sorted(set(int(i) for i in crit_points))
        bounds = [0] + crit_sorted + [len(cc) - 1]
        return {
            'surface': surface_label,
            'count': len(bounds) - 1,
            'critical_point_indices': crit_sorted,
            'step_ranges': [[bounds[i], bounds[i + 1]] for i in range(len(bounds) - 1)],
            'note': ('REG arrays are indexed [segment][term]; step_ranges are inclusive at both ends '
                     'and consecutive segments share their critical point'),
        }

    # The energy REG and the error REG do not necessarily share segments.  With
    # AUTO on, reg.reg finds critical points on whatever surface it is regressing
    # against, which for the *_error blocks is the recovery error, not the total
    # energy.  Reporting one set of segments for both would mislabel the error
    # blocks, so each surface gets its own.
    # The label must name the surface that was actually regressed against.  _err_ha
    # comes from integration_error(total_energy_wfn, iqa_for_error), and iqa_for_error
    # carries the D3 dispersion when it is on — so with dispersion the surface is
    # E_WFN - (E_IQA + E_disp), matching energies.recovery_error_definition.
    _err_surface = ('recovery error (E_WFN - (E_IQA + E_disp))' if DISPERSION
                    else 'recovery error (E_WFN - E_IQA)')
    energy_segments = _segment_block(critical_points, 'total energy (E_WFN)')
    error_segments = _segment_block(
        reg.find_critical(_err_ha, cc, min_points=POINTS, use_inflex=INFLEX) if AUTO else turning_points,
        _err_surface)

    lagrangian_values = [
        [lagrangians[s_i].get(str(atom).lower()) for s_i in range(len(reg_folders))]
        for atom in atoms
    ]
    l_over_t = None
    if T_raw is not None:
        _L_arr = np.array([[np.nan if v is None else v for v in row] for row in lagrangian_values], dtype=float)
        _T_arr = np.asarray(T_raw, dtype=float)[:len(atoms)]
        with np.errstate(divide='ignore', invalid='ignore'):
            l_over_t = np.abs(_L_arr) / np.abs(_T_arr)

    # ---- the per-level views -------------------------------------------------
    # Everything above this point is identical whichever partitioning was used, so
    # it is stored once at the top level.  A level only names its entities and
    # carries the REG results over them; its term tables are named by key into the
    # shared 'terms' block rather than copied, which is what keeps one file the
    # same size as the old two.
    def _level(entity_kind, entities, term_keys, reg_blocks, extra=None):
        level = {
            'entity_kind': entity_kind,
            'entities': [str(e) for e in entities],
            'entity_count': len(entities),
            'terms': term_keys,
            'terms_note': "values are keys into the top-level 'terms' block, not copies",
            'reg': {k: v for k, v in reg_blocks.items() if v is not None},
            'segments': {'energy': energy_segments, 'error': error_segments},
            'reg_surfaces': {
                'intra': energy_segments['surface'], 'inter': energy_segments['surface'],
                'dispersion': energy_segments['surface'], 'totals': energy_segments['surface'],
                'intra_error': error_segments['surface'], 'inter_error': error_segments['surface'],
                'entity_total_error': error_segments['surface'],
            },
        }
        if extra:
            level.update(extra)
        return level

    levels = {
        'REG_IQA': _level(
            'atom', atoms,
            {
                'intra': 'atom_intra',
                'inter': 'atom_inter',
                'dispersion': 'atom_dispersion' if atom_disp_block is not None else None,
                'totals': 'atom_totals',
            },
            {
                'intra': _reg_block(reg_atom_intra, iqa_intra_header_iqa),
                'inter': _reg_block(reg_atom_inter, iqa_inter_header_iqa),
                'dispersion': _reg_block(reg_atom_disp, iqa_disp_header_atomic)
                              if (DISPERSION and reg_atom_disp is not None) else None,
                'totals': _reg_block(reg_atom_totals, atom_entity_headers),
                'intra_error': _reg_block(reg_atom_intra_err, iqa_intra_header_iqa),
                'inter_error': _reg_block(reg_atom_inter_err, iqa_inter_header_iqa),
                'entity_total_error': _reg_block(reg_atom_err, _atom_err_headers),
                'charge_transfer': _reg_block(reg_ct, iqa_charge_transfer_headers)
                                   if CHARGE_TRANSFER_POLARISATION else None,
                'polarisation': _reg_block(reg_pl, iqa_polarisation_headers)
                                if CHARGE_TRANSFER_POLARISATION else None,
            },
            extra={'entity_labels_note': 'entities are atom labels, matching the top-level "atoms" list'},
        ),
    }

    if IQF:
        levels['REG_IQF'] = _level(
            'fragment', Frag_names,
            {
                'intra': 'fragment_intra',
                'intra_excl_dispersion': 'fragment_intra_excl_dispersion',
                'own_dispersion': 'fragment_own_dispersion' if fragment_own_dispersion is not None else None,
                'inter': 'fragment_inter',
                'dispersion': 'fragment_dispersion' if DISPERSION else None,
                'totals': 'fragment_totals',
            },
            {
                'intra': _reg_block(reg_intra, iqa_intra_header),
                'inter': _reg_block(reg_inter, iqa_inter_header),
                'dispersion': _reg_block(reg_disp, iqa_disp_header) if DISPERSION else None,
                'totals': _reg_block(reg_entity_totals, entity_headers),
                'intra_error': _reg_block(reg_intra_err, iqa_intra_header),
                'inter_error': _reg_block(reg_inter_err, iqa_inter_header),
            },
            extra={'entity_definitions': fragment_definitions,
                   'entity_labels_note': 'entities are fragment names, defined in entity_definitions'},
        )

    for _lvl in levels.values():
        _lvl['terms'] = {k: v for k, v in _lvl['terms'].items() if v is not None}

    transfer_bundle = {
        'schema_version': 3,
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'source_directory': cwd,
        'system': SYS,
        'analysis_kind': 'REG_IQF' if IQF else 'REG_IQA',
        # One bundle per run.  Shared data — geometry, energies, errors, charges,
        # multipoles, Lagrangians, and the per-atom and per-pair term tables — sits at
        # the top level exactly once.  Only what genuinely differs between the two
        # partitionings is nested under 'levels'.
        'primary_level': 'REG_IQF' if IQF else 'REG_IQA',
        'levels': levels,
        'levels_note': ('each level names its own entities and carries the REG results over them; '
                        'everything else is shared and lives at the top level'),
        'units': {
            'energy': 'hartree',
            'energy_to_kj_mol': HA_TO_KJ,
            'distance': 'angstrom',
            'charge': 'e',
            'note': 'every value is in hartree unless the key says kj_mol',
        },
        'settings': {
            'intra_properties': list(intra_prop),
            'intra_property_names': list(intra_prop_names),
            'inter_properties': list(inter_prop),
            'inter_property_names': list(inter_prop_names),
            'dispersion': bool(DISPERSION),
            'dispersion_functional': DISP_FUNCTIONAL if DISPERSION else None,
            'bj_damping': bool(BJ_DAMPING) if DISPERSION else None,
            'charge_transfer_polarisation': bool(CHARGE_TRANSFER_POLARISATION),
            'multipole_report': bool(MULTIPOLE_VISUALIZATION),
            'control_coordinate_type': CONTROL_COORDINATE_TYPE or 'folder-index',
            'auto_critical_points': bool(AUTO),
            'min_points_between_critical': POINTS,
            'turning_points': list(turning_points),
            'reverse': bool(REVERSE),
            'inflex': bool(INFLEX),
            'n_terms_ranked': n_terms,
            'min_table_rows': MIN_TABLE_ROWS,
            'max_table_rows': MAX_TABLE_ROWS,
            'property_min_table_rows': PROPERTY_MIN_TABLE_ROWS,
            'property_max_table_rows': PROPERTY_MAX_TABLE_ROWS,
            'r_threshold': R_THRESHOLD,
            'error_r_threshold': ERR_R_THRESHOLD,
            'zeros_for_missing': bool(option.use_zeros),
        },
        'control_coordinate_label': X_LABEL,
        'control_coordinates': _json_ready(cc.tolist()),
        'steps': [
            {
                'step': reg_folders[i],
                'folder': reg_root_list[i],
                'control_coordinate': float(cc[i]),
                'wfn_energy': float(total_energy_wfn[i]),
                'iqa_energy': float(total_energy_iqa[i]),
                'dispersion_energy': float(total_energy_dispersion[i]) if DISPERSION else None,
                'recovery_error_ha': float(per_step_errors_ha[i]),
                'recovery_error_kj_mol': float(per_step_errors_kj[i]),
                'closure_error_ha': float(closure_error_ha[i]),
                'closure_error_kj_mol': float(closure_error_kj[i]),
            }
            for i in range(len(reg_folders))
        ],
        'atoms': atoms,
        'atom_index': {str(atom): i for i, atom in enumerate(atoms)},
        'elements': [re.sub(r'\d+$', '', str(atom)) for atom in atoms],
        'pairs': {
            'n_total': n_pairs,
            'ordering': 'i < j over the atom list, i outer',
            'written_indices': kept_pairs,
            'written_labels': [[atoms[i], atoms[j]] for (i, j) in kept_pair_tuples],
            'distances_ang': _json_ready(pair_distances) if pair_distances is not None else None,
        },
        'fragments': {
            'source': FRAG_CONFIG_PATH if fragment_definitions else None,
            'definitions': fragment_definitions,
            'used_for_analysis': bool(IQF),
        },
        'energies': {
            'wfn': _json_ready(total_energy_wfn),
            'iqa': _json_ready(total_energy_iqa),
            'dispersion': _json_ready(total_energy_dispersion) if DISPERSION else None,
            'iqa_plus_dispersion': _json_ready(total_energy_iqa + total_energy_dispersion) if DISPERSION else None,
            'recovery_error_ha': _json_ready(per_step_errors_ha),
            'recovery_error_kj_mol': _json_ready(per_step_errors_kj),
            'recovery_error_definition': 'E_WFN - (E_IQA + E_disp): how far the IQA energy sits from the WFN energy',
            'closure_error_ha': _json_ready(closure_error_ha),
            'closure_error_kj_mol': _json_ready(closure_error_kj),
            # Each atom is credited half of every pair it belongs to, so summing over
            # all atoms counts each A<B pair once, in full — no /2 survives the sum.
            'closure_error_definition': ('sum_A E_IQA_Intra(A) + sum_{A<B} E_IQA_Inter(A,B) - sum_A E_IQA(A): '
                                         'whether the term decomposition re-sums to the reported IQA energy. '
                                         'A different quantity from recovery_error — neither replaces the other.'),
            'rmse_kj_mol': float(rmse_kj),
        },
        'terms': {
            'atom_intra': _term_block(iqa_intra_header_iqa, iqa_intra_iqa,
                                      properties=intra_prop,
                                      extra={'property_names': list(intra_prop_names),
                                             'layout': 'property-major: row = property_i * n_atoms + atom_i',
                                             'definition': 'E_IQA_Intra(A) per atom, as read from the .sum file',
                                             'includes_dispersion': False}),
            'atom_inter': atom_inter_block,
            'atom_dispersion': atom_disp_block,
            'atom_totals': _term_block(atom_entity_headers, atom_entity_totals,
                                       extra={'definition': 'E_IQA_Intra(A) + sum_B E_IQA_Inter(A,B)/2',
                                              'includes_dispersion': False}),
            'atom_iqa_reported': _term_block([str(a) for a in atoms], E_raw,
                                             extra={'definition': 'E_IQA(A) as reported in the .sum file'}),
            'fragment_intra': _term_block(
                iqf_intra_header, iqf_intra,
                extra={'definition': ('sum_{A in F} E_IQA_Intra(A) + sum_{A<B in F} E_IQA_Inter(A,B)'
                                      + (' + sum_{A<B in F} E_Disp(A,B)' if DISPERSION else '')),
                       'includes_own_dispersion': bool(DISPERSION),
                       'note': ('this is the array the REG_IQF intra REG was run on. With dispersion on it '
                                'folds in each fragment own-pair E_Disp, so it does NOT mean the same thing '
                                'as terms.atom_intra. Use fragment_intra_excl_dispersion for the atom-level '
                                'sense, or subtract fragment_own_dispersion.')}) if IQF else None,
            'fragment_intra_excl_dispersion': _term_block(
                iqf_intra_header, iqf_intra_no_disp,
                extra={'definition': 'sum_{A in F} E_IQA_Intra(A) + sum_{A<B in F} E_IQA_Inter(A,B)',
                       'includes_own_dispersion': False,
                       'note': 'same sense as terms.atom_intra; this is what fragment_totals is built from'}
                ) if IQF else None,
            'fragment_own_dispersion': _term_block(
                ['E_Disp_own(' + str(f) + ')' for f in Frag_names], fragment_own_dispersion,
                extra={'definition': 'sum_{A<B in F} E_Disp(A,B): dispersion between atoms of the same fragment',
                       'note': 'fragment_intra minus fragment_intra_excl_dispersion'}
                ) if fragment_own_dispersion is not None else None,
            'fragment_inter': _term_block(
                iqf_inter_header, iqf_inter,
                extra={'definition': 'sum_{A in F, B in G} E_IQA_Inter(A,B) for fragments F != G',
                       'includes_dispersion': False}) if IQF else None,
            'fragment_dispersion': _term_block(
                iqa_disp_header, iqa_disp,
                extra={'definition': 'sum_{A in F, B in G} E_Disp(A,B) for fragments F != G — cross-fragment only',
                       'note': 'own-fragment dispersion is in fragment_own_dispersion, not here'}
                ) if (IQF and DISPERSION) else None,
            'fragment_totals': _term_block(
                entity_headers, entity_totals,
                extra={'definition': ('fragment_intra_excl_dispersion(F) '
                                      '+ 1/2 sum_{G != F} E_IQA_Inter(F,G)'),
                       'includes_dispersion': False,
                       'note': ('contains no dispersion at all, neither own nor cross — unlike '
                                'fragment_intra. Add fragment_own_dispersion and half of '
                                'fragment_dispersion to get a dispersion-inclusive total.')}) if IQF else None,
            'charge_transfer': _term_block(iqa_charge_transfer_headers, iqa_charge_transfer_terms)
                               if CHARGE_TRANSFER_POLARISATION else None,
            'polarisation': _term_block(iqa_polarisation_headers, iqa_polarisation_terms)
                            if CHARGE_TRANSFER_POLARISATION else None,
        },
        'charge_distribution': {
            'per_atom_q_e': _json_ready(q_raw) if q_raw is not None else None,
            'sum_q_e': _json_ready(np.nansum(q_raw, axis=0)) if q_raw is not None else None,
        },
        'direct_multipoles': _json_ready(direct_multipoles) if direct_multipoles is not None else None,
        'iqa_atom_total': _json_ready(E_raw) if E_raw is not None else None,
        'integration': {
            'lagrangian': _term_block([str(a) for a in atoms], lagrangian_values,
                                      extra={'definition': 'L(A) per atom per step; null where unreadable'}),
            'kinetic_energy': _term_block([str(a) for a in atoms], T_raw) if T_raw is not None else None,
            'charge': _term_block([str(a) for a in atoms], q_raw) if q_raw is not None else None,
            'abs_l_over_t': _term_block([str(a) for a in atoms], l_over_t) if l_over_t is not None else None,
            'thresholds': {
                'lagrangian_hydrogen': _L_THRESHOLD_H,
                'lagrangian_heavy': _L_THRESHOLD_HEAVY,
            },
        },
        'xyz_structures': structure_payload,
        'quality_report': {
            'missing_files': missing_files,
            'bad_lagrangians': [
                {
                    'step': entry[0],
                    'atom': entry[1],
                    'lagrangian': entry[2],
                    'threshold': entry[3],
                    'atomic_file': entry[4],
                }
                for entry in bad_L
            ],
            'resubmit_paths': resubmit,
        },
    }

    # Absent term tables stay as explicit nulls rather than being dropped, matching how
    # the rest of the bundle reports "this run did not produce that" — a reader hitting
    # a null knows it looked in the right place.  Only the per-level role maps below
    # drop their empty entries, since a role that does not exist has no key to name.

    # A reader can check this instead of probing for keys to find out what the run
    # actually produced.
    transfer_bundle['availability'] = {
        'levels': sorted(levels.keys()),
        'pair_energies': atom_inter_block is not None,
        'all_pairs_written': len(kept_pairs) == n_pairs,
        # REG values are one number per term, so they are always written for every
        # pair even when the per-step curves of the flat pairs are pruned away.
        'reg_covers_all_pairs': True,
        'dispersion': atom_disp_block is not None,
        'fragments': fragment_definitions is not None,
        'geometry': bool(structure_payload),
        'pair_distances': pair_distances is not None,
        'multipoles': direct_multipoles is not None,
        'lagrangians': any(v is not None for row in lagrangian_values for v in row),
        'charge_transfer_polarisation': bool(CHARGE_TRANSFER_POLARISATION),
        'separate_error_segments': (error_segments['critical_point_indices']
                                    != energy_segments['critical_point_indices']),
    }

    # ONE bundle per run.  An IQF run used to write a second, near-identical file under
    # REG_IQA_results/ that differed only in which level filled the top-level reg keys —
    # two files to keep in step for no extra information.  Both levels now sit in this
    # one file, over a single shared copy of the data they have in common.
    transfer_bundle_path = os.path.join(cwd, SYS + '_results', SYS + '_model_transfer.json')
    indent = option.bundle_indent if option.bundle_indent and option.bundle_indent > 0 else None
    with open(transfer_bundle_path, 'w', encoding='utf-8') as bundle_handle:
        json.dump(_json_ready(transfer_bundle), bundle_handle, indent=indent,
                  separators=(',', ':') if indent is None else None, allow_nan=False)
    print('  Model-transfer bundle written to: {p}  ({s:.1f} MB)'.format(
        p=transfer_bundle_path, s=os.path.getsize(transfer_bundle_path) / (1024.0 * 1024.0)))
    print('  Levels in this bundle: ' + ', '.join(
        '{n} ({c} {k}s)'.format(n=name, c=lvl['entity_count'], k=lvl['entity_kind'])
        for name, lvl in sorted(levels.items()))
        + '  [primary: ' + transfer_bundle['primary_level'] + ']')
    if len(kept_pairs) < n_pairs:
        print('  {k} of {n} atom pairs written; the {d} left out drift by {r:.4f} kJ/mol in total '
              '(budget {t} kJ/mol) and are summed into '
              'terms.atom_inter.pruning.residual_per_property. '
              'Use --bundle-pair-budget 0 to write every pair.'.format(
                  k=len(kept_pairs), n=n_pairs, d=n_pairs - len(kept_pairs),
                  r=residual_drift_ha * HA_TO_KJ, t=pair_budget_kj))

    # BUILD INTEGRATION ERROR REPORT (printed to stdout and written to file)
    lines_out = []
    lines_out.append(sep_wide)
    lines_out.append('  IQA INTEGRATION ERROR REPORT')
    lines_out.append(sep_wide)
    lines_out.append('')
    lines_out.append('  {:<25s} {:>12s}  {:>18s}  {:>18s}  {:>22s}'.format(
        'Step', 'CC', 'WFN energy (Ha)', 'IQA energy (Ha)', 'Recovery error (kJ/mol)'))
    lines_out.append('  ' + sep_inner)
    for i in range(len(reg_folders)):
        flag = '  ***' if abs(per_step_errors_kj[i]) > 1.0 else ''
        lines_out.append('  {:<25s} {:>12.4f}  {:>18.8f}  {:>18.8f}  {:>+22.4f}{}'.format(
            reg_folders[i], float(cc[i]),
            total_energy_wfn[i], iqa_for_error[i],
            per_step_errors_kj[i], flag))
    lines_out.append('  ' + sep_inner)
    lines_out.append('  Recovery error RMSE: {:.4f} kJ/mol'.format(rmse_kj))
    lines_out.append('')

    report_text = '\n'.join(lines_out)
    print(report_text)
    error_report_path = cwd + '/' + SYS + '_results/integration_errors.txt'
    with open(error_report_path, 'w') as _ef:
        _ef.write(report_text + '\n')

    ###############################################################################
    #                                                                             #
    #                             WRITE CSV FILES                                 #
    #                                                                             #
    ###############################################################################
    os.chdir(cwd + '/' + SYS + "_results")
    dataframe_list = []

    if WRITE:
        # initialise excel files
        writer = pd.ExcelWriter(path=cwd + '/' + SYS + "_results/REG.xlsx", engine='xlsxwriter')
        energy_writer = pd.ExcelWriter(path=cwd + '/' + SYS + "_results/Energy.xlsx", engine='xlsxwriter')
        error_writer = pd.ExcelWriter(path=cwd + '/' + SYS + "_results/REG_Error.xlsx", engine='xlsxwriter')
        if IQF:
            compare_writer = pd.ExcelWriter(path=cwd + '/' + SYS + "_results/REG_IQA_vs_IQF.xlsx", engine='xlsxwriter')
        _used_writer_names = set()
        # ENERGY and CONTROL  COORDINATE ONLY FILES
        df_energy_output = pd.DataFrame()
        df_energy_output['WFN'] = total_energy_wfn
        df_energy_output['IQA'] = total_energy_iqa
        df_energy_output.index = cc
        if DISPERSION:
            df_energy_output['D3'] = total_energy_dispersion
            df_energy_output['IQA+D3'] = total_energy_iqa + total_energy_dispersion
        df_energy_output.to_csv('total_energy.csv', sep=',')
        df_energy_output.to_excel(energy_writer, sheet_name="total_energies")

        # LAGRANGIAN |L(A)| PER ATOM PER STEP
        L_abs_data = {
            str(atom): [
                abs(lagrangians[i].get(atom.lower()))
                if lagrangians[i].get(atom.lower()) is not None
                else np.nan
                for i in range(len(reg_folders))
            ]
            for atom in atoms
        }
        df_lagrangian_out = pd.DataFrame(L_abs_data, index=cc)
        df_lagrangian_out.index.name = 'CC'
        df_lagrangian_out.to_csv('lagrangian_L.csv', sep=',')
        df_lagrangian_out.to_excel(energy_writer, sheet_name='Lagrangian_L')

        pd.DataFrame(data=np.array(iqa_intra), index=iqa_intra_header, columns=cc).rename_axis('TERM').to_excel(energy_writer,
                                                                                            sheet_name='intra-atomic_energies')
        pd.DataFrame(data=np.array(iqa_inter), index=iqa_inter_header, columns=cc).rename_axis('TERM').to_excel(energy_writer,
                                                                                            sheet_name='inter-atomic_energies')

        if IQF:
            for i,reg_intra_comp in enumerate(iqf_intra_comp_list):
                for j in range(len(reg_intra_comp[0])):
                    df_iqf_intra = rv.create_term_dataframe(reg_intra_comp, iqf_intra_comp_head[i],j)
                    df_iqf_intra_sorted = df_iqf_intra.sort_values('REG')
                    df_iqf_intra_sorted.to_excel(writer, sheet_name=_safe_sheet_name(iqf_intra_header[i] + "_seg_" + str(j + 1), _used_writer_names))


        # INTER AND INTRA PROPERTIES RE-ARRANGEMENT
        list_property_final = []
        final_properties_comparison = []
        for i in range(len(reg_inter[0])):
            list_property_sorted = []
            properties_comparison = []
            df_inter = rv.create_term_dataframe(reg_inter, iqa_inter_header,i)
            df_intra = rv.create_term_dataframe(reg_intra, iqa_intra_header, i)
            for j in range(len(inter_prop)):
                df_property = rv.filter_term_dataframe(df_inter, inter_prop[j], inter_prop_names[j])
                if j <= 1:
                    properties_comparison.append(df_property)
                df_property.to_csv(inter_prop_names[j] + "_seg_" + str(i + 1) + ".csv", sep=',')
                df_property.to_excel(writer, sheet_name=_safe_sheet_name(inter_prop_names[j] + "_seg_" + str(i + 1), _used_writer_names))
                list_property_sorted.append(df_property)  # selection deferred: the cut is not known until every term is pooled
            for j in range(len(intra_prop)):
                df_property = rv.filter_term_dataframe(df_intra, intra_prop[j], intra_prop_names[j])
                if j == 0:
                    properties_comparison.append(df_property)
                df_property.to_csv(intra_prop_names[j] + "_seg_" + str(i + 1) + ".csv", sep=',')
                df_property.to_excel(writer, sheet_name=_safe_sheet_name(intra_prop_names[j] + "_seg_" + str(i + 1), _used_writer_names))
                list_property_sorted.append(df_property)  # selection deferred: the cut is not known until every term is pooled
            list_property_final.append(list_property_sorted)
            final_properties_comparison.append(properties_comparison)

        # DISPERSION OUTPUT
        disp_dic = {}
        if DISPERSION:
            df_dispersion_sorted = pd.DataFrame()
            disp_name_old = 'E_Disp(A,B)'
            disp_name_new = 'Vdisp'
            for i in range(len(reg_inter[0])):
                df_disp = rv.create_term_dataframe(reg_disp, iqa_disp_header,i)
                df_disp_new = rv.filter_term_dataframe(df_disp, disp_name_old, disp_name_new)
                disp_dic["Seg_" + str(i)] = df_disp_new
                df_disp_new.to_csv(disp_name_new + "_seg_" + str(i + 1) + ".csv", sep=',')
                df_disp_new.to_excel(writer, sheet_name=_safe_sheet_name(disp_name_new + "_seg_" + str(i + 1), _used_writer_names))
                df_disp_new.dropna(axis=0, how='any', subset=None,
                            inplace=True)  # get rid of "NaN" terms which have a null REG Value
                df_dispersion_sorted = pd.concat([df_dispersion_sorted.reset_index(drop=True),
                                                rv.select_significant_terms(df_disp_new, PROPERTY_MIN_TABLE_ROWS,
                                                                            PROPERTY_MAX_TABLE_ROWS, R_THRESHOLD)], axis=1)
            df_dispersion_sorted.to_csv('REG_' + disp_name_new + '_analysis.csv', sep=',')
            df_dispersion_sorted.to_excel(writer, sheet_name="REG_" + disp_name_new)
            rv.pandas_REG_dataframe_to_table(df_dispersion_sorted, 'REG_' + disp_name_new + '_table', SAVE_FIG=SAVE_FIG)
            pd.DataFrame(data=np.array(iqa_disp), index=iqa_disp_header, columns=cc).rename_axis('TERM').to_excel(energy_writer, sheet_name='dispersion_energies')
        # ── REG_final SIGNIFICANCE CUT ───────────────────────────────────────
        # One |REG| cut per segment, calibrated on the pooled table of every term
        # (Vcl, Vxc, Eintra and dispersion together) rather than on any one property.
        # Every REG value here is a slope against the same total energy, so pooling
        # is what makes the headline table mean "significant against the whole
        # decomposition": a property whose terms are all small — dispersion, usually —
        # contributes nothing to it without needing to be special-cased.  The
        # per-property working tables are calibrated separately, further down.
        seg_pooled = []
        seg_threshold = []
        for i in range(len(reg_inter[0])):
            df_pooled = pd.concat(final_properties_comparison[i])
            if DISPERSION:
                df_pooled = pd.concat([df_pooled, disp_dic["Seg_" + str(i)]])
            df_pooled = df_pooled.sort_values('REG').reset_index(drop=True)
            df_significant = rv.select_significant_terms(df_pooled, MIN_TABLE_ROWS, MAX_TABLE_ROWS, R_THRESHOLD)
            seg_pooled.append(df_significant)
            seg_threshold.append(df_significant.attrs['reg_threshold'])
            print('Segment ' + str(i + 1) + ': significance cut |REG| >= '
                  + ('%.4f' % seg_threshold[i] if seg_threshold[i] is not None else 'n/a')
                  + ' (' + (('max/%.0f' % (1 / df_significant.attrs['reg_fraction']))
                            if df_significant.attrs['reg_fraction'] else 'rank-limited')
                  + '), ' + str(len(df_significant)) + ' significant terms')
        # ─────────────────────────────────────────────────────────────────────

        # CHARGE-TRANSFER and POLARISATION
        if CHARGE_TRANSFER_POLARISATION:
            df_ct_pl_sorted = pd.DataFrame()
            for i in range(len(reg_inter[0])):
                df_pl = rv.filter_term_dataframe(rv.create_term_dataframe(reg_pl, iqa_polarisation_headers,i),
                                                'Vpl_IQA(A,B)',
                                                'Vpl')
                df_ct = rv.filter_term_dataframe(rv.create_term_dataframe(reg_ct, iqa_charge_transfer_headers,i),
                                                'Vct_IQA(A,B)',
                                                'Vct')
                df_pl.to_csv("Vpl_seg_" + str(i + 1) + ".csv", sep=',')
                df_pl.to_excel(writer, sheet_name=_safe_sheet_name("Vpl_seg_" + str(i + 1), _used_writer_names))
                df_ct.to_csv("Vct_seg_" + str(i + 1) + ".csv", sep=',')
                df_ct.to_excel(writer, sheet_name=_safe_sheet_name("Vct_seg_" + str(i + 1), _used_writer_names))
                df_temp = pd.concat([df_pl, df_ct]).sort_values('REG').reset_index(drop=True)
                df_ct_pl_sorted = pd.concat([df_ct_pl_sorted.reset_index(drop=True),
                                            rv.select_significant_terms(df_temp, PROPERTY_MIN_TABLE_ROWS,
                                                                        PROPERTY_MAX_TABLE_ROWS, R_THRESHOLD)], axis=1)
            df_ct_pl_sorted.to_csv('REG_Vct-Vpl_analysis.csv', sep=',')
            df_ct_pl_sorted.to_excel(writer, sheet_name='REG_Vct-Vpl')
            rv.pandas_REG_dataframe_to_table(df_ct_pl_sorted, 'REG_Vct-Vpl_table', SAVE_FIG=SAVE_FIG)
            pd.DataFrame(
                data=np.concatenate((np.array(iqa_polarisation_terms), np.array(iqa_charge_transfer_terms))),
                index=np.concatenate((iqa_polarisation_headers, iqa_charge_transfer_headers)),
                columns=cc).rename_axis('TERM').to_excel(energy_writer, sheet_name='pl_ct_energies')

        # OUTPUT OF ALL INTER AND INTRA TERMS SELECTED BY THE USER
        # Calibrated on each property's own leading term, not on the pooled cut used
        # for REG_final.  These are working tables — they feed mechanism figures, which
        # need more terms than belong in the final table — so a property is ranked
        # against itself here and small contributions are kept rather than dropped.
        all_prop_names = inter_prop_names + intra_prop_names
        for i in range(len(inter_prop) + len(intra_prop)):
            df_property_sorted = pd.DataFrame()
            for j in range(len(reg_inter[0])):
                df_property_sorted = pd.concat([df_property_sorted,
                                                rv.select_significant_terms(list_property_final[j][i],
                                                                            PROPERTY_MIN_TABLE_ROWS,
                                                                            PROPERTY_MAX_TABLE_ROWS, R_THRESHOLD)], axis=1)
            df_property_sorted.to_csv('REG_' + all_prop_names[i] + '_analysis.csv', sep=',')
            df_property_sorted.to_excel(writer, sheet_name=_safe_sheet_name('REG_' + all_prop_names[i], _used_writer_names))
            rv.pandas_REG_dataframe_to_table(df_property_sorted, 'REG_' + all_prop_names[i] + '_table', SAVE_FIG=SAVE_FIG)

        # FINAL COMPARISON
        df_final_sorted = pd.DataFrame()
        for i in range(len(reg_inter[0])):
            df_final = pd.DataFrame()
            for j in range(3):
                df_final = pd.concat([df_final, final_properties_comparison[i][j]])
            if DISPERSION:
                df_disp_seg = disp_dic["Seg_" + str(i)]
                df_final = pd.concat([df_final, df_disp_seg])
            dataframe_list.append(df_final)
            df_final = df_final.sort_values('REG').reset_index(drop=True)
            df_final_sorted = pd.concat([df_final_sorted.reset_index(drop=True),
                                        seg_pooled[i]], axis=1)  # the pooled selection the threshold came from
            df_final.to_csv('REG_full_comparison_seg_' + str(i + 1) + '.csv', sep=',')
            df_final.to_excel(writer, sheet_name=_safe_sheet_name('REG_full_comparison_seg_' + str(i+1), _used_writer_names))
        df_final_sorted.to_csv('REG_final_analysis.csv', sep=',')
        df_final_sorted.to_excel(writer, sheet_name='REG_final')
        rv.pandas_REG_dataframe_to_table(df_final_sorted, 'REG_final_table', SAVE_FIG=SAVE_FIG)

        # ── REG_IQA RANKING ──────────────────────────────────────────────────
        # Ranks each atom (or fragment, if IQF) by the REG of its IQA total energy,
        # E_intra(A) + sum_B (1/2 * E_inter(A,B)), against the control coordinate.
        df_entity_final_sorted = pd.DataFrame()
        for i in range(len(reg_entity_totals[0])):
            df_entity = rv.create_term_dataframe(reg_entity_totals, entity_headers, i)
            df_entity_sorted = df_entity.sort_values('REG').reset_index(drop=True)
            df_entity_sorted.to_csv('REG_IQA_ranking_seg_' + str(i + 1) + '.csv', sep=',')
            df_entity_sorted.to_excel(writer, sheet_name=_safe_sheet_name('REG_IQA_ranking_seg_' + str(i + 1), _used_writer_names))
            # Self-calibrated, not sharing the pooled cut: these are whole-entity IQA
            # totals, each a sum over many pair terms, so they sit on a different scale.
            df_entity_final_sorted = pd.concat([df_entity_final_sorted.reset_index(drop=True),
                                                rv.select_significant_terms(df_entity_sorted, MIN_TABLE_ROWS, MAX_TABLE_ROWS, R_THRESHOLD)], axis=1)
        df_entity_final_sorted.to_csv('REG_IQA_ranking_analysis.csv', sep=',')
        df_entity_final_sorted.to_excel(writer, sheet_name='REG_IQA_ranking')
        rv.pandas_REG_dataframe_to_table(df_entity_final_sorted, 'REG_IQA_ranking_table', SAVE_FIG=SAVE_FIG)
        # ─────────────────────────────────────────────────────────────────────

        # ── RECOVERY ERROR REG OUTPUT ─────────────────────────────────────────
        # Each IQA term is correlated against E_WFN − E_IQA (the recovery error)
        # rather than E_WFN itself.  Segments are defined on the error surface.
        # All output goes to the dedicated REG_Error.xlsx file.
        df_err_final_sorted = pd.DataFrame()
        for i in range(len(reg_inter_err[0])):
            df_err_inter = rv.create_term_dataframe(reg_inter_err, iqa_inter_header, i)
            df_err_intra = rv.create_term_dataframe(reg_intra_err, iqa_intra_header, i)
            df_err_inter.to_csv('REG_err_inter_seg_' + str(i + 1) + '.csv', sep=',')
            df_err_intra.to_csv('REG_err_intra_seg_' + str(i + 1) + '.csv', sep=',')
            df_err_inter.to_excel(error_writer, sheet_name='REG_err_inter_seg_' + str(i + 1))
            df_err_intra.to_excel(error_writer, sheet_name='REG_err_intra_seg_' + str(i + 1))
            df_err_combined = pd.concat([df_err_inter, df_err_intra]).sort_values('REG').reset_index(drop=True)
            df_err_combined.to_csv('REG_err_full_comparison_seg_' + str(i + 1) + '.csv', sep=',')
            df_err_combined.to_excel(error_writer, sheet_name='REG_err_full_seg_' + str(i + 1))
            df_err_filtered = df_err_combined[df_err_combined['R'].abs() >= ERR_R_THRESHOLD].reset_index(drop=True)
            df_err_filtered.to_excel(error_writer, sheet_name='REG_err_highR_seg_' + str(i + 1))
            df_err_atom = rv.create_term_dataframe(reg_atom_err, _atom_err_headers, i)
            df_err_atom_filtered = df_err_atom[df_err_atom['R'].abs() >= ERR_R_THRESHOLD].sort_values('REG').reset_index(drop=True)
            df_err_atom_filtered.to_excel(error_writer, sheet_name='REG_err_atomE_highR_seg_' + str(i + 1))
            # Self-calibrated too: these REG values are slopes against the recovery
            # error, not against the total energy, so the pooled cut does not apply.
            df_err_final_sorted = pd.concat([
                df_err_final_sorted.reset_index(drop=True),
                rv.select_significant_terms(df_err_combined, MIN_TABLE_ROWS, MAX_TABLE_ROWS, R_THRESHOLD)
            ], axis=1)
        df_err_final_sorted.to_csv('REG_err_final_analysis.csv', sep=',')
        df_err_final_sorted.to_excel(error_writer, sheet_name='REG_err_final')
        rv.pandas_REG_dataframe_to_table(df_err_final_sorted, 'REG_err_final_table', SAVE_FIG=SAVE_FIG)
        # ─────────────────────────────────────────────────────────────────────

        # ── IQF INTER-FRAGMENT BREAKDOWN OUTPUT ──────────────────────────────
        # For each segment, the top n_terms IQF inter-fragment terms (by |REG|)
        # are broken down to their constituent atom-pair contributions, written to
        # REG_IQA_vs_IQF.xlsx.  This mirrors the intra breakdown already in REG.xlsx.
        if IQF:
            _used_cmp_names = set()
            n_segs = len(reg_inter[0])
            n_segs_intra = len(reg_intra[0])
            for seg_j in range(n_segs):
                # Summary sheet: all IQF inter terms for this segment ranked by |REG|
                df_iqf_inter_seg = rv.create_term_dataframe(reg_inter, iqa_inter_header, seg_j)
                df_iqf_inter_seg_sorted = df_iqf_inter_seg.reindex(
                    df_iqf_inter_seg['REG'].abs().sort_values(ascending=False).index
                ).reset_index(drop=True)
                df_iqf_inter_seg_sorted.to_excel(compare_writer,
                    sheet_name=_safe_sheet_name('IQF_inter_seg' + str(seg_j + 1), _used_cmp_names))

                # Breakdown sheets for the top n_terms IQF inter terms
                top_indices = df_iqf_inter_seg_sorted.index[:n_terms]
                for rank, df_row_idx in enumerate(top_indices):
                    term_name = df_iqf_inter_seg_sorted.loc[df_row_idx, 'TERM']
                    # Map term name back to iqf_inter row index
                    matches = [k for k, h in enumerate(iqa_inter_header) if h == term_name]
                    if not matches:
                        continue
                    inter_idx = matches[0]
                    comps = iqf_inter_comp_list[inter_idx]
                    if not comps[0][seg_j]:
                        continue
                    df_bkdn = rv.create_term_dataframe(comps, iqf_inter_comp_head[inter_idx], seg_j)
                    df_bkdn_sorted = df_bkdn.sort_values('REG').reset_index(drop=True)
                    sheet = _safe_sheet_name(
                        'bkdn' + str(rank + 1) + '_' + term_name + '_s' + str(seg_j + 1),
                        _used_cmp_names)
                    df_bkdn_sorted.to_excel(compare_writer, sheet_name=sheet)

            for seg_j in range(n_segs_intra):
                df_iqf_intra_seg = rv.create_term_dataframe(reg_intra, iqa_intra_header, seg_j)
                df_iqf_intra_seg_sorted = df_iqf_intra_seg.reindex(
                    df_iqf_intra_seg['REG'].abs().sort_values(ascending=False).index
                ).reset_index(drop=True)
                df_iqf_intra_seg_sorted.to_excel(compare_writer,
                    sheet_name=_safe_sheet_name('IQF_intra_seg' + str(seg_j + 1), _used_cmp_names))

                for rank, df_row in df_iqf_intra_seg_sorted.iterrows():
                    if rank >= n_terms:
                        break
                    term_name = df_row['TERM']
                    matches = [k for k, h in enumerate(iqa_intra_header) if h == term_name]
                    if not matches:
                        continue
                    intra_idx = matches[0]
                    grp_reg = iqf_intra_prop_list[intra_idx]
                    grp_heads = iqf_intra_prop_heads[intra_idx]
                    if not grp_reg[0][seg_j]:
                        continue
                    df_grp = rv.create_term_dataframe(grp_reg, grp_heads, seg_j)
                    df_grp_sorted = df_grp.sort_values('REG').reset_index(drop=True)
                    sheet = _safe_sheet_name(
                        'intra' + str(rank + 1) + '_' + term_name + '_s' + str(seg_j + 1),
                        _used_cmp_names)
                    df_grp_sorted.to_excel(compare_writer, sheet_name=sheet)

            compare_writer.close()
        # ─────────────────────────────────────────────────────────────────────

        writer.close()
        energy_writer.close()
        error_writer.close()
        #rv.plot_violin([dataframe_list[i]['R'] for i in range(len(reg_inter[0]))], save=SAVE_FIG,
                    #file_name='violin.png')  # Violing plot of R vs Segments

    ###############################################################################
    #                                                                             #
    #                                   GRAPHS                                    #
    #                                                                             #
    ###############################################################################
    if AUTO:
        critical_points = reg.find_critical(total_energy_wfn, cc, min_points=POINTS, use_inflex=INFLEX)
    else:
        critical_points = turning_points

    rv.plot_segment(cc, 2625.50 * (total_energy_wfn - (sum(total_energy_wfn) / len(total_energy_wfn))), critical_points,
                    annotate=ANNOTATE,
                    label=LABELS,
                    y_label=r'Relative Energy [$kJ.mol^{-1}$]', x_label=X_LABEL, title=SYS,
                    save=SAVE_FIG, file_name='REG_analysis.png')

    _err_kj = np.array(per_step_errors_kj)
    rv.plot_segment(cc, _err_kj - _err_kj.mean(), critical_points,
                    annotate=ANNOTATE,
                    label=LABELS,
                    y_label=r'$E_\mathrm{WFN} - E_\mathrm{IQA}$ [$kJ.mol^{-1}$]',
                    x_label=X_LABEL,
                    title=SYS + ' — Recovery Error',
                    save=SAVE_FIG, file_name='REG_err_analysis.png')

    if DETAILED_ANALYSIS:
        for i in range(len(reg_inter[0])):
            rv.generate_data_vis(dataframe_list[i], [dataframe_list[i]['R'] for i in range(len(reg_inter[0]))],
                                n_terms, save=SAVE_FIG, file_name='detailed_seg_' + str(i + 1) + '.png',
                                title=SYS + ' seg. ' + str(i + 1))

    ###############################################################################
    #                                                                             #
    #                    NAME THE OUTPUT AFTER THE SYSTEM                         #
    #                                                                             #
    ###############################################################################
    # Done once everything this analysis writes is on disk, and before REG_Multi
    # starts, so that a single sweep over the finished directory covers every file
    # — including any added to the analysis later — rather than each of the
    # forty-odd writes above having to remember the prefix for itself.  Nothing
    # here is read back, so renaming is safe.
    if PREFIX:
        os.chdir(cwd)
        reg_setup.apply_output_prefix(cwd + '/' + SYS + '_results', PREFIX, verbose=True)

    ###############################################################################
    #                                                                             #
    #                           REG_Multi ANALYSIS                                #
    #                                                                             #
    ###############################################################################
    # Run last, once everything this analysis owns is written, so a REG_Multi
    # problem — most often AIMAll having been run without pairwise IQA — cannot
    # cost the user their REG-IQA results.  It re-reads the directory rather than
    # being handed this run's arrays, which keeps the two analyses independent;
    # the .sum parse is cached process-wide, so the exact V_cl it needs is free
    # here and only the .int moments are genuinely new I/O.
    if REG_MULTI:
        os.chdir(cwd)
        print('')
        print(sep_wide)
        print('  RUNNING REG_Multi — rank-resolved multipolar electrostatics')
        print(sep_wide)
        # A level-specific results directory, so running IQF after IQA does not
        # overwrite what the IQA run produced — the same reason REG_IQA_results and
        # REG_IQF_results are kept apart.
        multi_results = reg_setup.prefixed(
            'REG_Multi_' + ('IQF' if IQF else 'IQA') + '_results', PREFIX)
        multi_argv = ['-d', reg_dir, '-c', FRAG_CONFIG_PATH, '-o', multi_results,
                      '--no-collect']
        if PREFIX:
            multi_argv += ['--prefix', PREFIX]
        multi_extra = shlex.split(option.multi_options) if option.multi_options else []
        if option.ignore_fragments and not any(
                arg == '-i' or arg.startswith('--ignore-fragments') for arg in multi_extra):
            multi_argv.append('--ignore-fragments')
        # Under IQF, default REG_Multi to the scope that mirrors what IQF does with
        # the IQA terms: atom pairs inside a fragment summed into that fragment's own
        # channel, atom pairs between fragments summed into the fragment-pair channel.
        # Only a default — an explicit scope in auto_reg.config or in --multi-options
        # is left alone.
        _scope_given = ('scope' in reg_setup.read_multipole_config(FRAG_CONFIG_PATH)
                        or any(arg == '-s' or arg.startswith('--scope') or arg.startswith('-s')
                               for arg in multi_extra))
        if not _scope_given:
            if IQF:
                multi_argv += ['--scope', 'all']
                print('  IQF run: REG_Multi will use --scope all, so atom pairs inside a fragment')
                print('  are summed into that fragment own channel as well as between fragments.')
            else:
                # An IQA run is atom-wise, so its REG_Multi is too: one set of rank
                # terms per atom pair, not per fragment pair.  The config may still
                # define fragments for a later IQF run, and they are ignored here for
                # the same reason auto_reg ignores them without -f T.
                multi_argv += ['--scope', 'pairs']
                print('  IQA run: REG_Multi will use --scope pairs, reporting one set of multipole')
                print('  terms per atom pair. Use --multi-options to choose a fragment scope instead.')
        multi_argv += multi_extra
        try:
            import reg_multipole  # type: ignore
            reg_multipole.main(argv=multi_argv)
        except Exception as multi_err:
            print('')
            print(sep_wide)
            print('  REG_Multi ANALYSIS FAILED — the REG-IQA results above are unaffected')
            print(sep_wide)
            print('  ' + str(multi_err))
            print('')
            print('  Nothing was written to ' + multi_results + '. See REG_MULTI.md; the usual cause')
            print('  is AIMAll having been run without pairwise IQA, which REG_Multi needs for its')
            print('  convergence test.')
        finally:
            os.chdir(cwd)

    ###############################################################################
    #                                                                             #
    #                          COLLECT THIS RUN                                   #
    #                                                                             #
    ###############################################################################
    # Last, once REG_Multi has written its bundle too, so the folder holds
    # everything this run produced rather than everything it had produced by the
    # time the IQA analysis finished.  It is rebuilt from the directory each time,
    # so running IQF or REG_Multi later refreshes it instead of leaving it stale.
    if not option.no_collect:
        os.chdir(cwd)
        try:
            import reg_batch  # type: ignore
            reg_batch.collect_run(cwd, name=PREFIX, dir_prefix=PREFIX,
                                  level='REG_IQF' if IQF else 'REG_IQA',
                                  dir_name=option.collect_dir,
                                  command=[sys.argv[0]] + list(sys.argv[1:]))
        except Exception as collect_error:
            # A naming convenience must not cost a completed analysis.
            print('WARNING: could not collect this run — ' + str(collect_error))

    ###ENDING TIMER ###
    print("--- Total time for REG Analysis: {s} minutes ---".format(s=((time.time() - start_time) / 60)))


if __name__ == "__main__":
    # The exit code matters for a -R sweep: it is non-zero when any system failed,
    # so a sweep can be chained in a script without reading its output.
    sys.exit(main())

