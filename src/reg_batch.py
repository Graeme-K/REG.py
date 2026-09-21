"""
reg_batch.py

Run a REG driver over a folder of systems: every system below a root directory
is analysed exactly as it would be if you cd'd into it and ran the driver there,
and its output is named after the system so the results can be gathered together.

Each system is analysed in its own process.  That is what makes a long sweep
survivable: one system whose AIMAll run is missing a .sum, or whose wavefunction
never converged, fails on its own and the sweep carries on, and the memory a
238-atom system needs is handed back when it finishes rather than accumulating
across forty of them.

coded for the REG.py package
"""

import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
import time

from concurrent.futures import ThreadPoolExecutor

import reg_setup


SEP = 79 * '-'

COLLECTION_DIR = 'REG_sweep_collection'      # a sweep, at the root it swept
RUN_COLLECTION_DIR = 'REG_collection'        # a single run, beside its results
OVERVIEW_NAME = 'REG_collection_overview'    # the same file in both, so one reader serves both


# ---------------------------------------------------------------------------
# Naming the systems
# ---------------------------------------------------------------------------

def disambiguate(systems, root):
    """Make every system's name unique by pulling in the folders above it.

    Two systems called BEN under Chlorine/ and Fluorine/ would otherwise write
    BEN_REG.xlsx each, and collecting the sweep would silently lose one of them.
    The duplicated names only are extended — Chlorine_BEN, Fluorine_BEN — so a
    tree with no clashes keeps its short names.
    """
    root = os.path.abspath(root)
    for _ in range(8):  # bounded: each pass climbs one directory
        by_name = {}
        for system in systems:
            by_name.setdefault(system['name'], []).append(system)
        clashes = [group for group in by_name.values() if len(group) > 1]
        if not clashes:
            break
        progressed = False
        for group in clashes:
            for system in group:
                current = system.get('_naming_parent', system['path'])
                parent = os.path.dirname(current)
                # Climbing stops at the sweep root: its name is shared by every
                # system and so cannot tell two of them apart.
                if parent == current or parent == root or not parent.startswith(root):
                    continue
                system['_naming_parent'] = parent
                stem = reg_setup.sanitise_prefix(os.path.basename(parent))
                if not stem:
                    continue
                system['name'] = reg_setup.prefixed(system['name'], stem)
                progressed = True
        if not progressed:
            break

    # Whatever is left after climbing to the root (the same system name twice in
    # the same place cannot happen, but a sanitised name can still collide) is
    # made unique with a counter, so no two runs can write over each other.
    seen = {}
    for system in systems:
        name = system['name']
        if name in seen:
            seen[name] += 1
            system['name'] = '{n}_{i}'.format(n=name, i=seen[name])
        else:
            seen[name] = 1
    for system in systems:
        system.pop('_naming_parent', None)
    return systems


def print_plan(systems, root, results_names=None):
    """Show what a sweep would analyse, and under what name, before it starts."""
    print(SEP)
    print('  {n} SYSTEM(S) FOUND UNDER {r}'.format(n=len(systems), r=root))
    print(SEP)
    if not systems:
        print('  Nothing to analyse. A system is a directory holding an auto_reg.config')
        print('  with numbered geometry folders beneath it, or one holding at least two')
        print('  numbered folders that each contain a .wfn/.wfx.')
        return
    width = max(len(system['name']) for system in systems)
    print('  {name:<{w}}  {steps:>5}  {cfg:<7}  {path}'.format(
        name='NAME', w=width, steps='STEPS', cfg='CONFIG', path='PATH'))
    for system in systems:
        line = '  {name:<{w}}  {steps:>5}  {cfg:<7}  {path}'.format(
            name=system['name'], w=width, steps=system['steps'],
            cfg='yes' if system['config'] else 'no',
            path=os.path.relpath(system['path'], root))
        if results_names:
            existing = results_names(system)
            if existing:
                line += '   [already has ' + ', '.join(existing) + ']'
        print(line)
    print('')


# ---------------------------------------------------------------------------
# Running them
# ---------------------------------------------------------------------------

def _stream(process, log_handle, echo):
    """Copy the child's output to its log file, and to our own stdout when asked.

    The log is written whether or not it is echoed, so a parallel sweep — where
    echoing would interleave forty analyses into nonsense — still leaves every
    system's full output on disk to read afterwards.
    """
    for line in process.stdout:
        log_handle.write(line)
        log_handle.flush()
        if echo:
            sys.stdout.write(line)
            sys.stdout.flush()


def log_directory(root, name='reg_batch_logs'):
    """Where a sweep keeps its per-system logs.

    At the sweep root rather than inside each system: a .log sitting next to a
    system's geometry folders is picked up by the walk that looks for Gaussian
    outputs, and one collecting place is easier to read through afterwards.
    """
    path = os.path.join(os.path.abspath(root), name)
    try:
        os.makedirs(path, 0o755)
    except OSError:
        pass
    return path


def run_system(system, command, log_path, echo=True):
    """Run one system to completion in its own directory. Never raises."""
    started = time.time()
    print('  START  {n}  ({p})'.format(n=system['name'], p=system['path']))
    sys.stdout.flush()
    try:
        with open(log_path, 'w', encoding='utf-8', errors='replace') as log_handle:
            log_handle.write('$ ' + ' '.join(command) + '\n')
            log_handle.write('  in ' + system['path'] + '\n\n')
            process = subprocess.Popen(
                command, cwd=system['path'], stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
            _stream(process, log_handle, echo)
            returncode = process.wait()
    except Exception as run_error:  # a missing interpreter, an unreadable directory
        returncode = -1
        print('  ERROR  {n} — {e}'.format(n=system['name'], e=run_error))

    minutes = (time.time() - started) / 60.0
    print('  {status}  {n}  ({m:.1f} min)  log: {l}'.format(
        status='DONE  ' if returncode == 0 else 'FAILED', n=system['name'],
        m=minutes, l=log_path))
    sys.stdout.flush()
    return {'system': system, 'returncode': returncode, 'minutes': minutes,
            'log': log_path}


def exit_code(results):
    """0 when every system succeeded, 1 otherwise, so a sweep can be used in a
    script without reading its output."""
    return 1 if any(result.get('returncode') not in (0, None) for result in results) else 0


def run_batch(systems, command_for, log_path_for, jobs=1):
    """Analyse every system, in parallel when asked, and report what happened.

    Returns one result dict per system: what it was, whether it succeeded, how
    long it took and where its log is.
    """
    if not systems:
        return []

    jobs = max(1, int(jobs or 1))
    started = time.time()
    print(SEP)
    print('  ANALYSING {n} SYSTEM(S){j}'.format(
        n=len(systems),
        j='' if jobs == 1 else ' — {j} at a time'.format(j=jobs)))
    print(SEP)
    if jobs > 1:
        print('  Output of each system goes to its own log file rather than here,')
        print('  so the runs do not interleave.')
    print('')

    def work(system):
        return run_system(system, command_for(system), log_path_for(system),
                          echo=(jobs == 1))

    if jobs == 1:
        results = [work(system) for system in systems]
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(work, systems))

    failed = [result for result in results if result['returncode'] != 0]
    print('')
    print(SEP)
    print('  SWEEP COMPLETE — {ok} of {n} systems analysed in {m:.1f} minutes'.format(
        ok=len(results) - len(failed), n=len(results),
        m=(time.time() - started) / 60.0))
    print(SEP)
    width = max(len(result['system']['name']) for result in results)
    for result in results:
        print('  {name:<{w}}  {status:<7}  {m:>7.1f} min'.format(
            name=result['system']['name'], w=width,
            status='ok' if result['returncode'] == 0 else 'FAILED',
            m=result['minutes']))
    if failed:
        print('')
        print('  {n} system(s) failed. What went wrong is at the end of each log:'.format(
            n=len(failed)))
        for result in failed:
            print('    ' + result['log'])
    print('')
    return results


# ---------------------------------------------------------------------------
# Collecting a sweep
# ---------------------------------------------------------------------------
# A sweep leaves forty self-contained analyses in forty directories.  What it
# does not leave is a dataset: something a comparison across systems can be
# written against without first knowing where each system lives and what its
# results directory is called.  The collection directory is that dataset — every
# system's transfer bundle and config, under the system's own name, beside one
# overview file that says what each system is and what its analysis found.
#
# The bundles are copied verbatim rather than summarised or linked: the whole
# point of a transfer bundle is that a tool reading it needs nothing else, and a
# collection that can be moved to a laptop keeps that property.  The overview is
# a new file and never a substitute for them — it carries what a first pass over
# forty systems needs (size, path, energy range, recovery, segments, the leading
# REG terms), and points at the bundle for everything else.

def _load_json(path):
    """Read a JSON file, or return None and say why rather than stopping a sweep."""
    try:
        with open(path, encoding='utf-8') as handle:
            return json.load(handle)
    except Exception as read_error:
        print('  WARNING: could not read {p} — {e}'.format(p=path, e=read_error))
        return None


def _formula(elements):
    """'C6H6Cl' from the per-atom element list, carbon and hydrogen first."""
    if not elements:
        return None
    counts = {}
    for element in elements:
        symbol = str(element).strip().capitalize()
        counts[symbol] = counts.get(symbol, 0) + 1
    order = ([symbol for symbol in ('C', 'H') if symbol in counts]
             + sorted(symbol for symbol in counts if symbol not in ('C', 'H')))
    return ''.join(symbol + (str(counts[symbol]) if counts[symbol] > 1 else '')
                   for symbol in order)


def _span_kj(values, to_kj):
    """Peak-to-peak of an energy curve, in kJ/mol."""
    numbers = [value for value in (values or []) if isinstance(value, (int, float))]
    if len(numbers) < 2:
        return None
    return (max(numbers) - min(numbers)) * to_kj


def _top_reg_terms(level, r_threshold=0.0, per_segment=5):
    """The leading REG terms of each segment, ranked by |REG| across the blocks.

    Ranking intra, inter and dispersion together, behind the same weak |R| guard
    the run itself applied, is what the REG_final table does, so the overview
    agrees with the table in the results directory rather than offering a third
    opinion.  It is a shortlist and not that table: the significance cut, which
    decides how many terms are worth reporting, stays in the results directory.
    """
    reg_blocks = (level or {}).get('reg') or {}
    top = []
    for block_name in ('intra', 'inter', 'dispersion'):
        block = reg_blocks.get(block_name)
        if not block or not block.get('headers'):
            continue
        headers = block['headers']
        for segment_index, segment in enumerate(block.get('reg') or []):
            pearson = (block.get('pearson') or [])
            pearson_row = pearson[segment_index] if segment_index < len(pearson) else []
            for term_index, value in enumerate(segment):
                if value is None or term_index >= len(headers):
                    continue
                r_value = (pearson_row[term_index]
                           if term_index < len(pearson_row) else None)
                if r_threshold and (r_value is None or abs(r_value) < r_threshold):
                    continue
                top.append({
                    'segment': segment_index + 1,
                    'block': block_name,
                    'term': headers[term_index],
                    'reg': value,
                    'pearson': r_value,
                })

    by_segment = {}
    for entry in top:
        by_segment.setdefault(entry['segment'], []).append(entry)
    ranked = []
    for segment in sorted(by_segment):
        entries = sorted(by_segment[segment], key=lambda e: abs(e['reg']), reverse=True)
        ranked.extend(entries[:per_segment])
    return ranked


def summarise_reg_bundle(bundle):
    """What a comparison across systems needs from a REG-IQA/IQF bundle.

    Everything here is read defensively: a bundle from an older schema, or one
    whose run produced less than usual, gives fewer keys rather than an error.
    """
    if not bundle:
        return None
    to_kj = ((bundle.get('units') or {}).get('energy_to_kj_mol')) or 2625.5
    energies = bundle.get('energies') or {}
    primary = bundle.get('primary_level')
    level = (bundle.get('levels') or {}).get(primary) or {}
    segments = (level.get('segments') or {}).get('energy') or {}
    wfn = energies.get('wfn') or []

    summary = {
        'schema_version': bundle.get('schema_version'),
        'analysis_kind': bundle.get('analysis_kind'),
        'level': primary,
        'entity_kind': level.get('entity_kind'),
        'entity_count': level.get('entity_count'),
        'n_atoms': len(bundle.get('atoms') or []) or None,
        'formula': _formula(bundle.get('elements')),
        'n_steps': len(bundle.get('control_coordinates') or []) or None,
        'control_coordinate_label': bundle.get('control_coordinate_label'),
        'control_coordinates': bundle.get('control_coordinates'),
        'dispersion': bool((bundle.get('availability') or {}).get('dispersion')),
        'energy': {
            'wfn_span_kj_mol': _span_kj(wfn, to_kj),
            'iqa_span_kj_mol': _span_kj(energies.get('iqa'), to_kj),
            # The curve itself, relative to its own minimum: eleven numbers per
            # system, which is what a correlation across systems is run over.
            'relative_wfn_kj_mol': ([round((value - min(wfn)) * to_kj, 6) for value in wfn]
                                    if wfn else None),
            'recovery_rmse_kj_mol': energies.get('rmse_kj_mol'),
            'recovery_error_kj_mol': energies.get('recovery_error_kj_mol'),
            'closure_error_kj_mol': energies.get('closure_error_kj_mol'),
        },
        'segments': {
            'count': segments.get('count'),
            'step_ranges': segments.get('step_ranges'),
            'critical_point_indices': segments.get('critical_point_indices'),
        },
        'fragments': [{'name': definition.get('name'),
                       'atom_numbers': definition.get('atom_numbers')}
                      for definition in ((bundle.get('fragments') or {}).get('definitions') or [])],
        'top_reg_terms': _top_reg_terms(
            level, r_threshold=(bundle.get('settings') or {}).get('r_threshold') or 0.0),
        'quality': {
            'missing_files': len((bundle.get('quality_report') or {}).get('missing_files') or []),
            'bad_lagrangians': len((bundle.get('quality_report') or {}).get('bad_lagrangians') or []),
        },
    }
    return summary


def summarise_multipole_bundle(bundle):
    """What a comparison across systems needs from a REG_Multi bundle."""
    if not bundle:
        return None
    settings = bundle.get('settings') or {}
    accounting = bundle.get('accounting') or {}
    pairs = (bundle.get('admission') or {}).get('pairs') or []
    return {
        'schema_version': bundle.get('schema_version'),
        'scope': settings.get('scope'),
        'lmax_used': settings.get('lmax_used'),
        'tolerance': settings.get('tolerance'),
        'n_pairs_tested': len(pairs) or None,
        'n_pairs_admitted': sum(1 for pair in pairs if pair.get('admitted')) or None,
        'span_exact_kj_mol': accounting.get('span_exact_kj_mol'),
        'unresolved_fraction': accounting.get('unresolved_fraction'),
        'residual_fraction': accounting.get('residual_fraction'),
    }


BUNDLE_SUFFIX = '_model_transfer.json'

# The names a bundle carried before the results directories were named for the
# analysis level.  One of these is the older spelling of whatever level-specific
# bundle sits beside it, so it is superseded by a newer one rather than being a
# kind of its own.
_LEVEL_AGNOSTIC_KINDS = ('REG', 'REG_Multi')


def _bundle_kind(path, name=None):
    """'REG_IQA', 'REG_Multi_IQF', ... — what a bundle is, from its file name."""
    base = os.path.basename(path)
    if base.endswith(BUNDLE_SUFFIX):
        base = base[:-len(BUNDLE_SUFFIX)]
    if name and base.startswith(name + '_'):
        base = base[len(name) + 1:]
    return base or 'REG'


def _select_latest_bundles(paths, name=None):
    """The newest bundle of each kind, and what that leaves behind.

    A directory worked in for a while holds several runs of the same analysis —
    REG_IQA_results/ beside REG_IQA_results_n10/, a REG_Multi run remade at the
    IQA and IQF levels after a level-agnostic one.  Collecting all of them makes
    the folder ambiguous about which analysis it describes, so the newest of each
    kind is taken and the rest are recorded as superseded: they stay where they
    are, and the overview says they were passed over and when they were written.

    Returns (kept, superseded), both lists of paths, newest first in *superseded*.
    """
    by_kind = {}
    for path in paths:
        by_kind.setdefault(_bundle_kind(path, name), []).append(path)

    kept, superseded = {}, []
    for kind, candidates in by_kind.items():
        candidates = sorted(candidates, key=os.path.getmtime, reverse=True)
        kept[kind] = candidates[0]
        superseded.extend(candidates[1:])

    # A level-agnostic bundle is the older spelling of its family, so a newer
    # level-specific one in the same family replaces it.
    for legacy in _LEVEL_AGNOSTIC_KINDS:
        if legacy not in kept:
            continue
        newer = [path for kind, path in kept.items()
                 if kind != legacy and kind.startswith(legacy)
                 and os.path.getmtime(path) > os.path.getmtime(kept[legacy])]
        if newer:
            superseded.append(kept.pop(legacy))

    return (sorted(kept.values()),
            sorted(superseded, key=os.path.getmtime, reverse=True))


def _system_files(system_path, name=None, latest_only=True):
    """The transfer bundles and the config a system directory holds.

    Found by pattern rather than by name, so a rename of the results directory
    does not lose them.  *latest_only* keeps the newest bundle of each kind; see
    _select_latest_bundles.
    """
    # '*results*' rather than '*_results': a directory worked in for a while holds
    # REG_IQA_results_n10/ and REG_results_n10/ beside REG_IQA_results/, and a
    # bundle in one of those is still a bundle — it is either the newest of its
    # kind or reported as passed over, rather than going unseen.
    found = sorted(glob.glob(os.path.join(system_path, '*results*',
                                          '*model_transfer.json')))
    bundles, superseded = (_select_latest_bundles(found, name) if latest_only
                           else (found, []))
    config = os.path.join(system_path, reg_setup.CONFIG_NAME)
    return bundles, superseded, (config if os.path.isfile(config) else None)


def collect_sweep(results, root, dir_name=COLLECTION_DIR, level=None, command=None,
                  flat=False, scope='sweep', latest_only=True):
    """Gather every system's bundle and config, and write the overview beside them.

    *results* is what run_batch returns, or the same shape with 'returncode' None
    for a system that was not run in this sweep.  *flat* puts the copies straight
    into the folder instead of under bundles/ and configs/, which is what a single
    run wants: one folder holding a handful of files, to be downloaded whole.
    Returns the path of the overview file, or None when there was nothing to
    collect.
    """
    if not results:
        return None

    root = os.path.abspath(root)
    collection = os.path.join(root, dir_name)
    bundles_dir = collection if flat else os.path.join(collection, 'bundles')
    configs_dir = collection if flat else os.path.join(collection, 'configs')
    for directory in (collection, bundles_dir, configs_dir):
        try:
            os.makedirs(directory, 0o755)
        except OSError:
            pass

    print(SEP)
    print('  COLLECTING {w} INTO {d}'.format(
        w='THE SWEEP' if scope == 'sweep' else 'THIS RUN', d=dir_name))
    print(SEP)

    # A REG_Multi run made on its own does not know which level the IQA analysis
    # was run at, so it keeps what the folder already says rather than guessing
    # and demoting an IQF bundle to a bystander.
    if level is None:
        try:
            with open(os.path.join(collection, OVERVIEW_NAME + '.json'),
                      encoding='utf-8') as handle:
                level = json.load(handle).get('level')
        except Exception:
            level = None

    entries = []
    used_targets = set()
    copied_bytes = 0
    n_bundles = 0
    for result in results:
        system = result['system']
        name = system['name']
        found_bundles, superseded, config = _system_files(
            system['path'], name, latest_only=latest_only)

        entry = {
            'name': name,
            # For a single run the root is the system itself, and '.' says less
            # than the folder's own name does.
            'path': (os.path.basename(system['path']) if scope == 'run'
                     else os.path.relpath(system['path'], root)),
            'steps': system.get('steps'),
            'status': ('ok' if result.get('returncode') == 0
                       else 'not run in this sweep' if result.get('returncode') is None
                       else 'failed'),
            'minutes': round(result['minutes'], 3) if result.get('minutes') is not None else None,
            'log': (os.path.relpath(result['log'], root)
                    if result.get('log') and os.path.exists(result['log']) else None),
            'config': None,
            'bundles': [],
            'bundle_sources': [],
            'superseded_bundles': [
                {'path': os.path.relpath(path, root),
                 'kind': _bundle_kind(path, name),
                 'written': datetime.datetime.fromtimestamp(
                     os.path.getmtime(path)).isoformat(timespec='seconds')}
                for path in superseded],
            'primary_bundle': None,
            'summary': None,
            'multipole': None,
        }

        if config:
            target = os.path.join(configs_dir,
                                  reg_setup.prefixed(reg_setup.CONFIG_NAME, name))
            try:
                shutil.copy2(config, target)
                entry['config'] = os.path.relpath(target, collection)
            except OSError as copy_error:
                print('  WARNING: could not copy {c} — {e}'.format(c=config, e=copy_error))

        for source in found_bundles:
            # A system can hold bundles from more than one run — a level it was
            # analysed at earlier, or a run made before the outputs were named
            # after the system — and two of those can want the same name here.
            # The results directory they came from tells them apart, so nothing
            # is quietly overwritten by something it is not a copy of.
            target_name = reg_setup.prefixed(os.path.basename(source), name)
            if target_name in used_targets:
                source_dir = os.path.basename(os.path.dirname(source))
                target_name = reg_setup.prefixed(
                    source_dir + '__' + os.path.basename(source), name)
            suffix = 2
            while target_name in used_targets:
                stem, extension = os.path.splitext(target_name)
                target_name = '{s}_{i}{e}'.format(s=stem, i=suffix, e=extension)
                suffix += 1
            used_targets.add(target_name)

            target = os.path.join(bundles_dir, target_name)
            try:
                shutil.copy2(source, target)
            except OSError as copy_error:
                print('  WARNING: could not copy {b} — {e}'.format(b=source, e=copy_error))
                used_targets.discard(target_name)
                continue
            copied_bytes += os.path.getsize(target)
            n_bundles += 1
            entry['bundles'].append(os.path.relpath(target, collection))
            entry['bundle_sources'].append(os.path.relpath(source, root))

        # Which of them is this run's own: the one from the results directory the
        # requested level writes.  Falling back to the first non-multipole bundle
        # keeps a collection of an older tree useful.
        multipole_bundles, reg_bundles = [], []
        for relative in entry['bundles']:
            (multipole_bundles if 'REG_Multi' in os.path.basename(relative)
             else reg_bundles).append(relative)
        preferred = [relative for relative in reg_bundles
                     if level and os.path.basename(relative).endswith(
                         level + BUNDLE_SUFFIX)]
        if not preferred and reg_bundles:
            # No level asked for — collecting a directory rather than finishing a
            # run — so the analysis done last speaks for the system.
            preferred = sorted(reg_bundles, reverse=True,
                               key=lambda rel: os.path.getmtime(
                                   os.path.join(collection, rel)))
        primary = (preferred or [None])[0]
        entry['primary_bundle'] = primary

        if primary:
            entry['summary'] = summarise_reg_bundle(
                _load_json(os.path.join(collection, primary)))
        if multipole_bundles:
            entry['multipole'] = summarise_multipole_bundle(
                _load_json(os.path.join(collection, multipole_bundles[0])))
            if entry['multipole'] is not None:
                entry['multipole']['bundle'] = multipole_bundles[0]

        entries.append(entry)
        print('  {name}: {n} bundle(s){s}{c}'.format(
            name=name, n=len(entry['bundles']),
            s=('' if not superseded
               else ', {n} older one(s) passed over'.format(n=len(superseded))),
            c='' if entry['config'] else ', no auto_reg.config'))

    overview = {
        'schema': 'reg_collection_overview',
        'schema_version': 1,
        'scope': scope,
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'root': root,
        'level': level,
        'command': list(command) if command else None,
        'counts': {
            'systems': len(entries),
            'analysed': sum(1 for entry in entries if entry['status'] == 'ok'),
            'failed': sum(1 for entry in entries if entry['status'] == 'failed'),
            'not_run': sum(1 for entry in entries
                           if entry['status'] == 'not run in this sweep'),
            'with_bundle': sum(1 for entry in entries if entry['primary_bundle']),
        },
        'layout': {
            'bundles': ('every transfer bundle, copied verbatim, beside this file'
                        if flat else
                        'bundles/ — each system transfer bundle, copied verbatim'),
            'configs': ('each auto_reg.config, copied verbatim, beside this file'
                        if flat else
                        'configs/ — each system auto_reg.config, copied verbatim'),
            'overview_csv': OVERVIEW_NAME + '.csv',
            'selection': ('the newest bundle of each kind; anything older of the same '
                          'kind is left where it is and listed under '
                          '"superseded_bundles"' if latest_only
                          else 'every bundle found'),
            'note': 'paths in this file are relative to the directory holding it, '
                    'except "path" and "log", which are relative to "root"',
        },
        'systems': entries,
    }

    overview_path = os.path.join(collection, OVERVIEW_NAME + '.json')
    with open(overview_path, 'w', encoding='utf-8') as handle:
        json.dump(overview, handle, indent=1, allow_nan=False)
    csv_path = write_overview_csv(overview, os.path.join(collection, OVERVIEW_NAME + '.csv'))

    print('')
    print('  {n} bundle(s) and {c} config(s) collected ({m:.1f} MB)'.format(
        n=n_bundles, c=sum(1 for entry in entries if entry['config']),
        m=copied_bytes / (1024.0 * 1024.0)))
    print('  Overview: ' + overview_path)
    if csv_path:
        print('            ' + csv_path)
    print('')
    return overview_path


CSV_COLUMNS = [
    ('system', lambda e, s, m: e['name']),
    ('status', lambda e, s, m: e['status']),
    ('path', lambda e, s, m: e['path']),
    ('level', lambda e, s, m: s.get('level')),
    ('n_atoms', lambda e, s, m: s.get('n_atoms')),
    ('formula', lambda e, s, m: s.get('formula')),
    ('n_steps', lambda e, s, m: s.get('n_steps')),
    ('n_fragments', lambda e, s, m: len(s.get('fragments') or []) or None),
    ('n_segments', lambda e, s, m: (s.get('segments') or {}).get('count')),
    ('wfn_span_kj_mol', lambda e, s, m: (s.get('energy') or {}).get('wfn_span_kj_mol')),
    ('iqa_span_kj_mol', lambda e, s, m: (s.get('energy') or {}).get('iqa_span_kj_mol')),
    ('recovery_rmse_kj_mol', lambda e, s, m: (s.get('energy') or {}).get('recovery_rmse_kj_mol')),
    ('top_term', lambda e, s, m: (s.get('top_reg_terms') or [{}])[0].get('term')),
    ('top_term_reg', lambda e, s, m: (s.get('top_reg_terms') or [{}])[0].get('reg')),
    ('top_term_pearson', lambda e, s, m: (s.get('top_reg_terms') or [{}])[0].get('pearson')),
    ('multipole_lmax', lambda e, s, m: m.get('lmax_used')),
    ('multipole_pairs_admitted', lambda e, s, m: m.get('n_pairs_admitted')),
    ('multipole_unresolved_fraction', lambda e, s, m: m.get('unresolved_fraction')),
    ('bundle', lambda e, s, m: e['primary_bundle']),
    ('config', lambda e, s, m: e['config']),
]


def write_overview_csv(overview, path):
    """One row per system of the scalar fields, for reading straight into pandas.

    The overview JSON is the record; this is the same thing with the per-step
    curves and term lists left out, because a first look across forty systems is
    usually a table.
    """
    try:
        import csv
        with open(path, 'w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow([column for column, _ in CSV_COLUMNS])
            for entry in overview['systems']:
                summary = entry.get('summary') or {}
                multipole = entry.get('multipole') or {}
                writer.writerow([getter(entry, summary, multipole)
                                 for _, getter in CSV_COLUMNS])
        return path
    except Exception as csv_error:
        print('  WARNING: could not write the overview CSV — ' + str(csv_error))
        return None


def collect_run(system_path, name=None, level=None, dir_name=None, command=None,
                dir_prefix=''):
    """Gather one system's bundles and config into a folder beside its results.

    The same folder a sweep builds, for a system analysed on its own: everything
    worth taking away from the run in one place, so it can be downloaded whole
    instead of picked out of REG_IQA_results/, REG_IQF_results/ and
    REG_Multi_IQA_results/ one file at a time.

    It is rebuilt from what the directory holds each time, not added to, so a
    REG_Multi run made after the IQA run leaves a folder with both bundles in it
    rather than a stale one.
    """
    system_path = os.path.abspath(system_path)
    # The files are named after the system whether or not the run itself was, so a
    # folder downloaded from anywhere says which system it came from.  The folder
    # takes the run's own prefix, so an unprefixed run keeps unprefixed names in
    # its own directory, as the rest of its output does.
    name = reg_setup.sanitise_prefix(name or os.path.basename(system_path))
    system = {
        'path': system_path,
        'name': name,
        'steps': None,
        'config': os.path.isfile(os.path.join(system_path, reg_setup.CONFIG_NAME)),
    }
    result = {'system': system, 'returncode': 0, 'minutes': None, 'log': None}
    return collect_sweep(
        [result], system_path,
        dir_name=dir_name or reg_setup.prefixed(RUN_COLLECTION_DIR, dir_prefix),
        level=level, command=command, flat=True, scope='run')
