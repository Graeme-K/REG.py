"""
reg_multipole.py  (REG_Multi)

A REG analysis of the rank-resolved multipolar electrostatics between fragments.

What it does.  The IQA classical interaction V_cl(A,B) between two atoms is one
number.  Expanding it over the atoms' nucleus-centred spherical multipole moments
splits it into a series — charge-charge, charge-dipole, dipole-dipole and so on —
and summing that series over the atom pairs of two fragments gives the same split
for a fragment-fragment interaction.  Running REG over those rank terms says which
*kind* of electrostatics drives a process, not merely that electrostatics does.

Scope.  With fragments defined in auto_reg.config the default is every
interfragment atom pair, grouped by fragment pair.  Asked for an analysis within
a fragment (--scope intra, or --within NAME), or run with no fragments defined at
all, it instead takes every atom pair inside that fragment, or in the whole
system.  The fragment definitions are the same ones REG-IQF uses, read from the
same file, so the two analyses describe the same partition.

Why the gating matters.  The multipole series is asymptotic.  It is exact only
while the two basins' convergence spheres stay apart, and for close pairs it can
look settled at low rank and then turn around.  "Intermolecular means far enough"
does not hold: a hydrogen bond puts the donor H and acceptor O 1.7-2.0 A apart,
shorter than many intramolecular 1,3 distances — and those are exactly the pairs
a fragment analysis is most interested in.  So every candidate pair, interfragment
or not, faces the same five gates:

  1. Topology (cheap, not authoritative).  1,2 and 1,3 pairs are rejected
     outright; 1,4 and beyond are candidates.
  2. Geometry (cheap, not authoritative).  R_AB >= R_A + R_B.  With beta-sphere
     radii, R_A is a strict lower bound, so passing this proves nothing —
     necessary, not sufficient.
  3. Numerical convergence (authoritative).  Partial sums against the exact IQA
     V_cl, requiring both a small truncation residual *and* rank increments that
     are still decreasing at the top of the series.  The second condition is what
     catches a series that diverges after looking converged.
  4. Path-wide admission.  A pair is admitted only if it passes gate 3 at every
     geometry, at one fixed L_max.  A pair that converges at the reactant and
     fails near the transition state puts a kink in its term that would dominate
     the gradient and read as chemistry.
  5. Accounting.  Every rejected pair's exact IQA V_cl is reported as a single
     unresolved channel, with the fraction of the total change in V_cl along the
     coordinate that lives in it.

Tolerance.  The residual test is relative to variation, not to absolute energy:
REG fits a gradient, so what matters is that the truncation error does not vary
across the segment.  The default asks for the residual's peak-to-peak variation
to be under 5% of the pair's own peak-to-peak V_cl.  An absolute kJ/mol threshold
would wrongly reject large, flat, well-converged interfragment terms and wrongly
accept small ones whose error wanders.

On L_max.  Solano et al.'s errors at l_tot = l_A + l_B = 10 are 0.03-0.3 kJ/mol
for 1,3 and 1,4 pairs but 4-18 kJ/mol for 1,2 — the size of the signal itself.
Note that l_tot is the *summed* rank: L_max here is the rank per atom, so L_max=5
gives l_tot up to 10.  AIMAll must actually have been asked for moments that high;
the run reports what it found and caps L_max at it, because a low ceiling leaves
gate 3's increment condition untestable.

Not done here: the MMS shift that would bring 1,2 and 1,3 pairs back.  It moves
the expansion centres, so a REG built on shifted moments is not comparable
term-by-term with this one.  That belongs in a separate analysis, not this file.

Usage:  python3 reg_multipole.py -d /path/to/REG.py/src [options]

coded for the REG.py package
"""

import json
import os
import sys
import time
from optparse import OptionParser

import numpy as np

HA_TO_KJ = 2625.5
_BUNDLE_DECIMALS = 10


def _json_ready(obj, decimals=_BUNDLE_DECIMALS):
    """Convert numpy/python data into something json.dump can write literally.

    NaN and +/-Inf become null: Python writes them as bare NaN/Infinity tokens,
    which are not valid JSON and make JSON.parse throw in a browser.
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
        if obj != obj or obj in (float('inf'), float('-inf')):
            return None
        return round(obj, decimals)
    return obj


def _span(values):
    """Peak-to-peak variation of *values*.

    Returns NaN when nothing in the array is readable, rather than 0.0: a term
    that could not be read has an unknown span, and reporting it as zero would
    read as "this channel does not move" — the opposite of what is known.
    """
    values = np.asarray(values, dtype=float)
    if not np.any(np.isfinite(values)):
        return float('nan')
    return float(np.nanmax(values) - np.nanmin(values))


def _fmt(value, spec='{:.3f}', width=12, missing='n/a'):
    """Format a number that may be NaN, right-aligned in *width*."""
    if value is None or value != value:
        return missing.rjust(width)
    return spec.format(value).rjust(width)


def _table(rows, n_steps):
    """Stack per-step rows into a (n_rows, n_steps) array, empty included.

    numpy gives an empty list shape (0,), which pandas then reads as one column
    rather than none, so an analysis that admitted nothing would fail on the
    shape rather than report that it admitted nothing.
    """
    if len(rows) == 0:
        return np.zeros((0, n_steps))
    return np.asarray(rows, dtype=float)


def _segment_bounds(n_points, critical_points):
    """Inclusive [start, stop] index ranges of the REG segments.

    reg.split_segm makes consecutive segments share their critical point, so
    these ranges overlap by one step by design.
    """
    crit = sorted(set(int(i) for i in critical_points))
    bounds = [0] + crit + [n_points - 1]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


# Excel's hard limits.  A frame past either of these cannot be written at all,
# and xlsxwriter raises rather than truncating — which, left unguarded, aborts the
# whole workbook and leaves a 0-byte file behind.
_EXCEL_MAX_ROWS = 1048576
_EXCEL_MAX_COLS = 16384
# Well under the hard row limit: a sheet of a few hundred thousand rows is slow to
# write, enormous, and not something anyone reads in Excel.  The CSV has it all.
_EXCEL_ROW_BUDGET = 100000


def _to_excel_safe(dataframe, writer, name, used_names, index=True, skipped=None):
    """Write *dataframe* to a sheet, or record why it could not be.

    Nothing that goes in the workbook is unavailable elsewhere — every sheet has
    a CSV beside it — so a frame too big for Excel is skipped with a note rather
    than being allowed to take the workbook down with it.
    """
    # A table this run did not produce is simply absent, not an error: guarding
    # here as well as at each call site means one missed check cannot cost the
    # whole workbook, which is exactly how a 0-byte file happened before.
    if dataframe is None:
        return False
    rows, cols = dataframe.shape
    cols += 1 if index else 0
    if rows > _EXCEL_MAX_ROWS or cols > _EXCEL_MAX_COLS or rows > _EXCEL_ROW_BUDGET:
        if skipped is not None:
            skipped.append((name, rows, cols))
        return False
    dataframe.to_excel(writer, sheet_name=_safe_sheet_name(name, used_names), index=index)
    return True


def _safe_sheet_name(name, used_names, max_len=31):
    """Excel-safe sheet name (<= 31 chars, unique, no reserved characters)."""
    for bad in '[]:*?/\\':
        name = name.replace(bad, '-')
    if len(name) <= max_len and name not in used_names:
        used_names.add(name)
        return name
    base = name[:max_len - 3]
    for idx in range(1, 10000):
        candidate = base + '~' + str(idx)
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
    raise RuntimeError('Could not generate a unique sheet name for: ' + name)


# ---------------------------------------------------------------------------
# Pair grouping
# ---------------------------------------------------------------------------

def build_groups(atoms, frag_names, frag_atom_lists, scope, within):
    """Decide which atom pairs are analysed, and how they are grouped.

    Returns (groups, pair_index) where *groups* is a list of dicts with 'label',
    'kind' and 'pairs' (indices into *pair_index*), and *pair_index* is the list
    of (i, j) atom index pairs actually needed, i < j.

    scope
        'inter'  every interfragment atom pair, grouped by fragment pair.
        'intra'  every atom pair inside a fragment, one group per fragment,
                 restricted to *within* when that names fragments.
        'all'    both of the above; with no fragments defined, one group holding
                 every atom pair in the system.
        'pairs'  every atom pair in the system, each its own group — the
                 atom-wise analysis, matching what REG-IQA reports per pair.
                 Fragment definitions are ignored, as they are by an IQA run.
    """
    n_atoms = len(atoms)
    has_fragments = bool(frag_atom_lists)

    if scope == 'pairs':
        pairs = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)]
        return ([{'label': str(atoms[i]) + '-' + str(atoms[j]), 'kind': 'pair',
                  'members': None, 'pairs': [k]}
                 for k, (i, j) in enumerate(pairs)], pairs)

    if not has_fragments:
        pairs = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)]
        return ([{'label': 'System', 'kind': 'system', 'members': None,
                  'pairs': list(range(len(pairs)))}], pairs)

    members = [[a - 1 for a in atom_list] for atom_list in frag_atom_lists]
    wanted = None
    if within:
        lowered = [w.lower() for w in within]
        wanted = [f_i for f_i, name in enumerate(frag_names) if str(name).lower() in lowered]
        missing = [w for w in within
                   if w.lower() not in [str(n).lower() for n in frag_names]]
        if missing:
            raise ValueError('--within names fragments that are not in the config: '
                             + ', '.join(missing))

    pair_index = []
    pair_lookup = {}

    def _pair_id(i, j):
        key = (i, j) if i < j else (j, i)
        if key not in pair_lookup:
            pair_lookup[key] = len(pair_index)
            pair_index.append(key)
        return pair_lookup[key]

    groups = []
    if scope in ('intra', 'all'):
        for f_i, frag in enumerate(members):
            if wanted is not None and f_i not in wanted:
                continue
            ids = [_pair_id(frag[a], frag[b])
                   for a in range(len(frag)) for b in range(a + 1, len(frag))]
            if ids:
                groups.append({'label': str(frag_names[f_i]) + '(intra)', 'kind': 'intra',
                               'members': [f_i], 'pairs': ids})
    if scope in ('inter', 'all'):
        for f_i in range(len(members)):
            for f_j in range(f_i + 1, len(members)):
                if wanted is not None and not (f_i in wanted or f_j in wanted):
                    continue
                ids = [_pair_id(a, b) for a in members[f_i] for b in members[f_j]]
                if ids:
                    groups.append({'label': str(frag_names[f_i]) + '|' + str(frag_names[f_j]),
                                   'kind': 'inter', 'members': [f_i, f_j], 'pairs': ids})

    if not groups:
        raise ValueError('No atom pairs selected — check --scope and --within against '
                         'the fragment definitions in the config file')
    return groups, pair_index


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------

def admit_pairs(mp, atoms, coords, pair_index, exact_vcl, rank_energies,
                beta_radii, segments, options, separation=None, bonds=None):
    """Run the five gates over every candidate pair.

    *rank_energies* is (n_pairs, n_points, L+1, L+1) or None for pairs that the
    cheap gates already rejected — those are never computed, which is the only
    reason gates 1 and 2 exist.

    Returns (verdicts, diagnostics): one verdict dict per pair, and a dict of
    the path-level information the report and the bundle need.
    """
    n_pairs = len(pair_index)
    n_points = coords.shape[0]

    # ---- gate 1: topology ------------------------------------------------
    # The caller has usually built these already, to decide which pairs were
    # worth computing a tensor for at all; rebuilding the bond graph here would
    # repeat an O(points x atoms^2) pass for nothing.
    if bonds is None:
        bonds = mp.bond_graph(atoms, coords, tolerance=options['bond_tolerance'])
    if separation is None:
        separation = mp.topology_separation(len(atoms), bonds, max_depth=3)

    # ---- gate 2: geometry -------------------------------------------------
    radii, radii_are_lower_bound = mp.basin_radii(atoms, beta_radii, source=options['radii'])

    distances = np.zeros((n_pairs, n_points))
    for p_i, (i, j) in enumerate(pair_index):
        distances[p_i] = np.linalg.norm(coords[:, i, :] - coords[:, j, :], axis=1)

    verdicts = []
    for p_i, (i, j) in enumerate(pair_index):
        verdict = {
            'pair': (i, j),
            'label': str(atoms[i]) + '-' + str(atoms[j]),
            'min_distance_ang': float(distances[p_i].min()),
            'max_distance_ang': float(distances[p_i].max()),
            'separation': separation.get((i, j)),
            'admitted': False,
            'rejected_by': None,
            'note': '',
        }

        # gate 1
        if options['topology'] and verdict['separation'] in (2, 3):
            verdict['rejected_by'] = 'topology'
            verdict['note'] = ('1,{n} pair — nucleus-centred moments of bonded or '
                               'geminal atoms share a convergence sphere'
                               .format(n=verdict['separation']))
            verdicts.append(verdict)
            continue

        # gate 2
        radius_sum = radii[:, i] + radii[:, j]
        if np.all(np.isfinite(radius_sum)):
            margin = distances[p_i] - radius_sum
            verdict['geometry_margin_ang'] = float(margin.min())
            if margin.min() < 0:
                verdict['rejected_by'] = 'geometry'
                verdict['note'] = ('R_AB < R_A + R_B at {n} geometr{y} ({src} radii)'
                                   .format(n=int(np.sum(margin < 0)),
                                           y='y' if int(np.sum(margin < 0)) == 1 else 'ies',
                                           src=options['radii']))
                verdicts.append(verdict)
                continue
        else:
            verdict['geometry_margin_ang'] = None
            verdict['note'] = 'basin radii unavailable — geometric gate not applied'

        # gates 3 and 4
        if rank_energies is None or options['skip_convergence']:
            verdict['admitted'] = True
            verdict['note'] = (verdict['note'] + '; ' if verdict['note'] else '') + \
                'convergence gate skipped'
            verdicts.append(verdict)
            continue

        increments, partial = mp.partial_sums_by_total_rank(rank_energies[p_i])
        report = mp.convergence_verdict(
            partial, increments, exact_vcl[p_i], segments=segments,
            residual_tolerance=options['tolerance'],
            increment_ranks=options['increment_ranks'],
            absolute_floor_kj=options['floor'])
        verdict.update({
            'admitted': report['admitted'],
            'convergence': report,
            'max_abs_residual_kj_mol': report['max_abs_residual_kj_mol'],
            'mean_abs_residual_kj_mol': report['mean_abs_residual_kj_mol'],
        })
        if not report['admitted']:
            if not report['values_readable']:
                verdict['rejected_by'] = 'no_reference'
                verdict['note'] = 'exact IQA V_cl unavailable for this pair'
            elif not report['residual_pass'] and report['increment_failures']:
                verdict['rejected_by'] = 'convergence'
                verdict['note'] = 'residual varies too much and the series is not decreasing'
            elif not report['residual_pass']:
                worst = max(report['segments'], key=lambda s: s['ratio'])
                verdict['rejected_by'] = 'residual'
                verdict['note'] = ('residual varies by {r:.1%} of V_cl on segment {s}'
                                   .format(r=worst['ratio'], s=worst['segment'] + 1))
            else:
                verdict['rejected_by'] = 'increments'
                verdict['note'] = ('series not decreasing at {n} of {t} geometries'
                                   .format(n=len(report['increment_failures']), t=n_points))
        verdicts.append(verdict)

    return verdicts, {
        'bonds': sorted(tuple(sorted(b)) for b in bonds),
        'radii_are_lower_bound': radii_are_lower_bound,
        'radii_source': options['radii'],
        'distances': distances,
    }


def fragment_moment_analysis(mp, atoms, coords, moments, frag_names, frag_atom_lists,
                             groups, pair_index, exact_vcl, basin_radii_ang, segments,
                             l_max, options, with_energies=True):
    """Fragment-centred ("grouped-atom") multipole analysis.

    Every atom's moments are translated onto one centre per fragment and summed,
    so each fragment carries a single set of moments and a fragment pair gets one
    expansion rather than a sum over atom pairs.  This is a different
    decomposition of the same energy, not a regrouping of the atom-centred one:
    for two neutral fragments the monopole and charge-dipole terms vanish
    identically here, while the atom-centred series puts large charge-charge
    terms against each other that mostly cancel.  Having both is what lets a
    reader say whether a net dipole-dipole interaction drives a step, or whether
    the atomic contributions simply cancel and the net is weak.

    The gate changes with the expansion.  A fragment-centred series needs its
    convergence sphere to enclose the whole fragment, so the geometric condition
    is R_FG >= extent(F) + extent(G) with the extent measured from the centre —
    a demand that grows with fragment size however innocuous the individual atom
    pairs look.  The convergence test is the same one the atom pairs face, run
    against the exact IQA V_cl summed over the fragment pair.

    Returns a dict, or None when there are no fragment pairs to describe.
    """
    # The moments themselves are worth having whenever fragments are defined —
    # they are the coefficients a fragment-level discussion quotes — so they are
    # built either way.  *with_energies* controls only the expansion of fragment
    # pairs against each other, which is the part that needs a gate and a flag.
    inter_groups = [g for g in groups if g['kind'] == 'inter'] if with_energies else []
    if not frag_atom_lists:
        return None

    n_steps = coords.shape[0]
    n_frags = len(frag_atom_lists)
    members = [[a - 1 for a in atom_list] for atom_list in frag_atom_lists]
    weights = [mp.centre_weights([atoms[a] for a in group], options['fragment_centre'])
               for group in members]

    centres = np.zeros((n_steps, n_frags, 3))
    frag_moments = np.zeros((n_steps, n_frags, mp.n_components(l_max)))
    extents = np.zeros((n_steps, n_frags))
    # Each atom's contribution to the fragment moments, kept separately so that
    # cancellation between them can be measured below.
    contributions = [np.zeros((n_steps, len(group), mp.n_components(l_max)))
                     for group in members]
    for f_i, group in enumerate(members):
        for step in range(n_steps):
            centre = mp.fragment_centre(coords[step][group], weights[f_i])
            centres[step, f_i] = centre
            displacement = (coords[step][group] - centre) * mp.BOHR_PER_ANGSTROM
            translated = mp.translate_moments(moments[step][group], displacement, l_max)
            contributions[f_i][step] = translated
            frag_moments[step, f_i] = translated.sum(axis=0)
            extents[step, f_i] = mp.fragment_extents(
                coords[step][group], centre, basin_radii_ang[step][group])

    # How far the atoms' contributions to a fragment rank cancel each other.
    #
    # Both sides of the ratio are measured *after* translation onto the fragment
    # centre, which is the only way to compare like with like: an atom sitting a
    # distance d from the centre contributes q*d^l to rank l, so comparing the
    # fragment's moments against the atoms' own nucleus-centred ones would report
    # that re-expansion as though it were structure, and the ratio would not even
    # be bounded.  Measured this way it is bounded by 1 (triangle inequality), and
    # a value well below 1 is the case worth naming: the atomic contributions
    # oppose one another, so the net fragment term of that rank is weak however
    # large the individual contributions are.
    cancellation = {}
    for f_i, group in enumerate(members):
        rows = []
        for l in range(l_max + 1):
            fragment_size = mp.rank_magnitude(frag_moments[:, f_i, :], l)
            atomic_sum = mp.rank_magnitude(contributions[f_i], l).sum(axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(atomic_sum > 0, fragment_size / atomic_sum, np.nan)
            rows.append({'l': l, 'fragment': fragment_size, 'atomic_sum': atomic_sum,
                         'ratio': ratio})
        cancellation[str(frag_names[f_i])] = rows

    pair_lookup = {pair: k for k, pair in enumerate(pair_index)}
    results = []
    for group in inter_groups:
        f_i, f_j = group['members']
        ranks = mp.pair_rank_energies(
            frag_moments[:, f_i, :], frag_moments[:, f_j, :],
            centres[:, f_i, :], centres[:, f_j, :], l_max)
        separation = np.linalg.norm(centres[:, f_i, :] - centres[:, f_j, :], axis=1)
        margin = separation - (extents[:, f_i] + extents[:, f_j])

        # The whole fragment pair is described at once, so its reference is every
        # atom pair between the two fragments, admitted or not.
        exact = np.sum([exact_vcl[pair_lookup[p]] for p in
                        [(min(a, b), max(a, b)) for a in members[f_i] for b in members[f_j]]],
                       axis=0)

        increments, partial = mp.partial_sums_by_total_rank(ranks)
        report = mp.convergence_verdict(
            partial, increments, exact, segments=segments,
            residual_tolerance=options['tolerance'],
            increment_ranks=options['increment_ranks'],
            absolute_floor_kj=options['floor'])

        admitted = bool(report['admitted'] and margin.min() >= 0)
        rejected_by = None
        note = ''
        if margin.min() < 0:
            rejected_by = 'fragment_extent'
            note = ('the two expansion spheres overlap by {o:.2f} A at closest approach — a '
                    'fragment-centred expansion cannot converge here even where the individual '
                    'atom pairs do'.format(o=-margin.min()))
        elif not report['admitted']:
            rejected_by = 'convergence'
            note = 'residual or increments fail against the exact V_cl for this fragment pair'

        results.append({
            'label': group['label'], 'members': [f_i, f_j],
            'ranks': ranks, 'exact': exact,
            'total': ranks.sum(axis=(1, 2)),
            'residual': exact - ranks.sum(axis=(1, 2)),
            'separation_ang': separation, 'extent_margin_ang': margin,
            'admitted': admitted, 'rejected_by': rejected_by, 'note': note,
            'convergence': report,
        })

    return {'centres': centres, 'moments': frag_moments, 'extents': extents,
            'cancellation': cancellation, 'pairs': results,
            'centre_scheme': options['fragment_centre'],
            'fragment_names': [str(n) for n in frag_names]}


def main(argv=None):
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage)
    parser.add_option('-d', '--directory', action='store', type='string', dest='reg_dir',
                      help='PLEASE INSERT THE PATH OF REG.py folder installation')
    parser.add_option('-c', '--config', action='store', type='string', dest='config',
                      default='auto_reg.config',
                      help='fragment definition file, shared with auto_reg (default: auto_reg.config)')
    parser.add_option('-l', '--lmax', action='store', type='int', dest='lmax', default=None,
                      help='highest multipole rank per atom; l_tot reaches 2*LMAX. Capped at the '
                           'rank AIMAll actually wrote (default: 5)')
    parser.add_option('-s', '--scope', action='store', type='string', dest='scope', default=None,
                      help="'inter' (every interfragment atom pair, the default when fragments are "
                           "defined), 'intra' (every atom pair inside a fragment), 'all' (both), or "
                           "'pairs' (every atom pair in the system, each reported on its own — the "
                           "atom-wise analysis, which ignores the fragment definitions). With no "
                           "fragments defined, every atom pair in the system is used")
    parser.add_option('-w', '--within', action='store', type='string', dest='within', default=None,
                      help='comma-separated fragment names to analyse within; implies --scope intra '
                           'unless --scope says otherwise')
    parser.add_option('-t', '--tolerance', action='store', type='float', dest='tolerance', default=None,
                      help='allowed peak-to-peak variation of the truncation residual, as a fraction '
                           'of the pair own peak-to-peak V_cl over a segment (default: 0.05)')
    parser.add_option('--floor', action='store', type='float', dest='floor', default=None,
                      help='kJ/mol below which a term has no gradient worth resolving, so the '
                           'residual is judged in absolute terms instead (default: 0.05)')
    parser.add_option('--increment-ranks', action='store', type='int', dest='increment_ranks',
                      default=None,
                      help='how many of the top complete l_tot shells are examined for growth. '
                           'Only shells with l_tot <= L_max are testable — above that a shell is '
                           'missing most of the blocks its rank needs. Default: every complete '
                           'shell above l_tot = 1')
    parser.add_option('--radii', action='store', type='string', dest='radii', default=None,
                      help="basin radius source for the geometric gate: 'beta' (AIMAll beta spheres, "
                           "a strict lower bound — necessary but not sufficient) or 'vdw' (tabulated "
                           "van der Waals radii, an estimate of the 0.001 au envelope) (default: beta)")
    parser.add_option('--no-topology-filter', action='store_true', dest='no_topology', default=False,
                      help='do not reject 1,2 and 1,3 pairs outright; let the convergence gate decide. '
                           'Slower, and those pairs are expected to fail it')
    parser.add_option('--skip-convergence-gate', action='store_true', dest='skip_convergence',
                      default=False,
                      help='admit every pair that passes the cheap gates, WITHOUT testing convergence '
                           'against the exact IQA V_cl. Exploratory use only — the rank decomposition '
                           'it produces is not known to be meaningful')
    parser.add_option('--test-all-pairs', action='store_true', dest='test_all', default=False,
                      help='run the convergence test on pairs the cheap gates rejected too, so the '
                           'report can show how far off they are. Diagnostic; slower')
    parser.add_option('-i', '--ignore-fragments', action='store_true', dest='ignore_fragments',
                      default=False,
                      help='run as if auto_reg.config defined no fragments: which pairs are '
                           'described is then decided by the admission gates alone, not by any '
                           'partition. Defaults to one system-wide group; add --scope pairs to '
                           'report each atom pair separately')
    parser.add_option('--fragment-moments', action='store_true', dest='fragment_moments',
                      default=False,
                      help='also run the fragment-centred analysis: translate every atom multipole '
                           'moments onto one centre per fragment and expand each fragment pair '
                           'about those, giving "fragment dipole against fragment dipole". A '
                           'different decomposition of the same energy, written to its own files '
                           'and never mixed with the atom-centred terms')
    parser.add_option('--fragment-centre', action='store', type='string', dest='fragment_centre',
                      default=None,
                      help="where to put a fragment expansion centre: 'centroid' (default, close "
                           "to the centre that minimises the fragment radial extent, which is what "
                           "limits convergence), 'mass' or 'nuclear-charge'")
    parser.add_option('--pair-terms', action='store_true', dest='pair_terms', default=False,
                      help='also run REG over individual atom-pair totals, not only fragment-pair '
                           'rank terms')
    parser.add_option('-o', '--results-dir', action='store', type='string', dest='results_dir',
                      default=None,
                      help='name of the results directory (default: REG_Multi_results). '
                           'auto_reg passes a level-specific name so an IQF run does not '
                           'overwrite what an IQA run produced')
    parser.add_option('--bundle-indent', action='store', type='int', dest='bundle_indent', default=0,
                      help='indentation for the model-transfer JSON bundle; 0 writes it compact')

    (option, _args) = parser.parse_args(args=argv)

    reg_dir = option.reg_dir or os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(1, reg_dir)

    import pandas as pd  # type: ignore

    import aimall_utils as aim_u  # type: ignore
    import default_settings  # type: ignore
    import gaussian_utils as gauss_u  # type: ignore
    import multipole_utils as mp  # type: ignore
    import reg  # type: ignore
    import reg_setup  # type: ignore
    import reg_vis as rv  # type: ignore

    start_time = time.time()
    SYS = 'REG_Multi'
    cwd = str(os.getcwd())

    # ---- settings: config file first, command line wins --------------------
    config_options = reg_setup.read_multipole_config(option.config)
    frag_names, frag_atom_lists = reg_setup.read_fragment_config(option.config)

    # Fragments are a claim about which pairs belong together, and it is a
    # separate claim from the one the gates make about which pairs the multipole
    # series can describe.  --ignore-fragments drops the first so the second
    # decides alone: every atom pair in the system is a candidate, and what
    # survives is whatever the admission criteria admit.
    ignore_fragments = bool(option.ignore_fragments
                            or config_options.get('ignore_fragments', False))
    if ignore_fragments and frag_atom_lists:
        print('Ignoring the {n} fragment definition(s) in {c} — pair selection is left to the '
              'admission gates.'.format(n=len(frag_atom_lists), c=option.config))
    if ignore_fragments:
        frag_names, frag_atom_lists = None, None

    within = option.within.split(',') if option.within else config_options.get('within')
    within = [w.strip().strip('<>') for w in within] if within else None
    scope = (option.scope or config_options.get('scope')
             or ('intra' if within else ('inter' if frag_atom_lists else 'all')))
    scope = scope.lower()
    if scope not in ('inter', 'intra', 'all', 'pairs'):
        raise ValueError("--scope must be 'inter', 'intra', 'all' or 'pairs', not " + repr(scope))
    if not frag_atom_lists and scope != 'pairs':
        if scope != 'all':
            print('{why} — analysing every atom pair in the system.'.format(
                why=('Fragments ignored' if ignore_fragments
                     else 'No fragment definitions in ' + str(option.config))))
        scope = 'all'

    settings = {
        'lmax': option.lmax or config_options.get('lmax') or 5,
        'tolerance': option.tolerance if option.tolerance is not None
                     else config_options.get('tolerance', 0.05),
        'floor': option.floor if option.floor is not None else config_options.get('floor', 0.05),
        # None means "every complete shell above l_tot = 1", which is the widest
        # testable window and the default.
        'increment_ranks': (option.increment_ranks if option.increment_ranks is not None
                            else config_options.get('increment_ranks')),
        'radii': (option.radii or config_options.get('radii') or 'beta').lower(),
        'topology': (not option.no_topology) and config_options.get('topology', True),
        'skip_convergence': bool(option.skip_convergence),
        'bond_tolerance': 1.2,
        'fragment_moments': bool(option.fragment_moments
                                 or config_options.get('fragment_moments', False)),
        'fragment_centre': (option.fragment_centre or config_options.get('fragment_centre')
                            or 'centroid'),
    }
    if settings['fragment_moments'] and not frag_atom_lists:
        raise ValueError('--fragment-moments builds one expansion per fragment, so it cannot run '
                         + ('with --ignore-fragments' if ignore_fragments
                            else 'without fragment definitions; none were found in '
                                 + str(option.config)))
    if within and not frag_atom_lists:
        raise ValueError('--within names fragments, so it cannot run '
                         + ('with --ignore-fragments' if ignore_fragments
                            else 'without fragment definitions'))

    POINTS = default_settings.POINTS
    AUTO = default_settings.AUTO
    turning_points = default_settings.turning_points
    INFLEX = default_settings.INFLEX
    REVERSE = default_settings.REVERSE
    CONTROL_COORDINATE_TYPE = default_settings.CONTROL_COORDINATE_TYPE
    Scan_Atoms = default_settings.Scan_Atoms
    IRC_output = default_settings.IRC_output
    SAVE_FIG = default_settings.SAVE_FIG
    WRITE = default_settings.WRITE
    MIN_TABLE_ROWS = default_settings.MIN_TABLE_ROWS
    MAX_TABLE_ROWS = default_settings.MAX_TABLE_ROWS
    R_THRESHOLD = default_settings.R_THRESHOLD

    ###########################################################################
    #                        FILES, GEOMETRY, ENERGIES                        #
    ###########################################################################

    points = reg_setup.discover_reg_points('.')
    reg_folders = points['reg_folders']
    reg_root_list = points['reg_roots']
    wf_files = points['wf_files']
    g16_files = points['g16_files']
    if not reg_folders:
        raise FileNotFoundError('No geometry points found — each REG step should be its own '
                                'numbered folder holding a .wfn/.wfx, a Gaussian output and an '
                                'AIMAll _atomicfiles folder')

    # Created only when there is something to put in it, so a run that stops at a
    # missing prerequisite does not leave an empty REG_Multi_results/ behind
    # looking like output.
    results_dir = os.path.join(cwd, option.results_dir or (SYS + '_results'))
    results_dir_ready = []

    def ensure_results_dir():
        if results_dir_ready:
            return results_dir
        try:
            os.mkdir(results_dir, 0o755)
        except OSError:
            print('Directory {d} already exists — results will be overwritten'.format(d=results_dir))
        else:
            print('Successfully created the directory {d}'.format(d=results_dir))
        results_dir_ready.append(True)
        return results_dir

    atoms = (aim_u.get_atom_list_wfx(wf_files[0]) if points['wfx']
             else aim_u.get_atom_list(wf_files[0]))
    n_atoms = len(atoms)
    atomic_files = [wf[:-4] + '_atomicfiles' for wf in wf_files]
    xyz_files = [gauss_u.get_xyz_file(f) for f in g16_files]

    structures = [reg_setup.load_xyz_structure(path) for path in xyz_files]
    if any(s is None for s in structures):
        raise ValueError('Could not read geometry for every point — the multipole expansion '
                         'needs the nuclear positions at every step')
    coords = np.array([[[a['x'], a['y'], a['z']] for a in s['atoms']] for s in structures])
    if coords.shape[1] != n_atoms:
        raise ValueError('The geometries hold {g} atoms but the wavefunction holds {w}'
                         .format(g=coords.shape[1], w=n_atoms))

    if CONTROL_COORDINATE_TYPE == 'Scan':
        cc = gauss_u.get_control_coordinates_PES_Scan(g16_files, Scan_Atoms)
        X_LABEL = r'Control Coordinate [$\AA$]'
    elif CONTROL_COORDINATE_TYPE == 'IRC':
        cc = gauss_u.get_control_coordinates_IRC_g16(IRC_output)
        X_LABEL = r'Control Coordinate r[$\AA$]'
    else:
        cc = [reg_setup.folder_value(folder) for folder in reg_folders]
        X_LABEL = 'Control Coordinate [REG step]'
    cc = np.array(cc, dtype=float)
    if REVERSE:
        cc = -cc

    total_energy_wfn = np.array(aim_u.get_aimall_wfx_energies(wf_files) if points['wfx']
                                else aim_u.get_aimall_wfn_energies(wf_files))

    critical_points = (reg.find_critical(total_energy_wfn, cc, min_points=POINTS, use_inflex=INFLEX)
                       if AUTO else turning_points)
    segments = _segment_bounds(len(cc), critical_points)

    ###########################################################################
    #                      MOMENTS AND THE EXACT REFERENCE                    #
    ###########################################################################

    print('Reading atomic multipole moments from {n} .int files ...'
          .format(n=len(atomic_files) * n_atoms))
    moments, beta_radii, moment_info = mp.get_atomic_multipoles(
        atomic_files, atoms, settings['lmax'])
    if moment_info['missing']:
        raise ValueError('Could not read multipole moments from {n} .int file(s), the first being '
                         '{f}. REG_Multi needs every atom at every geometry.'
                         .format(n=len(moment_info['missing']), f=moment_info['missing'][0]))

    l_available = moment_info['l_available']
    l_max = settings['lmax']
    if l_available < l_max:
        print('WARNING: AIMAll wrote moments only up to l = {a}, below the requested L_max = {r}. '
              'Capping L_max at {a} (l_tot <= {t}). The increment condition of the convergence '
              'gate needs several ranks above the leading term to mean anything, so consider '
              'rerunning AIMAll with higher moments.'
              .format(a=l_available, r=l_max, t=2 * l_available))
        l_max = l_available
        moments = moments[:, :, :mp.n_components(l_max)]
    settings['lmax_used'] = l_max

    # V_cl is the reference the gates test against.  V_xc and the total E_inter
    # come along for context: the multipole series describes the classical part
    # only, and how much of a fragment interaction that part accounts for is a
    # different question from how well the series reproduces it.  All three come
    # out of the same cached .sum parse, so the extra two are free.
    print('Reading exact IQA V_cl(A,B), V_xc(A,B) and E_inter(A,B) from the AIMAll output ...')
    _reference_props = ['VC_IQA(A,B)', 'VX_IQA(A,B)', 'E_IQA_Inter(A,B)']
    _, _, exact_inter, _exact_headers, _missing_inter = aim_u.get_iqa_properties(
        atomic_files, [], _reference_props, atoms)
    all_pairs = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)]
    _reference_all = np.array([[np.nan if v is None else v for v in row] for row in exact_inter],
                              dtype=float)
    _n_all = len(all_pairs)
    exact_all = _reference_all[0:_n_all]
    vxc_all = _reference_all[_n_all:2 * _n_all]
    einter_all = _reference_all[2 * _n_all:3 * _n_all]
    exact_lookup = {pair: exact_all[k] for k, pair in enumerate(all_pairs)}
    vxc_lookup = {pair: vxc_all[k] for k, pair in enumerate(all_pairs)}
    einter_lookup = {pair: einter_all[k] for k, pair in enumerate(all_pairs)}
    reference_available = bool(np.any(np.isfinite(exact_all)))
    if not reference_available and not settings['skip_convergence']:
        raise ValueError(
            'No pairwise IQA data found — the convergence gate compares the multipole series '
            'against the exact V_cl(A,B), which AIMAll only writes when it is run with IQA '
            'enabled (the same requirement auto_reg.py has). Rerun AIMAll with IQA, or pass '
            '--skip-convergence-gate to produce an ungated decomposition whose accuracy is '
            'then unknown.')

    ###########################################################################
    #                             PAIR SELECTION                              #
    ###########################################################################

    groups, pair_index = build_groups(atoms, frag_names, frag_atom_lists, scope, within)
    n_pairs = len(pair_index)
    exact_vcl = np.array([exact_lookup[p] for p in pair_index])
    exact_vxc = np.array([vxc_lookup[p] for p in pair_index])
    exact_einter = np.array([einter_lookup[p] for p in pair_index])
    print('Scope: {s} — {g} group(s), {p} atom pair(s), L_max = {l} (l_tot <= {t})'
          .format(s=scope, g=len(groups), p=n_pairs, l=l_max, t=2 * l_max))

    # Cheap gates first, so the expensive tensor work is only done for pairs that
    # could still be admitted.  --test-all-pairs computes everything instead, which
    # is what lets the report quantify how far the rejected pairs actually are.
    bonds = mp.bond_graph(atoms, coords, tolerance=settings['bond_tolerance'])
    separation = mp.topology_separation(n_atoms, bonds, max_depth=3)
    if option.test_all or not settings['topology']:
        compute_mask = np.ones(n_pairs, dtype=bool)
    else:
        compute_mask = np.array([separation.get(p) not in (2, 3) for p in pair_index])

    # The rank table is the one array that scales with the pair count, at
    # 8 * (L_max+1)^2 bytes per pair per geometry — 288 bytes at L_max = 5.  Say
    # how big it is rather than let a large system meet it as a surprise; the
    # tensor itself is chunked and does not grow with the system.
    table_mb = n_pairs * len(cc) * (l_max + 1) ** 2 * 8 / (1024.0 * 1024.0)
    print('Building interaction tensors for {n} of {t} pair(s) — rank table {m:.0f} MB ...'
          .format(n=int(compute_mask.sum()), t=n_pairs, m=table_mb))
    if table_mb > 2048:
        print('WARNING: that table is large. --scope inter analyses only interfragment pairs, '
              'and a lower --lmax shrinks it quadratically.')
    rank_energies = np.full((n_pairs, len(cc), l_max + 1, l_max + 1), np.nan)
    compute_ids = np.flatnonzero(compute_mask)
    if compute_ids.size:
        # One flat batch over (pair, geometry); the tensor code chunks it internally.
        pair_i = np.repeat(compute_ids, len(cc))
        step_i = np.tile(np.arange(len(cc)), compute_ids.size)
        idx_a = np.array([pair_index[p][0] for p in pair_i])
        idx_b = np.array([pair_index[p][1] for p in pair_i])
        flat = mp.pair_rank_energies(
            moments[step_i, idx_a], moments[step_i, idx_b],
            coords[step_i, idx_a], coords[step_i, idx_b], l_max)
        rank_energies[pair_i, step_i] = flat

    ###########################################################################
    #                              ADMISSION                                  #
    ###########################################################################

    verdicts, admission_info = admit_pairs(
        mp, atoms, coords, pair_index, exact_vcl,
        None if settings['skip_convergence'] else rank_energies,
        beta_radii, segments, settings, separation=separation, bonds=bonds)

    admitted_mask = np.array([v['admitted'] for v in verdicts])
    print('Admitted {a} of {t} pair(s); {r} rejected'
          .format(a=int(admitted_mask.sum()), t=n_pairs, r=int((~admitted_mask).sum())))

    ###########################################################################
    #                        GROUP TERMS AND ACCOUNTING                       #
    ###########################################################################

    n_steps = len(cc)

    # Term labels lead with the multipole names and keep the rank indices beside
    # them, so a table can be read as "dipole-quadrupole" without losing which
    # (la, lb) block it actually is.
    def rank_header(group_label, la, lb, prefix=''):
        return '{g} {p}{n} [{a},{b}]'.format(g=group_label, p=prefix,
                                             n=mp.rank_pair_name(la, lb), a=la, b=lb)

    def shell_header(group_label, l_tot, prefix=''):
        return '{g} {p}l_tot={t} (R^-{r})'.format(g=group_label, p=prefix, t=l_tot, r=l_tot + 1)

    headers = []
    term_values = []
    shell_headers = []
    shell_values = []
    group_reports = []

    for group in groups:
        ids = np.array(group['pairs'])
        admitted_ids = ids[admitted_mask[ids]]
        rejected_ids = ids[~admitted_mask[ids]]

        # Plain sums, not nansum: an unreadable pair makes the group total unknown,
        # and quietly counting it as zero would turn a gap in the data into a
        # statement that the channel is flat.
        blocks = (np.sum(rank_energies[admitted_ids], axis=0) if admitted_ids.size
                  else np.zeros((n_steps, l_max + 1, l_max + 1)))
        multipolar_total = blocks.sum(axis=(1, 2))
        exact_admitted = (np.sum(exact_vcl[admitted_ids], axis=0) if admitted_ids.size
                          else np.zeros(n_steps))
        unresolved = (np.sum(exact_vcl[rejected_ids], axis=0) if rejected_ids.size
                      else np.zeros(n_steps))
        residual = exact_admitted - multipolar_total
        exact_total = exact_admitted + unresolved

        label = group['label']
        # A group with no admitted pair has no multipole description at all: its
        # rank terms would be a block of zeros, which would pad every table and
        # rank as "no effect" rather than as "not described".  Its energy is in
        # REG_Multi_energy.csv as an unresolved channel.
        if admitted_ids.size:
            for la in range(l_max + 1):
                for lb in range(l_max + 1):
                    headers.append(rank_header(label, la, lb))
                    term_values.append(blocks[:, la, lb])
        # The truncation residual and the unresolved channel are *not* REG terms.
        # They are bookkeeping — what the series left behind and what the gates
        # refused — and putting them in a ranked table of multipole terms would
        # invite reading them as a kind of multipole.  They are reported on the
        # energies instead: REG_Multi_energy.csv and excluded_accounting.csv,
        # where the balance still closes exactly.
            increments, _ = mp.partial_sums_by_total_rank(blocks)
            for l_tot in range(2 * l_max + 1):
                shell_headers.append(shell_header(label, l_tot))
                shell_values.append(increments[:, l_tot])

        # Context channels: the exchange-correlation part of the same pairs, and
        # their total IQA interaction.  The multipole series describes V_cl only,
        # so "how well is V_cl reproduced" and "how much of the interaction is
        # V_cl in the first place" are separate questions, and both get answered.
        vxc_total = np.sum(exact_vxc[ids], axis=0) if ids.size else np.zeros(n_steps)
        einter_total = np.sum(exact_einter[ids], axis=0) if ids.size else np.zeros(n_steps)

        # These describe the admitted pairs only.  With nothing admitted they are
        # undefined, not zero: such a group still has a V_cl, and reporting 0.000
        # would read as "no classical energy here" rather than "not described".
        if admitted_ids.size:
            with np.errstate(divide='ignore', invalid='ignore'):
                mean_abs_exact = float(np.nanmean(np.abs(exact_admitted)))
                mean_abs_residual = float(np.nanmean(np.abs(residual)))
                recovered = (1.0 - mean_abs_residual / mean_abs_exact
                             if mean_abs_exact > 0 else float('nan'))
        else:
            mean_abs_exact = mean_abs_residual = recovered = float('nan')

        group_reports.append({
            'label': label,
            'kind': group['kind'],
            'blocks': blocks,
            'vxc': vxc_total,
            'einter': einter_total,
            'mean_abs_exact_kj_mol': mean_abs_exact * HA_TO_KJ,
            'mean_abs_residual_kj_mol': mean_abs_residual * HA_TO_KJ,
            'max_abs_residual_kj_mol': (float(np.nanmax(np.abs(residual))) * HA_TO_KJ
                                        if np.any(np.isfinite(residual)) else float('nan')),
            'recovered_fraction': recovered,
            'n_pairs': int(ids.size),
            'n_admitted': int(admitted_ids.size),
            'n_rejected': int(rejected_ids.size),
            'multipolar_total': multipolar_total,
            'exact_admitted': exact_admitted,
            'exact_total': exact_total,
            'residual': residual,
            'unresolved': unresolved,
            'span_exact_kj_mol': _span(exact_total) * HA_TO_KJ,
            'span_unresolved_kj_mol': _span(unresolved) * HA_TO_KJ,
            'span_residual_kj_mol': _span(residual) * HA_TO_KJ,
            'unresolved_fraction': (_span(unresolved) / _span(exact_total)
                                    if _span(exact_total) > 0 else float('nan')),
            'residual_fraction': (_span(residual) / _span(exact_total)
                                  if _span(exact_total) > 0 else float('nan')),
        })

    system_exact = np.sum([g['exact_total'] for g in group_reports], axis=0)
    system_unresolved = np.sum([g['unresolved'] for g in group_reports], axis=0)
    system_residual = np.sum([g['residual'] for g in group_reports], axis=0)
    accounting = {
        'span_exact_kj_mol': _span(system_exact) * HA_TO_KJ,
        'span_unresolved_kj_mol': _span(system_unresolved) * HA_TO_KJ,
        'span_residual_kj_mol': _span(system_residual) * HA_TO_KJ,
        'unresolved_fraction': (_span(system_unresolved) / _span(system_exact)
                                if _span(system_exact) > 0 else float('nan')),
        'residual_fraction': (_span(system_residual) / _span(system_exact)
                              if _span(system_exact) > 0 else float('nan')),
        'end_to_end_exact_kj_mol': float(system_exact[-1] - system_exact[0]) * HA_TO_KJ,
        'end_to_end_unresolved_kj_mol': float(system_unresolved[-1] - system_unresolved[0]) * HA_TO_KJ,
        'reference_available': bool(reference_available),
    }

    ###########################################################################
    #                    FRAGMENT-CENTRED ("GROUPED ATOM") VIEW               #
    ###########################################################################
    # Kept in its own arrays and its own files throughout.  The two views answer
    # different questions and their rank terms are different quantities, so they
    # must never end up in one ranked table.

    fragment_view = None
    frag_headers, frag_values = [], []
    frag_shell_headers, frag_shell_values = [], []
    if frag_atom_lists:
        print('Building fragment moments about each fragment {c}{e} ...'.format(
            c=settings['fragment_centre'],
            e=' and expanding the fragment pairs' if settings['fragment_moments'] else ''))
        basin_radii_ang, _ = mp.basin_radii(atoms, beta_radii, source=settings['radii'])
        fragment_view = fragment_moment_analysis(
            mp, atoms, coords, moments, frag_names, frag_atom_lists, groups, pair_index,
            exact_vcl, basin_radii_ang, segments, l_max, settings,
            with_energies=settings['fragment_moments'])

    if fragment_view:
        for entry in fragment_view['pairs']:
            label = entry['label']
            if entry['admitted']:
                for la in range(l_max + 1):
                    for lb in range(l_max + 1):
                        frag_headers.append(rank_header(label, la, lb, prefix='frag: '))
                        frag_values.append(entry['ranks'][:, la, lb])
                increments, _ = mp.partial_sums_by_total_rank(entry['ranks'])
                for l_tot in range(2 * l_max + 1):
                    frag_shell_headers.append(shell_header(label, l_tot, prefix='frag: '))
                    frag_shell_values.append(increments[:, l_tot])
            # A fragment pair that fails contributes no terms at all: there is no
            # partial credit, because the expansion that would be credited is the
            # one that does not converge.  Its energy is in REG_Multi_energy.csv.

    ###########################################################################
    #                              REG ANALYSIS                               #
    ###########################################################################

    term_values = _table(term_values, n_steps)
    shell_values = _table(shell_values, n_steps)

    # No admitted pair anywhere means there is nothing to regress — which is a
    # real answer, not a failure: asked to work inside a fragment, every pair is
    # 1,2 or 1,3 and none of them can carry a nucleus-centred expansion.  The
    # admission report and the energies still say so, and are still written.
    has_terms = term_values.size > 0
    if not has_terms:
        print('No atom pair was admitted, so there are no multipole terms to rank. The')
        print('admission report and the energy files still record every pair and why it was')
        print('rejected; there is simply no decomposition to put beside them.')
    reg_terms = (reg.reg(total_energy_wfn, cc, term_values, np=POINTS, critical=AUTO,
                         inflex=INFLEX, critical_index=turning_points) if has_terms else None)
    reg_shells = (reg.reg(total_energy_wfn, cc, shell_values, np=POINTS, critical=AUTO,
                          inflex=INFLEX, critical_index=turning_points) if has_terms else None)

    reg_frag = None
    reg_frag_shells = None
    if frag_values:
        reg_frag = reg.reg(total_energy_wfn, cc, _table(frag_values, n_steps), np=POINTS,
                           critical=AUTO, inflex=INFLEX, critical_index=turning_points)
        reg_frag_shells = reg.reg(total_energy_wfn, cc, _table(frag_shell_values, n_steps),
                                  np=POINTS, critical=AUTO, inflex=INFLEX,
                                  critical_index=turning_points)

    reg_pairs = None
    pair_headers = []
    pair_values = []
    if option.pair_terms:
        for p_i, (i, j) in enumerate(pair_index):
            if not admitted_mask[p_i]:
                continue
            pair_headers.append('{a}-{b} V_multi'.format(a=atoms[i], b=atoms[j]))
            pair_values.append(np.nansum(rank_energies[p_i], axis=(1, 2)))
        if pair_values:
            reg_pairs = reg.reg(total_energy_wfn, cc, _table(pair_values, n_steps), np=POINTS,
                                critical=AUTO, inflex=INFLEX, critical_index=turning_points)

    n_segments = len(reg_terms[0]) if reg_terms is not None else len(segments)

    ###########################################################################
    #                                 REPORT                                  #
    ###########################################################################

    sep_wide = '=' * 96
    lines = [sep_wide,
             '  REG_Multi — RANK-RESOLVED MULTIPOLAR ELECTROSTATICS',
             sep_wide, '']
    lines.append('  Scope                : {s}{f}'.format(
        s=scope, f='  (fragments ignored — the gates alone decide)' if ignore_fragments else ''))
    lines.append('  Groups               : {n}'.format(n=len(groups)))
    lines.append('  Candidate atom pairs : {n}'.format(n=n_pairs))
    lines.append('  L_max per atom       : {l}  (l_tot = l_A + l_B <= {t}; AIMAll wrote l <= {a})'
                 .format(l=l_max, t=2 * l_max, a=l_available))
    lines.append('  Geometry points      : {n}, in {s} segment(s)'.format(n=n_steps, s=len(segments)))
    lines.append('')
    lines.append('  Gate settings')
    lines.append('    topology pre-filter : {s}'.format(
        s='on — 1,2 and 1,3 rejected' if settings['topology'] else 'OFF'))
    lines.append('    basin radii         : {s} ({b})'.format(
        s=settings['radii'],
        b='strict lower bound — necessary, not sufficient'
          if admission_info['radii_are_lower_bound'] else 'outer estimate'))
    if settings['skip_convergence']:
        lines.append('    convergence gate    : SKIPPED — the decomposition below is NOT known to '
                     'be numerically meaningful')
    else:
        lines.append('    residual tolerance  : {t:.1%} of each pair own peak-to-peak V_cl per segment'
                     .format(t=settings['tolerance']))
        lines.append('    flat-term floor     : {f} kJ/mol'.format(f=settings['floor']))
        lines.append('    increment condition : |dV| not growing across l_tot shells {w}'
                     .format(w=('2..{m} (every complete shell)'.format(m=l_max)
                                if settings['increment_ranks'] is None
                                else 'the top {n} complete ones'.format(
                                    n=settings['increment_ranks']))))
        if l_max < 3:
            lines.append('    WARNING: L_max = {m} leaves too little series to test for growth; '
                         'the increment'.format(m=l_max))
            lines.append('             condition is effectively inactive at this rank ceiling.')
        lines.append('    path-wide admission : a pair must pass at every geometry')
    lines.append('')

    if not reference_available:
        lines.append('  WARNING: no exact IQA V_cl was available anywhere in this run. Nothing')
        lines.append('  below has been checked against a reference, the residual is unknown, and')
        lines.append('  the rank terms may be describing a series that does not converge.')
        lines.append('')

    rejected = [v for v in verdicts if not v['admitted']]
    by_reason = {}
    for v in rejected:
        by_reason.setdefault(v['rejected_by'], []).append(v)
    lines.append('  ' + '-' * 92)
    lines.append('  ADMISSION: {a} admitted, {r} rejected'.format(
        a=int(admitted_mask.sum()), r=len(rejected)))
    lines.append('  ' + '-' * 92)
    for reason in ('topology', 'geometry', 'residual', 'increments', 'convergence', 'no_reference'):
        if reason in by_reason:
            lines.append('    {r:<14s} {n:>6d}'.format(r=reason, n=len(by_reason[reason])))
    lines.append('')

    # Pairs rejected across a fragment boundary are worth naming: a fragmentation
    # that cuts a covalent bond sends that bond's whole electrostatic channel into
    # the unresolved bucket, and the reader should know it was the partition that
    # put it there rather than the physics.
    interfragment_rejects = []
    for group in groups:
        if group['kind'] != 'inter':
            continue
        for p_id in group['pairs']:
            v = verdicts[p_id]
            if not v['admitted'] and v['rejected_by'] in ('topology', 'geometry'):
                interfragment_rejects.append((group['label'], v))
    if interfragment_rejects:
        lines.append('  Interfragment pairs rejected by a cheap gate — the partition cuts through')
        lines.append('  a bond or a contact too close for nucleus-centred moments:')
        for label, v in interfragment_rejects[:25]:
            lines.append('    {g:<28s} {p:<12s} {n}'.format(g=label, p=v['label'], n=v['note']))
        if len(interfragment_rejects) > 25:
            lines.append('    ... and {n} more (see multipole_admission.csv)'
                         .format(n=len(interfragment_rejects) - 25))
        lines.append('')

    lines.append('  ' + '-' * 92)
    lines.append('  WHAT THE RANK DECOMPOSITION DOES NOT COVER')
    lines.append('  ' + '-' * 92)
    lines.append('  V_cl(total, in scope) = rank terms + residual + unresolved, at every step.')
    lines.append('  Spans below are peak-to-peak along the whole control coordinate, and the')
    lines.append('  fraction is the unresolved span over the total span. It can exceed 100%:')
    lines.append('  that means the excluded channel moves more than the net does, so the part')
    lines.append('  that was resolved is partly cancelling it.')
    lines.append('')
    lines.append('  {g:<30s} {p:>10s} {a:>10s} {e:>12s} {u:>12s} {f:>9s}'.format(
        g='Group', p='pairs', a='admitted', e='span V_cl', u='unresolved', f='fraction'))
    lines.append('  ' + '-' * 92)
    # In 'pairs' scope every atom pair is its own group, so this table would run
    # to thousands of rows on a real system.  Print the most active ones and send
    # the reader to the CSV for the rest; the totals below are over all of them.
    _TABLE_LIMIT = 30
    _ordered = sorted(group_reports,
                      key=lambda r: (r['span_exact_kj_mol']
                                     if r['span_exact_kj_mol'] == r['span_exact_kj_mol'] else -1),
                      reverse=True)
    _shown = _ordered[:_TABLE_LIMIT] if len(_ordered) > _TABLE_LIMIT else group_reports
    for report in _shown:
        lines.append('  {g:<30s} {p:>10d} {a:>10d} {e} {u} {f}'.format(
            g=report['label'][:30], p=report['n_pairs'], a=report['n_admitted'],
            e=_fmt(report['span_exact_kj_mol']), u=_fmt(report['span_unresolved_kj_mol']),
            f=_fmt(report['unresolved_fraction'], '{:.1%}', 9)))
    if len(_ordered) > _TABLE_LIMIT:
        lines.append('  ... {n} more, ranked by span in excluded_accounting.csv'
                     .format(n=len(_ordered) - _TABLE_LIMIT))
    lines.append('  ' + '-' * 92)
    lines.append('  {g:<30s} {p:>10s} {a:>10s} {e} {u} {f}'.format(
        g='ALL GROUPS', p='', a='', e=_fmt(accounting['span_exact_kj_mol']),
        u=_fmt(accounting['span_unresolved_kj_mol']),
        f=_fmt(accounting['unresolved_fraction'], '{:.1%}', 9)))
    lines.append('')
    if accounting['span_residual_kj_mol'] == accounting['span_residual_kj_mol']:
        lines.append('  Truncation residual of the admitted pairs spans {r:.3f} kJ/mol ({f} of '
                     'the total).'.format(
                         r=accounting['span_residual_kj_mol'],
                         f=_fmt(accounting['residual_fraction'], '{:.1%}', 0)))
    else:
        lines.append('  Truncation residual: not known — the exact IQA V_cl was unavailable, so '
                     'there is nothing to compare the series against.')
    unresolved_fraction = accounting['unresolved_fraction']
    if unresolved_fraction == unresolved_fraction and unresolved_fraction > 1.0:
        lines.append('')
        lines.append('  NOTE: the excluded pairs move {f:.1f} times as much as the total change in '
                     'V_cl.'.format(f=unresolved_fraction))
        lines.append('  The rank decomposition below describes a part of the electrostatics that is')
        lines.append('  smaller than what it leaves out, and the two partly cancel. Read it as an')
        lines.append('  account of that part only, and quote the unresolved channel beside it.')
    elif unresolved_fraction == unresolved_fraction and unresolved_fraction > 0.25:
        lines.append('')
        lines.append('  NOTE: more than a quarter of the change in V_cl sits in pairs this method')
        lines.append('  cannot resolve. The rank decomposition describes the rest, not the whole,')
        lines.append('  and any interpretation of it has to say so.')
    lines.append('')
    lines.append('  ' + '-' * 92)
    lines.append('  ELECTROSTATIC RECOVERY — is V_cl reproduced by the multipole series?')
    lines.append('  ' + '-' * 92)
    lines.append('  Path averages in kJ/mol, over the admitted pairs of each group. "recovered" is')
    lines.append('  1 - mean|residual| / mean|V_cl|, so it says how much of the classical energy the')
    lines.append('  series accounts for. V_xc is the exchange-correlation energy of the same pairs:')
    lines.append('  the series does not describe it, and it is here to show how much of the')
    lines.append('  interaction is not classical in the first place.')
    lines.append('')
    lines.append('  {g:<26s} {e:>12s} {m:>12s} {r:>11s} {x:>10s} {v:>12s} {i:>12s}'.format(
        g='Group', e='<|V_cl|>', m='<|V_multi|>', r='<|resid|>', x='recovered',
        v='<V_xc>', i='<E_inter>'))
    lines.append('  ' + '-' * 92)
    # Ranked among the groups that actually have a multipole description, biggest
    # first.  Ranking by V_cl span instead — as the table above does, where the
    # excluded pairs are the point — fills this one with rejected pairs and 'n/a'.
    _described = [r for r in group_reports if r['n_admitted']]
    _recovery_rows = sorted(_described, key=lambda r: r['mean_abs_exact_kj_mol'],
                            reverse=True)[:_TABLE_LIMIT]
    _total_exact = np.sum([r['exact_total'] - r['unresolved'] for r in _described], axis=0) \
        if _described else np.zeros(n_steps)
    _total_multi = np.sum([r['multipolar_total'] for r in _described], axis=0) \
        if _described else np.zeros(n_steps)
    _total_abs_exact = float(np.nanmean(np.abs(_total_exact)))
    _total_abs_resid = float(np.nanmean(np.abs(_total_exact - _total_multi)))
    lines.append('  {g:<26s} {e} {m} {r} {x} {v} {i}'.format(
        g='TOTAL (included pairs)',
        e=_fmt(_total_abs_exact * HA_TO_KJ, '{:.3f}', 12),
        m=_fmt(float(np.nanmean(np.abs(_total_multi))) * HA_TO_KJ, '{:.3f}', 12),
        r=_fmt(_total_abs_resid * HA_TO_KJ, '{:.3f}', 11),
        x=_fmt(1.0 - _total_abs_resid / _total_abs_exact if _total_abs_exact > 0 else float('nan'),
               '{:.2%}', 10),
        v=_fmt(float(np.nanmean(np.sum([r['vxc'] for r in group_reports], axis=0))) * HA_TO_KJ,
               '{:+.3f}', 12),
        i=_fmt(float(np.nanmean(np.sum([r['einter'] for r in group_reports], axis=0))) * HA_TO_KJ,
               '{:+.3f}', 12)))
    lines.append('  ' + '-' * 92)
    for report in _recovery_rows:
        lines.append('  {g:<26s} {e} {m} {r} {x} {v} {i}'.format(
            g=report['label'][:26],
            e=_fmt(report['mean_abs_exact_kj_mol'], '{:.3f}', 12),
            m=_fmt(float(np.nanmean(np.abs(report['multipolar_total']))) * HA_TO_KJ
                   if (report['n_admitted'] and np.any(np.isfinite(report['multipolar_total'])))
                   else float('nan'), '{:.3f}', 12),
            r=_fmt(report['mean_abs_residual_kj_mol'], '{:.3f}', 11),
            x=_fmt(report['recovered_fraction'], '{:.2%}', 10),
            v=_fmt(float(np.nanmean(report['vxc'])) * HA_TO_KJ
                   if np.any(np.isfinite(report['vxc'])) else float('nan'), '{:+.3f}', 12),
            i=_fmt(float(np.nanmean(report['einter'])) * HA_TO_KJ
                   if np.any(np.isfinite(report['einter'])) else float('nan'), '{:+.3f}', 12)))
    if len(_described) > _TABLE_LIMIT:
        lines.append('  ... {n} more described groups, in REG_Multi_recovery.csv'
                     .format(n=len(_described) - _TABLE_LIMIT))
    if len(_described) < len(group_reports):
        lines.append('  ({n} group(s) have no admitted pair and so no multipole description; '
                     'their'.format(n=len(group_reports) - len(_described)))
        lines.append('   energy is the excluded channel in the table above.)')
    _worst = max(_described, key=lambda r: r['mean_abs_residual_kj_mol']) if _described else None
    if _worst is not None and _worst['mean_abs_residual_kj_mol'] > 0:
        lines.append('')
        lines.append('  Weakest recovery of any described group: {g} at {p}, mean residual '
                     '{r:.3f} kJ/mol.'.format(
                         g=_worst['label'],
                         p=_fmt(_worst['recovered_fraction'], '{:.2%}', 0),
                         r=_worst['mean_abs_residual_kj_mol']))
    lines.append('')
    lines.append('  Per-step energies are in REG_Multi_energy.csv: exact V_cl, the multipole sum,')
    lines.append('  the truncation residual and the unresolved channel, per group and in total,')
    lines.append('  with V_xc and E_inter beside them.')
    lines.append('')

    if fragment_view:
        lines.append('  ' + '-' * 92)
        lines.append('  FRAGMENT MOMENTS (grouped-atom moments, centre: {c})'
                     .format(c=fragment_view['centre_scheme']))
        lines.append('  ' + '-' * 92)
        lines.append('  Each fragment moments are translated onto one centre and summed, so a')
        lines.append('  fragment pair gets a single expansion. This is a different decomposition of')
        lines.append('  the same energy, not a regrouping of the terms above: for neutral fragments')
        lines.append('  its monopole terms vanish identically, while the atom-centred series puts')
        lines.append('  large charge-charge terms against each other that mostly cancel. The two')
        lines.append('  are reported separately and must not be ranked in one table.')
        lines.append('')
        if not fragment_view['pairs']:
            lines.append('  Moments are reported per fragment in fragment_moments.csv. Expanding')
            lines.append('  fragment pairs against each other is a separate analysis — add')
            lines.append('  --fragment-moments for it.')
            lines.append('')
        lines.append('  {g:<26s} {s:>9s} {e:>10s} {m:>9s} {r:>11s}  {v}'.format(
            g='Fragment pair', s='R(F-G)', e='needs', m='margin', r='residual', v='verdict')
            if fragment_view['pairs'] else '')
        lines.append('  ' + '-' * 92)
        for entry in fragment_view['pairs']:
            residual_kj = _span(entry['residual']) * HA_TO_KJ
            lines.append('  {g:<26s} {s:>9.2f} {e:>10.2f} {m:>9.2f} {r:>11.3f}  {v}'.format(
                g=entry['label'][:26], s=float(entry['separation_ang'].min()),
                e=float((entry['separation_ang'] - entry['extent_margin_ang']).max()),
                m=float(entry['extent_margin_ang'].min()), r=residual_kj,
                v='admitted' if entry['admitted'] else 'REJECTED (' + str(entry['rejected_by']) + ')'))
            if entry['note']:
                lines.append('      ' + entry['note'])
        if fragment_view['pairs']:
            lines.append('')
            lines.append('  R(F-G) is the closest approach of the two centres and "needs" is the sum')
            lines.append('  of the fragment extents there, so margin < 0 means the spheres overlap.')
        lines.append('')
        lines.append('  How far the atoms contributions to each fragment rank cancel, path average:')
        lines.append('  |sum_A Q_l(A)| / sum_A |Q_l(A)|, both after translation onto the fragment')
        lines.append('  centre, so the comparison is like with like and the ratio cannot exceed 1.')
        lines.append('  A small value is the case worth naming: the contributions oppose each other,')
        lines.append('  so the net fragment term of that rank is weak however large the atomic ones.')
        lines.append('')
        header = '  {f:<26s}'.format(f='Fragment') + ''.join(
            '{h:>9s}'.format(h='l=' + str(l)) for l in range(l_max + 1))
        lines.append(header)
        lines.append('  ' + '-' * 92)
        for name, rows in list(fragment_view['cancellation'].items())[:_TABLE_LIMIT]:
            lines.append('  {f:<26s}'.format(f=name[:26]) + ''.join(
                '{v:>9s}'.format(v=('%.2f' % np.nanmean(row['ratio']))
                                 if np.any(np.isfinite(row['ratio'])) else 'n/a')
                for row in rows))
        lines.append('')

    lines.append('  Bringing the rejected 1,2 and 1,3 pairs back would need the MMS shift, which')
    lines.append('  moves the expansion centres — a REG built on shifted moments is not comparable')
    lines.append('  term-by-term with this one and belongs in its own analysis.')
    lines.append('')

    report_text = '\n'.join(lines)
    print(report_text)
    ensure_results_dir()
    with open(os.path.join(results_dir, 'multipole_admission.txt'), 'w') as handle:
        handle.write(report_text + '\n')

    ###########################################################################
    #                                 OUTPUT                                  #
    ###########################################################################

    os.chdir(results_dir)

    admission_rows = []
    for p_i, v in enumerate(verdicts):
        i, j = v['pair']
        row = {
            'atom_A': atoms[i], 'atom_B': atoms[j],
            'separation': v['separation'], 'admitted': v['admitted'],
            'rejected_by': v['rejected_by'], 'note': v['note'],
            'min_R_ang': v['min_distance_ang'], 'max_R_ang': v['max_distance_ang'],
            'geometry_margin_ang': v.get('geometry_margin_ang'),
            'max_abs_residual_kj_mol': v.get('max_abs_residual_kj_mol'),
            'mean_abs_residual_kj_mol': v.get('mean_abs_residual_kj_mol'),
            'exact_vcl_span_kj_mol': _span(exact_vcl[p_i]) * HA_TO_KJ,
        }
        if 'convergence' in v:
            worst = max(v['convergence']['segments'], key=lambda s: s['ratio'])
            row['worst_segment'] = worst['segment'] + 1
            row['worst_residual_ratio'] = worst['ratio']
            row['n_geometries_not_decreasing'] = len(v['convergence']['increment_failures'])
        admission_rows.append(row)
    df_admission = pd.DataFrame(admission_rows)
    df_admission.to_csv('multipole_admission.csv', sep=',', index=False)

    df_terms = pd.DataFrame(data=term_values, index=headers, columns=cc).rename_axis('TERM')
    df_terms.to_csv('multipole_terms.csv', sep=',')
    df_shells = pd.DataFrame(data=shell_values, index=shell_headers, columns=cc).rename_axis('TERM')
    df_shells.to_csv('multipole_shells.csv', sep=',')

    # ---- energies -------------------------------------------------------
    # One row per series, one column per geometry point.  The other way round —
    # which is how this started — puts every channel in its own column, and on a
    # 238-atom system in 'pairs' scope that is 169,226 columns: past Excel's
    # 16,384 and unreadable besides.  Rows scale the same way but Excel allows a
    # million of them, and a long thin table is what a plotting tool wants.
    #
    # Fragment groupings are built here from the config whatever the scope is, so
    # an atom-wise run still gets a fragment-level energy breakdown.  That is a
    # reporting sum over atom-pair terms; it does not change the analysis.
    frag_energy_groups = []
    if frag_atom_lists:
        frag_members = [[a - 1 for a in atom_list] for atom_list in frag_atom_lists]
        pair_position = {pair: k for k, pair in enumerate(pair_index)}

        def _ids(pairs_wanted):
            return [pair_position[p] for p in pairs_wanted if p in pair_position]

        for f_i in range(len(frag_members)):
            ids = _ids([(min(a, b), max(a, b))
                        for x, a in enumerate(frag_members[f_i])
                        for b in frag_members[f_i][x + 1:]])
            if ids:
                frag_energy_groups.append((str(frag_names[f_i]) + '(intra)', np.array(ids)))
        for f_i in range(len(frag_members)):
            for f_j in range(f_i + 1, len(frag_members)):
                ids = _ids([(min(a, b), max(a, b))
                            for a in frag_members[f_i] for b in frag_members[f_j]])
                if ids:
                    frag_energy_groups.append(
                        (str(frag_names[f_i]) + '|' + str(frag_names[f_j]), np.array(ids)))
    else:
        frag_energy_groups.append(('System', np.arange(n_pairs)))

    def _channels(ids):
        """Energies of one set of pairs, split into included and excluded."""
        ids = np.asarray(ids, dtype=int)
        admitted_ids = ids[admitted_mask[ids]] if ids.size else ids
        rejected_ids = ids[~admitted_mask[ids]] if ids.size else ids
        multipolar = (np.sum(rank_energies[admitted_ids], axis=(0, 2, 3)) if admitted_ids.size
                      else np.zeros(n_steps))
        included = (np.sum(exact_vcl[admitted_ids], axis=0) if admitted_ids.size
                    else np.zeros(n_steps))
        excluded = (np.sum(exact_vcl[rejected_ids], axis=0) if rejected_ids.size
                    else np.zeros(n_steps))
        return {'included': included, 'multipolar': multipolar, 'excluded': excluded,
                'n_admitted': int(admitted_ids.size), 'n_rejected': int(rejected_ids.size),
                'vxc': np.sum(exact_vxc[ids], axis=0) if ids.size else np.zeros(n_steps),
                'einter': np.sum(exact_einter[ids], axis=0) if ids.size else np.zeros(n_steps)}

    all_ids = np.arange(n_pairs)
    # Each grouping's channels are wanted twice, for the recovery file and for the
    # balance.  On a large system each call is a fancy-indexed sum over the whole
    # rank table, so they are computed once and kept.
    channel_cache = {label: _channels(ids) for label, ids in frag_energy_groups}
    totals = _channels(all_ids)

    # (1) The recovery comparison: exact V_cl against the multipole sum, over the
    # *same* pairs — the admitted ones — and nothing else.  Residual and excluded
    # channels are deliberately not in this file: the question it answers is
    # whether the series reproduces what it claims to describe, and adding the
    # part it does not describe to either side would make the two curves agree or
    # disagree for the wrong reason.
    recovery_rows = {
        'TOTAL Vcl_IQA (included pairs)': totals['included'],
        'TOTAL Vcl_multipole (included pairs)': totals['multipolar'],
    }
    for label, ids in frag_energy_groups:
        channels = channel_cache[label]
        if not channels['n_admitted']:
            continue
        recovery_rows[label + ' Vcl_IQA (included pairs)'] = channels['included']
        recovery_rows[label + ' Vcl_multipole (included pairs)'] = channels['multipolar']
    if fragment_view:
        for entry in fragment_view['pairs']:
            if entry['admitted']:
                recovery_rows[entry['label'] + ' frag: Vcl_IQA'] = entry['exact']
                recovery_rows[entry['label'] + ' frag: Vcl_multipole'] = entry['total']
    df_energy_recovery = pd.DataFrame(
        data=list(recovery_rows.values()), index=list(recovery_rows.keys()),
        columns=cc).rename_axis('SERIES')
    df_energy_recovery.to_csv('energy_recovery.csv', sep=',')

    # (2) The same energy summed into fragment pairs and split by multipole type,
    # whatever the scope of the analysis was.
    type_rows, type_index = [], []
    for label, ids in frag_energy_groups:
        admitted_ids = ids[admitted_mask[ids]]
        if not admitted_ids.size:
            continue
        blocks = np.sum(rank_energies[admitted_ids], axis=0)
        for la in range(l_max + 1):
            for lb in range(l_max + 1):
                type_index.append(rank_header(label, la, lb))
                type_rows.append(blocks[:, la, lb])
        type_index.append(label + ' total (included pairs)')
        type_rows.append(blocks.sum(axis=(1, 2)))
    df_energy_by_type = pd.DataFrame(data=_table(type_rows, n_steps), index=type_index,
                                     columns=cc).rename_axis('TERM')
    df_energy_by_type.to_csv('energy_by_type.csv', sep=',')

    # (3) Every admitted atom pair's multipole energy, on its own.  No IQA
    # columns: this one is the multipole analysis alone.
    pair_rows, pair_labels = [], []
    for p_i, (i, j) in enumerate(pair_index):
        if not admitted_mask[p_i]:
            continue
        pair_labels.append('{a}-{b}'.format(a=atoms[i], b=atoms[j]))
        pair_rows.append(np.sum(rank_energies[p_i], axis=(1, 2)))
    df_energy_by_pair = pd.DataFrame(data=_table(pair_rows, n_steps), index=pair_labels,
                                     columns=cc).rename_axis('PAIR')
    df_energy_by_pair.to_csv('energy_by_pair.csv', sep=',')

    # The full balance, per fragment grouping and in total, for the accounting
    # rather than for the recovery plot.
    balance_rows, balance_index = [], []
    for label, ids in [('TOTAL', all_ids)] + frag_energy_groups:
        channels = totals if label == 'TOTAL' else channel_cache[label]
        for key, suffix in (('included', 'Vcl_IQA (included pairs)'),
                            ('multipolar', 'Vcl_multipole'),
                            ('excluded', 'Vcl_IQA (excluded pairs)'),
                            ('vxc', 'Vxc_IQA (all pairs)'),
                            ('einter', 'Einter_IQA (all pairs)')):
            balance_index.append(label + ' ' + suffix)
            balance_rows.append(channels[key])
        balance_index.append(label + ' Vcl_residual (included pairs)')
        balance_rows.append(channels['included'] - channels['multipolar'])
    balance_index.append('E_WFN')
    balance_rows.append(total_energy_wfn)
    df_energy = pd.DataFrame(data=balance_rows, index=balance_index,
                             columns=cc).rename_axis('SERIES')
    df_energy.to_csv('REG_Multi_energy.csv', sep=',')

    df_recovery = pd.DataFrame([
        {'group': r['label'], 'kind': r['kind'],
         'pairs': r['n_pairs'], 'admitted': r['n_admitted'],
         'mean_abs_Vcl_kj_mol': r['mean_abs_exact_kj_mol'],
         'mean_abs_residual_kj_mol': r['mean_abs_residual_kj_mol'],
         'max_abs_residual_kj_mol': r['max_abs_residual_kj_mol'],
         'recovered_fraction': r['recovered_fraction'],
         'mean_Vxc_kj_mol': (float(np.nanmean(r['vxc'])) * HA_TO_KJ
                             if np.any(np.isfinite(r['vxc'])) else float('nan')),
         'mean_Einter_kj_mol': (float(np.nanmean(r['einter'])) * HA_TO_KJ
                                if np.any(np.isfinite(r['einter'])) else float('nan'))}
        for r in group_reports])
    df_recovery.to_csv('REG_Multi_recovery.csv', sep=',', index=False)

    df_accounting = pd.DataFrame([
        {'group': r['label'], 'kind': r['kind'], 'pairs': r['n_pairs'],
         'admitted': r['n_admitted'], 'rejected': r['n_rejected'],
         'span_exact_kj_mol': r['span_exact_kj_mol'],
         'span_unresolved_kj_mol': r['span_unresolved_kj_mol'],
         'span_residual_kj_mol': r['span_residual_kj_mol'],
         'unresolved_fraction': r['unresolved_fraction'],
         'residual_fraction': r['residual_fraction']}
        for r in group_reports])
    df_accounting.to_csv('excluded_accounting.csv', sep=',', index=False)

    # The coefficients themselves, always: these are what a discussion of "the
    # charge, dipole and quadrupole of this atom" quotes, and they were read from
    # the .int files anyway.
    component_labels_atomic = mp.component_labels(l_max)
    atomic_rows, atomic_index = [], []
    for a_i, atom in enumerate(atoms):
        for c_i, component in enumerate(component_labels_atomic):
            atomic_index.append(str(atom) + ' ' + component)
            atomic_rows.append(moments[:, a_i, c_i])
        atomic_index.append(str(atom) + ' beta_sphere_radius (bohr)')
        atomic_rows.append(beta_radii[:, a_i])
    df_atomic_moments = pd.DataFrame(data=atomic_rows, index=atomic_index,
                                     columns=cc).rename_axis('TERM')
    df_atomic_moments.to_csv('atomic_moments.csv', sep=',')

    df_fragment_terms = None
    df_fragment_moments = None
    df_cancellation = None
    df_comparison = None
    if fragment_view:
        component_labels = mp.component_labels(l_max)
        rows, index = [], []
        for f_i, name in enumerate(fragment_view['fragment_names']):
            for c_i, component in enumerate(component_labels):
                index.append(name + ' ' + component)
                rows.append(fragment_view['moments'][:, f_i, c_i])
            for axis, axis_name in enumerate('xyz'):
                index.append(name + ' centre_' + axis_name + ' (Ang)')
                rows.append(fragment_view['centres'][:, f_i, axis])
            index.append(name + ' extent (Ang)')
            rows.append(fragment_view['extents'][:, f_i])
        df_fragment_moments = pd.DataFrame(data=rows, index=index, columns=cc).rename_axis('TERM')
        df_fragment_moments.to_csv('fragment_moments.csv', sep=',')

        cancellation_rows = []
        for name, entries in fragment_view['cancellation'].items():
            for row in entries:
                cancellation_rows.append({
                    'fragment': name, 'l': row['l'],
                    'fragment_rank_magnitude_mean': float(np.nanmean(row['fragment'])),
                    'atomic_rank_magnitude_sum_mean': float(np.nanmean(row['atomic_sum'])),
                    'ratio_mean': (float(np.nanmean(row['ratio']))
                                   if np.any(np.isfinite(row['ratio'])) else float('nan')),
                    'ratio_min': (float(np.nanmin(row['ratio']))
                                  if np.any(np.isfinite(row['ratio'])) else float('nan')),
                    'ratio_max': (float(np.nanmax(row['ratio']))
                                  if np.any(np.isfinite(row['ratio'])) else float('nan')),
                })
        df_cancellation = pd.DataFrame(cancellation_rows)
        df_cancellation.to_csv('fragment_moment_cancellation.csv', sep=',', index=False)

        if frag_values:
            df_fragment_terms = pd.DataFrame(data=_table(frag_values, n_steps),
                                             index=frag_headers, columns=cc).rename_axis('TERM')
            df_fragment_terms.to_csv('fragment_moment_terms.csv', sep=',')

        # The comparison the two views exist for: the same l_tot shell, seen as a
        # sum of atom-centred terms and as one fragment-centred term.  They are
        # different quantities and the columns say so; putting them side by side
        # is what shows a net interaction being weak because the atomic
        # contributions cancel, rather than because nothing is happening.
        atom_shell_index = {h: i for i, h in enumerate(shell_headers)}
        frag_shell_index = {h: i for i, h in enumerate(frag_shell_headers)}
        comparison_rows = []
        for entry in fragment_view['pairs']:
            label = entry['label']
            for l_tot in range(2 * l_max + 1):
                atom_key = shell_header(label, l_tot)
                frag_key = shell_header(label, l_tot, prefix='frag: ')
                if atom_key not in atom_shell_index:
                    continue
                atom_curve = np.asarray(shell_values[atom_shell_index[atom_key]], dtype=float)
                row = {
                    'group': label, 'l_tot': l_tot,
                    # Says why the fragment-centred columns are empty when they
                    # are, so a blank reads as "that expansion was not admitted"
                    # rather than as a gap in the file.
                    'fragment_centred_status': ('admitted' if entry['admitted']
                                                else 'rejected: ' + str(entry['rejected_by'])),
                    'atom_centred_mean_kj_mol': float(np.nanmean(atom_curve)) * HA_TO_KJ,
                    'atom_centred_span_kj_mol': _span(atom_curve) * HA_TO_KJ,
                }
                if frag_key in frag_shell_index:
                    frag_curve = np.asarray(frag_shell_values[frag_shell_index[frag_key]],
                                            dtype=float)
                    row['fragment_centred_mean_kj_mol'] = float(np.nanmean(frag_curve)) * HA_TO_KJ
                    row['fragment_centred_span_kj_mol'] = _span(frag_curve) * HA_TO_KJ
                else:
                    row['fragment_centred_mean_kj_mol'] = float('nan')
                    row['fragment_centred_span_kj_mol'] = float('nan')
                for seg_i in range(n_segments):
                    row['atom_REG_seg' + str(seg_i + 1)] = (
                        reg_shells[0][seg_i][atom_shell_index[atom_key]]
                        if reg_shells is not None else float('nan'))
                    row['fragment_REG_seg' + str(seg_i + 1)] = (
                        reg_frag_shells[0][seg_i][frag_shell_index[frag_key]]
                        if (reg_frag_shells is not None and frag_key in frag_shell_index)
                        else float('nan'))
                comparison_rows.append(row)
        # Only meaningful when both views exist; without --fragment-moments there is
        # nothing to compare the atom-centred terms against, so no empty file.
        df_comparison = pd.DataFrame(comparison_rows) if comparison_rows else None
        if df_comparison is not None:
            df_comparison.to_csv('fragment_moment_comparison.csv', sep=',', index=False)

    df_final = pd.DataFrame()
    if WRITE:
        writer = pd.ExcelWriter(path=os.path.join(results_dir, 'REG_Multi.xlsx'),
                                engine='xlsxwriter')
        used_sheets = set()
        _skipped_sheets = []
        # xlsxwriter only writes the file when the writer is closed, so anything
        # that escapes this block would leave a 0-byte REG_Multi.xlsx behind. The
        # workbook is closed either way and the failure is reported; every sheet
        # has a CSV beside it, so nothing is lost with it.
        try:
            _to_excel_safe(df_admission, writer, 'admission', used_sheets, index=False,
                           skipped=_skipped_sheets)
            _to_excel_safe(df_accounting, writer, 'accounting', used_sheets, index=False,
                           skipped=_skipped_sheets)
            _to_excel_safe(df_energy_recovery, writer, 'energy_recovery', used_sheets,
                           skipped=_skipped_sheets)
            _to_excel_safe(df_energy_by_type, writer, 'energy_by_type', used_sheets,
                           skipped=_skipped_sheets)
            _to_excel_safe(df_energy_by_pair, writer, 'energy_by_pair', used_sheets,
                           skipped=_skipped_sheets)
            _to_excel_safe(df_energy, writer, 'energy_balance', used_sheets,
                           skipped=_skipped_sheets)
            _to_excel_safe(df_recovery, writer, 'recovery', used_sheets, index=False,
                           skipped=_skipped_sheets)
            _to_excel_safe(df_atomic_moments, writer, 'atomic_moments', used_sheets,
                           skipped=_skipped_sheets)
            _to_excel_safe(df_terms, writer, 'rank_terms', used_sheets, skipped=_skipped_sheets)
            _to_excel_safe(df_shells, writer, 'shell_terms', used_sheets, skipped=_skipped_sheets)
            if df_fragment_moments is not None:
                _to_excel_safe(df_fragment_moments, writer, 'fragment_moments', used_sheets,
                               skipped=_skipped_sheets)
                _to_excel_safe(df_cancellation, writer, 'fragment_cancellation', used_sheets,
                               index=False, skipped=_skipped_sheets)
                if df_comparison is not None:
                    _to_excel_safe(df_comparison, writer, 'atom_vs_fragment', used_sheets,
                                   index=False, skipped=_skipped_sheets)
            if df_fragment_terms is not None:
                _to_excel_safe(df_fragment_terms, writer, 'fragment_rank_terms', used_sheets,
                               skipped=_skipped_sheets)

            for seg_i in range(n_segments if reg_terms is not None else 0):
                df_seg = rv.create_term_dataframe(reg_terms, headers, seg_i)
                df_seg = df_seg.sort_values('REG').reset_index(drop=True)
                df_seg.to_csv('REG_Multi_seg_' + str(seg_i + 1) + '.csv', sep=',')
                _to_excel_safe(df_seg, writer, 'REG_seg_' + str(seg_i + 1), used_sheets,
                               skipped=_skipped_sheets)
                df_final = pd.concat([df_final.reset_index(drop=True),
                                      rv.select_significant_terms(df_seg, MIN_TABLE_ROWS,
                                                                  MAX_TABLE_ROWS, R_THRESHOLD)], axis=1)

                df_shell_seg = rv.create_term_dataframe(reg_shells, shell_headers, seg_i)
                df_shell_seg = df_shell_seg.sort_values('REG').reset_index(drop=True)
                _to_excel_safe(df_shell_seg, writer, 'REG_shells_seg_' + str(seg_i + 1),
                               used_sheets, skipped=_skipped_sheets)

                if reg_frag is not None:
                    df_frag_seg = rv.create_term_dataframe(reg_frag, frag_headers, seg_i)
                    df_frag_seg = df_frag_seg.sort_values('REG').reset_index(drop=True)
                    df_frag_seg.to_csv('REG_Multi_fragment_seg_' + str(seg_i + 1) + '.csv', sep=',')
                    _to_excel_safe(df_frag_seg, writer, 'REG_frag_seg_' + str(seg_i + 1),
                                   used_sheets, skipped=_skipped_sheets)

                if reg_pairs is not None:
                    df_pair_seg = rv.create_term_dataframe(reg_pairs, pair_headers, seg_i)
                    df_pair_seg = df_pair_seg.sort_values('REG').reset_index(drop=True)
                    _to_excel_safe(df_pair_seg, writer, 'REG_pairs_seg_' + str(seg_i + 1),
                                   used_sheets, skipped=_skipped_sheets)

            df_final.to_csv('REG_Multi_final_analysis.csv', sep=',')
            _to_excel_safe(df_final, writer, 'REG_final', used_sheets, skipped=_skipped_sheets)
            rv.pandas_REG_dataframe_to_table(df_final, 'REG_Multi_final_table', SAVE_FIG=SAVE_FIG)
            # Always close: xlsxwriter only writes the file on close, so an exception
            # anywhere above would otherwise leave a 0-byte workbook behind.
        except Exception as excel_error:
            print('WARNING: REG_Multi.xlsx could not be completed — '
                  + str(excel_error))
            print('  Every sheet it would have held is written as a CSV beside it.')
        finally:
            writer.close()
            if _skipped_sheets:
                print('  Some tables were too large for a worksheet and were left out of '
                      'REG_Multi.xlsx; each one is complete in its CSV:')
                for name, rows, cols in _skipped_sheets:
                    print('    {n:<24s} {r} rows x {c} columns'.format(n=name, r=rows, c=cols))

    ###########################################################################
    #                        MODEL-TRANSFER JSON BUNDLE                       #
    ###########################################################################

    bundle = {
        'schema_version': 1,
        'analysis_kind': 'REG_Multi',
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'source_directory': cwd,
        'units': {
            'energy': 'hartree',
            'energy_to_kj_mol': HA_TO_KJ,
            'distance': 'angstrom',
            'note': 'every value is in hartree unless the key says kj_mol',
        },
        'settings': dict(settings, scope=scope, within=within, config=option.config,
                         ignore_fragments=ignore_fragments,
                         control_coordinate_type=CONTROL_COORDINATE_TYPE or 'folder-index',
                         auto_critical_points=bool(AUTO), turning_points=list(turning_points),
                         l_available=l_available),
        'definition': {
            'expansion': ('V_cl(A,B) = sum_{la,ka} sum_{lb,kb} Q[la,ka](A) T[la,ka;lb,kb](R) '
                          'Q[lb,kb](B), nucleus-centred, moments as AIMAll prints them '
                          '(Condon-Shortley phase included, normalisation excluded), with Q[0,0] '
                          'replaced by the atom net charge'),
            'rank_terms': 'V[la,lb] is summed over ka and kb, so each term is rotation-invariant',
            'closure': 'rank terms + V_residual + V_unresolved = exact IQA V_cl over the pairs in scope',
            'l_tot': ('shell terms collect every V[la,lb] with la + lb = n; that shell carries '
                      'the R^-(n+1) distance dependence'),
            'term_labels': ('terms are named by the multipoles they couple, with the rank indices '
                            'in brackets: "dipole-quadrupole [1,2]" is rank 1 on the first named '
                            'group against rank 2 on the second'),
            'moment_convention': ('real spherical-harmonic moments as AIMAll prints them '
                                  '(Condon-Shortley phase included, sqrt((2l+1)/4pi) excluded); '
                                  'k = +|m| is the cosine component and k = -|m| the sine one, so '
                                  'k = +1, -1, 0 are x, y, z'),
            'vxc_note': ('V_xc and E_inter are carried for context only. The multipole series '
                         'describes V_cl; V_xc is what it does not describe'),
        },
        'control_coordinate_label': X_LABEL,
        'control_coordinates': _json_ready(cc),
        'steps': [{'step': reg_folders[i], 'folder': reg_root_list[i],
                   'control_coordinate': float(cc[i]),
                   'wfn_energy': float(total_energy_wfn[i])} for i in range(n_steps)],
        'atoms': atoms,
        'elements': [mp.element_of(a) for a in atoms],
        'fragments': {
            'source': None if ignore_fragments else option.config,
            'ignored': ignore_fragments,
            'ignored_note': ('fragment definitions were present but deliberately not used; pair '
                             'selection was left to the admission gates'
                             if ignore_fragments else None),
            'definitions': ([{'name': frag_names[f], 'atom_numbers': frag_atom_lists[f],
                              'atom_labels': [atoms[a - 1] for a in frag_atom_lists[f]
                                              if 0 < a <= n_atoms]}
                             for f in range(len(frag_atom_lists))] if frag_atom_lists else None),
            'shared_with': 'auto_reg.py (REG-IQF)',
        },
        'groups': [{'label': r['label'], 'kind': r['kind'], 'n_pairs': r['n_pairs'],
                    'n_admitted': r['n_admitted'], 'n_rejected': r['n_rejected'],
                    'exact_vcl': _json_ready(r['exact_total']),
                    'multipolar_total': _json_ready(r['multipolar_total']),
                    'residual': _json_ready(r['residual']),
                    'unresolved': _json_ready(r['unresolved']),
                    'vxc': _json_ready(r['vxc']),
                    'einter': _json_ready(r['einter']),
                    'mean_abs_vcl_kj_mol': r['mean_abs_exact_kj_mol'],
                    'mean_abs_residual_kj_mol': r['mean_abs_residual_kj_mol'],
                    'recovered_fraction': r['recovered_fraction'],
                    'span_exact_kj_mol': r['span_exact_kj_mol'],
                    'span_unresolved_kj_mol': r['span_unresolved_kj_mol'],
                    'unresolved_fraction': r['unresolved_fraction']}
                   for r in group_reports],
        'accounting': _json_ready(accounting),
        'terms': {
            'rank': {'headers': headers, 'values': _json_ready(term_values),
                     'layout': 'row = term, column = geometry point'},
            'shell': {'headers': shell_headers, 'values': _json_ready(shell_values)},
            'fragment_rank': ({'headers': frag_headers, 'values': _json_ready(frag_values)}
                              if frag_values else None),
            'fragment_shell': ({'headers': frag_shell_headers,
                                'values': _json_ready(frag_shell_values)}
                               if frag_shell_values else None),
        },
        'reg': {
            'rank': ({'headers': headers, 'reg': _json_ready(reg_terms[0]),
                      'pearson': _json_ready(reg_terms[1])} if reg_terms is not None else None),
            'shell': ({'headers': shell_headers, 'reg': _json_ready(reg_shells[0]),
                       'pearson': _json_ready(reg_shells[1])} if reg_shells is not None else None),
            'pairs': ({'headers': pair_headers, 'reg': _json_ready(reg_pairs[0]),
                       'pearson': _json_ready(reg_pairs[1])} if reg_pairs is not None else None),
            'fragment_rank': ({'headers': frag_headers, 'reg': _json_ready(reg_frag[0]),
                               'pearson': _json_ready(reg_frag[1])}
                              if reg_frag is not None else None),
            'fragment_shell': ({'headers': frag_shell_headers,
                                'reg': _json_ready(reg_frag_shells[0]),
                                'pearson': _json_ready(reg_frag_shells[1])}
                               if reg_frag_shells is not None else None),
        },
        'segments': {'critical_point_indices': [int(i) for i in sorted(set(critical_points))],
                     'step_ranges': [[int(a), int(b)] for a, b in segments],
                     'note': ('inclusive at both ends; consecutive segments share their '
                              'critical point')},
        'admission': {
            'pairs': [{'atoms': [atoms[v['pair'][0]], atoms[v['pair'][1]]],
                       'indices': [int(v['pair'][0]), int(v['pair'][1])],
                       'separation': v['separation'], 'admitted': bool(v['admitted']),
                       'rejected_by': v['rejected_by'], 'note': v['note'],
                       'min_distance_ang': v['min_distance_ang'],
                       'max_abs_residual_kj_mol': v.get('max_abs_residual_kj_mol'),
                       'segments': (v['convergence']['segments'] if 'convergence' in v else None)}
                      for v in verdicts],
            'radii_source': admission_info['radii_source'],
            'radii_are_lower_bound': admission_info['radii_are_lower_bound'],
            'bonds': [[int(a), int(b)] for a, b in admission_info['bonds']],
            'convergence_gate_applied': not settings['skip_convergence'],
        },
        'moments': {
            'l_max': l_max,
            'component_labels': mp.component_labels(l_max),
            'values': _json_ready(moments),
            'layout': 'values[step][atom][component]',
            'beta_sphere_radii_bohr': _json_ready(beta_radii),
        },
        'geometry': {'coordinates_ang': _json_ready(coords),
                     'layout': 'coordinates[step][atom][xyz]'},
        'fragment_view': ({
            'definition': ('every atom moments translated onto one centre per fragment and '
                           'summed, so a fragment pair gets a single expansion. A different '
                           'decomposition of the same energy from the atom-centred terms, never '
                           'comparable with them term by term'),
            'centre_scheme': fragment_view['centre_scheme'],
            'fragment_names': fragment_view['fragment_names'],
            'centres_ang': _json_ready(fragment_view['centres']),
            'extents_ang': _json_ready(fragment_view['extents']),
            'moments': _json_ready(fragment_view['moments']),
            'component_labels': mp.component_labels(l_max),
            'moments_layout': 'moments[step][fragment][component]',
            'cancellation': {
                name: [{'l': row['l'],
                        'fragment_rank_magnitude': _json_ready(row['fragment']),
                        'atomic_rank_magnitude_sum': _json_ready(row['atomic_sum']),
                        'ratio': _json_ready(row['ratio'])} for row in rows]
                for name, rows in fragment_view['cancellation'].items()},
            'cancellation_note': ('sqrt(sum_k Q[l,k]^2) is rotation-invariant; the ratio is the '
                                  'fragment value over the sum of its atoms, so a small ratio '
                                  'means the atomic moments of that rank cancel'),
            'pairs': [{'label': entry['label'],
                       'admitted': bool(entry['admitted']),
                       'rejected_by': entry['rejected_by'],
                       'note': entry['note'],
                       'separation_ang': _json_ready(entry['separation_ang']),
                       'extent_margin_ang': _json_ready(entry['extent_margin_ang']),
                       'exact_vcl': _json_ready(entry['exact']),
                       'multipolar_total': _json_ready(entry['total']),
                       'residual': _json_ready(entry['residual']),
                       'max_abs_residual_kj_mol': entry['convergence']['max_abs_residual_kj_mol']}
                      for entry in fragment_view['pairs']],
        } if fragment_view else None),
    }

    bundle_path = os.path.join(results_dir, os.path.basename(results_dir).replace(
        '_results', '') + '_model_transfer.json')
    indent = option.bundle_indent if option.bundle_indent and option.bundle_indent > 0 else None
    with open(bundle_path, 'w', encoding='utf-8') as handle:
        json.dump(_json_ready(bundle), handle, indent=indent,
                  separators=(',', ':') if indent is None else None, allow_nan=False)
    print('  Model-transfer bundle written to: {p}  ({s:.1f} MB)'.format(
        p=bundle_path, s=os.path.getsize(bundle_path) / (1024.0 * 1024.0)))

    os.chdir(cwd)
    print('--- Total time for REG_Multi analysis: {s:.3f} minutes ---'
          .format(s=(time.time() - start_time) / 60))


if __name__ == '__main__':
    main()
