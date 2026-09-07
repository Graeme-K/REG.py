"""
iqa_diagnostics.py
Extended IQA integration quality diagnostics, reported separately from the
|L(A)| check in auto_reg.py.

Three tests are run:
  1. |L(A)| / |T(A)|  —  normalised integration error ranked per atom.
  2. Electron-population conservation  —  Σ q(A) per geometry step should
     equal the molecular charge (0 for neutral).
  3. Per-atom ΔE_IQA(A) plot  —  shows which atoms drive shape deviations
     between the IQA sum and the WFN energy curve.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ── thresholds for the normalised ratio ──────────────────────────────────────
_LT_THRESH_H     = 1e-3   # hydrogen
_LT_THRESH_HEAVY = 1e-2   # all other atoms
_HA_TO_KJ        = 2625.5


def run(lagrangians, T_vals, q_vals,
        iqa_atom_total, atoms, cc,
        total_energy_wfn, reg_folders,
        results_dir, save_fig=True,
        n_flagged=5, expected_charge=0.0):
    """Run all three extended diagnostics and write outputs to *results_dir*.

    Parameters
    ----------
    lagrangians     : list[dict]  — one dict per step, atom → L(A) or None.
    T_vals          : ndarray (n_atoms, n_steps)  — T(A) kinetic energies.
    q_vals          : ndarray (n_atoms, n_steps)  — q(A) net charges.
    iqa_atom_total  : ndarray (n_atoms, n_steps)  — E_IQA(A) per atom.
    atoms           : list[str]   — atom labels (lower-case, e.g. 'c1').
    cc              : ndarray     — control-coordinate values.
    total_energy_wfn: ndarray     — WFN total energy per step.
    reg_folders     : list[str]   — geometry-step folder names.
    results_dir     : str         — directory where outputs are written.
    save_fig        : bool        — save the ΔE plot to a PNG file.
    n_flagged       : int         — max flagged atoms shown in per-step table.
    expected_charge : float       — expected molecular net charge (default 0).

    Returns
    -------
    flagged_atoms : list[str]  — atoms that exceeded the |L/T| threshold.
    """
    n_atoms = len(atoms)
    n_steps = len(reg_folders)

    flagged_atoms = _lt_ratio_report(
        lagrangians, T_vals, atoms, cc, reg_folders,
        n_flagged, n_steps, results_dir)

    _electron_count_report(
        q_vals, atoms, cc, reg_folders,
        expected_charge, n_steps, results_dir)

    _delta_E_plot(
        iqa_atom_total, atoms, cc,
        total_energy_wfn, flagged_atoms,
        n_atoms, n_steps, results_dir, save_fig)

    return flagged_atoms


# ─────────────────────────────────────────────────────────────────────────────
# 1.  |L(A)| / |T(A)|
# ─────────────────────────────────────────────────────────────────────────────

def _lt_ratio_report(lagrangians, T_vals, atoms, cc, reg_folders,
                     n_flagged, n_steps, results_dir):
    n_atoms = len(atoms)
    LT = np.full((n_atoms, n_steps), np.nan)

    for si in range(n_steps):
        folder_L = lagrangians[si]
        for ai, atom in enumerate(atoms):
            L = folder_L.get(atom)
            T = T_vals[ai, si] if T_vals.ndim == 2 else T_vals[ai][si]
            if L is not None and T is not None and T != 0.0:
                LT[ai, si] = abs(L) / abs(T)

    max_LT   = np.nanmax(LT, axis=1)
    mean_LT  = np.nanmean(LT, axis=1)
    sort_idx = np.argsort(max_LT)[::-1]

    sep   = '=' * 90
    inner = '-' * 90
    lines = [
        sep,
        '  EXTENDED IQA DIAGNOSTICS — |L(A)| / |T(A)|  NORMALISED INTEGRATION ERROR',
        sep,
        '',
        '  Atoms ranked by worst-case |L(A)|/|T(A)| across all geometry points.',
        '  Thresholds:  H atoms < {:.0e}   heavy atoms < {:.0e}'.format(
            _LT_THRESH_H, _LT_THRESH_HEAVY),
        '',
        '  {:<10s}  {:>14s}  {:>14s}  {:>14s}  {}'.format(
            'Atom', 'max |L/T|', 'mean |L/T|', 'max |L|', 'Flag'),
        '  ' + inner,
    ]

    flagged = []
    for ai in sort_idx:
        atom = atoms[ai]
        thr  = _LT_THRESH_H if atom.startswith('h') else _LT_THRESH_HEAVY
        L_abs_vals = [lagrangians[si].get(atom) for si in range(n_steps)]
        L_abs_max  = max((abs(v) for v in L_abs_vals if v is not None),
                         default=np.nan)
        flag = '  ***' if max_LT[ai] > thr else ''
        if flag:
            flagged.append(atom)
        lines.append('  {:<10s}  {:>14.4e}  {:>14.4e}  {:>14.4e}{}'.format(
            atom, max_LT[ai], mean_LT[ai], L_abs_max, flag))

    lines.append('')

    # Per-step breakdown for the worst flagged atoms
    show_atoms = flagged[:n_flagged]
    if show_atoms:
        lines.append('  Per-step |L/T| for flagged atoms:')
        col_w = 14
        hdr = ('  {:<25s}  {:>10s}' +
               ('  {:>' + str(col_w) + 's}') * len(show_atoms)).format(
            'Step', 'CC', *[a.upper() for a in show_atoms])
        lines.append(hdr)
        lines.append('  ' + inner)
        for si in range(n_steps):
            row = '  {:<25s}  {:>10.4f}'.format(reg_folders[si], float(cc[si]))
            for atom in show_atoms:
                ai = atoms.index(atom)
                val = LT[ai, si]
                row += ('  {:>' + str(col_w) + '.4e}').format(val) \
                    if not np.isnan(val) \
                    else ('  {:>' + str(col_w) + 's}').format('N/A')
            lines.append(row)
        lines.append('')

    text = '\n'.join(lines)
    print(text)
    out_path = os.path.join(results_dir, 'diagnostics_LT_ratio.txt')
    with open(out_path, 'w') as f:
        f.write(text + '\n')

    return flagged


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Electron-population (charge) conservation
# ─────────────────────────────────────────────────────────────────────────────

def _electron_count_report(q_vals, atoms, cc, reg_folders,
                            expected_charge, n_steps, results_dir):
    # Σ q(A) per step — should equal expected_charge for a well-integrated density.
    q_sum = np.nansum(q_vals, axis=0)          # shape (n_steps,)
    deviation = q_sum - expected_charge

    sep   = '=' * 90
    inner = '-' * 90
    lines = [
        sep,
        '  EXTENDED IQA DIAGNOSTICS — ELECTRON POPULATION CONSERVATION',
        sep,
        '',
        '  Σ q(A) over all atoms should equal the molecular charge '
        '({:+.1f} e).'.format(expected_charge),
        '  Deviations indicate electrons lost or gained in basin integration.',
        '',
        '  {:<25s}  {:>10s}  {:>14s}  {:>16s}  {}'.format(
            'Step', 'CC', 'Σ q(A) [e]', 'Deviation [e]', 'Flag'),
        '  ' + inner,
    ]

    _CHARGE_THRESH = 0.01  # electrons
    for si in range(n_steps):
        flag = '  ***' if abs(deviation[si]) > _CHARGE_THRESH else ''
        lines.append('  {:<25s}  {:>10.4f}  {:>14.6f}  {:>+16.6f}{}'.format(
            reg_folders[si], float(cc[si]),
            q_sum[si], deviation[si], flag))

    lines += [
        '  ' + inner,
        '  Max |deviation|: {:+.6f} e'.format(float(np.max(np.abs(deviation)))),
        '',
    ]

    text = '\n'.join(lines)
    print(text)
    out_path = os.path.join(results_dir, 'diagnostics_charge_conservation.txt')
    with open(out_path, 'w') as f:
        f.write(text + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Per-atom ΔE_IQA(A) plot
# ─────────────────────────────────────────────────────────────────────────────

def _delta_E_plot(iqa_atom_total, atoms, cc,
                  total_energy_wfn, flagged_atoms,
                  n_atoms, n_steps, results_dir, save_fig):
    ref = 0   # relative to first geometry point

    dE_wfn  = (total_energy_wfn - total_energy_wfn[ref]) * _HA_TO_KJ
    dE_iqa_total = (np.sum(iqa_atom_total, axis=0) -
                    np.sum(iqa_atom_total[:, ref])) * _HA_TO_KJ
    error_kj = (np.sum(iqa_atom_total, axis=0) - total_energy_wfn) * _HA_TO_KJ

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    cmap = plt.get_cmap('tab20')
    for ai, atom in enumerate(atoms):
        dE_a = (iqa_atom_total[ai] - iqa_atom_total[ai, ref]) * _HA_TO_KJ
        if atom in flagged_atoms:
            ax1.plot(cc, dE_a, label=atom.upper(),
                     color=cmap(ai % 20), linewidth=2.0, linestyle='-', zorder=3)
        else:
            ax1.plot(cc, dE_a,
                     color=cmap(ai % 20), linewidth=0.7, linestyle='--',
                     alpha=0.5, zorder=2)

    ax1.plot(cc, dE_wfn,      'k-',  linewidth=2.5, label=r'$\Delta E_\mathrm{WFN}$',       zorder=5)
    ax1.plot(cc, dE_iqa_total,'k--', linewidth=1.8, label=r'$\Delta E_\mathrm{IQA}$ (sum)', zorder=5)
    ax1.axhline(0, color='grey', linewidth=0.5, zorder=1)
    ax1.set_ylabel(r'$\Delta E$ [kJ mol$^{-1}$]')
    ax1.set_title(
        r'Per-atom $\Delta E_\mathrm{IQA}(A)$ along control coordinate'
        '\n(solid/labelled = flagged atoms; dashed grey = unflagged)')
    ax1.legend(fontsize=8, ncol=max(1, n_atoms // 5 + 1), loc='best')

    ax2.plot(cc, error_kj, 'r-', linewidth=1.8,
             label=r'$\Sigma E_\mathrm{IQA}(A) - E_\mathrm{WFN}$')
    ax2.axhline(0, color='grey', linewidth=0.5)
    ax2.set_xlabel('Control Coordinate')
    ax2.set_ylabel(r'Recovery error [kJ mol$^{-1}$]')
    ax2.set_title('Total IQA recovery error along control coordinate')
    ax2.legend(fontsize=9)

    plt.tight_layout()

    if save_fig:
        out_path = os.path.join(results_dir, 'diagnostics_delta_E.png')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print('  Per-atom ΔE plot saved to: ' + out_path)

    plt.close(fig)
