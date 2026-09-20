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
        n_flagged=5, expected_charge=0.0,
        direct_multipoles=None):
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

    _direct_multipole_report(
        q_vals, direct_multipoles,
        cc, reg_folders, results_dir, save_fig,
        atoms=atoms, iqa_atom_total=iqa_atom_total,
        total_energy_wfn=total_energy_wfn)

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
# 3.  Direct charge / dipole / quadrupole visualization
# ─────────────────────────────────────────────────────────────────────────────


def _aggregate_component(component_data):
    data = np.asarray(component_data, dtype=float)
    if data.size == 0:
        return np.array([], dtype=float)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return np.nansum(data, axis=0)


def _direct_multipole_report(q_vals, direct_multipoles, cc, reg_folders,
                             results_dir, save_fig, atoms=None,
                             iqa_atom_total=None, total_energy_wfn=None):
    if q_vals is None and direct_multipoles is None:
        return

    n_steps = len(reg_folders)

    q_vals_arr = np.asarray(q_vals, dtype=float) if q_vals is not None else None
    if q_vals_arr is not None and q_vals_arr.ndim == 1:
        q_vals_arr = q_vals_arr[np.newaxis, :]

    charge_sum = np.nansum(q_vals_arr, axis=0) if q_vals_arr is not None else np.zeros(n_steps)

    if direct_multipoles:
        dipole = {
            'x': _aggregate_component(direct_multipoles.get('mu_x')),
            'y': _aggregate_component(direct_multipoles.get('mu_y')),
            'z': _aggregate_component(direct_multipoles.get('mu_z')),
        }
        dipole_mag = np.linalg.norm(
            np.stack([dipole.get('x', np.zeros(n_steps)),
                      dipole.get('y', np.zeros(n_steps)),
                      dipole.get('z', np.zeros(n_steps))], axis=1),
            axis=1,
        ) if any(v.size for v in dipole.values()) else np.zeros(n_steps)

        quadrupole = {
            'Qxx': _aggregate_component(direct_multipoles.get('q_xx')),
            'Qxy': _aggregate_component(direct_multipoles.get('q_xy')),
            'Qxz': _aggregate_component(direct_multipoles.get('q_xz')),
            'Qyy': _aggregate_component(direct_multipoles.get('q_yy')),
            'Qyz': _aggregate_component(direct_multipoles.get('q_yz')),
            'Qzz': _aggregate_component(direct_multipoles.get('q_zz')),
        }
    else:
        dipole = {'x': np.zeros(n_steps), 'y': np.zeros(n_steps), 'z': np.zeros(n_steps)}
        dipole_mag = np.zeros(n_steps)
        quadrupole = {
            'Qxx': np.zeros(n_steps), 'Qxy': np.zeros(n_steps), 'Qxz': np.zeros(n_steps),
            'Qyy': np.zeros(n_steps), 'Qyz': np.zeros(n_steps), 'Qzz': np.zeros(n_steps)
        }

    sep = '=' * 90
    lines = [
        sep,
        '  EXTENDED IQA DIAGNOSTICS — DIRECT CHARGE / DIPOLE / QUADRUPOLE ANALYSIS',
        sep,
        '',
        '  Per-step molecular quantities reconstructed directly from AIMAll .sum tables.',
        '  Charge: Σ q(A)    Dipole: Σ Mu_X(A), Σ Mu_Y(A), Σ Mu_Z(A)    '
        'Quadrupole: Σ Q_XX(A), ...\n',
        '  {:<25s}  {:>10s}  {:>12s}  {:>12s}  {:>12s}  {:>12s}  {:>12s}  {:>12s}  {:>12s}  {:>12s}  {:>12s}  {:>12s}  {:>12s}'.format(
            'Step', 'CC', 'Σq(A)', 'μx', 'μy', 'μz', '|μ|', 'Qxx', 'Qxy', 'Qxz', 'Qyy', 'Qyz', 'Qzz'),
        '  ' + '-' * 145,
    ]

    for si in range(n_steps):
        lines.append(
            '  {:<25s}  {:>10.4f}  {:>12.6f}  {:>12.6f}  {:>12.6f}  {:>12.6f}  {:>12.6f}  {:>12.6f}  {:>12.6f}  {:>12.6f}  {:>12.6f}  {:>12.6f}  {:>12.6f}'.format(
                reg_folders[si], float(cc[si]),
                charge_sum[si],
                dipole.get('x', np.zeros(n_steps))[si],
                dipole.get('y', np.zeros(n_steps))[si],
                dipole.get('z', np.zeros(n_steps))[si],
                dipole_mag[si],
                quadrupole.get('Qxx', np.zeros(n_steps))[si],
                quadrupole.get('Qxy', np.zeros(n_steps))[si],
                quadrupole.get('Qxz', np.zeros(n_steps))[si],
                quadrupole.get('Qyy', np.zeros(n_steps))[si],
                quadrupole.get('Qyz', np.zeros(n_steps))[si],
                quadrupole.get('Qzz', np.zeros(n_steps))[si],
            )
        )

    lines.append('')

    if q_vals_arr is not None and atoms is not None:
        lines.append('  PER-STEP PER-ATOM CHARGE DISTRIBUTION [e]')
        lines.append('  ' + '-' * 90)
        header = '  {:<25s}  {:>10s}'.format('Step', 'CC')
        header += ''.join('  {:>12s}'.format(atom.upper()) for atom in atoms)
        lines.append(header)
        lines.append('  ' + '-' * len(header))
        for si in range(n_steps):
            row = '  {:<25s}  {:>10.4f}'.format(reg_folders[si], float(cc[si]))
            for ai in range(q_vals_arr.shape[0]):
                val = q_vals_arr[ai, si]
                if np.isnan(val):
                    row += '  {:>12s}'.format('N/A')
                else:
                    row += '  {:>12.6f}'.format(val)
            lines.append(row)
        lines.append('')

    if iqa_atom_total is not None or total_energy_wfn is not None:
        lines.append('  PER-STEP ENERGY SUMMARY')
        lines.append('  ' + '-' * 90)
        lines.append('  {:<25s}  {:>10s}  {:>14s}  {:>14s}  {:>14s}  {:>14s}'.format(
            'Step', 'CC', 'E_WFN', 'ΣE_IQA(A)', 'ΔE_WFN', 'ΔE_IQA'))
        lines.append('  ' + '-' * 90)

        wfn_energy = np.asarray(total_energy_wfn, dtype=float) if total_energy_wfn is not None else np.full(n_steps, np.nan)
        iqa_energy_sum = np.nansum(np.asarray(iqa_atom_total, dtype=float), axis=0) if iqa_atom_total is not None else np.full(n_steps, np.nan)

        ref_iqa = iqa_energy_sum[0] if np.isfinite(iqa_energy_sum[0]) else np.nan
        ref_wfn = wfn_energy[0] if np.isfinite(wfn_energy[0]) else np.nan
        for si in range(n_steps):
            dE_wfn = wfn_energy[si] - ref_wfn if np.isfinite(wfn_energy[si]) and np.isfinite(ref_wfn) else np.nan
            dE_iqa = iqa_energy_sum[si] - ref_iqa if np.isfinite(iqa_energy_sum[si]) and np.isfinite(ref_iqa) else np.nan
            lines.append(
                '  {:<25s}  {:>10.4f}  {:>14.6f}  {:>14.6f}  {:>14.6f}  {:>14.6f}'.format(
                    reg_folders[si], float(cc[si]),
                    wfn_energy[si],
                    iqa_energy_sum[si],
                    dE_wfn,
                    dE_iqa,
                )
            )
        lines.append('')

    text = '\n'.join(lines)
    print(text)
    out_path = os.path.join(results_dir, 'diagnostics_multipoles.txt')
    with open(out_path, 'w') as f:
        f.write(text + '\n')

    if save_fig:
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        ax = axes[0]
        if q_vals_arr is not None and atoms is not None:
            for ai, atom in enumerate(atoms):
                ax.plot(cc, q_vals_arr[ai], label=atom.upper(), linewidth=1.0, alpha=0.45)
            ax.plot(cc, charge_sum, 'k-', linewidth=2.0, label='Σ q(A)')
        else:
            ax.plot(cc, charge_sum, 'k-o', linewidth=1.8)
        ax.axhline(0.0, color='gray', linewidth=0.5)
        ax.set_ylabel('q(A) [e]')
        ax.set_title('Direct charge / dipole / quadrupole diagnostics')
        if q_vals_arr is not None and atoms is not None:
            ax.legend(loc='best', fontsize=8, ncol=min(4, max(1, len(atoms))))

        ax = axes[1]
        ax.plot(cc, dipole.get('x', np.zeros_like(charge_sum)), label='μx', linewidth=1.8)
        ax.plot(cc, dipole.get('y', np.zeros_like(charge_sum)), label='μy', linewidth=1.8)
        ax.plot(cc, dipole.get('z', np.zeros_like(charge_sum)), label='μz', linewidth=1.8)
        ax.plot(cc, dipole_mag, 'k--', linewidth=2.0, label='|μ|')
        ax.axhline(0.0, color='gray', linewidth=0.5)
        ax.set_ylabel('Dipole components')
        ax.legend(loc='best', fontsize=8)

        ax = axes[2]
        quad_keys = ['Qxx', 'Qxy', 'Qxz', 'Qyy', 'Qyz', 'Qzz']
        styles = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
        for key, style in zip(quad_keys, styles):
            ax.plot(cc, quadrupole.get(key, np.zeros_like(charge_sum)), label=key, color=style, linewidth=1.8)
        ax.axhline(0.0, color='gray', linewidth=0.5)
        ax.set_ylabel('Quadrupole tensor')
        ax.set_xlabel('Control Coordinate')
        ax.legend(loc='best', fontsize=8, ncol=3)

        fig.tight_layout()

        out_path = os.path.join(results_dir, 'diagnostics_multipoles.png')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print('  Direct multipole plot saved to: ' + out_path)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Per-atom ΔE_IQA(A) plot
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
