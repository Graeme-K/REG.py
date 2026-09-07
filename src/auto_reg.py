"""
auto_reg.py v0.1
F. Falcioni, P. L. A. Popelier

Library with function to run a REG analysis
Check for updates at github.com/FabioFalcioni
For details about the method, please see XXXXXXX

Please, report bugs and issues to fabio.falcioni@manchester.ac.uk
coded by F.Falcioni

NOTE: The automatic analysis works if this file is run with python3 inside a folder containing all the REG points (
saved in numbered folders) """

# IMPORT LIBRARIES
from optparse import OptionParser
usage = "usage: %prog [options] arg"
parser = OptionParser(usage)
parser.add_option("-d", "--directory", action='store', type='string', dest='reg_dir',
                    help="PLEASE INSERT THE PATH OF REG.py folder installation")
parser.add_option("-f", "--IQF", action='store', type='string', dest='IQF_TF',
                    help="Select T or F based on if you want to run IQF or not")
parser.add_option("-z", "--zeros", action='store_true', dest='use_zeros', default=False,
                    help="Continue REG analysis using zeros for missing or poor-quality atoms instead of aborting")

(option, args) = parser.parse_args()


import sys
sys.path.insert(1, option.reg_dir)  # PLEASE INSERT THE PATH OF REG.py folder installation using -d option

import reg
import aimall_utils as aim_u # type: ignore
import numpy as np
import pandas as pd # type: ignore
import reg_vis as rv # type: ignore
import gaussian_utils as gauss_u # type: ignore
import dftd3_utils as disp_u # type: ignore
import re
import os
import time

import default_settings # type: ignore
import iqa_diagnostics # type: ignore

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


def main():

    ### STARTING TIMER ###
    start_time = time.time()
    ##############################    VARIABLES    ##################################

    SYS = 'REG'  # name of the system

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
    n_terms = default_settings.n_terms  # number of terms to rank in figures and tables
    ERR_R_THRESHOLD = default_settings.ERR_R_THRESHOLD  # |R| filter for error REG output

    ###### REG-IQF
    print(option.IQF_TF)
    if option.IQF_TF == 'T':
        IQF = True
    else:
        IQF = False
    if IQF:
        SYS = 'REG_IQF' 
        if os.path.exists('auto_reg.config'):
            with open('auto_reg.config') as f:
                config_file = f.read()
                Frag_names = re.findall(r'FRAG\s*ID\s*\d+\s*<(.*?)>',config_file) # Names of fragments
                Frag_lists = re.findall(r'FRAG\s*ATOMS\s*\[([\d,]+)\]',config_file) # Atoms by number for fragments
                List_of_frags = []
                for f_list_str in Frag_lists:
                    atom_list = f_list_str.split(",")
                    int_atom_list = [int(a) for a in atom_list]
                    List_of_frags.append(int_atom_list)
        else:
            raise FileNotFoundError(f"Error: auto_reg.config file is missing from directory")

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

    # Finding file paths and folders
    wf_file = []
    gau16_file = []
    reg_folders = []
    reg_folder_list = []
    WFX = False
    for root,_,files in os.walk("."):
        for name in files:
            if name.endswith(".wfn"):
                wf_file.append(os.path.join(root,name))
                reg_fold = root.split('/')[-1]
                reg_folders.append(reg_fold)
                reg_folder_list.append(root)
            elif name.endswith(".wfx"):
                wf_file.append(os.path.join(root,name))
                reg_fold = root.split('/')[-1]
                reg_folders.append(reg_fold)
                reg_folder_list.append(root)
                WFX = True
                
            elif (name.endswith(".out") or name.endswith(".log") or name.endswith(".gaussianoutput")) and not (name.startswith("dft-d3") or name.startswith("slurm")):
                gau16_file.append(os.path.join(root,name))

    # Sorting all folders based on REG folders
    def _folder_val(name):
        name = re.sub(r'^neg_?', '-', name)
        name = re.sub(r'(?<=\d)_', '.', name)
        match = re.search(r'-?\d+\.?\d*', name)
        return float(match.group()) if match else float('inf')

    all_files_sorted = sorted(zip(reg_folders,reg_folder_list,wf_file,gau16_file), key=lambda f: _folder_val(f[0]))

    # If folder values look like angles (dihedral range), roll to start after the
    # largest gap — mirrors roll_to_critical_point in setup_rdp.py to avoid
    # splitting a continuous sequence at the -180/180 boundary.
    vals = [_folder_val(t[0]) for t in all_files_sorted]
    if vals and all(-360 <= v <= 360 for v in vals) and (max(vals) - min(vals)) > 90:
        gaps = [vals[i+1] - vals[i] for i in range(len(vals) - 1)]
        wrap_gap = vals[0] + 360 - vals[-1]
        all_gaps = gaps + [wrap_gap]
        max_gap_idx = all_gaps.index(max(all_gaps))
        if max_gap_idx < len(gaps):  # wrap-around gap is not the largest — roll needed
            all_files_sorted = all_files_sorted[max_gap_idx+1:] + all_files_sorted[:max_gap_idx+1]

    reg_folders,reg_root_list,wf_files,g16_out_files = list(zip(*all_files_sorted))
        
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
    try:
        _T_raw,   _, _, _, _ = aim_u.get_iqa_properties(atomic_files, ['T(A)'],     [], atoms)
        _q_raw,   _, _, _, _ = aim_u.get_iqa_properties(atomic_files, ['q(A)'],     [], atoms)
        _E_raw,   _, _, _, _ = aim_u.get_iqa_properties(atomic_files, ['E_IQA(A)'], [], atoms)
        iqa_diagnostics.run(
            lagrangians=lagrangians,
            T_vals=np.array(_T_raw),
            q_vals=np.array(_q_raw),
            iqa_atom_total=np.array(_E_raw),
            atoms=atoms,
            cc=cc,
            total_energy_wfn=total_energy_wfn,
            reg_folders=reg_folders,
            results_dir=cwd + '/' + SYS + '_results',
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

    if IQF:
        iqf_inter, iqf_inter_header, iqf_inter_comps, iqf_inter_comp_head, iqf_intra, iqf_intra_header, iqf_intra_comps, iqf_intra_comp_head = sum_into_fragments(Frag_names,List_of_frags,atoms,True,iqa_inter,inter_prop,[],iqa_intra,intra_prop)
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
    if DISPERSION:
        for i in range(0, len(reg_folders)):
            xyz_file = xyz_files[i]
            disp_u.run_DFT_D3(DFT_D3_PATH, reg_root_list[i], xyz_file, DISP_FUNCTIONAL,BJ_DAMPING)
        folders_disp = [reg_root_list[i] + '/dft-d3.log' for i in range(0, len(reg_folders))]
        # GET INTER-ATOMIC DISPERSION TERMS:
        iqa_disp, iqa_disp_header = disp_u.disp_property_from_dftd3_file(folders_disp, atoms)
        iqa_disp_header = np.array(iqa_disp_header)  # used for reference
        iqa_disp = np.array(iqa_disp)
        # Total D3 must be captured here — the IQF block below absorbs intra-fragment
        # pairs into iqf_intra, so summing iqa_disp afterwards gives inter-fragment only.
        total_energy_dispersion = sum(iqa_disp)
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

    # CALCULATE RECOVERY ERROR
    iqa_for_error = total_energy_iqa + total_energy_dispersion if DISPERSION else total_energy_iqa
    per_step_errors_ha, rmse_kj = reg.integration_error(total_energy_wfn, iqa_for_error)
    per_step_errors_kj = [2625.5 * e for e in per_step_errors_ha]

    # REG AGAINST RECOVERY ERROR — which IQA terms track E_WFN − E_IQA?
    # Segments match the total energy surface so error and energy analyses are directly comparable.
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

    # REG of per-atom E_IQA(A) totals against the recovery error.
    # E_IQA(A) already includes the correct intra + inter correction from the .sum file,
    # so this ranks atoms by how much their total energy tracks the integration gap.
    _atom_err_headers = np.array([str(a) for a in atoms])
    reg_atom_err = reg.reg(_err_ha, cc, _iqa_atom_total[:len(atoms)], np=POINTS, critical=AUTO,
                           inflex=INFLEX, critical_index=turning_points)

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
                list_property_sorted.append(
                    pd.concat([df_property[-n_terms:], df_property[:n_terms]], axis=0).sort_values('REG'))
            for j in range(len(intra_prop)):
                df_property = rv.filter_term_dataframe(df_intra, intra_prop[j], intra_prop_names[j])
                if j == 0:
                    properties_comparison.append(df_property)
                df_property.to_csv(intra_prop_names[j] + "_seg_" + str(i + 1) + ".csv", sep=',')
                df_property.to_excel(writer, sheet_name=_safe_sheet_name(intra_prop_names[j] + "_seg_" + str(i + 1), _used_writer_names))
                list_property_sorted.append(pd.concat([df_property], axis=0).sort_values('REG'))
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
                                                pd.concat([df_disp_new[-n_terms:], df_disp_new[:n_terms]],
                                                            axis=0).sort_values(
                                                    'REG').reset_index(drop=True)], axis=1)
            df_dispersion_sorted.to_csv('REG_' + disp_name_new + '_analysis.csv', sep=',')
            df_dispersion_sorted.to_excel(writer, sheet_name="REG_" + disp_name_new)
            rv.pandas_REG_dataframe_to_table(df_dispersion_sorted, 'REG_' + disp_name_new + '_table', SAVE_FIG=SAVE_FIG)
            pd.DataFrame(data=np.array(iqa_disp), index=iqa_disp_header, columns=cc).rename_axis('TERM').to_excel(energy_writer, sheet_name='dispersion_energies')
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
                                            pd.concat([df_temp[-n_terms:], df_temp[:n_terms]], axis=0).sort_values(
                                                'REG').reset_index(drop=True)], axis=1)
            df_ct_pl_sorted.to_csv('REG_Vct-Vpl_analysis.csv', sep=',')
            df_ct_pl_sorted.to_excel(writer, sheet_name='REG_Vct-Vpl')
            rv.pandas_REG_dataframe_to_table(df_ct_pl_sorted, 'REG_Vct-Vpl_table', SAVE_FIG=SAVE_FIG)
            pd.DataFrame(
                data=np.concatenate((np.array(iqa_polarisation_terms), np.array(iqa_charge_transfer_terms))),
                index=np.concatenate((iqa_polarisation_headers, iqa_charge_transfer_headers)),
                columns=cc).rename_axis('TERM').to_excel(energy_writer, sheet_name='pl_ct_energies')

        # OUTPUT OF ALL INTER AND INTRA TERMS SELECTED BY THE USER
        all_prop_names = inter_prop_names + intra_prop_names
        for i in range(len(inter_prop) + len(intra_prop)):
            df_property_sorted = pd.DataFrame()
            for j in range(len(reg_inter[0])):
                df_property_sorted = pd.concat([df_property_sorted, list_property_final[j][i]], axis=1)
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
                                        pd.concat([df_final[-n_terms:], df_final[:n_terms]], axis=0).sort_values(
                                            'REG').reset_index(drop=True)], axis=1)
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
            df_entity_final_sorted = pd.concat([df_entity_final_sorted.reset_index(drop=True),
                                                pd.concat([df_entity_sorted[-n_terms:], df_entity_sorted[:n_terms]], axis=0)
                                                  .sort_values('REG').reset_index(drop=True)], axis=1)
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
            df_err_final_sorted = pd.concat([
                df_err_final_sorted.reset_index(drop=True),
                pd.concat([df_err_combined[-n_terms:], df_err_combined[:n_terms]], axis=0)
                  .sort_values('REG').reset_index(drop=True)
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

    ###ENDING TIMER ###
    print("--- Total time for REG Analysis: {s} minutes ---".format(s=((time.time() - start_time) / 60)))


if __name__ == "__main__":
    main()

