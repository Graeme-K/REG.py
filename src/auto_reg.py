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
import sys
sys.path.insert(1, '/mnt/iusers01/pp01/w06498gk/1Software/REG/src')  # PLEASE INSERT THE PATH OF REG.py folder installation
import reg
import aimall_utils as aim_u
import numpy as np
import pandas as pd
import reg_vis as rv
import gaussian_utils as gauss_u
import dftd3_utils as disp_u
import re
import os
import time

### STARTING TIMER ###
start_time = time.time()
##############################    VARIABLES    ##################################

SYS = 'REG'  # name of the system

### PES Critical points options ###
POINTS = 2  # number of points for "find_critical" function
AUTO = True  # Search for critical points
turning_points = []  # manually put critical points in the PES if necessary
# NOTE: If analysing only a single segment (i.e. the PES has no critical points) please put AUTO=False and tp=[]

# DEFINE THE DESIRED TERMS:
intra_prop = ['E_IQA_Intra(A)']  # chose the AIMAll intra atomic properties to analyse
intra_prop_names = ['Eintra']  # names of the properties shown in the output
inter_prop = ['VC_IQA(A,B)', 'VX_IQA(A,B)', 'E_IQA_Inter(A,B)']  # chose the AIMAll inter atomic properties to analyse
inter_prop_names = ['Vcl', 'Vxc', 'Einter']  # names of the properties shown in the output

REVERSE = False  # Reverse the REG points

INFLEX = False

### CONTROL COORDINATE OPTIONS ###
CONTROL_COORDINATE_TYPE = ''  # 'Scan' or 'IRC'. If empty ('') then default will be used
Scan_Atoms = [1, 6]  # list of the atoms used for PES Scan (i.e. ModRedundant option in Gaussian)
IRC_output = ''  # insert the g16 output file path if using IRC as control coordinate

CHARGE_TRANSFER_POLARISATION = False  # Split the classical electrostatic term into polarisation and monopolar charge-transfer

DISPERSION = True  # Run DFT-D3 program to consider dispersion
# NOTE: This only works if you have DFT-D3 program installed in the same machine https://www.chemie.uni-bonn.de/pctc/mulliken-center/software/dft-d3/
### DISPERSION OPTIONS ###
DFT_D3_PATH = '/mnt/iusers01/pp01/w06498gk/1Software/DFT-D3/dftd3'  # insert path of DFT-D3 program
DISP_FUNCTIONAL = 'B3-LYP'  # insert the functional used for D3 correction
BJ_DAMPING = True  # Becke-Johnson Damping

WRITE = True  # write csv files for energy values and REG analysis
SAVE_FIG = True  # save figures
ANNOTATE = True  # annotate figures
DETAILED_ANALYSIS = False
LABELS = True  # label figures
n_terms = 4  # number of terms to rank in figures and tables


###### REG-IQF
IQF = True
if IQF:
    SYS = 'REG_IQF' 
    with open('auto_reg.config') as f:
        config_file = f.read()
        Frag_names = re.findall(r'FRAG\s*ID\s*\d+\s*<(.*?)>',config_file) # Names of fragments
        Frag_lists = re.findall(r'FRAG\s*ATOMS\s*\[([\d,]+)\]',config_file) # Atoms by number for fragments
        List_of_frags = []
        for f_list_str in Frag_lists:
            atom_list = f_list_str.split(",")
            int_atom_list = [int(a) for a in atom_list]
            List_of_frags.append(int_atom_list)

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
            
        elif name.endswith(".out"):
            gau16_file.append(os.path.join(root,name))

# Sorting all folders based on REG folders
all_files_sorted = sorted(zip(reg_folders,reg_folder_list,wf_file,gau16_file),key=lambda f: int(re.sub('\D', '', f[0])))

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
    cc = [int(reg_folders[i]) for i in range(0, len(reg_folders))]
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
            iqf_intra_comp_head[f_indx].append(str(fragment[0]))
            for atom_ind in fragment[1:]:
                frag_e += intra_terms[(int(atom_ind) - 1)].copy()
                iqf_intra_comps[f_indx].append(intra_terms[(int(atom_ind) - 1)].copy())
                iqf_intra_comp_head[f_indx].append(str(atom_ind))
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
    
    iqf_inter = np.zeros((int(((N_frag*(N_frag - 1)) / 2) * n_prop),int(len(inter_terms[0]))),dtype=float)
    iqf_inter_comps = [[] for _ in range(int(((N_frag*(N_frag - 1)) / 2) * n_prop))]
    iqf_inter_comp_head = [[] for _ in range(int(((N_frag*(N_frag - 1)) / 2) * n_prop))]
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
                            iqf_intra_comp_head[f1_indx].append(str(atom1) + "-" + str(atom2) + "_" + str(inter_prop[prop_indx]))
                        elif (f1_indx != f2_indx) and (atom1 < atom2):
                            F1_ID = f1_indx + 1
                            F2_ID = f2_indx + 1
                            iqf_inter[int((prop_indx*N_frag)+((F1_ID-1)*(2*N_frag-F1_ID))/2+(F2_ID-F1_ID-1))] += inter_terms[int((prop_indx*no_inter)+((atom1-1)*(2*len(atoms)-atom1))/2+(atom2-atom1-1))]
                            iqf_inter_comps[int((prop_indx*N_frag)+((F1_ID-1)*(2*N_frag-F1_ID))/2+(F2_ID-F1_ID-1))].append(inter_terms[int((prop_indx*no_inter)+((atom1-1)*(2*len(atoms)-atom1))/2+(atom2-atom1-1))])
                            iqf_inter_comp_head[int((prop_indx*N_frag)+((F1_ID-1)*(2*N_frag-F1_ID))/2+(F2_ID-F1_ID-1))].append(str(atom1) + "-" + str(atom2) + "_" + str(inter_prop[prop_indx]))

    # Creating list of iqf inter headers
    for prop in inter_prop:
        for f1_indx in range(len(fragment_atom_list)):
            for f2_indx in range((f1_indx + 1),len(fragment_atom_list)):   
                iqf_inter_header.append(str(prop) + "_" + str(fragment_names[f1_indx] + "_" + fragment_names[f2_indx]))

    return iqf_inter, iqf_inter_header, iqf_inter_comps, iqf_inter_comp_head, iqf_intra, iqf_intra_header, iqf_intra_comps, iqf_intra_comp_head

if IQF:
    # GET INTRA-ATOMIC TERMS:
    iqa_intra, iqa_intra_header,missing_intra = aim_u.intra_property_from_int_file(atomic_files, intra_prop, atoms)
    iqa_intra_header = np.array(iqa_intra_header)  # used for reference
    iqa_intra = np.array(iqa_intra)
    # GET INTER-ATOMIC TERMS:
    iqa_inter, iqa_inter_header,missing_inter = aim_u.inter_property_from_int_file(atomic_files, inter_prop, atoms)
    iqa_inter_header = np.array(iqa_inter_header)  # used for reference
    iqa_inter = np.array(iqa_inter)

    iqf_inter, iqf_inter_header, iqf_inter_comps, iqf_inter_comp_head, iqf_intra, iqf_intra_header, iqf_intra_comps, iqf_intra_comp_head = sum_into_fragments(Frag_names,List_of_frags,atoms,True,iqa_inter,inter_prop,[],iqa_intra,intra_prop)

    iqa_intra_header = np.array(iqf_intra_header)
    iqa_inter_header = np.array(iqf_inter_header)


    iqa_intra = iqf_intra
    iqa_inter = iqf_inter


else:
    # GET INTRA-ATOMIC TERMS:
    iqa_intra, iqa_intra_header,missing_intra = aim_u.intra_property_from_int_file(atomic_files, intra_prop, atoms)
    iqa_intra_header = np.array(iqa_intra_header)  # used for reference
    iqa_intra = np.array(iqa_intra)
    # GET INTER-ATOMIC TERMS:
    iqa_inter, iqa_inter_header,missing_inter = aim_u.inter_property_from_int_file(atomic_files, inter_prop, atoms)
    iqa_inter_header = np.array(iqa_inter_header)  # used for reference
    iqa_inter = np.array(iqa_inter)


# RAISING VALUE ERROR FOR MISSING FILE
missing_files = missing_intra + missing_inter
if len(missing_files) > 0:
    missing_file_message = 'The following files are missing:\n'
    for missing_file in missing_files:
        missing_file_message += missing_file + "\n"
    raise ValueError(missing_file_message)

###############################################################################
#                                                                             #
#                               REG ANALYSIS                                  #
#                                                                             #
###############################################################################

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
        disp_u.run_DFT_D3(DFT_D3_PATH, reg_root_list[i], xyz_file, DISP_FUNCTIONAL)
    folders_disp = [reg_root_list[i] + '/dft-d3.log' for i in range(0, len(reg_folders))]
    # GET INTER-ATOMIC DISPERSION TERMS:
    iqa_disp, iqa_disp_header = disp_u.disp_property_from_dftd3_file(folders_disp, atoms)
    iqa_disp_header = np.array(iqa_disp_header)  # used for reference
    iqa_disp = np.array(iqa_disp)
    if IQF:
        iqa_disp, iqa_disp_header, iqa_disp_comps, iqf_inter_comp_head, iqf_intra, _, iqf_intra_comps, iqf_intra_comp_hea = sum_into_fragments(Frag_names,List_of_frags,atoms,False,iqa_disp,['E_Disp(A,B)'],[iqf_intra,iqf_intra_comps,iqf_intra_comp_head])  #### To remove
        iqa_intra = iqf_intra
    # REG
    reg_disp = reg.reg(total_energy_wfn, cc, iqa_disp, np=POINTS, critical=AUTO, inflex=INFLEX,
                       critical_index=turning_points)
    total_energy_dispersion = sum(iqa_disp)

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

### REG breakdown IQF ######

if IQF:
    iqf_intra_comp_list = []
    for i in range(len(iqa_intra)):
        iqf_val = iqa_intra[i]
        iqf_comp_np = np.array(iqf_intra_comps[i])
        reg_int = reg.reg(iqf_val,cc,iqf_comp_np,np=POINTS, critical=False, inflex=INFLEX,
                        critical_index=[int(len(reg_intra[0][0])) - 1])
        iqf_intra_comp_list.append(reg_int)

# CALCULATE TOTAL ENERGIES
total_energy_iqa = sum(iqa_inter[:(len(atoms)*(len(atoms)-1))]) + sum(iqa_intra[:len(atoms)])  # used to calculate the integration error

# CALCULATE RECOVERY ERROR
if DISPERSION:
    rmse_integration = reg.integration_error(total_energy_wfn, total_energy_iqa + total_energy_dispersion)
else:
    rmse_integration = reg.integration_error(total_energy_wfn, total_energy_iqa)
print('Integration error [kJ/mol](RMSE)')
print(rmse_integration[1])

###############################################################################
#                                                                             #
#                             WRITE CSV FILES                                 #
#                                                                             #
###############################################################################
os.chdir(cwd + '/' + SYS + "_results")
dataframe_list = []

if WRITE:
    # initialise excel file
    writer = pd.ExcelWriter(path=cwd + '/' + SYS + "_results/REG.xlsx", engine='xlsxwriter')
    energy_writer = pd.ExcelWriter(path=cwd + '/' + SYS + "_results/Energy.xlsx", engine='xlsxwriter')
    # ENERGY and CONTROL  COORDINATE ONLY FILES
    df_energy_output = pd.DataFrame()
    df_energy_output['WFN'] = total_energy_wfn
    df_energy_output['IQA'] = total_energy_iqa
    df_energy_output.index = cc
    if DISPERSION:
        df_energy_output['D3'] = total_energy_dispersion
    df_energy_output.to_csv('total_energy.csv', sep=',')
    df_energy_output.to_excel(energy_writer, sheet_name="total_energies")

    pd.DataFrame(data=np.array(iqa_intra).transpose(), columns=iqa_intra_header).to_excel(energy_writer,
                                                                                          sheet_name='intra-atomic_energies')
    pd.DataFrame(data=np.array(iqa_inter).transpose(), columns=iqa_inter_header).to_excel(energy_writer,
                                                                                          sheet_name='inter-atomic_energies')

    if IQF:
        for i,reg_intra_comp in enumerate(iqf_intra_comp_list):
            for j in range(len(reg_intra_comp[0])):
                df_iqf_intra = rv.create_term_dataframe(reg_intra_comp, iqf_intra_comp_head[i],j)
                df_iqf_intra_sorted = df_iqf_intra.sort_values('REG')
                df_iqf_intra_sorted.to_excel(writer, sheet_name= iqf_intra_header[i]+ "_seg_" + str(j + 1))


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
            df_property.to_excel(writer, sheet_name=inter_prop_names[j] + "_seg_" + str(i + 1))
            list_property_sorted.append(
                pd.concat([df_property[-n_terms:], df_property[:n_terms]], axis=0).sort_values('REG'))
        for j in range(len(intra_prop)):
            df_property = rv.filter_term_dataframe(df_intra, intra_prop[j], intra_prop_names[j])
            if j == 0:
                properties_comparison.append(df_property)
            df_property.to_csv(intra_prop_names[j] + "_seg_" + str(i + 1) + ".csv", sep=',')
            df_property.to_excel(writer, sheet_name=intra_prop_names[j] + "_seg_" + str(i + 1))
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
            df_disp_new.to_excel(writer, sheet_name=disp_name_new + "_seg_" + str(i + 1))
            df_disp_new.dropna(axis=0, how='any', subset=None,
                           inplace=True)  # get rid of "NaN" terms which have a null REG Value
            df_dispersion_sorted = pd.concat([df_dispersion_sorted.reset_index(drop=True),
                                              pd.concat([df_disp_new[-n_terms:], df_disp_new[:n_terms]],
                                                        axis=0).sort_values(
                                                  'REG').reset_index(drop=True)], axis=1)
        df_dispersion_sorted.to_csv('REG_' + disp_name_new + '_analysis.csv', sep=',')
        df_dispersion_sorted.to_excel(writer, sheet_name="REG_" + disp_name_new)
        rv.pandas_REG_dataframe_to_table(df_dispersion_sorted, 'REG_' + disp_name_new + '_table', SAVE_FIG=SAVE_FIG)
        pd.DataFrame(data = np.array(iqa_disp).transpose(), columns=iqa_disp_header).to_excel(energy_writer, sheet_name='dispersion_energies')
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
            df_pl.to_excel(writer, sheet_name="Vpl_seg_" + str(i + 1))
            df_ct.to_csv("Vct_seg_" + str(i + 1) + ".csv", sep=',')
            df_ct.to_excel(writer, sheet_name="Vct_seg_" + str(i + 1))
            df_temp = pd.concat([df_pl, df_ct]).sort_values('REG').reset_index(drop=True)
            df_ct_pl_sorted = pd.concat([df_ct_pl_sorted.reset_index(drop=True),
                                         pd.concat([df_temp[-n_terms:], df_temp[:n_terms]], axis=0).sort_values(
                                             'REG').reset_index(drop=True)], axis=1)
        df_ct_pl_sorted.to_csv('REG_Vct-Vpl_analysis.csv', sep=',')
        df_ct_pl_sorted.to_excel(writer, sheet_name='REG_Vct-Vpl')
        rv.pandas_REG_dataframe_to_table(df_ct_pl_sorted, 'REG_Vct-Vpl_table', SAVE_FIG=SAVE_FIG)
        pd.DataFrame(
            data=np.concatenate((np.array(iqa_polarisation_terms), np.array(iqa_charge_transfer_terms))).transpose(),
            columns=np.concatenate((iqa_polarisation_headers, iqa_charge_transfer_headers))).to_excel(energy_writer,
                                                                                                      sheet_name='pl_ct_energies')

    # OUTPUT OF ALL INTER AND INTRA TERMS SELECTED BY THE USER
    all_prop_names = inter_prop_names + intra_prop_names
    for i in range(len(inter_prop) + len(intra_prop)):
        df_property_sorted = pd.DataFrame()
        for j in range(len(reg_inter[0])):
            df_property_sorted = pd.concat([df_property_sorted, list_property_final[j][i]], axis=1)
        df_property_sorted.to_csv('REG_' + all_prop_names[i] + '_analysis.csv', sep=',')
        df_property_sorted.to_excel(writer, sheet_name='REG_' + all_prop_names[i])
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
        df_final.to_excel(writer, sheet_name='REG_full_comparison_seg_' + str(i+1))
    df_final_sorted.to_csv('REG_final_analysis.csv', sep=',')
    df_final_sorted.to_excel(writer, sheet_name='REG_final')
    rv.pandas_REG_dataframe_to_table(df_final_sorted, 'REG_final_table', SAVE_FIG=SAVE_FIG)

    writer.save()
    energy_writer.save()
    rv.plot_violin([dataframe_list[i]['R'] for i in range(len(reg_inter[0]))], save=SAVE_FIG,
                   file_name='violin.png')  # Violing plot of R vs Segments

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

if DETAILED_ANALYSIS:
    for i in range(len(reg_inter[0])):
        rv.generate_data_vis(dataframe_list[i], [dataframe_list[i]['R'] for i in range(len(reg_inter[0]))],
                             n_terms, save=SAVE_FIG, file_name='detailed_seg_' + str(i + 1) + '.png',
                             title=SYS + ' seg. ' + str(i + 1))

###ENDING TIMER ###
print("--- Total time for REG Analysis: {s} minutes ---".format(s=((time.time() - start_time) / 60)))
