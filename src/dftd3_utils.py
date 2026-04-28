"""
dft-d3_utils.py v0.0
F. Falcioni, P. L. A. Popelier

Library with function to submit job to DFT-D3 and get properties values from output
Check for updates at github.com/FabioFalcioni

Please, report bugs and issues to fabio.falcioni@manchester.ac.uk
coded by F.Falcioni
"""

import os

ATOMIC_NUMBER_TO_SYMBOL = {
    1: 'H',   2: 'He',  3: 'Li',  4: 'Be',  5: 'B',   6: 'C',   7: 'N',   8: 'O',
    9: 'F',  10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',  16: 'S',
   17: 'Cl', 18: 'Ar', 19: 'K',  20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V',  24: 'Cr',
   25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge',
   33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y',  40: 'Zr',
   41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd',
   49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I',  54: 'Xe', 55: 'Cs', 56: 'Ba',
   57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd',
   65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf',
   73: 'Ta', 74: 'W',  75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
   81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra',
   89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U',  93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm',
   97: 'Bk', 98: 'Cf', 99: 'Es',100: 'Fm',101: 'Md',102: 'No',103: 'Lr',104: 'Rf',
  105: 'Db',106: 'Sg',107: 'Bh',108: 'Hs',109: 'Mt',110: 'Ds',111: 'Rg',112: 'Cn',
  113: 'Nh',114: 'Fl',115: 'Mc',116: 'Lv',117: 'Ts',118: 'Og',
}


def ensure_xmol_format(xyz_file):
    """
###########################################################################################################
FUNCTION: ensure_xmol_format
          Detects if an XYZ file uses atomic numbers instead of element symbols (non-xmol format)
          and converts it to xmol format, writing a new file with '_xmol' appended to the name.

INPUT:
    xyz_file = path to the xyz file to check/convert

OUTPUT:
    path to the (possibly new) xmol-format file. If the file was already xmol, the original
    path is returned unchanged. If conversion was needed, the new '_xmol.xyz' path is returned.

RAISES:
    ValueError if an atomic number in the file has no entry in ATOMIC_NUMBER_TO_SYMBOL
###########################################################################################################
    """
    print(xyz_file)
    with open(xyz_file, 'r') as f:
        lines = f.readlines()

    # Lines 0 and 1 are the atom count and title; coordinates start at line 2
    coord_lines = lines[2:]

    # Detect whether the first token of each coordinate line is an integer (atomic number)
    needs_conversion = False
    for line in coord_lines:
        stripped = line.strip()
        if not stripped:
            continue
        first_token = stripped.split()[0]
        if first_token.lstrip('-').isdigit():
            needs_conversion = True
            break

    if not needs_conversion:
        return xyz_file

    # Convert atomic numbers to element symbols
    new_coord_lines = []
    for line in coord_lines:
        stripped = line.strip()
        if not stripped:
            new_coord_lines.append(line)
            continue
        parts = line.split()
        atomic_number = int(parts[0])
        if atomic_number not in ATOMIC_NUMBER_TO_SYMBOL:
            raise ValueError('Unknown atomic number {} in file: {}'.format(atomic_number, xyz_file))
        symbol = ATOMIC_NUMBER_TO_SYMBOL[atomic_number]
        # Rebuild the line replacing the atomic number with the symbol, preserving the rest
        new_line = line.replace(parts[0], symbol, 1)
        new_coord_lines.append(new_line)

    # Build the new file path: insert '_xmol' before the extension
    base, ext = os.path.splitext(xyz_file)
    new_xyz_file = base + '_xmol' + (ext if ext else '.xyz')

    with open(new_xyz_file, 'w') as f:
        f.writelines(lines[:2])        # atom count line and title line unchanged
        f.writelines(new_coord_lines)

    print('Converted {} to xmol format -> {}'.format(xyz_file, new_xyz_file))
    return new_xyz_file


def disp_property_from_dftd3_file(folders, atom_list):
    """
###########################################################################################################
Added by Aël Cador - 17 March 2020
FUNCTION: disp_property_from_dftd3_file
          get dispersion interatomic properties from DFT-D3 result files (Grimme DFT-D3)

INPUT:
    folders = path to DFT-D3 result files
    atom_list = list of atomic labels* e.g.: [n1, c2, h3, ...]
    disp_lim = list of atomic label numbers which separate the fragments (the first atom of each fragment starting with the 2nd fragment)

OUTPUT: [intra_properties, contributions_list]
    disp_properties = array of array containing the dispersion energy values for each atom for each geometry
    contributions_list = list of contributions  organized in the same order as in disp_properties

ERROR:
    'File is empty or does not exist:'

###########################################################################################################
"""
    # INTERNAL VARIABLES:
    temp1 = []  # Temporary array
    temp2 = []  # Temporary array
    #   temp3 = [] #Temporary array
    disp_properties = []  # Output
    contributions_list = []  # Output
    n = len(atom_list)

    for i in range(len(atom_list)):
        for j in range(i + 1, len(atom_list)):
            contributions_list.append('E_Disp(A,B)-' + atom_list[i] + '_' + atom_list[j])

    for path in folders:
        #        for i in range(len(atom_list)):
        #            atom1 = atom_list[i]
        #            for j in range(i+1, len(atom_list)):
        #                atom2 = atom_list[j]
        file = open(path, "r")  # +"/" + atom1 + "_" + atom2 + ".int", "r")
        lines = file.readlines()
        file.close()
        for i in lines:
            if 'analysis of pair-wise terms (in kcal/mol)' in i:
                start = lines.index(i) + 2
            elif 'distance range (Angstroem) analysis' in i:

                end = lines.index(i) - 1  # check if both are valid*

        if end >= len(lines):  # Checks the DFT-D3 file.
            raise ValueError("File is empty or does not exist: " + path)  # +"/" + atom1 + "_" + atom2 + ".int")

        lines = [lines[i] for i in range(start, end)]
        #               for term in prop:
        for i in lines:
            #                   if (term + '  ') in i:
            temp1.append(float(i.split()[-1]))
    # ORGANIZE ARRAY ORDER
    #    for j in range(len(prop)):
    #        for i in range(j, len(temp1), len(prop)):
    #            temp2.append(temp1[i])
    for j in range(int(n * (n - 1) / 2)):
        temp2.append([temp1[i] / 627.503 for i in range(j, len(temp1), int(n * (n - 1) / 2))])  # temp2 ?
    start = 0
    #   for j in range(len(prop)):
    for atom_prop in temp2:
        disp_properties.append([atom_prop[i] for i in range(start, len(folders))])  # (j+1)*
    # start = (j+1)*len(folders)
    # CREATE CONTRIBUTIONS LIST ARRAY:

    return disp_properties, contributions_list


#def run_DFT_D3(program_path, xyz_file, functional, BJ_Damping=True):
    '''
###########################################################################################################
FUNCTION: run_DFT_D3
          run Grimme DFT-D3 program for selected xyz file given the path of the program

INPUT:
    program_path = path to DFT-D3 program
    xyz_file = xyz format file
    functional = functional used for
    BJ_Dumping = Becke Johnson dumping (default=True)

ERRORS:
    'Insert a functional that works with BJ dumping and AIMAll'
    'Insert a functional that works with AIMAll'

Note (06/12/2020): AIMAll works with LSDA, B3LYP, PBE, PBE0, M062X
###########################################################################################################
    '''
def run_DFT_D3(program_path,reg_folder, xyz_file, functional, BJ_Damping=True):
    '''
###########################################################################################################
FUNCTION: run_DFT_D3
          run Grimme DFT-D3 program for selected xyz file given the path of the program

INPUT:
    program_path = path to DFT-D3 program
    xyz_file = xyz format file
    functional = functional used for
    BJ_Dumping = Becke Johnson dumping (default=True)

ERRORS:
    'Insert a functional that works with BJ dumping and AIMAll'
    'Insert a functional that works with AIMAll'

Note (06/12/2020): AIMAll works with LSDA, B3LYP, PBE, PBE0, M062X
###########################################################################################################
    '''
    # Ensure that input XYZ file is in xmol format (element symbols instead of atomic numbers) required by DFT-D3
    xyz_file = ensure_xmol_format(xyz_file)

    # Run DFT-D3 with the appropriate command line options based on the functional and damping choice
    if BJ_Damping==True:
        functional_list = ['B3-LYP','PBE0','PBE']
        if functional.upper() in functional_list:
            os.system(program_path + ' ' + xyz_file + ' -func ' + functional.lower() + ' -bj -anal > {}/dft-d3.log'.format(reg_folder))
        else:
            raise ValueError('Insert a functional that works with BJ dumping and AIMAll')

    # If no BJ damping, check for functionals that work with AIMAll and run DFT-D3 accordingly
    else:
        functional_list = ['B3-LYP','M062X','PBE', 'PBE0']
        d2_functional_list = ['b97-d']
        if functional.upper() in functional_list:
            os.system(program_path + ' ' + xyz_file + ' -zero -func ' + functional.lower() + ' -anal > {}/dft-d3.log'.format(reg_folder))
        elif functional.lower() in d2_functional_list:
            os.system(program_path + ' ' + xyz_file + ' -func ' + functional.lower() + '-old -anal > {}/dft-d3.log'.format(reg_folder))
        else:
            raise ValueError('Insert a functional that works with AIMAll')
