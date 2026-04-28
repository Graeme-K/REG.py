"""
gaussian_utils.py v0.1
L. J. Duarte, F.Falcioni, P. L. A. Popelier

Library with function to submit job to Gaussian and get properties values from output_files
GAUSSIAN version: G09 / G16
Check for updates at https://github.com/FabioFalcioni/REG.py
For details about the method, please see XXXXXXX

Please, report bugs and issues to fabio.falcioni@manchester.ac.uk
coded by L. J. Duarte and F. Falcioni
"""

import re
import numpy as np

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

def get_atom_list_wfn_g09(wfn_file):
    """
    ###########################################################################################################
    FUNCTION: get_atom_list_wfn_g09
              get atomic labels from g09 wfn file

    INPUT: wfn_file
        wfn_file = Any wfn file of the desired PES

    OUTPUT: atom_list
        list of each atom label for all atoms in molecule

    ERROR:
        "Atomic labels not found" : Atom list does not exist in wfn_file
    ###########################################################################################################
    """
    #INTERNAL VARIABLES:
    atom_list = []
    
    #OPEN FILE:
    file = open(wfn_file, "r")
    lines = file.readlines() #Convert file into a array of lines
    file.close() #Close file
    
    #ERRORS:
    if "(CENTRE " not in lines[2]:
        raise ValueError("Atomic labels not found")  #Checks if atomic list exist inside file
        
    #GET ATOM LIST:
    for i in range(len(lines)):
        if "(CENTRE " in lines[i]:
            split_line = lines[i].split()
            atom_list.append(split_line[0].lower() + str(split_line[1])) # uppercase to lowercase
    
    return atom_list


def get_atom_list_wfx_g09(wfx_file):
    """
    ###########################################################################################################
    FUNCTION: get_atom_list_wfx_g09
              get atomic labels from g09 wfn file

    INPUT: wfn_file
        wfx_file = Any double wavefunction file of the desired PES

    OUTPUT: atom_list
        list of each atom label for all atoms in molecule

    ERROR:
        "Atomic labels not found" : Atom list does not exist in wfn_file
    ###########################################################################################################
    """
    #INTERNAL VARIABLES:
    atom_list = []
    
    #OPEN FILE:
    file = open(wfx_file, "r")
    lines = file.readlines() #Convert file into a array of lines
    file.close() #Close file
    
    #ERRORS:
    if " <Nuclear Names>" not in lines[33]:
        raise ValueError("Atomic labels not found")  #Checks if atomic list exist inside file
        
    #GET ATON LINES NUMBER
    for i in range(len(lines)):
        if "<Nuclear Names>" in lines[i]:
            start_line = i+1
        if "</Nuclear Names>" in lines[i]:
            end_line = i

    #GET ATOM LIST:
    for i in range(start_line, end_line):
        split_line = lines[i].split()
        atom_list.append(split_line[0].lower())
    
    return atom_list


def get_atom_list_wfn_g16(wfn_file):
    """
    ###########################################################################################################
    FUNCTION: get_atom_list_wfn_g16
              get atomic labels from g16 wfn file

    INPUT: wfn_file
        wfn_file = Any wfn file of the desired PES

    OUTPUT: atom_list
        list of each atom label for all atoms in molecule

    ERROR:
        "Atomic labels not found" : Atom list does not exist in wfn_file
    ###########################################################################################################
    """
    #INTERNAL VARIABLES:
    atom_list = []
    
    #OPEN FILE:
    file = open(wfn_file, "r")
    lines = file.readlines() #Convert file into a array of lines
    file.close() #Close file
    
    #ERRORS:
    if "(CENTRE " not in lines[2]:
        raise ValueError("Atomic labels not found")  #Checks if atomic list exist inside file
        
    #GET ATOM LIST:
    for i in range(len(lines)):
        if "(CENTRE " in lines[i]:
            split_line = lines[i].split()
            atom_list.append(split_line[0].lower()) # uppercase to lowercase
    
    return atom_list

def get_control_coordinates_IRC_g16(output_file):
    '''
    ###########################################################################################################
    FUNCTION: get_control-coordinates
        get control coordinates from g16 output file
    INPUT: output_file
        output_file = Gaussian16 output file for Intrinsic Reaction Coordinate (IRC) scan.

    OUTPUT: coordinates
        list of the control coordinate of the IRC scan

    ERROR:
        "Control coordinates not found. Please, check that you have the g16 IRC output in the running folder"
    ###########################################################################################################

    '''
    #INTERNAL VARIABLES
    coordinates = []
    start = 0
    end = 0
    found = False

    #WORKING IN THE FILE
    with open(output_file, 'r') as f:
        lines = f.readlines()
        # GET THE FIRST/LAST COORDINATE POSITION
        for i in range(len(lines)):
            if "Summary of reaction path following" in lines[i]:
                start = i + 3
                found = True

        # ERRORS
        if found is False:
            raise ValueError("Control coordinates not found. Please, check that you have the g16 IRC output in the running folder")

        for j in range(start, len(lines)):
            if "---" in lines[j]:
                end = j
                break

        # GET COORDINATES LIST
        for line in lines[start: end]:
            coordinates.append(float(line.split()[2]))

    return coordinates


def get_control_coordinates_PES_Scan(g16_output_filelist, atom_list):
    '''
    ###########################################################################################################
    FUNCTION: get_control_coordinates_PES_Scan
        get REG control coordinates from g16 single point energy output files
    INPUT:
        g16_output_filelist = Gaussian16 output file for Single Point Energy
        atom_list = list of atoms (as strings) used for the ModRedundant option in Gaussian16

    OUTPUT:
        cc = control coordinates list

    ERROR:
        "Single Point Energy File not found"
    ###########################################################################################################
    '''
    cc = []
    xyz_files = []
    for i in range(0, len(g16_output_filelist)):
        xyz_files.append(get_xyz_file(g16_output_filelist[i]))

    # Control Coordinates search for BOND movement
    for i in range(0, len(xyz_files)):
        f1 = open(xyz_files[i], 'r')
        coordinates1_list = f1.readlines()[2:]  # temporary remove the first 2 lines of xyz file
        atom1 = [float(c) for c in re.findall(r"[-+]?\d*\.\d+|\d+", coordinates1_list[atom_list[0]])]
        atom2 = [float(c) for c in re.findall(r"[-+]?\d*\.\d+|\d+", coordinates1_list[atom_list[1]])]
        x = 0
        y = 1
        z = 2
        cc.append(np.sqrt((atom2[x] - atom1[x]) ** 2 + (atom2[y] - atom1[y]) ** 2 + (atom2[z] - atom1[z]) ** 2))
    return cc

def get_xyz_file(g16_single_point_output):
    '''
    ###########################################################################################################
    FUNCTION: get_xyz_file
        get xyz coordinates from g16 single point energy output file
    INPUT: g16_single_point_output
        g16_single_point_output = Gaussian16 output file for Single Point Energy

    OUTPUT: xyz_output
        xyz_output = xyz file format

    ERROR:
        "Single Point Energy File not found"
    ###########################################################################################################
    '''
    start = 0
    end = 0
    xyz_output = str(g16_single_point_output[:-4]) + "_gauout.xyz"

    openold= open(g16_single_point_output, 'r')
    opennew= open(xyz_output, 'w')
    rlines = openold.readlines()
    for i in range (len(rlines)):
            if "Standard orientation:" in rlines[i] or "Input orientation:" in rlines[i]:
                start = i
    for m in range (start + 5, len(rlines)):
        if "---" in rlines[m]:
            end = m
            break
    opennew.write("\n{}\n\n".format(str(end-start-5)))
    ## Convert to Cartesian coordinates format
    ## convert atomic number to atomic symbol
    for line in rlines[start+5 : end] :
        words = line.split()
        word1 = ATOMIC_NUMBER_TO_SYMBOL[int(words[1])]
        word3 = str(words[3])

        ## copy from atom list.

        opennew.write("{}{}\n".format(word1,line[30:-1]))
    openold.close()
    opennew.close()
    return xyz_output




