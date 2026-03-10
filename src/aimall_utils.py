"""
aimall_utils.py v0.1
L. J. Duarte, XXXXXXX, P. L. A. Popelier 

Library with function to submit job to AIMAll and get properties values from output
AIMAll version: 17.11.14
Check for updates at github.com/ljduarte
For details about the method, please see XXXXXXX

Please, report bugs and issues to leo.j.duarte@hotmail.com.br
coded by L. J. Duarte
"""

import os
import warnings

import numpy as np


def _extract_section(lines, start_marker, end_marker):
    """Extract lines between start_marker and end_marker (exclusive).

    Returns the list of lines in the section, or None if start_marker is
    not found (signals that a fallback read is needed).
    """
    recording = False
    result = []
    for line in lines:
        if start_marker in line:
            recording = True
        elif end_marker in line:
            if recording:
                return result
        elif recording:
            result.append(line)
    return None  # start_marker not found (or file was truncated before end_marker)


def read_int_section_fast(filepath, start_marker, end_marker, max_bytes=8192):
    """Read a section between markers by seeking from the end of the file.

    IQA energy sections sit near the end of .int files, after megabytes of
    wavefunction coefficients.  Seeking from the end avoids transferring all
    that data over the NFS mount.

    If the section is not found in the tail chunk (e.g. max_bytes too small,
    or the section is near the start of the file) a full file read is
    performed with a warning so correctness is never compromised.

    Returns a list of lines inside the section, or None if not found anywhere.
    """
    with open(filepath, 'rb') as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        read_size = min(max_bytes, file_size)
        f.seek(-read_size, os.SEEK_END)
        chunk = f.read(read_size).decode('utf-8', errors='ignore')

    result = _extract_section(chunk.split('\n'), start_marker, end_marker)

    if result is None and read_size < file_size:
        # Section not found in the tail; fall back to reading the whole file.
        warnings.warn(
            "Section '{}' not found in last {} bytes of '{}'; "
            "falling back to full file read. "
            "Consider increasing max_bytes.".format(start_marker, max_bytes, filepath),
            RuntimeWarning,
            stacklevel=3
        )
        with open(filepath, 'r', errors='ignore') as f:
            result = _extract_section(f.read().split('\n'), start_marker, end_marker)

    return result  # None if not found; list of lines if found


def read_last_line_fast(file_path, max_bytes=200):
    """Return the last non-empty line by seeking from the end of the file.

    .wfn files only need their final TOTAL ENERGY line, so there is no point
    reading the entire file across NFS.
    """
    with open(file_path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        read_size = min(max_bytes, file_size)
        f.seek(-read_size, os.SEEK_END)
        chunk = f.read(read_size)

    for line in reversed(chunk.split(b'\n')):
        if line.strip():
            return line.decode('utf-8', errors='ignore')
    return ""


# ---------------------------------------------------------------------------
# .sum file fast-path helpers
# ---------------------------------------------------------------------------

# Maps user-facing inter prop names to a function of the 7-column row from
# the .sum E_IQA_Inter(A,B)/2 table.
# Column order: [E_IQA_Inter(A,B)/2, Vne(A,B)/2, Ven(A,B)/2,
#                Vee(A,B)/2, Vnn(A,B)/2, VeeC(A,B)/2, VeeX(A,B)/2]
_INTER_SUM_DERIVATIONS = {
    'E_IQA_Inter(A,B)':   lambda c: 2.0 * c[0],
    'E_IQA_Inter(A,B)/2': lambda c: c[0],
    'V_IQA(A,B)':         lambda c: 2.0 * c[0],   # V_IQA_Inter = E_IQA_Inter
    'V_IQA(A,B)/2':       lambda c: c[0],
    'VX_IQA(A,B)':        lambda c: 2.0 * c[6],   # VX = 2 * VeeX/2
    'VX_IQA(A,B)/2':      lambda c: c[6],
    'VC_IQA(A,B)':        lambda c: 2.0 * (c[0] - c[6]),  # VC = E - VX
    'VC_IQA(A,B)/2':      lambda c: c[0] - c[6],
    'Vne(A,B)':           lambda c: 2.0 * c[1],
    'Vne(A,B)/2':         lambda c: c[1],
    'Ven(A,B)':           lambda c: 2.0 * c[2],
    'Ven(A,B)/2':         lambda c: c[2],
    'Vee(A,B)':           lambda c: 2.0 * c[3],
    'Vee(A,B)/2':         lambda c: c[3],
    'Vnn(A,B)':           lambda c: 2.0 * c[4],
    'Vnn(A,B)/2':         lambda c: c[4],
    'VeeC(A,B)':          lambda c: 2.0 * c[5],
    'VeeC(A,B)/2':        lambda c: c[5],
    'VeeX(A,B)':          lambda c: 2.0 * c[6],
    'VeeX(A,B)/2':        lambda c: c[6],
    'Vneen(A,B)':         lambda c: 2.0 * (c[1] + c[2]),
    'Vneen(A,B)/2':       lambda c: c[1] + c[2],
}


def _get_sum_path(atomicfiles_folder):
    """Derive the AIMAll .sum file path from an _atomicfiles folder path.

    e.g. './RUN.pointdir/RUN_atomicfiles' -> './RUN.pointdir/RUN.sum'
    """
    parent = os.path.dirname(atomicfiles_folder)
    stem = os.path.basename(parent)
    for suffix in ('.pointdir', '.pointsdir'):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    return os.path.join(parent, stem + '.sum')

def _parse_sum_file(sum_path, atom_list):
    """Parse an AIMAll .sum file and return structured data for all atoms.

    The .sum file collects all per-geometry IQA results in one place, making
    it possible to replace 190+ individual .int file reads with a single read.

    Returns a dict:
      result['intra'][atom_label] = {column_name: float}
          All per-atom property tables are merged into a flat dict per atom.
          Keys match the column header strings exactly as in the .sum file,
          e.g. 'E_IQA_Intra(A)', 'T(A)', 'q(A)', 'Vne(A,A)', etc.

      result['inter'][(atom_a, atom_b)] = [7 floats]
          Seven columns from the E_IQA_Inter(A,B)/2 table for the canonical
          pair ordering (atom_a precedes atom_b in atom_list).
          Order: [E_IQA_Inter/2, Vne/2, Ven/2, Vee/2, Vnn/2, VeeC/2, VeeX/2]

    Returns None on I/O error (file not found, unreadable, etc.).
    """
    try:
        with open(sum_path, 'r', errors='ignore') as f:
            lines = f.readlines()
    except OSError:
        return None

    atom_set = set(a.lower() for a in atom_list)
    atom_idx = {a.lower(): i for i, a in enumerate(atom_list)}
    result = {'intra': {a: {} for a in atom_set}, 'inter': {}}

    i = 0
    while i < len(lines):
        tokens = lines[i].split()

        # ---- Generic per-atom property table --------------------------------
        # Detected by: first two tokens are "Atom A", third token is NOT "Atom"
        # (that would be the inter pair table "Atom A  Atom B  ..."),
        # and the following line is a dashed separator.
        if (len(tokens) >= 3
                and tokens[0] == 'Atom' and tokens[1] == 'A'
                and tokens[2] != 'Atom'
                and i + 1 < len(lines)
                and lines[i + 1].strip().startswith('---')):
            col_names = tokens[2:]   # column names follow "Atom A" on the header
            i += 2                   # skip header line + separator line
            while i < len(lines):
                row = lines[i].split()
                if not row or lines[i].strip().startswith('---'):
                    break            # blank line or closing separator
                atom = row[0].lower()
                if atom == 'total':
                    i += 1
                    break
                if atom in atom_set:
                    for j_col, col in enumerate(col_names):
                        if j_col + 1 < len(row):
                            try:
                                result['intra'][atom][col] = float(row[j_col + 1])
                            except ValueError:
                                pass
                i += 1
            continue

        # ---- Inter pair table -----------------------------------------------
        # Detected by: "Atom A  Atom B  E_IQA_Inter(A,B)/2 ..."
        if (len(tokens) >= 4
                and tokens[0] == 'Atom' and tokens[1] == 'A'
                and tokens[2] == 'Atom' and tokens[3] == 'B'
                and i + 1 < len(lines)
                and lines[i + 1].strip().startswith('---')):
            i += 2  # skip header + separator
            while i < len(lines):
                row = lines[i].split()
                if not row:
                    break           # blank line ends this atom-A block
                if len(row) < 9:
                    i += 1
                    continue
                atom_a = row[0].lower()
                atom_b = row[1].lower()
                # Skip summary rows (SumB, A'=Mol-A, SumB-A')
                if atom_b not in atom_set or atom_a not in atom_set:
                    i += 1
                    continue
                # Store only the canonical (earlier index, later index) pair
                idx_a = atom_idx.get(atom_a, -1)
                idx_b = atom_idx.get(atom_b, -1)
                if 0 <= idx_a < idx_b:
                    try:
                        result['inter'][(atom_a, atom_b)] = [float(v) for v in row[2:9]]
                    except ValueError:
                        pass
                i += 1
            continue

        i += 1

    return result

def distance_A_B(xyz_file, atom_A, atom_B):
    """
    ###########################################################################################################
    FUNCTION: distance_A_B
        get the distance between atom A and atom B
    INPUT: wfn_file
        xyz_file = xyz file obtained from the get_xyz_file function
        atom_A = atomic labeling (integer)
        atom_B = atomic labeling (integer)

    OUTPUT: atom_list
         r_AB = distance between atom A and atom B

    ERROR:
        "Atomic labels not found" : Atom list does not exist in wfn_file
    ###########################################################################################################
    """
    # INTERNAL VARIABLES:
    all_coordinates = []

    # WORKING IN THE FILE:
    with open(xyz_file) as f:
        coordinates_list = f.readlines()[3:]  # remove the first 2 lines of xyz file
        for i in range(0, len(coordinates_list)):
            coordinates_of_atom = [float(c) for c in coordinates_list[i].split()[1:]]
            all_coordinates.append(coordinates_of_atom)

    coord_atom_A = all_coordinates[atom_A - 1]
    coord_atom_B = all_coordinates[atom_B - 1]
    x = 0
    y = 1
    z = 2
    # GET DISTANCE
    r_AB = np.sqrt((coord_atom_B[x] - coord_atom_A[x]) ** 2 + (coord_atom_B[y] - coord_atom_A[y]) ** 2 + (
                coord_atom_B[z] - coord_atom_A[z]) ** 2)
    return r_AB

def get_atom_list(wfn_file):
    """
    ###########################################################################################################
    FUNCTION: get_atom_list
              get atomic labels from wfn file

    INPUT: wfn_file
        wfn_file = Any wfn file of the desired PES

    OUTPUT: atom_list
        list of each atom label for all atoms in molecule

    ERROR:
        "Atomic labels not found" : Atom list does not exist in wfn_file
    ###########################################################################################################
    """

    # INTERNAL VARIABLES:
    atom_list = []

    # OPEN FILE:
    file = open(wfn_file, "r")
    lines = file.readlines()  # Convert file into a array of lines
    file.close()  # Close file

    # ERRORS:
    if "(CENTRE " not in lines[2]:
        raise ValueError("Atomic labels not found")  # Checks if atomic list exist inside file

    # GET ATOM LIST:
    for i in range(len(lines)):
        if "(CENTRE" in lines[i]:
            split_line = lines[i].split()
            atom_list.append(split_line[0].lower() + str(split_line[1]))  # uppercase to lowercase

    return atom_list

def get_atom_list_wfx(wfx_file):
    # INTERNAL VARIABLES:
    atom_list = []

    # OPEN FILE:
    file = open(wfx_file, "r")
    lines = file.readlines()  # Convert file into a array of lines
    file.close()  # Close file

    # GET ATOM LIST:
    for i in range(len(lines)):
        if "<Nuclear Names>" in lines[i]:
            i += 1
            while "</Nuclear Names>" not in lines[i]:
                split_line = lines[i].split()
                atom_list.append(split_line[0].lower())  # uppercase to lowercase
                i += 1

    return atom_list

def get_aimall_wfn_energies(A):
    """
    ###########################################################################################################
    FUNCTION: get_aimall_wfn_energies
              get all wfn energies from the wavefunction files (wfn)

    INPUT:
        A = list of all wfn files of the PES

    OUTPUT: wfn_energy
        wfn_energy = list of energies for each PES point

    ERROR:
        "Energy values not found in file : file_name" : No energy values in file_name
    ###########################################################################################################
    """
    # INTERNAL VARIABLES:
    wfn_energy = []  # List of wfn files

    # READ FILES: seek from end — only the last line (TOTAL ENERGY) is needed,
    # so there is no reason to transfer the entire wfn file over NFS.
    for path in A:
        last_line = read_last_line_fast(path)
        # ERRORS:
        if "TOTAL ENERGY " not in last_line:  # Checks if there is an energy value at the end of wfn file.
            raise ValueError("Energy values not found in file: ", path)
        wfn_energy.append(float(last_line.split()[3]))

    return wfn_energy

def get_aimall_wfx_energies(A):
    # INTERNAL VARIABLES:
    wfx_energy = []  # List of wfn files

    # READ FILES
    for path in A:
        file = open(path, "r")
        lines = file.readlines()
        # ERRORS:
        if "<Energy = T + Vne + Vee + Vnn>" not in lines[
            -6]:  # Checks if there is an energy value at the end of wfn file.
            raise ValueError("Energy values not found in file: ", path)
        wfx_energy.append(float(lines[-5].split()[0]))
        file.close()

    return wfx_energy

def intra_property_from_int_file(folders, prop, atom_list):
    """
    ###########################################################################################################
    FUNCTION: intra_property_from_int_file
              get IQA intra-atomic properties from int files

    INPUT:
        folders = path to _atomicfiles folders
        prop = list of IQA for each atoms e.g.: "['T(A)', 'Vee(A,A)', 'Vne(A,A)']"
        atom_list = list of atomic lables e.g.: [n1, c2, h3, ...]

    OUTPUT: [intra_properties, contributions_list]
        intra_properties = array of array containing the IQA values for eacha atom for each geometry
        contributions_list = list of contributions  organized in the same order as in intra_properties

    ERROR:
        File is empty or does not exist
    ###########################################################################################################
    """
    # INTERNAL VARIABLES:
    temp1 = []  # Temporary array
    temp2 = []  # Temporary array
    temp3 = []  # Temporary array
    intra_properties = []  # Output
    contributions_list = []  # Output
    missing_files = [] # Files that cannot be found

    # Pre-build patterns for the .int fallback path.
    prop_patterns = {p: p + '           ' for p in prop}

    # READ PROPERTIES: try the single .sum file first (one NFS read per geometry
    # instead of one per atom), fall back to individual .int files if unavailable.
    for folder in folders:
        sum_data = _parse_sum_file(_get_sum_path(folder), atom_list)

        # Check that every requested term is a direct column in the .sum intra tables.
        use_sum = (sum_data is not None
                   and all(p in sum_data['intra'].get(atom_list[0].lower(), {})
                           for p in prop))

        if sum_data is not None and not use_sum:
            warnings.warn(
                "Not all intra prop terms {} are present in '{}'; "
                "falling back to .int file reads for this geometry.".format(
                    prop, _get_sum_path(folder)),
                RuntimeWarning, stacklevel=2)

        if sum_data is None:
            warnings.warn(
                "No .sum file found at '{}'; "
                "falling back to individual .int file reads.".format(
                    _get_sum_path(folder)),
                RuntimeWarning, stacklevel=2)

        if use_sum:
            # Fast path: extract from the already-parsed .sum dict.
            for atom in atom_list:
                atom_data = sum_data['intra'].get(atom.lower(), {})
                if not all(p in atom_data for p in prop):
                    missing_files.append(folder + "/" + atom + ".int")
                    continue
                for term in prop:
                    temp1.append(atom_data[term])

        else:
            # Fallback: seek from the end of each .int file.
            for atom in atom_list:
                lines = read_int_section_fast(
                    folder + "/" + atom + ".int",
                    'IQA Energy Components (see "2EDM Note")',
                    '2EDM Note:'
                )
                if lines is None:
                    missing_files.append(folder + "/" + atom + ".int")
                    continue
                found = {}
                for line in lines:
                    for term in prop:
                        if term not in found and prop_patterns[term] in line:
                            found[term] = float(line.split()[-1])
                            break
                for term in prop:
                    if term in found:
                        temp1.append(found[term])

    # RAISING VALUE ERROR FOR MISSING FILE
    if len(missing_files) > 0:
        missing_file_message = 'The following files are missing:\n'
        for missing_file in missing_files:
            missing_file_message += missing_file + "\n"
        raise ValueError(missing_file_message)

    # ORGANIZE ARRAY ORDER
    for j in range(len(prop)):
        for i in range(j, len(temp1), len(prop)):
            temp2.append(temp1[i])
    for j in range(len(atom_list)):
        temp3.append([temp2[i] for i in range(j, len(temp2), len(atom_list))])

    start = 0
    for j in range(len(prop)):
        for atom_prop in temp3:
            intra_properties.append([atom_prop[i] for i in range(start, (j + 1) * len(folders))])
        start = (j + 1) * len(folders)

    # CREATE CONTRIBUTIONS LIST ARRAY:
    for a in prop:
        for b in atom_list:
            contributions_list.append(a + '-' + b)

    return intra_properties, contributions_list, missing_files

def inter_property_from_int_file(folders, prop, atom_list):
    """
    ###########################################################################################################
    FUNCTION: inter_property_from_int_file
              get IQA interatomic properties from int files

    INPUT:
        folders = path to _atomicfiles folders
        prop = list of IQA for each atoms e.g.: "['T(A)', 'Vee(A,A)', 'Vne(A,A)']"
        atom_list = list of atomic lables e.g.: [n1, c2, h3, ...]

    OUTPUT: [intra_properties, contributions_list]
        intra_properties = array of array containing the IQA values for eacha atom for each geometry
        contributions_list = list of contributions  organized in the same order as in intra_properties

    ERROR:
        "File is empty or does not exist"
    ###########################################################################################################
    """
    # INTERNAL VARIABLES:
    temp1 = []  # Temporary array
    temp2 = []  # Temporary array
    temp3 = []  # Temporary array
    inter_properties = []  # Output
    contributions_list = []  # Output
    missing_files = [] # Files that cannot be found

    # Pre-build patterns for the .int fallback path.
    prop_patterns = {p: p + '  ' for p in prop}

    # Check whether all requested terms can be derived from .sum inter table columns.
    _inter_derivable = all(p in _INTER_SUM_DERIVATIONS for p in prop)

    # READ PROPERTIES: try the single .sum file first, fall back to .int files.
    for path in folders:
        sum_data = _parse_sum_file(_get_sum_path(path), atom_list)
        use_sum = sum_data is not None and _inter_derivable

        if sum_data is not None and not use_sum:
            warnings.warn(
                "Not all inter prop terms {} are derivable from '{}'; "
                "falling back to .int file reads for this geometry.".format(
                    prop, _get_sum_path(path)),
                RuntimeWarning, stacklevel=2)

        if sum_data is None:
            warnings.warn(
                "No .sum file found at '{}'; "
                "falling back to individual .int file reads.".format(
                    _get_sum_path(path)),
                RuntimeWarning, stacklevel=2)

        for i in range(len(atom_list)):
            atom1 = atom_list[i]
            for j in range(i + 1, len(atom_list)):
                atom2 = atom_list[j]

                if use_sum:
                    # Fast path: derive values from the .sum inter pair table.
                    cols = sum_data['inter'].get((atom1.lower(), atom2.lower()))
                    if cols is None:
                        missing_files.append(path + "/" + atom1 + "_" + atom2 + ".int")
                        continue
                    for term in prop:
                        temp1.append(_INTER_SUM_DERIVATIONS[term](cols))

                else:
                    # Fallback: seek from the end of each inter .int file.
                    lines = read_int_section_fast(
                        path + "/" + atom1 + "_" + atom2 + ".int",
                        ' Energy Components (See "2EDM Note"):',
                        '2EDM Note:'
                    )
                    if lines is None:
                        missing_files.append(path + "/" + atom1 + "_" + atom2 + ".int")
                        continue
                    found = {}
                    for line in lines:
                        for term in prop:
                            if term not in found and prop_patterns[term] in line:
                                found[term] = float(line.split()[-1])
                                break
                    for term in prop:
                        if term in found:
                            temp1.append(found[term])

    # RAISING VALUE ERROR FOR MISSING FILE
    if len(missing_files) > 0:
        missing_file_message = 'The following files are missing:\n'
        for missing_file in missing_files:
            missing_file_message += missing_file + "\n"
        raise ValueError(missing_file_message)

    # ORGANIZE ARRAY ORDER
    for j in range(len(prop)):
        for i in range(j, len(temp1), len(prop)):
            temp2.append(temp1[i])
    for j in range(int(len(atom_list) * (len(atom_list) - 1) / 2)):
        temp3.append([temp2[i] for i in range(j, len(temp2), int(len(atom_list) * (len(atom_list) - 1) / 2))])
    start = 0
    for j in range(len(prop)):
        for atom_prop in temp3:
            inter_properties.append([atom_prop[i] for i in range(start, (j + 1) * len(folders))])
        start = (j + 1) * len(folders)
        # CREATE CONTRIBUTIONS LIST ARRAY:
    for a in prop:
        for i in range(len(atom_list)):
            for j in range(i + 1, len(atom_list)):
                contributions_list.append(a + '-' + atom_list[i] + '_' + atom_list[j])

    return inter_properties, contributions_list, missing_files

def charge_transfer_and_polarisation_from_int_file(folders, atom_list, inter_properties, xyz_files):
    """
    ###########################################################################################################
    FUNCTION: charge_transfer_and_polarisation_from_int_file
        get IQA monopolar charge-transfer and polarisation properties from int files

    INPUT:
        folders = path to _atomicfiles folders
        atom_list = list of atomic lables e.g.: [n1, c2, h3, ...]
        inter_properties = array of inter-atomic properties
        xyz_files = xyz files list obtained from the get_xyz_file function

    OUTPUT:
        charge_transfer_properties = array of array containing the IQA charge transfer values for each atom for each geometry
        contributions_list_CT = list of contributions  organized in the same order as in charge_transfer_properties
        polarisation_properties = array of array containing the IQA polarisation values for each atom for each geometry
        contributions_list_PL = list of contributions  organized in the same order as in polarisation_properties

    ERROR:
        "File is empty or does not exist"
    ###########################################################################################################
    """
    # INTERNAL VARIABLES:
    n = len(atom_list)
    f = len(folders)
    temp1 = []  # Temporary array
    temp2 = []  # Temporary array
    temp3 = []  # Temporary array
    net_charges = []
    charge_transfer_properties = []  # Output
    polarisation_properties = []  # Output
    contributions_list_CT = []  # Output
    contributions_list_PL = []  # Output

    # CREATE CONTRIBTUIONS LIST ARRAY
    for i in range(len(atom_list)):
        for j in range(i + 1, len(atom_list)):
            contributions_list_PL.append('Vpl_IQA(A,B)-' + atom_list[i] + '_' + atom_list[j])
            contributions_list_CT.append('Vct_IQA(A,B)-' + atom_list[i] + '_' + atom_list[j])

    # READ NET-CHARGE PROPERTIES: try .sum first (q(A) column), fall back to .int.
    for folder in folders:
        sum_data = _parse_sum_file(_get_sum_path(folder), atom_list)

        if sum_data is None:
            warnings.warn(
                "No .sum file found at '{}'; "
                "falling back to individual .int file reads for net charges.".format(
                    _get_sum_path(folder)),
                RuntimeWarning, stacklevel=2)

        net_charge_group = []
        for i in range(0, len(atom_list)):
            atom = atom_list[i]

            if sum_data is not None and 'q(A)' in sum_data['intra'].get(atom.lower(), {}):
                # Fast path: net charge is directly available as q(A) in the .sum file.
                Q = sum_data['intra'][atom.lower()]['q(A)']

            else:
                # Fallback: read the basin integration section from the .int file.
                # Note: this section is near the file start, so read_int_section_fast
                # will always fall back to a full read here.
                lines = read_int_section_fast(
                    folder + "/" + atom + ".int",
                    'Results of the basin integration:',
                    '|Dipole|'
                )
                if lines is None:
                    raise ValueError("File is empty or does not exist: " + folder + "/" + atom + ".int")
                Q = float(lines[0].split()[-4])

            net_charge_group.append(Q)
        net_charges.append(net_charge_group)

    # GET CHARGE TRANSFER TERMS
    for k in range(len(net_charges)):
        for i in range(len(atom_list)):
            for j in range(i + 1, len(atom_list)):
                temp1.append((net_charges[k][i] * net_charges[k][j]) / ((distance_A_B(xyz_files[k], i + 1, j + 1))*1.8897259886))

    # ORGANIZE CHARGE TRANSFER ARRAY ORDER
    for j in range(int(n * (n - 1) / 2)):
        temp2.append([temp1[i] for i in range(j, len(temp1), int(n * (n - 1) / 2))])
    start = 0
    for atom_prop in temp2:
        charge_transfer_properties.append([atom_prop[i] for i in range(start, f)])

    # Isolate Vcl terms
    classical_properties = inter_properties[:len(charge_transfer_properties)]

    # OBTAIN POLARISATION TERMS AS Vpl = Vcl - Vct
    for i in range(len(classical_properties)):
        for j in range(len(classical_properties[i])):
            temp3.append(classical_properties[i][j] - charge_transfer_properties[i][j])

    # ORGANIZE POLARISATION ARRAY ORDER
    polarisation_properties = [temp3[i * f:(i + 1) * f] for i in range((len(temp3) + f - 1) // f)]

    return charge_transfer_properties, contributions_list_CT, polarisation_properties, contributions_list_PL

