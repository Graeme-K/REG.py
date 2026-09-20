"""
reg_setup.py

Shared setup for the REG drivers: reading auto_reg.config, finding the geometry
points on disk and ordering them along the control coordinate.

auto_reg.py and reg_multipole.py both start from the same directory layout — one
numbered folder per geometry point, each holding a wavefunction, a Gaussian
output and an AIMAll _atomicfiles folder — and both read fragment definitions
from the same auto_reg.config.  Keeping that in one place is what lets a
REG_Multi run describe exactly the fragments an IQF run describes, rather than a
second definition that has to be kept in step by hand.

coded for the REG.py package
"""

import os
import re


# ---------------------------------------------------------------------------
# auto_reg.config
# ---------------------------------------------------------------------------

def read_fragment_config(path):
    """Parse an auto_reg.config fragment definition file.

    Accepts both spellings seen in the wild:
        FRAG ID 1 <backbone>   FRAG ATOMS [1,2,3]
        FRAG ID backbone       FRAG ATOMS [1,2,3]

    Returns (names, atom_lists) or (None, None) when the file is absent or
    holds no usable definition.  Atom numbers are kept 1-based, as written.
    """
    if not path or not os.path.exists(path):
        return None, None

    with open(path, encoding='utf-8', errors='ignore') as config_handle:
        config_text = config_handle.read()

    names = re.findall(r'FRAG\s*ID\s*\d+\s*<(.*?)>', config_text)
    if not names:
        names = re.findall(r'FRAG\s*ID\s*(?:\d+\s*)?<?([^<>\r\n]+?)>?\s*(?=FRAG\s*ATOMS)',
                           config_text, flags=re.IGNORECASE)
        names = [n.strip() for n in names]

    atom_lists = []
    for atoms_raw in re.findall(r'FRAG\s*ATOMS\s*\[([\d,\s]+)\]', config_text):
        atom_lists.append([int(a) for a in atoms_raw.split(',') if a.strip()])

    if not atom_lists:
        return None, None

    # Tolerate a missing or short name list rather than losing the definitions.
    if len(names) < len(atom_lists):
        names = list(names) + ['Frag_' + str(i + 1) for i in range(len(names), len(atom_lists))]

    return names[:len(atom_lists)], atom_lists


# Keys understood in a "MULTI <KEY> <value>" line of auto_reg.config.  These are
# the REG_Multi settings that belong with the system rather than with the run,
# so that the same command reproduces the same analysis from the directory
# alone.  A command-line flag overrides whatever the file says.
_MULTI_KEYS = {
    'LMAX': 'lmax',
    'SCOPE': 'scope',
    'WITHIN': 'within',
    'TOLERANCE': 'tolerance',
    'FLOOR': 'floor',
    'INCREMENT_RANKS': 'increment_ranks',
    'RADII': 'radii',
    'TOPOLOGY': 'topology',
    'IGNORE_FRAGMENTS': 'ignore_fragments',
    'FRAGMENT_MOMENTS': 'fragment_moments',
    'FRAGMENT_CENTRE': 'fragment_centre',
    'FRAGMENT_CENTER': 'fragment_centre',
}


def read_multipole_config(path):
    """Parse the optional 'MULTI <KEY> <value>' lines of auto_reg.config.

    Example::

        MULTI LMAX 5
        MULTI SCOPE inter
        MULTI WITHIN <Gua(pi)>, <NH2(group)>
        MULTI TOLERANCE 0.05
        MULTI FRAGMENT_MOMENTS on
        MULTI FRAGMENT_CENTRE centroid

    Returns a dict of the options that were present; unknown keys are ignored so
    that an older config, or one carrying settings for a future version, still
    reads.  Fragment names in WITHIN may be written with or without <angle
    brackets>.
    """
    options = {}
    if not path or not os.path.exists(path):
        return options

    with open(path, encoding='utf-8', errors='ignore') as config_handle:
        for line in config_handle:
            match = re.match(r'\s*MULTI\s+([A-Za-z_]+)\s+(.+?)\s*$', line)
            if not match:
                continue
            key = _MULTI_KEYS.get(match.group(1).upper().replace('-', '_'))
            if key is None:
                continue
            raw = match.group(2).strip()
            if key == 'within':
                names = re.findall(r'<([^<>]+)>', raw)
                if not names:
                    names = [n.strip() for n in raw.split(',') if n.strip()]
                options[key] = names
            elif key in ('lmax', 'increment_ranks'):
                try:
                    options[key] = int(raw)
                except ValueError:
                    pass
            elif key in ('tolerance', 'floor'):
                try:
                    options[key] = float(raw)
                except ValueError:
                    pass
            elif key in ('topology', 'fragment_moments', 'ignore_fragments'):
                options[key] = raw.strip().lower() not in ('off', 'false', 'no', '0')
            else:
                options[key] = raw.strip().strip('<>').lower()
    return options


# ---------------------------------------------------------------------------
# Finding and ordering the geometry points
# ---------------------------------------------------------------------------

def folder_value(name):
    """Numeric position of a REG folder along the control coordinate.

    Folder names carry the coordinate: '1', '2.15', '1_75' (= 1.75) or
    'neg_30' (= -30).  A name with no number sorts last.
    """
    name = re.sub(r'^neg_?', '-', name)
    name = re.sub(r'(?<=\d)_', '.', name)
    match = re.search(r'-?\d+\.?\d*', name)
    return float(match.group()) if match else float('inf')


def discover_reg_points(root='.'):
    """Find every geometry point below *root* and order it along the path.

    Returns a dict with parallel lists 'reg_folders' (folder names), 'reg_roots'
    (paths), 'wf_files', 'g16_files' and the flag 'wfx' saying which wavefunction
    format was found.

    Points are ordered by the number in their folder name.  When the numbers look
    like angles spanning a wide range, the sequence is rolled to start after the
    largest gap, which stops a dihedral scan being split at the -180/180
    boundary — the same treatment setup_rdp.py gives a scan.
    """
    wf_file = []
    g16_file = []
    reg_folders = []
    reg_folder_list = []
    wfx = False

    for walk_root, _, files in os.walk(root):
        for name in files:
            if name.endswith('.wfn') or name.endswith('.wfx'):
                wf_file.append(os.path.join(walk_root, name))
                reg_folders.append(walk_root.split('/')[-1])
                reg_folder_list.append(walk_root)
                if name.endswith('.wfx'):
                    wfx = True
            elif ((name.endswith('.out') or name.endswith('.log')
                   or name.endswith('.gaussianoutput'))
                  and not (name.startswith('dft-d3') or name.startswith('slurm'))):
                g16_file.append(os.path.join(walk_root, name))

    ordered = sorted(zip(reg_folders, reg_folder_list, wf_file, g16_file),
                     key=lambda entry: folder_value(entry[0]))

    values = [folder_value(entry[0]) for entry in ordered]
    if values and all(-360 <= v <= 360 for v in values) and (max(values) - min(values)) > 90:
        gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        wrap_gap = values[0] + 360 - values[-1]
        all_gaps = gaps + [wrap_gap]
        max_gap_idx = all_gaps.index(max(all_gaps))
        if max_gap_idx < len(gaps):   # the wrap-around gap is not the largest — roll
            ordered = ordered[max_gap_idx + 1:] + ordered[:max_gap_idx + 1]

    if not ordered:
        return {'reg_folders': [], 'reg_roots': [], 'wf_files': [], 'g16_files': [], 'wfx': wfx}

    reg_folders, reg_roots, wf_files, g16_files = (list(col) for col in zip(*ordered))
    return {
        'reg_folders': reg_folders,
        'reg_roots': reg_roots,
        'wf_files': wf_files,
        'g16_files': g16_files,
        'wfx': wfx,
    }


def load_xyz_structure(path):
    """Read an XYZ file into {'path', 'n_atoms', 'atoms': [{symbol,x,y,z}, ...]}.

    Returns None when the file is missing or too short to parse.  Blank lines are
    skipped first, because get_xyz_file() writes a leading one.
    """
    if not path or not os.path.exists(path):
        return None

    with open(path, 'r', encoding='utf-8', errors='ignore') as xyz_handle:
        xyz_lines = [line.strip() for line in xyz_handle if line.strip()]

    if len(xyz_lines) < 2:
        return None

    try:
        n_atoms = int(xyz_lines[0])
    except ValueError:
        return None

    atoms = []
    for line in xyz_lines[1:1 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        atoms.append({
            'symbol': parts[0],
            'x': float(parts[1]),
            'y': float(parts[2]),
            'z': float(parts[3]),
        })

    return {'path': path, 'n_atoms': n_atoms, 'atoms': atoms}
