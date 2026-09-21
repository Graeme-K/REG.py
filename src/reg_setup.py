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

    # A geometry point is a folder holding a wavefunction, and its Gaussian output
    # is the one in that same folder.  Pairing them folder by folder rather than
    # collecting two flat lists and zipping them matters: a stray .log anywhere in
    # the tree — the reg.log of a previous run, a batch log — used to join the
    # second list and shift every wavefunction onto the wrong output, which is
    # silent and gives wrong energies rather than an error.
    for walk_root, _, files in os.walk(root):
        wf_names = sorted(name for name in files
                          if name.endswith('.wfn') or name.endswith('.wfx'))
        if not wf_names:
            continue
        g16_names = sorted(name for name in files
                           if (name.endswith('.out') or name.endswith('.log')
                               or name.endswith('.gaussianoutput'))
                           and not (name.startswith('dft-d3') or name.startswith('slurm')))
        if not g16_names:
            raise FileNotFoundError(
                'No Gaussian output found in ' + walk_root + ', which holds '
                + wf_names[0] + '. Each geometry point needs its single point '
                'output (.out/.log) beside its wavefunction')
        for wf_index, wf_name in enumerate(wf_names):
            wf_file.append(os.path.join(walk_root, wf_name))
            reg_folders.append(os.path.basename(os.path.normpath(walk_root)))
            reg_folder_list.append(walk_root)
            g16_file.append(os.path.join(
                walk_root, g16_names[wf_index] if wf_index < len(g16_names) else g16_names[0]))
            if wf_name.endswith('.wfx'):
                wfx = True

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


# ---------------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------------
# A batch run analyses many systems that all produce the same file names, so
# every name a run creates is put behind the system's own token.  The whole
# results tree of a folder of systems can then be collected into one place —
# or opened side by side in the explorer — without CLOBEN's REG.xlsx and
# CLOPY's REG.xlsx being the same file name.

_PREFIX_UNSAFE = re.compile(r'[^A-Za-z0-9._-]+')


def sanitise_prefix(name):
    """Turn a system or folder name into a token safe to put in front of a file name.

    Anything that is not alphanumeric, dot, dash or underscore becomes '_', and
    leading/trailing separators are dropped, so 'Cl-pi run 2' gives 'Cl-pi_run_2'.
    Returns '' for a name with nothing usable in it, which every caller reads as
    "do not prefix".
    """
    if not name:
        return ''
    return _PREFIX_UNSAFE.sub('_', str(name).strip()).strip('_.')


def prefixed(name, prefix):
    """'<prefix>_<name>', leaving a name that already carries the prefix alone.

    Idempotent on purpose: the results directory and the transfer bundle are
    named with the prefix as they are created, and the sweep over the finished
    directory must not turn them into CLOBEN_CLOBEN_....
    """
    if not prefix or not name:
        return name
    if name == prefix or name.startswith(prefix + '_'):
        return name
    return prefix + '_' + name


def apply_output_prefix(directory, prefix, verbose=False):
    """Put *prefix* in front of every entry of a finished results directory.

    Renaming afterwards rather than threading the prefix through each of the
    forty-odd writes keeps the two in step by construction: a file added to the
    analysis later is prefixed without anyone remembering to do it.  It is safe
    because nothing in a REG run reads back what it wrote — the directory is
    output only.

    A rename that fails (an open handle, a read-only filesystem) is reported and
    skipped: a naming convenience must not cost a completed analysis.  Returns
    the list of (old_name, new_name) pairs actually renamed.
    """
    if not prefix or not directory or not os.path.isdir(directory):
        return []

    renamed = []
    for name in sorted(os.listdir(directory)):
        new_name = prefixed(name, prefix)
        if new_name == name:
            continue
        try:
            os.replace(os.path.join(directory, name), os.path.join(directory, new_name))
        except OSError as rename_error:
            print('WARNING: could not rename {a} to {b} — {e}'.format(
                a=name, b=new_name, e=rename_error))
            continue
        renamed.append((name, new_name))

    if verbose and renamed:
        print('  {n} file(s) in {d} renamed to start with "{p}_"'.format(
            n=len(renamed), d=os.path.basename(directory), p=prefix))
    return renamed


# ---------------------------------------------------------------------------
# Finding the systems in a folder of systems
# ---------------------------------------------------------------------------

CONFIG_NAME = 'auto_reg.config'

# Directories that are part of a system rather than a system of their own.
_SKIP_DIR_SUFFIXES = ('_atomicfiles', '_results')
# 'reg_batch_logs' and 'REG_sweep_collection' are what a sweep itself leaves at
# the root it was run from; neither holds a system.
_SKIP_DIR_NAMES = {'__pycache__', '.git', 'reg_batch_logs', 'REG_sweep_collection'}

# Names that say what the folder holds rather than which system it belongs to.
# A system root found under one of these takes its name from the folder above,
# so a tree of CLOBEN/REG-IQA/1..11 is called CLOBEN and not REG-IQA.
_GENERIC_DIR_NAME = re.compile(
    r'^(reg([-_ ]?iqa|[-_ ]?iqf|[-_ ]?multi)?|steps?|points?|geom(etries|etry)?|'
    r'structures?|scan|irc|calc(ulations?)?|run|runs|data)$', re.IGNORECASE)


def _is_skipped_dir(name):
    return (name.startswith('.') or name in _SKIP_DIR_NAMES
            or name.endswith(_SKIP_DIR_SUFFIXES))


def _holds_wavefunction(path):
    """True when the directory holds a .wfn/.wfx directly, i.e. it is a geometry point."""
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_file() and (entry.name.endswith('.wfn') or entry.name.endswith('.wfx')):
                    return True
    except OSError:
        pass
    return False


def _child_dirs(path):
    try:
        with os.scandir(path) as entries:
            return sorted(entry.name for entry in entries
                          if entry.is_dir() and not _is_skipped_dir(entry.name))
    except OSError:
        return []


def _count_step_folders(path):
    """Geometry points directly inside *path*."""
    return sum(1 for name in _child_dirs(path)
               if _holds_wavefunction(os.path.join(path, name)))


def _subtree_step_count(path, depth):
    """Geometry points inside *path* or below it, stopping at *depth* levels down."""
    total = _count_step_folders(path)
    if depth > 0:
        for name in _child_dirs(path):
            child = os.path.join(path, name)
            if not _holds_wavefunction(child):
                total += _subtree_step_count(child, depth - 1)
    return total


def _system_name(path, root):
    """The name a system root is known by: its folder, or the nearest folder above
    it that names a system rather than describing its contents."""
    candidate = os.path.abspath(path)
    root = os.path.abspath(root)
    while (_GENERIC_DIR_NAME.match(os.path.basename(candidate))
           and os.path.dirname(candidate).startswith(root)
           and os.path.dirname(candidate) != candidate
           and candidate != root):
        candidate = os.path.dirname(candidate)
    return os.path.basename(candidate)


def find_reg_systems(root='.', min_steps=2, max_depth=6):
    """Find every directory below *root* that a REG analysis can be run in.

    A directory is taken to be a system root when it holds at least *min_steps*
    geometry points directly, or when it holds an auto_reg.config with that many
    points somewhere beneath it — which is the layout in the wild, where the
    config sits with the system and the numbered folders sit one level down in a
    REG-IQA/ folder.  The search does not descend into a directory it has
    claimed, so a system is never analysed twice, once as itself and once as
    part of an ancestor.

    Returns a list of dicts ordered by path, each with 'path' (absolute),
    'name' (what the outputs are prefixed with), 'steps' and 'config'.
    """
    root = os.path.abspath(root)
    found = []

    def scan(directory, depth):
        direct = _count_step_folders(directory)
        has_config = os.path.isfile(os.path.join(directory, CONFIG_NAME))
        steps = direct
        if steps < min_steps and has_config:
            steps = _subtree_step_count(directory, max_depth - depth)
        if steps >= min_steps:
            found.append({
                'path': directory,
                'name': sanitise_prefix(_system_name(directory, root)),
                'steps': steps,
                'config': has_config,
            })
            return  # claimed — its own numbered folders are not separate systems
        if depth >= max_depth:
            return
        for name in _child_dirs(directory):
            child = os.path.join(directory, name)
            if not _holds_wavefunction(child):
                scan(child, depth + 1)

    scan(root, 0)
    found.sort(key=lambda entry: entry['path'])
    return found
