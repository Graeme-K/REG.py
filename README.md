# REG.py
This repository is a continuation of the development of the Relative Energy Gradient (REG) method [[1]](#1) implemented in Python3.

The manual.pdf contains the information regarding the theory and the code implementation plus a full walk-through tutorial on a simple system.  

Please, report bugs and issues to fabio.falcioni@manchester.ac.uk

Please, cite [[1]](#1) if using this code for your studies/research. The code is under MIT license.
# Dependencies
- [AdjustText](https://github.com/Phlya/adjustText)
- [xlsxwriter](https://github.com/jmcnamara/XlsxWriter)
# Setup
The `setup_rdp.py` is a standalone script that can be run with specific options to find the suitable number of points on which to run the following REG-IQA analysis.
It uses the Ramer-Douglas-Peucker algorithm [[2]](#2) on the PES obtained from electronic structure calculations. 
Run the command
 `python3 setup_rdp.py -h (or --help)`
 for the usage. 
 
Minimum requirements:
 -  A txt file containing the energies at each step of an IRC or PES scan (easily obtained in GaussView).
# Usage
***Manual has been update with a full tutorial. Functions section needs to be updated***

To run a REG analysis (with IQA and DFT-D3):
- Save this repository in your machine.
- Copy the `auto_reg.py` script in the folder where each REG step has been saved. Note: Each REG step should be saved as a numbered folder and contain the gaussian single point energy output, gaussian wavefunction output and the atomic-files folder obtained with AIMAll [[3]](#3). 
- Enter the script, change the REG installation path and all the possible options as explained in the script.
- Run the command : 
`python3 auto_reg.py > reg.log &`
  
# Results
All results of the REG analysis will be saved in a folder called "SYSTEM_results" with SYSTEM being the chosen system name. 
Inside the folder different files for each type of energy will be found together with png images of the REG tables and various csv files for later data analysis.
The main file is called "REG.xlsx" which contains all the results together in one place.

Each run also writes `SYSTEM_results/SYSTEM_model_transfer.json`, a self-contained JSON
bundle carrying everything the analysis produced — per-step term tables, labelled REG
results, fragment definitions, geometries and integration quality — so a downstream tool
needs no other file from the results directory. An IQF run puts its fragment-level and
atom-level results in that same single file, over one shared copy of everything the two
levels have in common. See
[MODEL_TRANSFER_BUNDLE.md](MODEL_TRANSFER_BUNDLE.md) for its layout and options.

# REG_Multi (rank-resolved electrostatics)
`reg_multipole.py` runs a REG analysis over the ranks of the multipole expansion of
the IQA classical interaction — charge-charge, charge-dipole, dipole-dipole and so on —
between the fragments defined in the same `auto_reg.config` that REG-IQF uses. Run it
the same way as `auto_reg.py`, from the directory holding the numbered geometry folders:

`python3 reg_multipole.py -d /path/to/REG.py/src > reg_multi.log &`

or as part of a normal REG run, with `-m`:

`python3 auto_reg.py -d /path/to/REG.py/src -m > reg.log &`

It runs after the IQA analysis is complete and written, so a problem there (most
often AIMAll having been run without pairwise IQA) cannot cost you your REG results.

It follows the level of the run: `-f F` reports one set of multipole terms per atom
pair, `-f T` sums them into fragment channels the way IQF sums the IQA terms.

Fragment terms are built from atom-centred moments summed at the energy level. Passing
`--fragment-moments` adds the other view, where each fragment's atomic moments are
translated onto one centre and summed, giving "fragment dipole against fragment dipole".
The two are different decompositions of the same energy and are kept in separate files.

Pass `--ignore-fragments` to leave the partition out entirely and let the admission
criteria alone decide which pairs are described.

The multipole series is asymptotic, so not every atom pair may legitimately be described
this way. Pairs are admitted by a per-pair numerical convergence test applied at every
geometry on the path, and the exact IQA V_cl of every rejected pair is reported as an
unresolved channel alongside the rank terms. See [REG_MULTI.md](REG_MULTI.md) for the
method, the settings and the output files.

Run `python3 tests/test_multipole.py` to check the installation: it verifies the
interaction tensor against closed-form energies, the full series against the exact
Coulomb energy of two charge clusters, and the admission gates against a synthetic
path whose convergence behaviour is known atom by atom.

## References
<a id="1">[1]</a> 
Thacker, Joseph CR, and Paul LA Popelier. "The ANANKE relative energy gradient (REG) method to automate IQA analysis over configurational change." Theoretical chemistry accounts 136.7 (2017): 1-13.

<a id="2">[2]</a> 
Douglas, D.H. and T.K. Peucker, Algorithms for the reduction of the number of points required to represent a digitized line or its caricature. Cartographica: the international journal for geographic information and geovisualization, 1973. 10(2): p. 112-122.

<a id="3">[3]</a> 
 AIMAll (Version 19.10.12), Todd A. Keith, TK Gristmill Software, Overland Park KS, USA, 2019 (aim.tkgristmill.com)
