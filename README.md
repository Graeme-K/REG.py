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
  
# A folder of systems
`-R` runs a full REG analysis on every system below a directory, each one exactly as
if you had cd'd into it and run the command there:

`python3 auto_reg.py -d /path/to/REG.py/src -R /path/to/folder_of_systems > sweep.log &`

or, from inside that folder, with the root left out:

`python3 auto_reg.py -d /path/to/REG.py/src -R > sweep.log &`

A *system* is a directory holding an `auto_reg.config` with numbered geometry folders
somewhere beneath it — the usual `SYSTEM/auto_reg.config` plus `SYSTEM/REG-IQA/1..11`
layout — or one holding at least two numbered folders that each contain a `.wfn`/`.wfx`.
The search does not descend into a directory it has claimed, so nesting is free: a
folder of ions, each holding ten molecules, gives the ten molecules of each ion and not
the ions themselves. Every other option applies to each system in turn, so `-R -f T -m`
runs IQF and REG_Multi on all of them. Each system's output is named after it, and the
sweep gathers every system's transfer bundle and config into one folder where the
command was run, with an overview of the whole set — see
[The collection folder](#the-collection-folder).

Check what a sweep will do before committing to it:

`python3 auto_reg.py -R --list-systems`

```
  40 SYSTEM(S) FOUND UNDER /.../2_REG_IQA_REDO_D3
  NAME        STEPS  CONFIG   PATH
  CLOADEFIVE     11  yes      Chlorine/CLOADEFIVE
  CLOBEN         11  yes      Chlorine/CLOBEN
  ...
```

Every folder and file a run creates is named after its own system, so the results of the
whole sweep can be collected in one place without forty `REG.xlsx` overwriting each
other:

```
Chlorine/CLOBEN/CLOBEN_REG_IQA_results/CLOBEN_REG_IQA_model_transfer.json
                                       CLOBEN_REG.xlsx
                                       CLOBEN_Energy.xlsx
                                       CLOBEN_REG_final_analysis.csv
                CLOBEN_REG_Multi_IQA_results/CLOBEN_REG_Multi.xlsx
```

Two systems that share a folder name — `Chlorine/BEN` and `Fluorine/BEN` — are named
`Chlorine_BEN` and `Fluorine_BEN`, so no two runs of a sweep can write the same file.

## The collection folder

A sweep leaves forty self-contained analyses in forty directories. What it does not
leave, by itself, is a *dataset*. So the sweep also makes one folder at the root it
swept — where the command was run, unless a root was given as the argument:

```
REG_sweep_collection/
  REG_collection_overview.json   the whole set described in one file
  REG_collection_overview.csv    the same, one row per system, for pandas or Excel
  bundles/CLOBEN_REG_IQA_model_transfer.json          verbatim copies
          CLOBEN_REG_Multi_IQA_model_transfer.json
          CLOPY_REG_IQA_model_transfer.json  ...
  configs/CLOBEN_auto_reg.config                      verbatim copies
          CLOPY_auto_reg.config  ...
```

The bundles are **copies, not summaries or links**: a transfer bundle is self-contained
by design, so the folder can be moved to a laptop and still answer everything about
every system. Each system's files carry its name.

Only the **newest bundle of each kind** is taken. A directory worked in for a while
holds several runs of the same analysis — `REG_IQA_results/` beside
`REG_IQA_results_n10/`, a level-agnostic `REG_Multi_results/` from before the same run
was remade at the IQA and IQF levels — and collecting all of them would leave the folder
ambiguous about which analysis it describes. The ones passed over stay exactly where
they are, and the overview lists them under `superseded_bundles` with their dates, so
nothing goes missing quietly.

`REG_collection_overview.json` is a new file that describes the set. Per system: where it
lives, whether the run succeeded and how long it took, its log, its collected bundle and
config, its fragment definitions, and a summary read from the bundle — atom count and
formula, number of steps and the control coordinate, the relative E_WFN curve in kJ/mol,
the WFN and IQA energy spans, the recovery RMSE, the segmentation, and the leading REG
terms of each segment with their Pearson R. With `-m`, the REG_Multi side adds L_max,
admitted pair count and the unresolved/residual fractions. The CSV carries the scalar
fields of that, one row per system, which is what a first look across forty systems
usually wants:

```
system,status,path,level,n_atoms,formula,n_steps,...,wfn_span_kj_mol,recovery_rmse_kj_mol,top_term,top_term_reg,...
CLOBEN,ok,Chlorine/CLOBEN,REG_IQA,13,C6H6Cl,11,...,1.857,0.0133,E_IQA_Intra(A)-cl13,3.004,...
LIBEN,ok,Lithium/LIBEN,REG_IQA,13,C6H6Li,11,...,135.349,0.0504,...
```

The overview is a shortlist and an index, never a replacement for a bundle: anything
deeper — per-pair curves, geometries, multipole moments — is read from the copies in
`bundles/`, which the overview points at by relative path.

`--collect-only` builds the folder from work that was done earlier without re-running
anything, so existing results can be turned into a dataset in seconds. With `-R` it
covers every system below the root; without it, the one directory given as the argument
or the current one:

```bash
reg-auto --collect-only                 # this system's REG_collection/, rebuilt from disk
reg-auto -R --collect-only /path/to/folder_of_systems
```

| option | |
|---|---|
| `-R`, `--recursive` | run on every system below the root given as the argument, or below the current directory |
| `--list-systems` | list the systems and the name each one's output would carry, then stop |
| `-j N`, `--jobs N` | analyse N systems at a time (default 1) |
| `--skip-existing` | leave alone any system that already has every results directory this run would produce, so an interrupted sweep can be restarted |
| `--collect-only` | analyse nothing; build the collection folder and overview from what the systems already hold |
| `--no-collect` | do not build the collection folder |
| `--collect-dir NAME` | name it something other than `REG_sweep_collection`; an absolute path puts it anywhere |
| `--no-prefix` | keep the usual unprefixed names in each system's own directory |
| `--prefix NAME` | name the output of a single run, analysed on its own, after NAME |

Each system is analysed in its own process, so one that fails — AIMAll run without
pairwise IQA, a step whose Gaussian output is missing — fails on its own and the sweep
carries on. What happened to each is printed at the end, and the full output of each
system is kept in `reg_batch_logs/SYSTEM_reg.log` at the root of the sweep. The command
exits non-zero if any system failed.

`reg_multipole.py` takes the same `-R`, `--list-systems` and `-j`, for running the
rank-resolved analysis alone over a folder of systems.

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

A run given a name with `--prefix`, and every run of a `-R` sweep, puts that name in
front of the results directory and of everything inside it —
`CLOBEN_REG_IQA_results/CLOBEN_REG.xlsx` — so the results of many systems can be
gathered together. Nothing else about the output changes.

## One folder to take away

Every run also gathers what is worth keeping into `REG_collection/`, beside the results
directories, so there is one folder to download instead of a file to find in each of
`REG_IQA_results/`, `REG_IQF_results/` and `REG_Multi_IQA_results/`:

```
CLOPY/REG_collection/CLOPY_REG_IQA_model_transfer.json
                     CLOPY_REG_IQF_model_transfer.json
                     CLOPY_REG_Multi_IQA_model_transfer.json
                     CLOPY_auto_reg.config
                     REG_collection_overview.json
                     REG_collection_overview.csv
```

The bundles are copies, named after the system's folder whether or not the run itself
was named, so folders downloaded from several systems never collide. The overview is the
same file a sweep writes, describing one system instead of forty — see
[The collection folder](#the-collection-folder) for what is in it.

It is **rebuilt from the directory each time**, not added to, so running IQF or
`reg_multipole.py` later leaves a folder holding everything the system has rather than a
stale one — and `reg-auto --collect-only` rebuilds it without running anything. Only the
newest bundle of each kind is taken; older runs of the same analysis stay where they are
and are listed in the overview as superseded. `--no-collect` turns it off; `--collect-dir NAME` (or an absolute path) puts
it elsewhere. A `-R` sweep collects centrally at its root instead, so its systems do not
each keep a copy of their own.

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
