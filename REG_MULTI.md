# REG_Multi — rank-resolved multipolar electrostatics

`reg_multipole.py` runs a REG analysis over the *ranks* of the multipole
expansion of the classical electrostatic interaction, between fragments defined
in the same `auto_reg.config` that REG-IQF uses.

IQA gives one number per atom pair, `V_cl(A,B)`. Expanding it over the two
atoms' nucleus-centred spherical multipole moments splits that number into a
series — charge–charge, charge–dipole, dipole–dipole, and so on — and summing
the series over the atom pairs of two fragments gives the same split for a
fragment–fragment interaction. Running REG over those rank terms says which
*kind* of electrostatics drives a process, not merely that electrostatics does.

The earlier prototype did one ion against one molecule, with the molecular side
taken from recovered molecular moments. This version works between fragments,
with every term built from nucleus-centred atomic moments, so fragment and atom
levels are directly comparable and nothing depends on a separate molecular
expansion.

## What you need

* The same directory layout `auto_reg.py` expects: one numbered folder per
  geometry point, each with a `.wfn`/`.wfx`, a Gaussian output and an AIMAll
  `_atomicfiles` folder.
* **AIMAll run with IQA**, because the convergence gate compares the series
  against the exact `V_cl(A,B)`. Without it the run stops and says so; the
  `--skip-convergence-gate` escape hatch exists but produces a decomposition of
  unknown accuracy, and every output it writes says that too.
* Moments as high as you can get. AIMAll writes to `l = 5` by default, giving
  `l_tot = l_A + l_B ≤ 10`. The run reports the rank it actually found and caps
  `L_max` there.

## Quick start

```bash
# from the directory holding the numbered geometry folders
python3 /path/to/REG.py/src/reg_multipole.py -d /path/to/REG.py/src
```

or, if the package is installed, `reg-multi -d /path/to/REG.py/src`. As with
`auto_reg.py`, the script can also be copied into the data directory and pointed
at its installation with `-d`.

It can also run as part of a normal REG-IQA or REG-IQF job, which is the usual
way to get both analyses from one command:

```bash
python3 auto_reg.py -d /path/to/REG.py/src -m
python3 auto_reg.py -d /path/to/REG.py/src -m --multi-options='--scope all --lmax 4'
```

`-m` / `--reg-multi` turns it on, or set `REG_MULTI = True` in
`default_settings.py`. It runs last, after everything the IQA analysis owns is on
disk, so a REG_Multi problem — most often AIMAll having been run without pairwise
IQA — cannot cost you your REG results: it reports the failure and leaves the
rest of the run intact. Settings are better placed in `auto_reg.config` as `MULTI`
lines than in `--multi-options`, so that the same command reproduces the same
analysis from the directory alone.

Results land in `REG_Multi_IQA_results/` or `REG_Multi_IQF_results/` according to
the level of the run, beside `REG_IQA_results/` and `REG_IQF_results/`, so
running IQF after IQA never overwrites what the IQA run produced. A standalone
`reg-multi` writes to `REG_Multi_results/` unless `-o/--results-dir` says
otherwise.

## Fragments and scope

Fragment definitions are read from `auto_reg.config`, unchanged:

```
FRAG ID 1 <Gua(pi)>
FRAG ATOMS [1,2,3,4,5,6,7,8,9,10]

FRAG ID 2 <F->
FRAG ATOMS [17]
```

| scope | what is analysed | grouped as |
|---|---|---|
| `inter` (default with fragments) | every interfragment atom pair | one group per fragment pair, `F|G` |
| `intra` | every atom pair inside a fragment | one group per fragment, `F(intra)` |
| `all` | both of the above | both sets of groups |
| `pairs` | every atom pair in the system | one group per atom pair, `a1-b2` — the atom-wise analysis; fragment definitions are ignored |
| — (no fragments defined) | every atom pair in the system | one group, `System` |

`--within NAME[,NAME]` restricts an intra-fragment analysis to the fragments
named. With no fragment definitions at all, every pair in the system is used.

### Leaving the fragments out

A fragment definition is a claim about which atoms belong together. The admission
gates make a separate claim, about which pairs the multipole series can describe.
`--ignore-fragments` (or `MULTI IGNORE_FRAGMENTS on`) drops the first so the
second decides alone: the run behaves as though `auto_reg.config` defined no
fragments, every atom pair in the system is a candidate, and what gets described
is whatever the criteria admit.

```bash
reg-multi -d /path/to/REG.py/src -i               # one system-wide group
reg-multi -d /path/to/REG.py/src -i --scope pairs # one group per atom pair
reg-auto  -d /path/to/REG.py/src -m -i            # same, from a normal REG run
```

`-i` is the short form, on both `reg-multi` and `reg-auto`. On `reg-auto` it
affects only the REG_Multi analysis; the IQA/IQF analysis itself is untouched.

The default grouping is then a single `System` channel, which is the compact form
— one set of 36 rank terms for the whole system, answering "which multipole
channel drives this step" without reference to any partition. Add `--scope pairs`
to keep each atom pair separate instead.

`--fragment-moments` and `--within` both need fragments and are refused with
`--ignore-fragments`, since neither means anything without a partition. Note that
`--scope pairs` on its own already selects by the gates alone; what
`--ignore-fragments` adds is that the *energy* groupings stop using fragments
too.

Run from `auto_reg.py`, the scope follows the level of the run:

* `-f T -m` (IQF) defaults to `all`, so atom pairs are summed into fragment
  channels exactly the way IQF sums the IQA terms: pairs inside a fragment into
  that fragment's own channel, pairs between fragments into the fragment-pair
  channel.
* `-f F -m` (IQA) defaults to `pairs`, reporting one set of multipole terms per
  atom pair — `o1-o4 charge-charge [0,0]` and so on. Fragment definitions in the
  config are ignored, for the same reason `auto_reg` ignores them without `-f T`.

An explicit scope in `auto_reg.config` or in `--multi-options` is left alone
either way.

REG_Multi settings can also live in `auto_reg.config`, so the same command
reproduces the same analysis from the directory alone. Command-line flags
override the file:

```
MULTI LMAX 5
MULTI SCOPE inter
MULTI WITHIN <Gua(pi)>
MULTI TOLERANCE 0.05
MULTI FLOOR 0.05
MULTI RADII beta
MULTI TOPOLOGY on
MULTI FRAGMENT_MOMENTS on
MULTI FRAGMENT_CENTRE centroid
MULTI IGNORE_FRAGMENTS off
```

## Which pairs the analysis may describe

The multipole series is asymptotic. It is exact only while the two basins'
convergence spheres stay apart, and for close pairs it can look settled at low
rank and then turn around. "Intermolecular means far enough" does not hold: a
hydrogen bond puts the donor H and acceptor O 1.7–2.0 Å apart, shorter than many
intramolecular 1,3 distances — and those are exactly the pairs a fragment
analysis is most interested in. So every candidate pair faces the same five
gates, interfragment or not.

**1. Topology** — cheap, not authoritative. 1,2 and 1,3 pairs are rejected
outright; 1,4 and beyond are candidates. Bonds are taken from covalent radii, as
the union over the whole path, *not* from AIM bond critical points: a hydrogen
bond has a BCP but its donor and acceptor are two separate basins, and letting
AIM connectivity reject those pairs would throw away the signal. Disable with
`--no-topology-filter` and let gate 3 decide alone.

**2. Geometry** — cheap, not authoritative. `R_AB ≥ R_A + R_B`. With
`--radii beta` (the default) `R_A` is the AIMAll β-sphere radius, which is
inscribed inside the basin, so this is a **strict lower bound**: failing it
proves the basins overlap, passing it proves nothing. `--radii vdw` uses
tabulated van der Waals radii as a stand-in for the 0.001 au envelope — stricter
and closer to the condition actually wanted, but an estimate rather than a
measurement of these basins.

**3. Numerical convergence** — the authoritative one. Partial sums `Σ_L V_L` are
built up to `L_max` and two things are required at every geometry:

* the truncation residual against the exact IQA `V_cl` is within tolerance, and
* the rank increments `|ΔV_L|` are not growing across the complete shells.

The second condition is what catches divergence. Only shells with
`l_tot ≤ L_max` are tested: above that a shell is missing most of the blocks its
rank would need (`l_tot = 2·L_max` holds only the single `(L_max, L_max)` block),
so it is small for a bookkeeping reason rather than a physical one. This is why
the rank AIMAll actually wrote matters — at a low ceiling there is little series
left to test, and the run warns you.

**4. Path-wide admission.** A pair is admitted only if it passes gate 3 at
*every* geometry, at one fixed `L_max`. A pair that converges at the reactant and
fails near the transition state produces a kink that would dominate the gradient
and the R², and it would look like chemistry.

**5. Accounting.** Every rejected pair's exact IQA `V_cl` is reported as a single
unresolved channel, per fragment pair and overall, with the fraction of the total
change in `V_cl` that lives there. If half the electrostatic change sits in
excluded 1,2 pairs, the rank decomposition is describing the other half and the
report says so.

### Tolerance

The residual test is relative to *variation*, not to absolute energy. REG fits a
gradient, so what matters is that the truncation error does not vary across the
segment. The default (`--tolerance 0.05`) requires the residual's peak-to-peak
variation to stay under 5% of the pair's own peak-to-peak `V_cl` on every
segment. An absolute kJ/mol threshold would wrongly reject large, flat,
well-converged interfragment terms and wrongly accept small ones whose error
wanders.

A pair whose `V_cl` barely moves has no gradient worth resolving, so below
`--floor` (default 0.05 kJ/mol) the residual is judged in absolute terms instead.
That floor also sets the scale below which a growing increment is treated as
numerical noise rather than divergence.

For scale: Solano *et al.* report absolute errors at `l_tot = 10` of 0.03–0.3
kJ/mol for 1,3 and 1,4 pairs, negligible against typical REG term variations,
against 4–18 kJ/mol for 1,2 pairs — the size of the signal you would be trying to
interpret.

## Energies

All energy files are written **one row per series, one column per geometry
point**. The other orientation puts every channel in its own column, which on a
238-atom system in `pairs` scope is 169,226 columns — past Excel's limit of
16,384, and unreadable besides.

Three files, for three questions.

**`energy_recovery.csv` — was the electrostatics recovered?**

Two curves per grouping, and nothing else:

| row | meaning |
|---|---|
| `... Vcl_IQA (included pairs)` | exact IQA `V_cl`, summed over the admitted pairs |
| `... Vcl_multipole (included pairs)` | the multipole series, over those same pairs |

Both sums run over the *same* pairs — the admitted ones. No residual and no
excluded channel appear in this file at all, because the question it answers is
whether the series reproduces what it claims to describe, and adding the part it
does not describe to either side would make the curves agree or disagree for the
wrong reason. Rows are given for the total, for each fragment grouping, and for
each fragment-centred pair when `--fragment-moments` is on.

**`energy_by_type.csv` — which multipole did the work?**

The same energy summed into fragment channels and split by type:
`F|G charge-charge [0,0]`, `F|G charge-dipole [0,1]`, `F|G dipole-charge [1,0]`
and so on, with a `total (included pairs)` row per channel. The fragment
groupings are read from `auto_reg.config` **whatever the scope was** (unless
`--ignore-fragments` is on, when there is a single `System` channel), so an
atom-wise run still gets a fragment-level breakdown — it is a reporting sum over
the atom-pair terms and does not change the analysis.

**`energy_by_pair.csv` — the multipole energy of every pair.**

One row per admitted atom pair, its total multipole energy at each step. No IQA
columns: this one is the multipole analysis on its own. The rank-by-rank detail
for each pair is in `multipole_terms.csv`.

**`REG_Multi_energy.csv` — the full balance**, for accounting rather than for
plotting recovery: per grouping and in total, `Vcl_IQA (included pairs)`,
`Vcl_multipole`, `Vcl_residual (included pairs)`, `Vcl_IQA (excluded pairs)`,
`Vxc_IQA (all pairs)`, `Einter_IQA (all pairs)`, and `E_WFN`.

The REG term tables hold **multipole terms only** — `charge-charge [0,0]`,
`dipole-quadrupole [1,2]` and the rest. The truncation residual and the
unresolved channel are not terms and are not ranked beside them; they are
bookkeeping, and they live on the energies (and in `excluded_accounting.csv`),
where the balance still closes exactly at every step:

```
Vcl_IQA(included) = Vcl_multipole + Vcl_residual
Vcl_IQA(all)      = Vcl_IQA(included) + Vcl_IQA(excluded)
```

`V_xc` and `E_inter` are context, not part of the decomposition. The multipole
series describes `V_cl` only, and "how well is `V_cl` reproduced" is a different
question from "how much of this interaction is classical at all" — a fragment
pair whose `V_cl` is reproduced to 99.9% may still be dominated by `V_xc`. The
report's recovery table puts both in front of you:

```
  Group                          <|V_cl|>  <|V_multi|>   <|resid|>  recovered       <V_xc>    <E_inter>
  Acceptor|Donor                    0.108        0.108       0.000     99.95%      -47.259      -47.367
```

where `recovered = 1 - mean|residual| / mean|V_cl|` over the admitted pairs.

## How terms are named

Terms lead with the multipoles they couple and keep the rank indices beside them:

```
Gua(pi)|F- charge-dipole [0,1]
Gua(pi)|F- dipole-dipole [1,1]
Gua(pi)|F- quadrupole-octupole [2,3]
```

Order is preserved and matters: `charge-dipole [0,1]` is the first group's charge
against the second's dipole, which is a different term from `dipole-charge [1,0]`.
Ranks above the named multipoles (l > 5) print as `2^l-pole`. Shell terms are
labelled by their total rank and its distance dependence, `l_tot=3 (R^-4)`.
Fragment-centred terms carry a `frag:` prefix throughout, so the two views can
never collide in one table.

## Two ways to build a fragment term

There are two different things "the multipole interaction between two fragments"
can mean, and REG_Multi can produce both. They are never mixed in one table.

**Atom-centred (default).** Every atom keeps its own nucleus-centred moments and
the sum is taken at the *energy* level:

```
V[la,lb](F|G) = Σ_{A∈F} Σ_{B∈G}  Q_la(A) · T(R_AB) · Q_lb(B)
```

So `V[1,1]` is the sum of all atom–atom dipole–dipole interactions. Each atom
pair is gated individually, and a pair that fails goes to the unresolved channel
while the rest of the fragment pair is still described.

**Fragment-centred** (`--fragment-moments`, or `MULTI FRAGMENT_MOMENTS on`). Every
atom's moments are translated onto one centre per fragment and summed at the
*moment* level, so each fragment carries a single set of moments and a fragment
pair gets one expansion:

```
V[la,lb](F|G) = Q_la(F) · T(R_FG) · Q_lb(G)
```

Now `V[1,1]` is the interaction of the two fragment dipoles — "the dipole–dipole
interaction between these fragments drives this step".

### They are not the same decomposition

Both converge to the same energy where both converge, but they attribute it to
different ranks. Two neutral 3-atom fragments 4.53 Å apart, from the test fixture:

| l_tot | atom-centred | fragment-centred |
|---|---|---|
| 0 | **−1.7905** | −0.0000 |
| 1 | +1.0920 | −0.0000 |
| 2 | +0.1438 | **−0.2302** |
| 3 | −0.4012 | −0.3675 |
| total | −0.9908 | −0.9894 |

(exact: −0.9924 kJ/mol.) At fragment level the monopole terms vanish identically
because each fragment is neutral; at atom level large charge–charge terms sit
against each other and mostly cancel. Neither is wrong, but a `V[1,1]` from one
is not the same quantity as a `V[1,1]` from the other, so they go in separate
files and share no term name.

### The gate is stricter for fragment centres

A fragment-centred expansion needs its convergence sphere to enclose the whole
fragment, so the geometric condition becomes

```
R_FG  ≥  extent(F) + extent(G),    extent(F) = max over A∈F of (|r_A − c_F| + R_A)
```

which grows with fragment size however innocuous the individual atom pairs look.
The whole fragment pair is then admitted or rejected together — there is no
partial credit, because the expansion that would be credited is the one that does
not converge. The same convergence test (residual + increments, at every
geometry) then runs against the exact IQA `V_cl` summed over the fragment pair.

The expansion centre is the fragment centroid by default, which is close to the
centre that minimises the fragment's radial extent — and the radial extent is what
limits whether the expansion converges at all. `--fragment-centre mass` and
`nuclear-charge` are available. The centre is recomputed at every geometry from
the fragment's own atoms, so it travels with the fragment.

### Measuring cancellation

`fragment_moment_cancellation.csv` answers "did the atomic contributions cancel?"
directly, per fragment and per rank:

```
ratio_l  =  |Σ_A Q_l(A)|  /  Σ_A |Q_l(A)|
```

with both sides measured **after** translation onto the fragment centre, so the
comparison is like with like. (Comparing a fragment moment against the atoms'
own nucleus-centred moments would report the re-expansion — an atom a distance d
from the centre contributes q·d^l to rank l — as though it were structure, and the
ratio would not even be bounded.) Measured this way it lies in [0, 1] by the
triangle inequality: 0 is total cancellation, 1 is every contribution pulling the
same way. A neutral fragment has `ratio_0 = 0` exactly.

`fragment_moment_comparison.csv` puts the two views side by side per `l_tot`
shell, with each one's mean, span and REG value per segment — which is where a
net fragment interaction that is weak *because its atomic contributions cancel*
becomes visible, as against one that is weak because nothing is happening.

## Which ranking is which

With `reg-auto -f T -m`, the headline ranking in `REG_Multi_final_analysis.csv`
(and the `REG_final` sheet) is built from the **atom-centred** terms: each
fragment-pair term is the sum over its atom pairs of that rank's contribution,
every pair gated individually. Fragment-centred moments are not involved.

The fragment-centred view is a separate analysis, and it never joins that
ranking:

| file | which view |
|---|---|
| `REG_Multi_final_analysis.csv`, `REG_Multi_seg_N.csv` | atom-centred, summed into fragment channels |
| `REG_Multi_fragment_seg_N.csv` | fragment-centred, `frag:`-prefixed terms |
| `energy_by_type.csv` | atom-centred, summed into fragment channels |
| `fragment_moment_terms.csv` | fragment-centred |

Every fragment-centred term carries a `frag:` prefix, so the two can always be
told apart, and only `--fragment-moments` produces any of them at all.

The translation onto fragment centres is done here, through the addition theorem
for regular solid harmonics — ichor is not used, which is why it works to any
`L_max` rather than stopping at octupole.

## Moment coefficients

Two files, written whenever the moments are read — which is every run:

* `atomic_moments.csv` — every atom's `Q[l,k]` at every step, as read from the
  `.int` files, plus its β-sphere radius.
* `fragment_moments.csv` — each fragment's `Q[l,k]` about its own centre, with
  the centre coordinates and the fragment's radial extent. Written whenever
  fragments are defined, not only under `--fragment-moments`: these are the
  coefficients a fragment-level discussion quotes, and the flag only controls
  whether fragment *pairs* are expanded against each other.
  `fragment_moment_cancellation.csv` comes with it.

With `--ignore-fragments` neither fragment file is written, since there are no
fragments in play.

## What comes out

In the results directory — `REG_Multi_IQA_results/` or `REG_Multi_IQF_results/`
from a `reg-auto` run, `REG_Multi_results/` from a standalone one, or whatever
`-o/--results-dir` names:

| file | contents |
|---|---|
| `multipole_admission.txt` | the gate report: settings, admission counts by reason, interfragment pairs a cheap gate rejected, and the unresolved-channel accounting |
| `multipole_admission.csv` | every pair, its verdict, and the numbers behind it |
| `multipole_terms.csv` | per-step rank terms `V[la,lb]`, plus `V_residual` and `V_unresolved`, per group |
| `multipole_shells.csv` | the same energy collected by `l_tot` shell |
| `excluded_accounting.csv` | per group: spans of the exact, unresolved and residual channels |
| `REG_Multi_seg_N.csv`, `REG_Multi_final_analysis.csv` | the REG values per segment, and the significant-term table |
| `REG_Multi.xlsx` | all of the above in one workbook, except any table too large for a worksheet — those are listed at the end of the run and are complete in their CSVs |
| `energy_recovery.csv` | exact `V_cl` against the multipole sum, over the admitted pairs only — the recovery plot |
| `energy_by_type.csv` | the same energy in fragment channels, split by multipole type |
| `energy_by_pair.csv` | every admitted atom pair's multipole energy |
| `REG_Multi_energy.csv` | the full balance per grouping and in total, with `V_xc` and `E_inter` |
| `REG_Multi_recovery.csv` | per group: how much of `V_cl` the series accounts for, and the `V_xc` it does not |
| `atomic_moments.csv` | every atom's `Q[l,k]` and β-sphere radius at every step |
| `fragment_moments.csv` | each fragment's `Q[l,k]`, centre and extent at every step (whenever fragments are defined) |
| `fragment_moment_cancellation.csv` | per fragment and rank, how far the atomic contributions cancel |
| `fragment_moment_terms.csv`, `REG_Multi_fragment_seg_N.csv` | **`--fragment-moments` only** — the fragment-centred rank terms and their REG values |
| `fragment_moment_comparison.csv` | **`--fragment-moments` only** — the same `l_tot` shell seen both ways, with REG values |
| `REG_Multi_model_transfer.json` | self-contained bundle: terms, REG values, admission record, moments, geometry, fragment definitions, and the fragment-centred view |

Terms are reported per `(l_A, l_B)` rank block, summed over the `k` components,
which makes each one rotation-invariant. Individual `k` components depend on the
frame and are deliberately not the unit of output.

The decomposition closes exactly, at every step and for every group:

```
Σ over rank blocks  +  V_residual  +  V_unresolved  =  exact IQA V_cl over the pairs in scope
```

so nothing is quietly dropped, and a reader can always recover what the rank
terms leave out.

## Performance

The interaction tensor is built for the whole batch of (pair, geometry)
instances at once: every tensor element is an array rather than a float, and one
pass of Python control flow serves the entire batch, because which elements exist
and which branches are taken depend only on the `(l, k)` indices and never on the
values. Measured on this code at `L_max = 5`:

| | rate | a 133-atom path over 20 geometries (175,560 pair-geometries) |
|---|---|---|
| one pair at a time (the prototype's approach) | ~32 tensor builds/s | ~1.5 hours |
| batched, as implemented here | ~34,000 pair-geometries/s | ~6 s |

Reading the moments costs about the same again: ~760 `.int` files/s on 0.5 MB
files, since the file is streamed and abandoned as soon as the moment section
closes, and the atoms are read concurrently.

Memory is the trade, in two places. The tensor holds `(L_max+1)⁴` arrays of the
batch length, 10.4 kB per batch element at `L_max = 5`; the batch is chunked to a
~200 MB budget by default (`chunk_for_budget` in `multipole_utils.py`), which
sits just past the point where throughput stops improving, so this part does not
grow with the system. The rank table does: it holds `8·(L_max+1)²` bytes per pair
per geometry — 288 bytes at `L_max = 5`, so 160 MB for 28,000 pairs over 20
geometries. The run prints its size before allocating it. `--scope inter` cuts it
to interfragment pairs only, and a lower `--lmax` shrinks it quadratically.

If you ever do need more — a system of many hundreds of atoms with a long path,
or `L_max` pushed higher — the profitable step is to compile the recursion rather
than to parallelise around it: the inner work is a fixed, value-independent
program of a few thousand array operations, which is what a `numba.njit` kernel
over the batch, or a small Fortran/C extension called once per chunk, is good at.
Neither is a dependency today because the numpy version handles realistic jobs in
seconds.

## Moment convention

Moments are real spherical-harmonic moments, exactly as AIMAll prints them:
Condon-Shortley phase included, the `sqrt((2l+1)/4pi)` normalisation excluded.
The signed index `k` is `+|m|` for the cosine component and `-|m|` for the sine
one, so `k = +1, -1, 0` are `x, y, z`.

This convention does **not** affect any energy the analysis reports. Each rank
term `V[la,lb]` is summed over `ka` and `kb`, and that sum is a contraction over
complete irreducible blocks — it is rotation-invariant, and identical whether the
intermediate algebra is done in spherical or Cartesian form. A Cartesian
rendering would change only how individual *moment components* are displayed:
`fragment_moments.csv` would gain a dipole as `(mu_x, mu_y, mu_z)` and a
quadrupole as a traceless `3x3` tensor, in place of (or beside) `Q[1,k]` and
`Q[2,k]`. That is a reporting convenience, worth adding if fragment moments are
being quoted in a paper; it is not a change to the method.

## Validation

What the implementation has been checked against, in order of how much it proves:

| check | result |
|---|---|
| interaction tensor vs closed-form charge–charge, charge–dipole, dipole–dipole | exact to 1e-17 |
| full series vs the exact Coulomb energy of two point-charge clusters, through `l_tot = 10`, with solid harmonics built independently | 5e-3 kJ/mol at 6 Å |
| moment translation vs group moments computed directly from the charge clouds | 5e-14 |
| admission gates vs a synthetic path where `R_AB > rho_A + rho_B` is known atom by atom | every verdict correct, including a pair that converges at one end of the path and fails at the other |
| **fragment-centred series vs exact IQA `V_cl`, real AIMAll data** (benzene + F⁻, 11 geometries, 21 fragment pairs) | residuals 0.001–0.021 kJ/mol |
| **whole pipeline vs the original prototype's published energies** (adenine + F⁻, `L_max = 3`, ungated) | exact `V_cl` to 4e-7 kJ/mol, multipole sum to 4e-6 kJ/mol |

The benzene + F⁻ run is also a check on the gating itself. The `C(pi)|Ion`
fragment pair *passes* the geometric gate — the centres clear the summed
β-sphere extents by 0.34 Å — and is then rejected by the convergence test with a
15.9 kJ/mol residual, while the `H|Ion` pairs at a similar separation are
admitted at 0.02 kJ/mol. That is the β-sphere lower bound behaving exactly as
documented: necessary, not sufficient, and no substitute for the numerical test.

`python3 tests/test_multipole.py` runs the first four of these (93 checks) with
no external data. The last two need the datasets they name.

## Limitations

* **1,2 and 1,3 pairs stay out.** Bringing them back needs the MMS shift, which
  moves the expansion centres. A REG built on shifted moments is not comparable
  term-by-term with one built on nuclear-centred moments, so that belongs in a
  separate analysis, not this one.
* Moments are read from `.int` files, so the analysis is only as good as the
  integration behind them. Check `auto_reg.py`'s Lagrangian report first.
* The `k`-resolved terms are computed but not published, for the reason above.
* The two views must not be ranked in one table. They share no term name, which
  makes that hard to do by accident, but nothing stops a reader concatenating the
  two CSVs — the totals agree, the rank attributions do not.
* Dispersion and exchange are outside the scope: this decomposes `V_cl` only.
