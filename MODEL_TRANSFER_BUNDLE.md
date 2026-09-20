# The model-transfer bundle

`auto_reg.py` writes `SYSTEM_results/SYSTEM_model_transfer.json` on every run. It is
meant to be a **self-contained** substitute for the whole results directory: a tool
reading it needs nothing else — not `Energy.xlsx`, not `REG.xlsx`, not
`auto_reg.config`, not the XYZ files.

**One file per run, shared data stored once.** Geometry, energies, recovery and closure
errors, charges, multipoles, Lagrangians and the per-atom and per-pair term tables are
identical whichever partitioning was used, so they sit at the top level exactly once.
Only what genuinely differs — the entities, their REG results and their segmentation —
is nested under `levels`.

Written as strict JSON: no `NaN` or `Infinity` tokens, so `JSON.parse` in a browser
works. Unreadable values are `null`.

## Conventions

- Energies are in **hartree** unless the key ends in `_kj_mol`. `units.energy_to_kj_mol`
  carries the conversion factor.
- Atom pairs run in the canonical order `i < j` over `atoms`, `i` outer.
- Term tables are **property-major**: row `= property_index * n_entities + entity_index`.
  Every table carries its own `headers`, so no row index has to be inferred.
- REG arrays are indexed `[segment][term]`.

## Top level

| Key | What it holds |
| --- | --- |
| `schema_version` | `3` for this layout |
| `primary_level`, `levels` | see below |
| `settings` | every analysis option the run used |
| `availability` | booleans saying what this run produced — check this rather than probing for keys |
| `control_coordinates`, `steps` | the path: per step, folder, CC, WFN/IQA/dispersion energy, both error measures |
| `atoms`, `atom_index`, `elements` | atom labels and their ordering |
| `pairs` | pair ordering, which pairs were written, per-step A–B distances in Å |
| `fragments` | fragment definitions read from `auto_reg.config`, present whether or not IQF ran |
| `energies` | totals plus `recovery_error_*` and `closure_error_*` |
| `terms` | **all** per-step term tables, at every resolution |
| `integration` | per-atom `L(A)`, `T(A)`, `q(A)`, `|L/T|` |
| `direct_multipoles` | per-atom dipole and quadrupole components, when the `.sum` files carry them |
| `xyz_structures` | full geometry at every step |
| `quality_report` | missing files, out-of-threshold Lagrangians, paths needing resubmission |

## `levels`

Keyed by analysis name. An IQF run has both `REG_IQA` and `REG_IQF`; a plain run has only
`REG_IQA`. `primary_level` names the one the run was actually asked for.

```jsonc
"levels": {
  "REG_IQA": {
    "entity_kind": "atom",
    "entities": ["c1", "c2", ...],            // 238 atoms — named explicitly
    "entity_count": 238,
    "terms": {"intra": "atom_intra", "inter": "atom_inter", ...},
    "reg": {"intra": {...}, "inter": {...}, ...},
    "segments": {"energy": {...}, "error": {...}}
  },
  "REG_IQF": {
    "entity_kind": "fragment",
    "entities": ["...", ...],                 // 10 fragments
    "entity_definitions": [{"name", "atom_indices", "atom_labels"}],
    "terms": {"intra": "fragment_intra", ...},
    ...
  }
}
```

`entities` is always the explicit list of what that level's REG runs over, so the row
meaning is never inferred from a count.

**`level.terms` values are keys into the top-level `terms` block, not copies.** Read a
level's intra table as `bundle.terms[level.terms.intra]`. This is what keeps one file the
size the old two were.

`level.reg` blocks are `{headers, reg, pearson}`: `intra`, `inter`, `dispersion`,
`totals`, `intra_error`, `inter_error`, and on the atom level `entity_total_error`
(`E_IQA(A)` against the recovery error) plus `charge_transfer` / `polarisation` when that
option is on.

## Segments

Each level carries two segmentations, because they are not always the same:

- `segments.energy` — critical points of the total energy surface, for `reg.intra`,
  `reg.inter`, `reg.dispersion`, `reg.totals`.
- `segments.error` — critical points of the recovery error surface, for the `*_error`
  blocks.

With `AUTO` on, `reg.reg` finds critical points on whichever surface it is regressing
against, so the two can genuinely differ. `availability.separate_error_segments` says
whether they did on this run. Each level also carries `reg_surfaces`, naming the surface
every REG block was fitted against by name — with dispersion on, the error surface is
`E_WFN − (E_IQA + E_disp)`, not `E_WFN − E_IQA`.

`step_ranges` are inclusive at both ends, and consecutive segments **share** their
critical point, so the ranges overlap by one step. (The current levels use the same
segmentation as each other; it is stored per level because that is where it belongs, not
because it varies.)

## The two error measures

They answer different questions and neither replaces the other:

- `recovery_error` — `E_WFN − (E_IQA + E_disp)`. How far the IQA energy sits from the
  wavefunction energy.
- `closure_error` — `Σ_A E_IQA_Intra(A) + Σ_{A<B} E_IQA_Inter(A,B) − Σ_A E_IQA(A)`.
  Whether the term decomposition re-sums to the IQA energy AIMAll reports. Note the pair
  sum is **not** halved: each atom is credited half of every pair it belongs to, so
  summing over all atoms counts each `A<B` pair once, in full.

## `terms`

Atom level, always present: `atom_intra`, `atom_inter`, `atom_dispersion`,
`atom_totals`, `atom_iqa_reported`. Fragment level, on an IQF run: `fragment_intra`,
`fragment_intra_excl_dispersion`, `fragment_own_dispersion`, `fragment_inter`,
`fragment_dispersion`, `fragment_totals`. Plus `charge_transfer` and `polarisation` when
that option is on. A block a run did not produce is present as `null`, so a reader
hitting one knows it looked in the right place.

**Every block carries a `definition` string**, and the ones that could be ambiguous carry
`includes_dispersion` / `includes_own_dispersion` too. Read them before combining terms.

#### Dispersion is not handled the same way by every fragment term

This is the one trap in the file. `sum_into_fragments` folds each fragment's *own-pair*
dispersion into the intra term, but the fragment totals are formed before that fold:

| Term | Contents |
| --- | --- |
| `fragment_intra` | `Σ_{A∈F} E_IQA_Intra(A) + Σ_{A<B∈F} E_IQA_Inter(A,B) + Σ_{A<B∈F} E_Disp(A,B)` |
| `fragment_intra_excl_dispersion` | the same **without** the own dispersion — same sense as `atom_intra` |
| `fragment_own_dispersion` | `Σ_{A<B∈F} E_Disp(A,B)` — the difference between the two above |
| `fragment_inter` | cross-fragment `E_IQA_Inter` only, no dispersion |
| `fragment_dispersion` | cross-fragment `E_Disp` only |
| `fragment_totals` | `fragment_intra_excl_dispersion(F) + ½ Σ_{G≠F} E_IQA_Inter(F,G)` — **no dispersion at all** |

So `fragment_intra` has dispersion folded in while `fragment_totals` has none: adding
intra and inter per fragment gives a mixed quantity. `fragment_intra` is what the
`REG_IQF` intra REG was run on, so it stays as analysed; use
`fragment_intra_excl_dispersion` for the atom-level sense of "intra", and add
`fragment_own_dispersion` plus half of `fragment_dispersion` to get a
dispersion-inclusive total.

`atom_inter` and `atom_dispersion` are the per-pair tables: `VC_IQA`, `VX_IQA`,
`E_IQA_Inter` and `E_Disp` for each pair at each step.

### Pair pruning

A few hundred atoms means tens of thousands of pairs, and writing every per-step curve
dominates the file. Some are dropped — but the cut is made against a **total drift
budget**, not a per-pair threshold.

That distinction matters. A per-pair threshold bounds each row and says nothing about
their sum: on a 238-atom system, 8105 pairs each varying by under 0.01 kJ/mol together
drifted 2.71 kJ/mol along the path — about a sixth of the total energy change. So pairs
are instead dropped least-active-first while the residual accumulates, stopping before
its own peak-to-peak variation would exceed `--bundle-pair-budget` (default **0.05
kJ/mol**) in any channel. `pruning.residual_drift_kj_mol` reports what was actually left
out, and it is bounded by the budget.

Nothing is silently lost. `pruning.residual_per_property` holds the per-step sum of the
dropped rows, so every property total still reconstructs exactly — add it back and the
closure above balances. `pruning.residual_by_fragment_pair` splits that same residual
over fragment pairs, so the omitted drift can still be *placed* even though the
individual atom pairs behind it were not written.

> **Never rebuild fragment energies from the stored atom pairs.** The dropped rows carry
> a large constant offset as well as the bounded drift, so summing only the written pairs
> can be tens of kJ/mol out in absolute terms. Fragment energies must come from the
> `fragment_*` terms, which include every pair.

`--bundle-pair-budget 0` writes every pair. File size is now data-dependent: a system
whose many small pairs drift coherently keeps most of them (correctly), so pruning saves
little. Raise the budget if size matters more than atom-level attribution.

**REG values are never pruned** — they are one number per term, so every pair is ranked
regardless (`availability.reg_covers_all_pairs`).

## Options

| Flag | Effect |
| --- | --- |
| `--bundle-pair-budget F` | total kJ/mol of drift that may be left out of the per-pair curves (default `0.05`; `0` writes all) |
| `--bundle-indent N` | JSON indentation; `0` (the default) writes compact |
| `--multipole-visualization` | produces the multipole *report and plots*. The multipole **data** is in the bundle either way — it comes free from the cached `.sum` tables |

## Changes from schema 2

- Shared data is stored once at the top level; `levels` holds only what differs.
- `reg.labelled.atom_*` / `reg.labelled.fragment_*` became `levels.<name>.reg.*`, with
  each level naming its `entities` explicitly.
- `segments` split into `segments.energy` and `segments.error` per level.
- `closure_error_definition` corrected: the pair sum is not halved.
- Pair pruning is bounded by total residual drift (`--bundle-pair-budget`) instead of a
  per-pair threshold, which could lose an unbounded amount in aggregate.
- `fragment_intra_excl_dispersion` and `fragment_own_dispersion` added, and every term
  block now carries a `definition`.
- Error-surface labels name the dispersion-inclusive surface when dispersion is on.
