# analysis/

Scripts behind numbers the manuscript quotes that no figure script prints, and
the data audits behind the archive. They were written in earlier Claude Code
sessions and lived only in `/tmp` scratch directories; they were moved here on
2026-09-22 and ported to the figure scripts as they now stand (the 2026-09-15
and 09-17 refactors had renamed or moved the variables several of them read).

Run each from this directory, e.g. `python case16_stats.py`. Scripts that
execute figure scripts do so in a throwaway copy of `figures/allcases`
(`_paths.scratch_copy()`), so they never write PNG/EPS files into the repo.
`output/` holds what each printed on 2026-09-22, against the repo and the
archive as they stood that day.

## Manuscript statistics

| Script | What it computes | Quoted in | Status 2026-09-22 |
|---|---|---|---|
| `case16_stats.py` | The pinned definitions of the Results statistics: sample-point Tglob ranges (all eight / six resolved), HEXTOR's departure from the six-resolved median, albedo ranges and darkest counts, the temperature spread (whole plane, in-band <= 1200 W/m2, subsets, LFRic withholding tests), and the Figure 17 areas, band extents, six-resolved band, 1-D cold-edge shift, largest step and invariants | Section 3.1 (surface temperature, HEXTOR, spread and Synthesis paragraphs) | Reproduces the text: 14.2-132.8 K, median 37.8 K (34.9 K six); HEXTOR -43.5 ... +32.2 K, +29.8 K at Case 16; spread 22.0 K (20.4 six, 22.7 +HEXTOR); areas 33.2 / 52.5 / 14.3%; band 1240-1740, 1080-1260, 1080-1220 W/m2; 12.6%; shift 30 / 110 W/m2; step 200 W/m2 |
| `stats.py` (+ `tables.py`) | Per-case ranges for all four variables from the figure scripts' embedded arrays: temperature, water vapor (dex), cloud fraction, albedo, brightest/darkest model counts, Case 4 albedo spread in W/m2 | Section 3.1; the Case 4 absorbed-flux spread (110 W/m2) | Temperature ranges agree with `case16_stats.py`. Its "HEXTOR vs median" lines use older definitions (median of all other models); the text uses the six-resolved median from `case16_stats.py` |
| `spreadstats.py` | Median and extrema of the Figure 7 spread fields, subsets, water vapor with and without ExoColumn | Spread paragraph | Reproduces 22.0 K; subsets as in `case16_stats.py` |
| `sumcheck.py` | The four Figure 17 invariants: partition, no gaps per pressure row, 3-D contested inside the total, sample-point shading equal to `fig_energy_balance.py`'s regime list | Re-check after any change to `fig_summary.py` | All four hold |
| `counterfactual.py` | LFRic temperature and albedo anisotropy fits with and without Case 7 and Cases 8/10/11 | LFRic Case 7 discussion (energy-balance paragraph) | Reproduces 32.3 K at ratio 15 with Case 7, 21.3 K at ratio 4 without, 35.5 K at 4 with it |
| `lfric_loo.py` | LFRic temperature LOO error by variogram family and anisotropy, with and without Case 7 | Same | Consistent with `counterfactual.py` |
| `hextor_cloud_tests.py` | HEXTOR rerun as submitted, without its cloud correction, and with it scaled by S/900 (HEXTOR's own `run_samosa.py`, into temporary directories), as departures from the six-resolved median; ExoColumn's greenhouse effect at Cases 1 and 10 from the archive | HEXTOR paragraph of Section 3.1 | Reproduces the text: without the correction -3.3 / +9.7 / +8.6 / +49.0 / +74.4 K at Cases 10 / 1 / 15 / 9 / 4 and Case 16 running away; scaled, -7.3 and -19.0 K at Cases 1 and 10; ExoColumn 11.6 and 20.4 W/m2 (quoted as 12 and 20). The submitted rerun matches the archive to 0.004 K |

## Regime diagnostics (Figures 15 and 16)

| Script | What it computes | Status 2026-09-22 |
|---|---|---|
| `recon.py` | Rhines length, deformation radius, RMS and dayside winds, tropopause per model and case, from the archive; imported by `jets2.py` | Runs |
| `jets2.py` | The single/double-jet label at sigma = 0.5, 0.3, 0.15, the tropopause and the upper troposphere | Reproduces "only 32 of the 69 model-case combinations" keeping one label |
| `regstats.py` | Statistics of the `extract_regimes.py` table: wind ratios, windiest/calmest models, jet unanimity, Rhines hit rate, jet latitudes, single-jet wind range | Reproduces the single-jet 5.4-31.2 m/s range and the LFRic Case 11 exception at 48.7 m/s |
| `sef.py`, `sef2lib.py`, `sef2.py` | Night-side / terminator static-energy flux convergence | **Does not reproduce the original check**: its reference-energy correction was never saved, which is why the manuscript says the terminator check predates LFRic Cases 8, 10 and 11. Kept as the record of the attempt; the Generic PCM values it prints are not meaningful |

## Data audits

| Script | What it checks |
|---|---|
| `validate_lfric.py` | LFRic NetCDF files reproduce the group's summary table (area-weighted) |
| `check_inst.py`, `check_inst2.py` | Every model's instellation and surface pressure against the protocol, from its files, namelists and fluxes (Cases 1-16). Flags PlaHab Case 11 as run at 400 rather than 900 W/m2, the one case at the wrong instellation; ExoCAM's `sol_tsi` rows flag everywhere because that field is a constant 1361.27 in every file, and its fluxes are what check out. `check_inst2.py` also shows ExoColumn's 3537 Pa of extra dry gas |
| `plahab_asr.py`, `plahab_map.py`, `plahab_mid.py` | PlaHab absorbed flux against its logs, per case |

## Elsewhere

- Section 4 (the extension sequences): `figures/allcases/fig_interpolation_temp_1d.py` prints every number, and `figures/allcases/fit_anisotropy.py` the leave-one-out errors; `figures/allcases/check_table4.py` checks the protocol's Table 4 against its Sobol draws.
- Lost before 2026-09-22, results kept only in the Claude memory notes: the HEXTOR step-by-step configuration ladder (`ladder.py`) that split the HEXTOR-ExoColumn gap into cloud, table and dimensionality terms, and the HEXTOR frozen-dayside cloud-mask experiments. No number from either is quoted in the manuscript. The cloud-correction reruns that are quoted came from the same lost sessions and are regenerated by `hextor_cloud_tests.py`.
