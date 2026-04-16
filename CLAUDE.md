# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAMOSA (Sparse Atmospheric Model Sampling Analysis) is an exoplanet model intercomparison project (exoMIP) under the CUISINES framework. The project uses quasi-Monte Carlo (QMC) sparse sampling to compare outputs from multiple climate models across a 2D parameter space of stellar instellation (400–2600 W/m²) and N₂ surface pressure (0.10–10.0 bar).

The six climate models being compared are: **ExoCAM**, **ExoPlaSim**, **ROCKE-3D**, **Generic PCM** (with/without ocean heat transport), **PlaHab**, and **LFRic**.

## Running Figure Scripts

Each figure script is standalone and must be run from its own directory so output files save correctly:

```bash
cd figures/allcases && python fig_interpolation_temp.py
cd figures/allcases && python fig_tally.py
cd figures/case4    && python fig_compare_temp.py
```

Scripts save both `.png` and `.eps` to their own directory.

## Key Dependencies

- `netCDF4` — reading model output files
- `numpy`, `matplotlib` — core numerics and plotting
- `pykrige` (`OrdinaryKriging`, `UniversalKriging`) — Kriging interpolation across the sparse sample grid
- `scipy` (`qmc`, `CubicSpline`, `CloughTocher2DInterpolator`) — QMC sampling and interpolation
- `mpl_toolkits.basemap` — map projections in Case 4 spatial comparison figures

## Repository Structure

```
figures/
  allcases/    # Full parameter-space analysis (all 16 QMC sample points)
  selectcases/ # Selected representative cases (Cases 1, 4, 16)
  case4/       # Detailed single-case comparison (S=1200 W/m², p=2.34 bar)
data/
  ExoCAM/      # samosa1_*.nc, samosa4_*.nc (time mean + std NetCDF files)
  ROCKE3D/     # same naming convention
  GENERICPCM/  # (empty — data lives at /models/data/samosa/genericpcm/)
  GENERICPCM_OHT/
```

## Data Layout

- **Reduced/summary data** (single scalar per model per case) is embedded directly in the figure scripts as NumPy arrays.
- **Full spatial data** (lat/lon grids) lives outside this repo at `/models/data/samosa/<model>/`. The `data/` directory here holds time-mean and time-std NetCDF summaries for ExoCAM and ROCKE-3D only.
- `global_output_TEMPLATE.dat` is a binary template for the standardized output format models submit to SAMOSA.

## Sampling Design

Two Sobol QMC sequences define the 16 sample points:
- `seed1 = 5936744`, `m=3` → 8 points spanning full (flux, pn2) space
- `seed2 = 397676`, `m=3` → 8 additional points with `flux ≤ 1800 W/m²`

These seeds and the parameter grids (`flux = np.arange(400, 2700, 100)`, log-spaced `pn2`) are replicated identically across multiple scripts — if you need to change the sampling, update all scripts.

## Runaway Cases

`runawaytemp = 600.0` (K) is used as a sentinel value for runaway greenhouse cases. ExoCAM and ROCKE-3D have several such cases at high instellation/pressure. These are masked or excluded in some analyses; see commented-out blocks in the Kriging scripts.

## Variable Naming Conventions

- Model output variables differ by model: `ts`/`TS`/`surface_temperature`/`tsurf` for surface temperature; `ua`/`U`/`u_wind_speed`/`ub`/`u_in_w3` for zonal wind.
- ROCKE-3D temperatures are in Celsius and require `+ 273.16` conversion.
- PlaHab data is in a plain-text `.out` format, not NetCDF.
