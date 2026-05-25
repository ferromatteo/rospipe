# ROSS2 Optical Pipeline

A Python pipeline for reducing and calibrating optical images from the **ROSS2** instrument on the [REM telescope](https://www.rem.inaf.it) (La Silla, ESO).

ROSS2 is a 4-quadrant imager covering *g, r, i, z* bands simultaneously.  
This pipeline handles all steps from raw frames to calibrated photometric catalogs.

---

## Repository structure

```
.
├── rospipe.py          # Main reduction + photometry pipeline
├── make_bpm.py         # Build bad-pixel masks from master bias/flat
├── make_fringe.py      # Build master fringe frame for z-band defringing
├── cal/                # Calibration files (master bias, flat, BPM, fringe)
└── notebooks/
    └── lightcurve.ipynb   # Multi-band light curve extraction notebook
```

---

## Requirements

```
python >= 3.9
astropy
numpy
scipy
matplotlib
sep
astroscrappy
requests
```

Install with:

```bash
pip install astropy numpy scipy matplotlib sep astroscrappy requests
```

---

## Calibration setup

### `cal/` directory

The pipeline expects the following files in `cal/` (or the directory passed via `--cal`):

| File | Description |
|---|---|
| `master_bias_*_{FILT}.fits` | Master bias per filter/quadrant |
| `master_flat_*_{FILT}*.fits` | Normalised master flat per filter |
| `bpm_{FILT}.fits` | Bad-pixel mask (0=good, 1=bad) |
| `master_fringe_z.fits` | Master fringe for z-band defringing |

ROSS2 quadrant–filter mapping: `g→UR`, `r→BR`, `i→UL`, `z→BL`.

### `make_bpm.py` — Build bad-pixel masks

Generates one `bpm_{filt}.fits` per filter by flagging:
- NaN/Inf pixels in the master flat (dead pixels, zero-padded regions)
- Low flat response below a threshold (vignetting)
- Hot pixels in the bias (N-sigma outliers)
- Deviant flat pixels (N-sigma from median in well-illuminated zone)

```bash
python make_bpm.py [--cal cal/] [--flat-min 0.5] [--bias-sigma 5] [--flat-sigma 7]
```

### `make_fringe.py` — Build master fringe frame

Stacks z-band frames to produce a zero-mean fractional fringe pattern used for defringing. Accepts raw or already-reduced frames.

```bash
python make_fringe.py <img_dir> [--bias BIAS] [--flat FLAT] [-o OUTPUT]
```

---

## Main pipeline: `rospipe.py`

### Basic usage

```bash
# Reduce all IMG*.fits in current directory, write to ./reduced/
python rospipe.py .

# Use a separate calibration directory
python rospipe.py /data/sci --cal /data/cal

# Download latest master bias/flat from the ROSS2 server before reducing
python rospipe.py . --live

# Re-run only photometry on already-reduced frames
python rospipe.py . --phot-only

# Forced photometry at known target positions
python rospipe.py . -f targets.txt
```

### Processing steps

1. **Bias subtraction** and **flat-field division**
2. **Bad-pixel masking** (pixels set to NaN)
3. **z-band defringing** — fits and subtracts a scaled, optionally shifted master fringe
4. **Cosmic-ray rejection** via `astroscrappy`
5. Write `red_*.fits` with `PROCSTAT=4` in the header (WCS preserved from the telescope's own astrometric solution)

6. **Aperture photometry** with SEP using the WCS from the header
8. **Catalog cross-match** against online reference catalogs (PanSTARRS, SDSS, APASS, SkyMapper via TOCats), with VSX variable-star filtering
9. **Iterative sigma-clipped zeropoint fit**
10. Write per-image `phot_*.txt` catalog and calibration plot `phot_*.png`
11. Append a summary row to `photcal.txt`

### Key options

| Flag | Default | Description |
|---|---|---|
| `--cal DIR` | `./cal` | Calibration directory |
| `--outdir DIR` | `sci_dir/reduced` | Output directory |
| `--live` | off | Download latest master bias/flat from ROSS2 server into `cal/` as `live_bias_{filt}.fits.gz` / `live_flat_{filt}.fits.gz`; falls back to local on failure |
| `--phot-only` | off | Skip reduction, re-run photometry on existing `red_*.fits` |
| `-f FILE` | — | Forced photometry target file (see below) |
| `--phot-radius` | 5.0 px | Aperture radius |
| `--phot-thresh` | 1.2 σ | Source detection threshold |
| `--phot-cat-radius` | 15 arcmin | Catalog search radius |
| `--no-phot` | off | Skip photometric calibration |
| `--no-crclean` | off | Skip cosmic-ray rejection |
| `--no-defringe` | off | Skip z-band defringing |
| `--no-bpm` | off | Skip bad-pixel masking |
| `-j N` | min(4, ncpu) | Parallel workers for reduction |

### PROCSTAT header keyword

| Value | Meaning |
|---|---|
| `4` | Frame reduced by rospipe — WCS preserved from the telescope's astrometric solution |

On `--phot-only`, all `red_*.fits` frames go directly to photometry (astrometry is never re-attempted).

### Forced photometry (`-f targets.txt`)

When a target is too faint for blind detection, `rospipe` performs forced aperture photometry at the exact sky position. Provide a plain-text file with one target per line:

```
# RA           Dec          radius_arcsec
209.5405   -64.7348    2.0
13:58:09.72  -64:44:05.26  2.0
```

Forced sources appear in `phot_*.txt` with **flag = 4**.

---

## Output files

### `phot_*.txt` — Per-image photometry catalog

Space-separated text, one row per source. Header lines start with `#`.

| Column | Description |
|---|---|
| `id` | Source index |
| `x_pix`, `y_pix` | Pixel coordinates (0-indexed) |
| `ra_deg`, `dec_deg` | Sky coordinates (ICRS, deg) |
| `flux_adu`, `flux_err_adu` | Aperture flux and error |
| `mag_inst`, `mag_inst_err` | Instrumental magnitude (ZP = 25) |
| `mag_cal`, `mag_cal_err` | Calibrated magnitude |
| `flag` | Quality flag (see below) |
| `mag_cat`, `mag_cat_err` | Matched catalog magnitude (`-99` if unmatched) |

**Flag values:** `0` = isolated + central (best), `1` = crowded, `2` = border, `3` = crowded + border, `4` = forced photometry.

### `photcal.txt` — Session calibration summary

One row per processed image: ZP, ZP error, RMS, limiting magnitude, number of calibration stars, quality label, etc.

---

## Light curve notebook

`notebooks/lightcurve.ipynb` reads the `phot_*.txt` files produced by `rospipe`, cross-matches each epoch against a target position, and produces:
- A multi-band light curve CSV
- Magnitude vs. time plots per filter

Configure `INPUT_FOLDER`, `TARGET_RA`, `TARGET_DEC`, and `TOLERANCE_ARCSEC` in the config cell. Supports both single-night and multi-night folder structures (`RECURSIVE = True`).

---

## Reference catalogs (via TOCats)

| Filter | Priority order |
|---|---|
| g, r, i | PanSTARRS DR1, SDSS DR9, APASS DR9, SkyMapper DR4 |
| z | PanSTARRS DR1, SDSS DR9, SkyMapper DR4 |

Variable stars are automatically removed using the VSX catalog before the zeropoint fit.

---

## License

MIT
