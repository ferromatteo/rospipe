#!/usr/bin/env python3
"""Pipeline ROS2: bias, flat, CR cleaning, z-band defringing."""

import argparse
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import time as _time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS as _WCS
from astroscrappy import detect_cosmics
import numpy as np
import requests as _requests
import sep as _sep
import heapq
import itertools
from scipy.ndimage import gaussian_filter, shift
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist as _cdist
from astropy.coordinates import SkyCoord as _SkyCoord
import astropy.units as _apu
from astropy.wcs.utils import fit_wcs_from_points as _fit_wcs_from_points


def _read_header(path):
    """Read image header, using ext=1 for fpack-compressed (.fz) files."""
    if path.endswith('.fz'):
        return fits.getheader(path, ext=1)
    return fits.getheader(path)


def load_fits(path):
    """Load FITS data; handles .fits.gz (transparent) and .fits.fz (ext=1)."""
    if path.endswith('.fz'):
        data, header = fits.getdata(path, ext=1, header=True)
    else:
        data, header = fits.getdata(path, header=True)
    return data.astype(np.float64), header


def collect_files(directory):
    """Separate IMG (science), master_bias, and master_flat by filename prefix.

    Recognises both plain FITS (.fits) and compressed files (.fits.gz, .fits.fz, .fz).
    """
    science, biases, flats = {}, {}, {}
    seen = set()
    all_files = []
    for pat in ('*.fits', '*.fits.gz', '*.fits.fz', '*.fz'):
        for f in glob.glob(os.path.join(directory, pat)):
            if f not in seen:
                seen.add(f)
                all_files.append(f)
    for f in sorted(all_files):
        base = os.path.basename(f)
        try:
            hdr = _read_header(f)
        except Exception:
            continue
        if "FILTER" not in hdr:
            continue
        filt = hdr["FILTER"].strip()
        if base.startswith("IMG"):
            science.setdefault(filt, []).append(f)
        elif base.startswith("master_bias"):
            biases[filt] = f
        elif base.startswith("master_flat"):
            flats[filt] = f
    return science, biases, flats


def robust_sigma(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if np.isfinite(mad) and mad > 0:
        return 1.4826 * mad
    sigma = np.nanstd(values)
    return float(sigma) if np.isfinite(sigma) else 0.0


def _odd_box(size):
    size = max(3, int(size))
    return size + 1 if size % 2 == 0 else size


def _fit_linear_fringe(image_hp, fringe_hp, mask, clip_sigma=4.0, max_iter=3):
    x = fringe_hp[mask].ravel()
    y = image_hp[mask].ravel()
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if x.size < 100:
        return 0.0, 0.0, np.inf, 0

    for _ in range(max_iter):
        design = np.column_stack([np.ones_like(x), x])
        coeff, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coeff
        sigma = robust_sigma(resid)
        if sigma <= 0:
            break
        keep = np.abs(resid - np.nanmedian(resid)) < clip_sigma * sigma
        if keep.sum() == x.size or keep.sum() < 100:
            break
        x = x[keep]
        y = y[keep]

    design = np.column_stack([np.ones_like(x), x])
    coeff, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coeff
    return float(coeff[1]), float(coeff[0]), robust_sigma(resid), int(x.size)


def _highpass(image, sigma):
    """Fast high-pass: subtract Gaussian-smoothed version."""
    return image - gaussian_filter(image, sigma=sigma, mode="nearest")


def precompute_fringe_hp(fringe, background_box=129):
    """Precompute high-pass fringe and filled fringe (call once, not per frame)."""
    gsigma = _odd_box(background_box) / 3.0
    fringe_fill = np.where(np.isfinite(fringe), fringe, 0.0)
    fringe_hp = _highpass(fringe_fill, gsigma)
    return fringe_fill, fringe_hp


def fit_fringe_model(image, fringe, background_box=129, max_shift=2.0,
                     shift_step=0.5, clip_sigma=4.0,
                     _fringe_precomp=None):
    """Fit a scaled, slightly shifted fringe pattern on sky pixels.

    The fit is done on high-pass filtered images to reduce sensitivity to
    large-scale sky gradients, with a small shift search to absorb mild
    night-to-night pattern drifts.

    Pass _fringe_precomp=(fringe_fill, fringe_hp) from precompute_fringe_hp()
    to avoid recomputing the fringe smoothing for every frame.
    """
    valid = np.isfinite(image) & np.isfinite(fringe)
    if valid.sum() < 100:
        return {
            "scale": 0.0,
            "offset": 0.0,
            "dx": 0.0,
            "dy": 0.0,
            "rms": np.inf,
            "npix": 0,
            "model": np.zeros_like(image),
        }

    gsigma = _odd_box(background_box) / 3.0
    fill_value = np.nanmedian(image[valid])
    img_fill = np.where(np.isfinite(image), image, fill_value)
    image_hp = _highpass(img_fill, gsigma)

    if _fringe_precomp is not None:
        fringe_fill, fringe_hp_base = _fringe_precomp
    else:
        fringe_fill = np.where(np.isfinite(fringe), fringe, 0.0)
        fringe_hp_base = _highpass(fringe_fill, gsigma)
    fringe_strength = np.abs(fringe_hp_base[valid])
    if fringe_strength.size == 0:
        return {
            "scale": 0.0,
            "offset": 0.0,
            "dx": 0.0,
            "dy": 0.0,
            "rms": np.inf,
            "npix": 0,
            "model": np.zeros_like(image),
        }

    fringe_floor = np.nanpercentile(fringe_strength, 35)
    sigma = robust_sigma(image_hp[valid])
    if sigma <= 0:
        sigma = np.nanstd(image_hp[valid])
    sky_mask = valid & (np.abs(image_hp - np.nanmedian(image_hp[valid])) < clip_sigma * max(sigma, 1e-6))
    sky_mask &= np.abs(fringe_hp_base) >= fringe_floor
    if sky_mask.sum() < 100:
        sky_mask = valid

    if max_shift <= 0 or shift_step <= 0:
        shifts = [0.0]
    else:
        steps = int(round(max_shift / shift_step))
        shifts = [round(step * shift_step, 3) for step in range(-steps, steps + 1)]

    best = None
    for dy in shifts:
        for dx in shifts:
            fringe_hp = shift(fringe_hp_base, shift=(dy, dx), order=1, mode="nearest", prefilter=False)
            scale, offset, rms, npix = _fit_linear_fringe(image_hp, fringe_hp, sky_mask, clip_sigma=clip_sigma)
            if best is None or rms < best["rms"]:
                fringe_model = shift(fringe_fill, shift=(dy, dx), order=1, mode="nearest", prefilter=False)
                best = {
                    "scale": scale,
                    "offset": offset,
                    "dx": dx,
                    "dy": dy,
                    "rms": rms,
                    "npix": npix,
                    "model": scale * fringe_model,
                }

    return best


# ============================================================================
# PHOTOMETRIC CALIBRATION
# ============================================================================

# ---------------------------------------------------------------------------
# ROS2 instrument constants
# ---------------------------------------------------------------------------
_ROS2_GAIN     = 1.0    # e/ADU  (GAIN keyword default)
_ROS2_RONOISE  = 4.5    # ADU/pix (RONOISE keyword default)
_ROS2_PIXSCALE = 0.587  # arcsec/pix (fixed plate scale)

# Live calibration base URLs (used with --live)
_ROS2_LIVE_BASE = "https://ross.oas.inaf.it/RossDB/ROS2"
_ROS2_LIVE_QUAD = {'g': 'UR', 'r': 'BR', 'i': 'UL', 'z': 'BL'}

_TOCATS_BASE    = "https://cats.oas.inaf.it"
_TOCATS_TIMEOUT = 30
_TOCATS_RETRIES = 3
_TOCATS_DEFAULT_ERR = 0.05   # mag: fallback when catalog has no per-star error

# Position columns per catalog (ra_col, dec_col)
# Note: sdss9_mini RAmas/DECmas column values are in degrees despite the name
_CAT_POS = {
    'ps1_mini':     ('raMean',   'decMean'),
    'sdss9_mini':   ('RAmas',    'DECmas'),
    'apass9':       ('RAJ2000',  'DEJ2000'),
    'skymapperdr4': ('raj2000',  'dej2000'),
}

# (mag_col, err_col_or_None, null_sentinel_or_None) per catalog per filter
# 2 observes in g, r, i, z only
_CAT_BANDS = {
    'ps1_mini': {
        'g': ('gMeanPSFMag', 'gMeanPSFMagErr', -999.0),
        'r': ('rMeanPSFMag', 'rMeanPSFMagErr', -999.0),
        'i': ('iMeanPSFMag', 'iMeanPSFMagErr', -999.0),
        'z': ('zMeanPSFMag', 'zMeanPSFMagErr', -999.0),
    },
    'sdss9_mini': {
        'g': ('GSDSSmm', None, None),
        'r': ('RSDSSmm', None, None),
        'i': ('ISDSSmm', None, None),
        'z': ('ZSDSSmm', None, None),
    },
    'apass9': {
        'g': ('g_mag', 'e_g_mag', None),
        'r': ('r_mag', 'e_r_mag', None),
        'i': ('i_mag', 'e_i_mag', None),
    },
    'skymapperdr4': {
        'g': ('g_psf', 'e_g_psf', None),
        'r': ('r_psf', 'e_r_psf', None),
        'i': ('i_psf', 'e_i_psf', None),
        'z': ('z_psf', 'e_z_psf', None),
    },
}

# Catalog priority per filter
_FILTER_PRIORITY = {
    'g': ['ps1_mini', 'sdss9_mini', 'apass9', 'skymapperdr4'],
    'r': ['ps1_mini', 'sdss9_mini', 'apass9', 'skymapperdr4'],
    'i': ['ps1_mini', 'sdss9_mini', 'apass9', 'skymapperdr4'],
    'z': ['ps1_mini', 'sdss9_mini', 'skymapperdr4'],
}

# ZP quality thresholds (rms in mag)
_ZP_QUALITY = [
    (0.100, 'VERY GOOD'),
    (0.175, 'GOOD'),
    (0.250, 'MEDIUM'),
    (0.350, 'POOR'),
    (1e9,   'VERY POOR'),
]


# ---------------------------------------------------------------------------
# Target file parser (forced photometry)
# ---------------------------------------------------------------------------

def parse_target_file(filepath):
    """Parse a target position file for forced photometry.

    Each line: RA DEC [radius_arcsec]
    Supported formats:
        Sexagesimal:  13:58:09.72 -64:44:05.26 1.5
        Decimal deg:  209.540 -64.735 2.0
        No radius:    209.540 -64.735          (defaults to 2.0 arcsec)
    Lines starting with '#' or empty lines are ignored.
    Returns list of dicts with 'ra_deg', 'dec_deg', 'radius_arcsec'.
    """
    DEFAULT_RADIUS_ARCSEC = 2.0
    targets = []
    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            ra_str, dec_str = parts[0], parts[1]
            radius = float(parts[2]) if len(parts) >= 3 else DEFAULT_RADIUS_ARCSEC
            if ':' in ra_str:
                coord = _SkyCoord(ra_str, dec_str,
                                  unit=(_apu.hourangle, _apu.deg))
            else:
                coord = _SkyCoord(float(ra_str), float(dec_str), unit=_apu.deg)
            targets.append({
                'ra_deg':        float(coord.ra.deg),
                'dec_deg':       float(coord.dec.deg),
                'radius_arcsec': radius,
            })
    return targets


# ---------------------------------------------------------------------------
# TOCats download helpers
# ---------------------------------------------------------------------------

def _tocats_query(catalog, ra_deg, dec_deg, radius_deg, limit=10000):
    """Query TOCats service with exponential-backoff retries.
    Returns parsed JSON dict (contains 'metadata' and 'data' keys).
    """
    url = (f"{_TOCATS_BASE}/{catalog}/"
           f"radius={radius_deg}&ra={ra_deg}&dec={dec_deg}&maxobjs={limit}&json")
    last_exc = None
    for attempt in range(_TOCATS_RETRIES):
        try:
            resp = _requests.get(url, timeout=_TOCATS_TIMEOUT)
            if resp.status_code == 200:
                d = resp.json()
                if isinstance(d, dict) and 'metadata' in d:
                    return d
                last_exc = RuntimeError(
                    f"unexpected response (no metadata): {str(d)[:120]}")
            else:
                last_exc = RuntimeError(f"HTTP {resp.status_code}")
        except Exception as exc:
            last_exc = exc
        if attempt < _TOCATS_RETRIES - 1:
            _time.sleep(2 ** attempt)
    raise RuntimeError(
        f"TOCats '{catalog}' failed after {_TOCATS_RETRIES} tries: {last_exc}")


def _tocats_extract(resp, fields):
    """Extract named columns from TOCats JSON into numpy float64 arrays.
    Returns dict: field_name -> np.ndarray  (NaN for None / missing / non-numeric).
    """
    def _safe_float(v):
        if v is None:
            return np.nan
        try:
            return float(v)
        except (ValueError, TypeError):
            return np.nan

    meta_cols = list(resp.get('metadata', {}).keys())
    objs      = resp.get('data', {}).get('objs', [])
    if not objs:
        return {f: np.array([], dtype=np.float64) for f in fields}
    col_idx = {c: i for i, c in enumerate(meta_cols)}
    result  = {}
    for f in fields:
        if f in col_idx:
            idx = col_idx[f]
            result[f] = np.array(
                [_safe_float(row[idx]) for row in objs],
                dtype=np.float64,
            )
        else:
            result[f] = np.full(len(objs), np.nan, dtype=np.float64)
    return result


def _download_optical_catalog(ra_deg, dec_deg, radius_deg, filter_name, verbose=False):
    """Download photometric reference catalog from TOCats for a given sky region.

    Tries catalogs in filter-appropriate priority order.
    Returns (catalog_name, ra, dec, mag, mag_err) as numpy arrays,
    or (None, None, None, None, None) when no suitable catalog is found.
    """
    priority = _FILTER_PRIORITY.get(filter_name,
                                    ['ps1_mini', 'apass9', 'skymapperdr4'])

    tried = []   # (cat_name, reason) collected for final diagnostic

    for cat in priority:
        if cat not in _CAT_BANDS or filter_name not in _CAT_BANDS[cat]:
            continue
        mag_col, err_col, null_val = _CAT_BANDS[cat][filter_name]
        ra_col, dec_col = _CAT_POS[cat]

        fields = [ra_col, dec_col, mag_col]
        if err_col:
            fields.append(err_col)
        if cat == 'sdss9_mini':
            fields.append('CLASS')       # 6 = star
        if cat == 'skymapperdr4':
            fields.append('class_star')  # 0-1 probability

        try:
            if verbose:
                print(f"    {cat}...", end=' ', flush=True)
            resp   = _tocats_query(cat, ra_deg, dec_deg, radius_deg)
            data   = _tocats_extract(resp, fields)
            n      = len(data.get(ra_col, []))
            if n == 0:
                reason = "0 objects"
                if verbose:
                    print(reason)
                tried.append((cat, reason))
                continue

            ra_arr  = data[ra_col]
            dec_arr = data[dec_col]
            mag_arr = data[mag_col]
            err_arr = (data[err_col]
                       if (err_col and err_col in data)
                       else np.full(n, _TOCATS_DEFAULT_ERR))
            # Replace missing/zero errors with default
            bad_err = ~np.isfinite(err_arr) | (err_arr <= 0)
            err_arr[bad_err] = _TOCATS_DEFAULT_ERR

            # Star / galaxy separation
            if cat == 'sdss9_mini' and 'CLASS' in data and len(data['CLASS']) == n:
                star_mask = data['CLASS'] == 6.0
            elif cat == 'skymapperdr4' and 'class_star' in data and len(data['class_star']) == n:
                star_mask = data['class_star'] > 0.5
            else:
                star_mask = np.ones(n, dtype=bool)

            # Validity filter
            valid = star_mask.copy()
            if null_val is not None:
                valid &= mag_arr > (null_val + 0.1)
            valid &= (np.isfinite(mag_arr) & (mag_arr > 8.0) & (mag_arr < 23.0)
                      & np.isfinite(err_arr) & (err_arr < 1.5)
                      & np.isfinite(ra_arr) & np.isfinite(dec_arr))

            ra_arr, dec_arr = ra_arr[valid], dec_arr[valid]
            mag_arr, err_arr = mag_arr[valid], err_arr[valid]

            n_valid = len(ra_arr)
            if verbose:
                print(f"{n_valid} stars")
            if n_valid < 5:
                reason = f"only {n_valid} valid stars after cuts"
                tried.append((cat, reason))
                continue

            return cat, ra_arr, dec_arr, mag_arr, err_arr

        except Exception as exc:
            reason = f"error: {exc}"
            if verbose:
                print(reason)
            tried.append((cat, reason))
            continue

    # All catalogs failed – always print a diagnostic so the user knows why
    reasons = '; '.join(f"{c}: {r}" for c, r in tried) if tried else "no catalogs in priority list"
    print(f"  [PHOT] No catalog found for filter '{filter_name}' "
          f"(RA={ra_deg:.3f}, Dec={dec_deg:.3f}) — tried: {reasons}")
    return None, None, None, None, None


def _download_vsx(ra_deg, dec_deg, radius_deg, verbose=False):
    """Download VSX variable stars for a region.
    Returns (ra, dec) numpy arrays (may be empty).
    """
    try:
        if verbose:
            print("    VSX...", end=' ', flush=True)
        resp  = _tocats_query('vsx', ra_deg, dec_deg, radius_deg, limit=50000)
        data  = _tocats_extract(resp, ['RAJ2000', 'DEJ2000'])
        ra    = data['RAJ2000']
        dec   = data['DEJ2000']
        valid = np.isfinite(ra) & np.isfinite(dec)
        if verbose:
            print(f"{valid.sum()} variables")
        return ra[valid], dec[valid]
    except Exception as exc:
        if verbose:
            print(f"ERROR ({exc})")
        return np.array([]), np.array([])


def _filter_variables(cat_ra, cat_dec, vsx_ra, vsx_dec, tol_arcsec=2.0):
    """Return boolean mask: True = star is NOT in VSX (keep it)."""
    if len(vsx_ra) == 0:
        return np.ones(len(cat_ra), dtype=bool)
    tol_deg = tol_arcsec / 3600.0
    tree    = cKDTree(np.column_stack([vsx_ra, vsx_dec]))
    dists, _ = tree.query(np.column_stack([cat_ra, cat_dec]))
    return dists >= tol_deg


def _group_files_by_position(file_infos, tolerance_arcmin=1.0):
    """Group (path, ra, dec, filt) tuples by sky proximity.

    Images within tolerance_arcmin of each other share one catalog download
    per filter.  Returns list of (center_ra, center_dec, [(path, ra, dec, filt), ...]).
    """
    if not file_infos:
        return []
    tol_deg  = tolerance_arcmin / 60.0
    assigned = [False] * len(file_infos)
    groups   = []
    for i, item_i in enumerate(file_infos):
        if assigned[i]:
            continue
        ra_i, dec_i = item_i[1], item_i[2]
        group = [item_i]
        assigned[i] = True
        for j in range(i + 1, len(file_infos)):
            if assigned[j]:
                continue
            ra_j, dec_j = file_infos[j][1], file_infos[j][2]
            dra = (ra_j - ra_i) * np.cos(np.radians(0.5 * (dec_i + dec_j)))
            if np.hypot(dra, dec_j - dec_i) < tol_deg:
                group.append(file_infos[j])
                assigned[j] = True
        cra  = float(np.mean([x[1] for x in group]))
        cdec = float(np.mean([x[2] for x in group]))
        groups.append((cra, cdec, group))
    return groups


# ---------------------------------------------------------------------------
# Source detection and aperture photometry helpers
# ---------------------------------------------------------------------------

def _detect_sources(data, threshold_sigma=3.0, min_pixels=5):
    """Background-subtract and detect sources with SEP.
    Returns (sources_recarray, background_subtracted_data, rms_map).
    """
    data_c   = np.ascontiguousarray(data, dtype=np.float64)
    nan_mask = ~np.isfinite(data_c)
    data_c[nan_mask] = 0.0
    bkg      = _sep.Background(data_c, bw=64, bh=64, fw=3, fh=3)
    data_sub = data_c - bkg
    data_sub[nan_mask] = 0.0
    rms      = bkg.rms()
    sources  = _sep.extract(data_sub, threshold_sigma, err=rms,
                            minarea=min_pixels, deblend_nthresh=32,
                            deblend_cont=0.005, clean=True)
    return sources, data_sub, rms


def _compute_lim_mag(zp, bkg_rms_map, aperture_radius, snr=3.0,
                     central_fraction=0.8):
    """3-sigma limiting magnitude from median background RMS map.

    Uses only the central `central_fraction` of the map to avoid noisy/vignetted
    edges inflating the RMS estimate.

    lim_mag = ZP + 25 - 2.5 * log10(snr * median_rms * sqrt(N_pix_in_aperture))
    where N_pix = pi * r^2 and 25 is the instrumental zeropoint.
    """
    if zp is None:
        return np.nan
    n_pix = np.pi * aperture_radius ** 2
    ny, nx = bkg_rms_map.shape
    my = int(ny * (1 - central_fraction) / 2)
    mx = int(nx * (1 - central_fraction) / 2)
    central = bkg_rms_map[my:ny - my, mx:nx - mx]
    rms_sky = float(np.nanmedian(central))
    if rms_sky <= 0:
        return np.nan
    flux_lim = snr * rms_sky * np.sqrt(n_pix)
    if flux_lim <= 0:
        return np.nan
    return float(zp + 25.0 - 2.5 * np.log10(flux_lim))


def _save_phot_plot(outpath, inst_mag, cat_mag, inst_err, cat_err,
                    fit_mask, zp, zp_err, rms, n_used, n_total,
                    filt, cat_name, lim_mag=None, filename=''):
    """Two-panel calibration PNG: inst_mag vs cat_mag + residuals."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 8),
        gridspec_kw={'height_ratios': [2, 1]})

    used   = fit_mask  & np.isfinite(inst_mag) & np.isfinite(cat_mag)
    unused = ~fit_mask & np.isfinite(inst_mag) & np.isfinite(cat_mag)

    if unused.sum():
        ax1.errorbar(cat_mag[unused], inst_mag[unused],
                     xerr=cat_err[unused], yerr=inst_err[unused],
                     fmt='o', color='#aaaaaa', ms=3, lw=0.5, alpha=0.5,
                     label='rejected')
    if used.sum():
        ax1.errorbar(cat_mag[used], inst_mag[used],
                     xerr=cat_err[used], yerr=inst_err[used],
                     fmt='o', color='steelblue', ms=4, lw=0.7, alpha=0.85,
                     label=f'used (N={n_used}/{n_total})')
        mx = np.linspace(cat_mag[used].min() - 0.3, cat_mag[used].max() + 0.3, 100)
        ax1.plot(mx, mx - zp, 'r-', lw=1.2,
                 label=f'ZP={zp:+.3f}\u00b1{zp_err:.3f}  rms={rms:.3f}')
    if lim_mag is not None and np.isfinite(lim_mag):
        ax1.axvline(lim_mag, color='orange', ls='--', lw=0.9,
                    label=f'3\u03c3 lim={lim_mag:.2f}')
    ax1.set_ylabel('Instrumental mag')
    ax1.set_xlabel('Catalog mag')
    ax1.invert_yaxis()
    ax1.invert_xaxis()
    ax1.legend(fontsize=8)
    ax1.set_title(f'ROS2 Photometry  |  {filt}  |  {cat_name}  |  {filename}',
                  fontsize=9)

    if used.sum():
        resid = cat_mag[used] - inst_mag[used] - zp
        ax2.scatter(cat_mag[used], resid, s=12, color='steelblue', alpha=0.8)
        ax2.axhline(0,    color='r', lw=1.0)
        ax2.axhline(+rms, color='r', lw=0.7, ls='--', alpha=0.6)
        ax2.axhline(-rms, color='r', lw=0.7, ls='--', alpha=0.6)
    ax2.set_ylabel('Residual (mag)')
    ax2.set_xlabel('Catalog mag')
    ax2.invert_xaxis()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Zeropoint fitting
# ---------------------------------------------------------------------------

def _fit_zeropoint(inst_mag, cat_mag, inst_err, cat_err,
                   sigma=3.0, max_iter=10, min_stars=5):
    """Iterative sigma-clipped weighted zeropoint fit.

    Returns (zp, zp_err, rms, n_used, mask).
    All values are None / 0 / mask=False on failure.
    """
    mask = (np.isfinite(inst_mag) & np.isfinite(cat_mag)
            & np.isfinite(inst_err) & np.isfinite(cat_err)
            & (inst_err > 0) & (cat_err > 0)
            & (inst_err < 1.0) & (cat_err < 1.0))
    n_used = int(mask.sum())
    if n_used < min_stars:
        return None, None, None, 0, mask

    for _ in range(max_iter):
        if n_used < min_stars:
            break
        w      = 1.0 / (inst_err[mask] ** 2 + cat_err[mask] ** 2)
        delta  = cat_mag[mask] - inst_mag[mask]
        zp     = float(np.sum(w * delta) / np.sum(w))
        resid  = cat_mag - inst_mag - zp
        rms_w  = float(np.sqrt(np.sum(w * resid[mask] ** 2) / np.sum(w)))
        new_mask = mask & (np.abs(resid) < sigma * max(rms_w, 1e-9))
        if new_mask.sum() == n_used:
            break
        mask   = new_mask
        n_used = int(mask.sum())

    if n_used < min_stars:
        return None, None, None, 0, mask

    w      = 1.0 / (inst_err[mask] ** 2 + cat_err[mask] ** 2)
    delta  = cat_mag[mask] - inst_mag[mask]
    zp     = float(np.sum(w * delta) / np.sum(w))
    zp_err = float(np.sqrt(1.0 / np.sum(w)))
    resid  = cat_mag[mask] - inst_mag[mask] - zp
    rms    = float(np.sqrt(np.sum(w * resid ** 2) / np.sum(w)))
    return zp, zp_err, rms, n_used, mask


def _zp_quality_label(rms):
    """Return quality string from ZP fit RMS."""
    if rms is None or not np.isfinite(rms):
        return 'UNKNOWN'
    for thresh, label in _ZP_QUALITY:
        if rms < thresh:
            return label
    return 'VERY POOR'


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _save_phot_catalog(outpath, sources, det_ra, det_dec,
                        flux, flux_err, ap_flag,
                        inst_mag, inst_mag_err,
                        matched, cat_idx, cat_mag, cat_mag_err,
                        cal_mag, cal_mag_err,
                        zp, zp_err, rms, n_used, n_total,
                        filt, cat_name, aperture_radius,
                        date_obs, exptime, obj_name,
                        image_shape=None, lim_mag=None,
                        min_isolation_pix=10.0, central_fraction=0.90,
                        force_flag_4=None):
    """Write per-image photometry catalog text file (phot_<stem>.txt).

    Flag convention (matches remirpipe):
      0 = isolated + central  (best quality)
      1 = crowded  (neighbour within min_isolation_pix)
      2 = border   (outside central_fraction of image, or SEP truncation bit)
      3 = crowded + border
      4 = forced photometry (position injected, not blind-detected)
    """
    def _f(v):
        return f"{v:.4f}" if (v is not None and np.isfinite(float(v))) else 'nan'

    quality = _zp_quality_label(rms)
    stem    = os.path.splitext(os.path.basename(outpath))[0].lstrip('phot_')

    # --- isolation flag: crowded if any neighbour within min_isolation_pix ---
    positions = np.column_stack([sources['x'], sources['y']])
    n_src = len(sources)
    if n_src >= 2:
        tree = cKDTree(positions)
        distances, _ = tree.query(positions, k=2,
                                  distance_upper_bound=min_isolation_pix)
        isolated = distances[:, 1] >= min_isolation_pix
    else:
        isolated = np.ones(n_src, dtype=bool)

    # --- border flag: outside central_fraction OR SEP truncation bit (16) ---
    if image_shape is not None:
        ny, nx = image_shape
        bx = (1.0 - central_fraction) / 2.0 * nx
        by = (1.0 - central_fraction) / 2.0 * ny
        x = positions[:, 0]
        y = positions[:, 1]
        central = (x >= bx) & (x <= nx - bx) & (y >= by) & (y <= ny - by)
    else:
        central = np.ones(n_src, dtype=bool)
    # SEP ap_flag bit 16 = aperture truncated at image boundary → also border
    sep_border = (ap_flag & 16) != 0
    is_border = (~central) | sep_border

    # --- combined flag ---
    flag = np.zeros(n_src, dtype=int)
    flag[~isolated]              += 1   # crowded
    flag[is_border]              += 2   # border
    if force_flag_4 is not None:
        flag[force_flag_4]        = 4   # forced photometry

    header = [
        "# ROS2 Photometry Catalog",
        f"# File:         {stem}",
        f"# Object:       {obj_name}",
        f"# Filter:       {filt}",
        f"# Date-obs:     {date_obs}",
        f"# Exptime:      {exptime:.1f} s",
        f"# Catalog:      {cat_name or 'N/A'}",
        f"# Zeropoint:    {_f(zp)} +/- {_f(zp_err)} mag",
        f"# RMS:          {_f(rms)} mag",
        f"# Lim_mag(3s):  {_f(lim_mag)} mag",
        f"# N_cal:        {n_used}/{n_total}",
        f"# Quality:      {quality}",
        f"# Aperture:     {aperture_radius:.1f} pix ({aperture_radius * _ROS2_PIXSCALE:.2f} arcsec)",
        "#",
        "# Flag: 0=isolated+central (best), 1=crowded, 2=border, 3=crowded+border, 4=forced",
        f"#   isolation radius: {min_isolation_pix:.1f} pix  "
        f"central fraction: {central_fraction:.2f}",
        "# Unmatched catalog values are -99.",
        "#",
        "# id  x_pix  y_pix  ra_deg  dec_deg  flux_adu  flux_err_adu"
        "  mag_inst  mag_inst_err  mag_cal  mag_cal_err  flag  mag_cat  mag_cat_err",
    ]

    rows = []
    for i in range(n_src):
        m_cat  = cat_mag[cat_idx[i]]     if (matched[i] and cat_idx[i] >= 0) else -99
        me_cat = cat_mag_err[cat_idx[i]] if (matched[i] and cat_idx[i] >= 0) else -99
        m_cat_s  = f"{m_cat:.4f}"  if m_cat  != -99 else "-99"
        me_cat_s = f"{me_cat:.4f}" if me_cat != -99 else "-99"
        rows.append(
            f"{i+1:5d}  {sources['x'][i]:9.3f}  {sources['y'][i]:9.3f}  "
            f"{det_ra[i]:11.6f}  {det_dec[i]:11.6f}  "
            f"{flux[i]:14.3f}  {flux_err[i]:12.3f}  "
            f"{_f(inst_mag[i])}  {_f(inst_mag_err[i])}  "
            f"{_f(cal_mag[i])}  {_f(cal_mag_err[i])}  "
            f"{flag[i]:1d}  "
            f"{m_cat_s}  {me_cat_s}"
        )

    with open(outpath, 'w') as fh:
        fh.write('\n'.join(header + rows) + '\n')


def _append_calib_summary(calib_file, stem, obj_name, filt, cat_name,
                           zp, zp_err, rms, n_used, n_total, quality,
                           aperture_radius, n_cat, n_vsx,
                           date_obs, exptime, lim_mag=None):
    """Append one row to the session-level photcal.txt summary file."""
    header_needed = not os.path.exists(calib_file)
    with open(calib_file, 'a') as fh:
        if header_needed:
            fh.write(
                "# ROS2 Photometric Calibration Summary\n"
                "# filename  object  filter  catalog"
                "  zp  zp_err  rms  lim_mag  n_used  n_total  quality"
                "  aperture_pix  n_cat_stars  n_vsx_removed  date_obs  exptime_s\n"
            )

        def _f(v):
            return f"{v:.4f}" if (v is not None and np.isfinite(float(v))) else 'N/A'

        fh.write(
            f"{stem}  {obj_name}  {filt}  {cat_name or 'N/A'}  "
            f"{_f(zp)}  {_f(zp_err)}  {_f(rms)}  {_f(lim_mag)}  "
            f"{n_used}  {n_total}  {quality}  "
            f"{aperture_radius:.1f}  "
            f"{n_cat}  {n_vsx}  {date_obs}  {exptime:.1f}\n"
        )


# ---------------------------------------------------------------------------
# Astrometry solver (quad-matching, adapted from remirpipe)
# ---------------------------------------------------------------------------

def _geo_hash(coords):
    """Scale/rotation-invariant geometric hash code for a 4-star quad."""
    coords = np.asarray(coords, dtype=np.float64)
    dists  = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    i, j   = np.unravel_index(np.argmax(dists), dists.shape)
    A, B   = coords[i], coords[j]
    others = [coords[k] for k in range(4) if k not in (i, j)]
    C, D   = others[0], others[1]
    vec    = B - A
    norm2  = float(np.dot(vec, vec))
    if norm2 < 1e-12:
        return np.zeros(4)
    perp = np.array([-vec[1], vec[0]])
    xC = float(np.dot(C - A, vec) / norm2)
    yC = float(np.dot(C - A, perp) / norm2)
    xD = float(np.dot(D - A, vec) / norm2)
    yD = float(np.dot(D - A, perp) / norm2)
    if xC > xD:
        xC, xD, yC, yD = xD, xC, yD, yC
    if xC + xD > 1.0:
        xC, xD, yC, yD = 1 - xC, 1 - xD, 1 - yC, 1 - yD
    return np.array([xC, yC, xD, yD])


def _build_quads(xy, mag, G=1000):
    """Build geometric quads from the G brightest star combinations.
    xy: Nx2 positions, mag: N magnitudes (lower = brighter).
    Returns list of {'coords': 4x2 array, 'hash': 4-vector}.
    """
    n = len(xy)
    if n < 4:
        return []
    heap = []
    cnt  = 0
    for i0, i1, i2, i3 in itertools.combinations(range(n), 4):
        sm = float(mag[i0] + mag[i1] + mag[i2] + mag[i3])
        if len(heap) < G:
            heapq.heappush(heap, (-sm, cnt, (i0, i1, i2, i3)))
        elif sm < -heap[0][0]:
            heapq.heapreplace(heap, (-sm, cnt, (i0, i1, i2, i3)))
        cnt += 1
    result = []
    for _, _, (i0, i1, i2, i3) in heap:
        c = xy[[i0, i1, i2, i3]]
        result.append({'coords': c, 'hash': _geo_hash(c)})
    return result


def _match_quads(det_quads, cat_quads, threshold=0.02):
    """Match detection quads to catalog quads by geometric hash similarity."""
    if not det_quads or not cat_quads:
        return []
    det_h = np.array([q['hash'] for q in det_quads])
    cat_h = np.array([q['hash'] for q in cat_quads])
    dists = _cdist(det_h, cat_h, metric='euclidean')
    mi, mj = np.where(dists < threshold)
    if len(mi) == 0:
        return []
    order = np.argsort(dists[mi, mj])
    return [{'det': det_quads[mi[k]], 'cat': cat_quads[mj[k]],
             'dist': float(dists[mi[k], mj[k]])} for k in order]


def _sim_transform(src, dst):
    """Least-squares similarity transform (scale + rotation + translation) src→dst."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    sm, dm = src.mean(0), dst.mean(0)
    X, Y   = src - sm, dst - dm
    C      = X.T @ Y
    U, S, Vt = np.linalg.svd(C)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1] *= -1
        R = Vt.T @ U.T
    scale = float(np.trace(R @ C) / np.sum(X ** 2))
    t     = dm - scale * (R @ sm)
    return scale, R, t


def _apply_sim(xy, scale, R, t):
    """Apply similarity transform to Nx2 array."""
    return (scale * (R @ xy.T)).T + t


def _solve_astrometry(sources, wcs_init, cat_ra, cat_dec, cat_mag,
                       image_shape, num_quads=1000, hash_tol=0.02,
                       pix_tol=3.0, min_matches=6, accept_rms_px=3.0,
                       n_det=30, n_cat=40, verbose=False):
    """Refine WCS using quad-matching of detected sources against a catalog.

    Algorithm (adapted from remirpipe):
      1. Project catalog to pixel space via initial WCS
      2. Build scale/rotation-invariant geometric quads from both sets
      3. Match quads by hash similarity, find consensus similarity transform
      4. Cross-match with refined positions, sigma-clip outliers
      5. Fit new TAN WCS via fit_wcs_from_points

    Returns (wcs_refined, n_matches, rms_px) or (None, 0, inf) on failure.
    """
    ny, nx = image_shape
    if len(sources) < 4:
        return None, 0, np.inf

    # --- project catalog to pixel space via initial WCS ---
    try:
        cat_pix = wcs_init.all_world2pix(
            np.column_stack([cat_ra, cat_dec]), 0)
    except Exception:
        return None, 0, np.inf

    margin   = 0.25 * max(nx, ny)
    in_frame = ((cat_pix[:, 0] >= -margin) & (cat_pix[:, 0] <= nx + margin) &
                (cat_pix[:, 1] >= -margin) & (cat_pix[:, 1] <= ny + margin))
    if in_frame.sum() < 4:
        if verbose:
            print(f"  [ASTRO] only {in_frame.sum()} catalog stars projected in frame")
        return None, 0, np.inf

    cat_pix_f = cat_pix[in_frame]
    cat_mag_f = cat_mag[in_frame]
    cat_ra_f  = cat_ra[in_frame]
    cat_dec_f = cat_dec[in_frame]

    # --- pick N brightest detected sources ---
    src_flux  = sources['flux']
    src_xy    = np.column_stack([sources['x'], sources['y']])
    det_order = np.argsort(-src_flux)[:n_det]
    det_xy    = src_xy[det_order]
    det_mag   = -2.5 * np.log10(np.maximum(src_flux[det_order], 1e-9))

    # --- pick N brightest catalog stars ---
    cat_order = np.argsort(cat_mag_f)[:n_cat]
    cat_pix_b = cat_pix_f[cat_order]
    cat_mag_b = cat_mag_f[cat_order]

    if len(det_xy) < 4 or len(cat_pix_b) < 4:
        return None, 0, np.inf

    # --- build geometric quads ---
    det_quads = _build_quads(det_xy,   det_mag,  G=num_quads)
    cat_quads = _build_quads(cat_pix_b, cat_mag_b, G=num_quads)
    matches   = _match_quads(det_quads, cat_quads, threshold=hash_tol)

    if not matches:
        if verbose:
            print("  [ASTRO] no quad matches found")
        return None, 0, np.inf

    # --- evaluate top transforms, find consensus ---
    transforms = []
    for m in matches[:min(200, len(matches))]:
        try:
            sc, R, t = _sim_transform(m['cat']['coords'], m['det']['coords'])
        except Exception:
            continue
        if not (0.5 < sc < 2.0):
            continue
        cat_t = _apply_sim(cat_pix_b, sc, R, t)
        d, _  = cKDTree(det_xy).query(cat_t, distance_upper_bound=pix_tol)
        n_m   = int((d < pix_tol).sum())
        if n_m > 0:
            transforms.append({'scale': sc, 'R': R, 't': t, 'n': n_m})

    if not transforms:
        if verbose:
            print("  [ASTRO] no valid transform candidates")
        return None, 0, np.inf

    best     = max(transforms, key=lambda x: x['n'])
    sc, R, t = best['scale'], best['R'], best['t']

    # --- final cross-match using all in-frame catalog stars ---
    cat_t_all = _apply_sim(cat_pix_f, sc, R, t)
    d_all, _  = cKDTree(det_xy).query(cat_t_all, distance_upper_bound=pix_tol)
    matched   = d_all < pix_tol

    if matched.sum() < min_matches:
        if verbose:
            print(f"  [ASTRO] only {matched.sum()} matches (need {min_matches})")
        return None, 0, np.inf

    cat_ra_m  = cat_ra_f[matched]
    cat_dec_m = cat_dec_f[matched]
    cat_t_m   = cat_t_all[matched]
    _, nn     = cKDTree(det_xy).query(cat_t_m)
    det_xy_m  = det_xy[nn]

    # --- sigma-clip outliers ---
    for _ in range(3):
        if len(det_xy_m) < 4:
            break
        resid = np.linalg.norm(det_xy_m - cat_t_m, axis=1)
        rms_c = float(np.sqrt(np.mean(resid ** 2)))
        keep  = resid < 3.0 * max(rms_c, 0.1)
        cat_ra_m  = cat_ra_m[keep]
        cat_dec_m = cat_dec_m[keep]
        det_xy_m  = det_xy_m[keep]
        cat_t_m   = cat_t_m[keep]

    if len(det_xy_m) < 4:
        return None, 0, np.inf

    # --- fit TAN WCS ---
    try:
        sky     = _SkyCoord(ra=cat_ra_m * _apu.deg, dec=cat_dec_m * _apu.deg)
        wcs_new = _fit_wcs_from_points(
            (det_xy_m[:, 0], det_xy_m[:, 1]),
            sky, projection='TAN', sip_degree=None)
    except Exception as exc:
        if verbose:
            print(f"  [ASTRO] WCS fit failed: {exc}")
        return None, 0, np.inf

    # --- compute reprojection RMS ---
    repix  = wcs_new.all_world2pix(
        np.column_stack([cat_ra_m, cat_dec_m]), 0)
    rms_px = float(np.sqrt(np.mean(np.sum(
        (det_xy_m - repix) ** 2, axis=1))))

    if rms_px > accept_rms_px:
        if verbose:
            print(f"  [ASTRO] rms={rms_px:.2f} px > {accept_rms_px} — rejected")
        return None, 0, np.inf

    return wcs_new, len(det_xy_m), rms_px


# ---------------------------------------------------------------------------
# Main photometry entry point
# ---------------------------------------------------------------------------

def photometric_calibration_ros(fits_path, out_dir,
                                  aperture_radius=5.0,
                                  threshold_sigma=1.2,
                                  min_pixels=5,
                                  catalog_radius_arcmin=15.0,
                                  min_cal_stars=5,
                                  sigma_clip=3.0,
                                  vsx_tol_arcsec=2.0,
                                  preloaded_catalog=None,
                                  preloaded_vsx=None,
                                  targets=None,
                                  verbose=False):
    """Fixed-aperture photometric calibration for a ROS2 reduced image.

    If preloaded_catalog / preloaded_vsx are provided (from run_photometry_pass)
    the network downloads are skipped and the cached data are reused.
    Writes phot_<stem>.txt, phot_<stem>.png, and appends to photcal.txt.
    Returns a result dict, or None if skipped/failed.
    """
    basename = os.path.basename(fits_path)
    stem     = os.path.splitext(basename)[0]

    data, hdr = fits.getdata(fits_path, header=True)
    data = data.astype(np.float64)

    # WCS is required for sky coordinates
    if 'CRVAL1' not in hdr or 'CRPIX1' not in hdr:
        print(f"  [PHOT] {basename}: no WCS (CRVAL1/CRPIX1 missing), skipping")
        return None
    try:
        wcs = _WCS(hdr, naxis=2)
        ny, nx = data.shape
        ra_ctr, dec_ctr = map(float, wcs.all_pix2world([[nx / 2, ny / 2]], 0)[0])
    except Exception as exc:
        print(f"  [PHOT] {basename}: WCS error ({exc}), skipping")
        return None

    filt     = hdr.get('FILTER', '').strip()
    exptime  = float(hdr.get('EXPTIME', hdr.get('EXPOSED', 1.0)))
    date_obs = hdr.get('DATE-OBS', hdr.get('DATE_OBS', ''))
    obj_name = hdr.get('OBJECT',   hdr.get('OBJNAME', ''))
    gain     = float(hdr.get('GAIN',    _ROS2_GAIN))
    rdnoise  = float(hdr.get('RONOISE', _ROS2_RONOISE))  # ADU/pix

    radius_deg = catalog_radius_arcmin / 60.0

    print(f"  [PHOT] {basename}  filt={filt}  "
          f"center=({ra_ctr:.4f}, {dec_ctr:.4f})")

    # --- reference catalog (preloaded or downloaded) ---
    if preloaded_catalog is not None:
        cat_name, cat_ra, cat_dec, cat_mag, cat_mag_err = preloaded_catalog
    else:
        if verbose:
            print(f"  [PHOT] Downloading catalog (filter='{filt}'):")
        cat_name, cat_ra, cat_dec, cat_mag, cat_mag_err = _download_optical_catalog(
            ra_ctr, dec_ctr, radius_deg, filt, verbose=verbose)

    if cat_name is None:
        return None

    # --- VSX filtering (preloaded or downloaded) ---
    if preloaded_vsx is not None:
        vsx_ra, vsx_dec = preloaded_vsx
    else:
        if verbose:
            print("  [PHOT] Downloading VSX:")
        vsx_ra, vsx_dec = _download_vsx(ra_ctr, dec_ctr, radius_deg, verbose=verbose)

    keep = _filter_variables(cat_ra, cat_dec, vsx_ra, vsx_dec,
                              tol_arcsec=vsx_tol_arcsec)
    n_vsx_removed = int((~keep).sum())
    cat_ra  = cat_ra[keep];  cat_dec     = cat_dec[keep]
    cat_mag = cat_mag[keep]; cat_mag_err = cat_mag_err[keep]
    n_cat   = len(cat_ra)
    if verbose and n_vsx_removed:
        print(f"  [PHOT] {n_vsx_removed} VSX vars removed; {n_cat} ref stars remain")

    # --- source detection ---
    try:
        sources, data_sub, rms_map = _detect_sources(data, threshold_sigma, min_pixels)
    except Exception as exc:
        print(f"  [PHOT] {basename}: detection failed ({exc}), skipping")
        return None

    n_src = len(sources)
    if verbose:
        print(f"  [PHOT] {n_src} sources detected")
    if n_src == 0:
        print(f"  [PHOT] {basename}: no sources detected, skipping")
        return None

    # --- astrometry refinement (WCS correction via quad-matching) ---
    # PROCSTAT = 1 → just reduced, WCS from header (may be poor) → try refinement
    # PROCSTAT > 1 → astrometry already solved in a previous run → skip refinement
    procstat = int(hdr.get('PROCSTAT', 1))
    if procstat > 1:
        if verbose:
            print(f"  [ASTRO] {basename}: PROCSTAT={procstat}, WCS already refined — skipping")
    elif n_src >= 4 and cat_name is not None:
        _astro_wcs, _astro_n, _astro_rms = _solve_astrometry(
            sources, wcs, cat_ra, cat_dec, cat_mag,
            data.shape, verbose=verbose)
        if _astro_wcs is not None:
            wcs = _astro_wcs
            # update PROCSTAT=2 in the FITS header on disk
            try:
                with fits.open(fits_path, mode='update') as hdul:
                    hdul[0].header['PROCSTAT'] = (2, "Processing status: 2=astrometry done")
            except Exception:
                pass
            print(f"  [ASTRO] WCS refined: {_astro_n} matches, "
                  f"rms={_astro_rms:.2f} px")
        else:
            print(f"  [ASTRO] {basename}: WCS refinement failed — skipping photometry")
            return None
    else:
        reason = "too few sources" if n_src < 4 else "no catalog"
        print(f"  [ASTRO] {basename}: cannot refine WCS ({reason}) — skipping photometry")
        return None

    # --- aperture photometry ---
    # per-pixel noise: sky RMS + Poisson source noise + read noise (all ADU)
    err_map = np.sqrt(rms_map ** 2
                      + np.maximum(data_sub, 0.0) / gain
                      + rdnoise ** 2)
    try:
        flux, flux_err, ap_flag = _sep.sum_circle(
            data_sub, sources['x'], sources['y'],
            aperture_radius, err=err_map)
    except Exception as exc:
        print(f"  [PHOT] {basename}: aperture photometry failed ({exc}), skipping")
        return None

    _INST_ZP = 25.0
    good = (flux > 0) & np.isfinite(flux) & (flux_err > 0) & np.isfinite(flux_err)
    inst_mag     = np.full(n_src, np.nan)
    inst_mag_err = np.full(n_src, np.nan)
    inst_mag[good]     = _INST_ZP - 2.5 * np.log10(flux[good])
    inst_mag_err[good] = (2.5 / np.log(10.0)) * flux_err[good] / flux[good]

    # --- pixel → sky ---
    try:
        sky     = wcs.all_pix2world(
            np.column_stack([sources['x'], sources['y']]), 0)
        det_ra  = sky[:, 0]
        det_dec = sky[:, 1]
    except Exception as exc:
        print(f"  [PHOT] {basename}: pix→sky failed ({exc}), skipping")
        return None

    # match tolerance: 3 pixels at ROS2 pixel scale
    match_tol_deg = (3.0 * _ROS2_PIXSCALE) / 3600.0

    # --- catalog cross-match ---
    cat_tree       = cKDTree(np.column_stack([cat_ra, cat_dec]))
    dists, cat_idx = cat_tree.query(np.column_stack([det_ra, det_dec]))
    matched        = dists < match_tol_deg

    # --- zeropoint fit ---
    fit_sel         = matched & good & np.isfinite(inst_mag_err)
    n_total_matched = int(fit_sel.sum())
    zp = zp_err = rms = None
    n_used = 0
    fit_mask_full = np.zeros(n_src, dtype=bool)

    if n_total_matched < min_cal_stars:
        print(f"  [PHOT] {basename}: only {n_total_matched} matched – no calibration")
    else:
        zp, zp_err, rms, n_used, fit_mask_sub = _fit_zeropoint(
            inst_mag[fit_sel],
            cat_mag[cat_idx[fit_sel]],
            inst_mag_err[fit_sel],
            cat_mag_err[cat_idx[fit_sel]],
            sigma=sigma_clip, min_stars=min_cal_stars)
        if fit_mask_sub is not None and zp is not None:
            sel_idx = np.where(fit_sel)[0]
            fit_mask_full[sel_idx[fit_mask_sub]] = True

    quality = _zp_quality_label(rms)

    # --- limiting magnitude ---
    lim_mag = _compute_lim_mag(zp, rms_map, aperture_radius)

    if zp is not None:
        lim_str = f"  lim(3\u03c3)={lim_mag:.2f}" if np.isfinite(lim_mag) else ""
        print(f"  [PHOT] ZP={zp:+.3f}\u00b1{zp_err:.3f}  rms={rms:.3f}"
              f"  N={n_used}/{n_total_matched}  [{quality}]"
              f"  cat={cat_name}{lim_str}")

    cal_mag = (inst_mag + zp if zp is not None
               else np.full(n_src, np.nan))
    cal_mag_err = (np.sqrt(inst_mag_err ** 2 + zp_err ** 2) if zp is not None
                   else np.full(n_src, np.nan))

    # --- forced photometry at target positions ---
    # For each target that is NOT already blind-detected within its error
    # radius, perform aperture photometry at the exact sky position.
    # These sources get flag=4 and are appended to the catalog.
    _INST_ZP = 25.0
    force_flag_4 = np.zeros(n_src, dtype=bool)
    forced_xy    = []   # (x, y) of forced sources to extend positions arrays

    if targets:
        ny_im, nx_im = data.shape
        for tgt in targets:
            tgt_ra  = tgt['ra_deg']
            tgt_dec = tgt['dec_deg']
            tol_deg = tgt['radius_arcsec'] / 3600.0

            # skip if already detected within radius
            dists_tgt = np.sqrt((det_ra - tgt_ra) ** 2 +
                                (det_dec - tgt_dec) ** 2)
            if dists_tgt.min() <= tol_deg:
                if verbose:
                    print(f"  [FORCED] ({tgt_ra:.5f},{tgt_dec:.5f}): "
                          f"already detected (sep={dists_tgt.min()*3600:.1f}\")")
                continue

            # project to pixel space
            try:
                tgt_x, tgt_y = wcs.all_world2pix([[tgt_ra, tgt_dec]], 0)[0]
            except Exception:
                continue
            if not (0 <= tgt_x < nx_im and 0 <= tgt_y < ny_im):
                if verbose:
                    print(f"  [FORCED] ({tgt_ra:.5f},{tgt_dec:.5f}): outside image")
                continue

            # aperture photometry
            try:
                f_f, fe_f, _ = _sep.sum_circle(
                    data_sub, [tgt_x], [tgt_y],
                    aperture_radius, err=err_map)
            except Exception as exc:
                if verbose:
                    print(f"  [FORCED] aperture failed: {exc}")
                continue

            flux_f  = float(f_f[0])
            fluxe_f = float(fe_f[0])
            if not (flux_f > 0 and np.isfinite(flux_f) and fluxe_f > 0):
                if verbose:
                    print(f"  [FORCED] ({tgt_ra:.5f},{tgt_dec:.5f}): "
                          f"non-positive flux (skipped)")
                continue

            im_f  = _INST_ZP - 2.5 * np.log10(flux_f)
            ime_f = (2.5 / np.log(10.0)) * fluxe_f / flux_f
            cm_f  = im_f + zp  if zp  is not None else np.nan
            cme_f = (float(np.sqrt(ime_f ** 2 + zp_err ** 2))
                     if zp is not None else np.nan)

            msg = (f"mag={cm_f:.3f}±{cme_f:.3f}"
                   if np.isfinite(cm_f) else f"inst={im_f:.3f} (no ZP)")
            print(f"  [FORCED] ({tgt_ra:.5f},{tgt_dec:.5f}): {msg}")

            # extend all per-source arrays with one extra row
            idx = len(det_ra)          # index of this new entry
            det_ra       = np.append(det_ra,       tgt_ra)
            det_dec      = np.append(det_dec,       tgt_dec)
            flux         = np.append(flux,         flux_f)
            flux_err     = np.append(flux_err,     fluxe_f)
            ap_flag      = np.append(ap_flag,      np.int32(0))
            inst_mag     = np.append(inst_mag,     im_f)
            inst_mag_err = np.append(inst_mag_err, ime_f)
            matched      = np.append(matched,      False)
            cat_idx      = np.append(cat_idx,      0)
            cal_mag      = np.append(cal_mag,      cm_f)
            cal_mag_err  = np.append(cal_mag_err,  cme_f)
            force_flag_4 = np.append(force_flag_4, True)
            forced_xy.append((tgt_x, tgt_y))

    # if any forced sources, extend the sources recarray for x/y lookup
    if forced_xy:
        extra = np.zeros(len(forced_xy),
                         dtype=np.dtype([('x', np.float64),
                                         ('y', np.float64),
                                         ('flux', np.float64)]))
        for k, (fx, fy) in enumerate(forced_xy):
            extra['x'][k]    = fx
            extra['y'][k]    = fy
            extra['flux'][k] = 0.0
        # build extended sources with only x, y, flux (enough for _save_phot_catalog)
        common_fields = [('x', np.float64), ('y', np.float64),
                         ('flux', np.float64)]
        src_ext = np.zeros(len(sources) + len(forced_xy),
                           dtype=np.dtype(common_fields))
        src_ext['x'][:len(sources)]    = sources['x']
        src_ext['y'][:len(sources)]    = sources['y']
        src_ext['flux'][:len(sources)] = sources['flux']
        for k, (fx, fy) in enumerate(forced_xy):
            src_ext['x'][len(sources) + k]    = fx
            src_ext['y'][len(sources) + k]    = fy
        sources_out = src_ext
    else:
        sources_out = sources

    # --- write text catalog ---
    phot_path  = os.path.join(out_dir, f"phot_{stem}.txt")
    calib_path = os.path.join(out_dir, "photcal.txt")

    _save_phot_catalog(
        phot_path, sources_out, det_ra, det_dec,
        flux, flux_err, ap_flag, inst_mag, inst_mag_err,
        matched, cat_idx, cat_mag, cat_mag_err,
        cal_mag, cal_mag_err,
        zp, zp_err, rms, n_used, n_total_matched,
        filt, cat_name, aperture_radius, date_obs, exptime, obj_name,
        image_shape=data.shape, lim_mag=lim_mag,
        force_flag_4=force_flag_4 if force_flag_4.any() else None)

    _append_calib_summary(
        calib_path, stem, obj_name, filt, cat_name,
        zp, zp_err, rms, n_used, n_total_matched, quality,
        aperture_radius, n_cat, n_vsx_removed,
        date_obs, exptime, lim_mag=lim_mag)

    # --- calibration plot PNG ---
    if zp is not None and n_total_matched >= min_cal_stars:
        plot_path = os.path.join(out_dir, f"phot_{stem}.png")
        _save_phot_plot(
            plot_path,
            inst_mag[fit_sel], cat_mag[cat_idx[fit_sel]],
            inst_mag_err[fit_sel], cat_mag_err[cat_idx[fit_sel]],
            fit_mask_full[fit_sel],
            zp, zp_err, rms, n_used, n_total_matched,
            filt, cat_name, lim_mag=lim_mag, filename=stem)

    return dict(zp=zp, zp_err=zp_err, rms=rms,
                n_used=n_used, n_total=n_total_matched,
                quality=quality, catalog=cat_name, filter=filt,
                lim_mag=lim_mag)


def run_photometry_pass(reduced_paths, out_dir,
                        aperture_radius=5.0, threshold_sigma=1.2,
                        min_pixels=5, catalog_radius_arcmin=15.0,
                        min_cal_stars=5, sigma_clip=3.0,
                        vsx_tol_arcsec=2.0,
                        group_tolerance_arcmin=1.0,
                        targets=None,
                        verbose=False):
    """Photometric calibration for a batch of reduced FITS files.

    Groups images by sky position so that catalog and VSX downloads are shared
    across dithered / repeated observations of the same field, avoiding
    redundant network queries.
    """
    if not reduced_paths:
        return

    # collect positions from WCS headers
    file_infos = []
    for path in reduced_paths:
        try:
            hdr  = fits.getheader(path)
            if 'CRVAL1' not in hdr or 'CRPIX1' not in hdr:
                continue
            filt = hdr.get('FILTER', '').strip()
            ra   = float(hdr['CRVAL1'])
            dec  = float(hdr['CRVAL2'])
            file_infos.append((path, ra, dec, filt))
        except Exception:
            continue

    if not file_infos:
        print("  [PHOT] No files with WCS – skipping photometry pass")
        return

    groups = _group_files_by_position(file_infos, group_tolerance_arcmin)
    print(f"  [PHOT] {len(file_infos)} file(s) → {len(groups)} sky region(s)")

    # remove stale summary so reruns don't duplicate rows
    calib_summary = os.path.join(out_dir, "photcal.txt")
    if os.path.exists(calib_summary):
        os.remove(calib_summary)

    radius_deg = catalog_radius_arcmin / 60.0

    for g_idx, (cra, cdec, group_files) in enumerate(groups, 1):
        if verbose:
            print(f"  [PHOT] Region {g_idx}/{len(groups)}: "
                  f"RA={cra:.4f} Dec={cdec:.4f}  ({len(group_files)} files)")

        # VSX: one download per region
        if verbose:
            print("  [PHOT] Downloading VSX:")
        vsx_ra, vsx_dec = _download_vsx(cra, cdec, radius_deg, verbose=verbose)
        preloaded_vsx = (vsx_ra, vsx_dec)

        # catalog: one download per filter per region
        catalogs = {}
        for path, ra, dec, filt in group_files:
            if filt not in catalogs:
                if verbose:
                    print(f"  [PHOT] Downloading catalog (filter='{filt}'):")
                catalogs[filt] = _download_optical_catalog(
                    cra, cdec, radius_deg, filt, verbose=verbose)

        # photometry per file
        for path, ra, dec, filt in group_files:
            photometric_calibration_ros(
                path, out_dir,
                aperture_radius=aperture_radius,
                threshold_sigma=threshold_sigma,
                min_pixels=min_pixels,
                catalog_radius_arcmin=catalog_radius_arcmin,
                min_cal_stars=min_cal_stars,
                sigma_clip=sigma_clip,
                vsx_tol_arcsec=vsx_tol_arcsec,
                preloaded_catalog=catalogs.get(filt),
                preloaded_vsx=preloaded_vsx,
                targets=targets,
                verbose=verbose,
            )


# ============================================================================
def _reduce_one_image(sf, bias_data, flat_data, bpm, fringe_data, fringe_precomp,
                      filt, out_dir, bias_basename, flat_basename,
                      no_crclean, fringe_box, fringe_max_shift, fringe_shift_step):
    """Process one science frame: bias/flat/BPM/defringe/CR → write red_*.fits.
    Returns (outpath, log_line).
    """
    sci_data, sci_hdr = load_fits(sf)
    reduced = (sci_data - bias_data) / flat_data
    if bpm is not None:
        reduced[bpm] = np.nan

    fringe_scale = fringe_dx = fringe_dy = 0.0
    fringe_rms  = np.inf
    fringe_npix = 0
    if filt == "z" and fringe_data is not None:
        fringe_fit  = fit_fringe_model(
            reduced, fringe_data,
            background_box=fringe_box,
            max_shift=fringe_max_shift,
            shift_step=fringe_shift_step,
            _fringe_precomp=fringe_precomp,
        )
        fringe_scale = fringe_fit["scale"]
        fringe_dx    = fringe_fit["dx"]
        fringe_dy    = fringe_fit["dy"]
        fringe_rms   = fringe_fit["rms"]
        fringe_npix  = fringe_fit["npix"]
        reduced      = reduced - fringe_fit["model"]

    ncr = 0
    if not no_crclean:
        gain    = sci_hdr.get("GAIN",    1.0)
        rdnoise = sci_hdr.get("RONOISE", 4.5) * gain
        nanmask = ~np.isfinite(reduced)
        reduced[nanmask] = np.nanmedian(reduced)
        crmask, reduced = detect_cosmics(
            reduced, inmask=nanmask,
            sigclip=3.5, sigfrac=0.3, objlim=3.0,
            gain=gain, readnoise=rdnoise, satlevel=65535.0,
            niter=6, sepmed=True, cleantype='meanmask',
            fsmode='median', psfmodel='gauss',
            psffwhm=2.5, psfsize=7, verbose=False,
        )
        ncr = int(crmask[~nanmask].sum())
        reduced[nanmask] = np.nan

    _base = os.path.basename(sf)
    for _suf in ('.fits.gz', '.fits.fz', '.fz', '.fits'):
        if _base.endswith(_suf):
            _base = _base[:-len(_suf)]
            break
    outname = "red_" + _base + ".fits"
    outpath = os.path.join(out_dir, outname)
    sci_hdr["HISTORY"] = f"Bias subtracted: {bias_basename}"
    sci_hdr["HISTORY"] = f"Flat divided: {flat_basename}"
    if filt == "z" and fringe_data is not None:
        sci_hdr["FRNGSCAL"] = (float(fringe_scale), "Best-fit fringe scale")
        sci_hdr["FRNGDX"]   = (float(fringe_dx),    "Fringe x-shift [pix]")
        sci_hdr["FRNGDY"]   = (float(fringe_dy),    "Fringe y-shift [pix]")
        sci_hdr["FRNGRMS"]  = (float(fringe_rms),   "Robust RMS after fringe fit")
        sci_hdr["FRNGNPX"]  = (int(fringe_npix),    "Pixels used in fringe fit")
        sci_hdr["HISTORY"]  = (
            f"Defringe scale={fringe_scale:.2f} shift=({fringe_dx:.1f},{fringe_dy:.1f})"
        )
    if not no_crclean:
        sci_hdr["HISTORY"] = f"CR cleaned (astroscrappy): {ncr} pixels"
    sci_hdr["PROCSTAT"] = (4, "Processing status: 1=reduced, 2=astrometry done, 4=rospipe reduced (WCS from telescope)")
    fits.writeto(outpath, reduced.astype(np.float32), header=sci_hdr, overwrite=True)

    cr_info = f", {ncr} CR" if not no_crclean else ""
    fr_info = ""
    if filt == "z" and fringe_data is not None:
        fr_info = (
            f", fringe scale={fringe_scale:.1f},"
            f" shift=({fringe_dx:.1f},{fringe_dy:.1f}),"
            f" rms={fringe_rms:.2f}"
        )
    return outpath, f"[{filt}] {os.path.basename(sf)} -> {outname}{cr_info}{fr_info}"


# ============================================================================
# Live calibration helpers
# ============================================================================

def _download_live_cal(url, dest_path, timeout=60, retries=2):
    """Download a calibration file from url to dest_path.
    Returns True on success, False on failure.
    """
    for attempt in range(retries):
        try:
            resp = _requests.get(url, timeout=timeout, stream=True)
            if resp.status_code == 200:
                with open(dest_path, 'wb') as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                return True
            print(f"  [LIVE] HTTP {resp.status_code}: {url}")
        except Exception as exc:
            print(f"  [LIVE] Attempt {attempt + 1} failed: {exc}")
        if attempt < retries - 1:
            _time.sleep(2 ** attempt)
    return False


def _apply_live_calibrations(biases, flats, filters, cal_dir, timeout=60):
    """Download latest bias/flat from the ROS2 server for each filter.
    Falls back silently to the existing local entry on failure.
    Returns updated (biases, flats) dicts.
    """
    biases = dict(biases)
    flats  = dict(flats)
    for filt in sorted(filters):
        quad = _ROS2_LIVE_QUAD.get(filt)
        if quad is None:
            print(f"  [LIVE] No quadrant mapping for filter '{filt}', skipping")
            continue
        for kind, url, dest_name, d in [
            ('flat',
             f"{_ROS2_LIVE_BASE}/masterflats/mf_latest_{quad}_{filt}.fits.gz",
             f"live_flat_{filt}.fits.gz", flats),
            ('bias',
             f"{_ROS2_LIVE_BASE}/masterbias/mb_latest_{quad}_{filt}.fits.gz",
             f"live_bias_{filt}.fits.gz", biases),
        ]:
            dest = os.path.join(cal_dir, dest_name)
            print(f"  [LIVE] {kind} ({filt}) <- {url}")
            if _download_live_cal(url, dest, timeout=timeout):
                d[filt] = dest
                print(f"  [LIVE] {kind} ({filt}) OK -> {dest_name}")
            else:
                fallback = d.get(filt)
                if fallback:
                    print(f"  [LIVE] {kind} ({filt}) failed — fallback: "
                          f"{os.path.basename(fallback)}")
                else:
                    print(f"  [LIVE] {kind} ({filt}) failed — no local fallback")
    return biases, flats


# ============================================================================
def main():
    # default calibration directory: <script_dir>/cal/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_cal = os.path.join(script_dir, "cal")

    ap = argparse.ArgumentParser(description="CCD reduction: (img - bias) / flat + CR cleaning")
    ap.add_argument("sci_dir", help="Folder with science files (IMG*.fits)")
    ap.add_argument("cal_dir", nargs="?", default=None,
                     help="Calibration folder (default: <script_dir>/cal/)")
    ap.add_argument("-o", "--outdir", default=None,
                     help="Output folder (default: sci_dir/reduced)")
    ap.add_argument("--phot-only", action="store_true",
                     help="Skip reduction: run photometry only on existing red_*.fits in --outdir")
    ap.add_argument("--no-crclean", action="store_true",
                     help="Skip cosmic-ray cleaning")
    ap.add_argument("--no-defringe", action="store_true",
                     help="Skip z-band fringe removal")
    ap.add_argument("--fringe-box", type=int, default=129,
                     help="Median filter box size for fringe isolation (default: 129)")
    ap.add_argument("--fringe-max-shift", type=float, default=2.0,
                     help="Maximum fringe shift [pixels] in the fit (default: 2.0)")
    ap.add_argument("--fringe-shift-step", type=float, default=0.5,
                     help="Fringe shift grid step [pixels] (default: 0.5)")
    ap.add_argument("--no-bpm", action="store_true",
                     help="Disable bad pixel mask")
    ap.add_argument("--live", action="store_true",
                     help="Download latest master bias/flat from ROS2 server; "
                          "fall back to local files on failure")
    ap.add_argument("--no-phot", action="store_true",
                     help="Skip photometric calibration after reduction")
    ap.add_argument("--phot-radius", type=float, default=5.0,
                     help="Aperture radius for photometry [pixels] (default: 5.0)")
    ap.add_argument("--phot-thresh", type=float, default=1.2,
                     help="Source detection threshold [sigma] for photometry (default: 1.2)")
    ap.add_argument("--phot-min-pixels", type=int, default=5,
                     help="Min connected pixels for photometry detection (default: 5)")
    ap.add_argument("--phot-group-radius", type=float, default=1.0,
                     help="Sky grouping radius [arcmin] for catalog sharing (default: 1.0)")
    ap.add_argument("--phot-cat-radius", type=float, default=15.0,
                     help="Catalog search radius [arcmin] for photometry (default: 15)")
    ap.add_argument("--phot-verbose", action="store_true",
                     help="Verbose output for photometry step")
    ap.add_argument("--target-file", "-f", default=None,
                     help="File with target positions for forced photometry "
                          "(RA DEC [radius_arcsec] per line, decimal or sexagesimal)")
    ap.add_argument("--jobs", "-j", type=int, default=min(4, os.cpu_count() or 1),
                     help="Parallel workers for image reduction (default: min(4,cpu_count))")
    args = ap.parse_args()

    sci_dir = os.path.abspath(args.sci_dir)
    cal_dir = os.path.abspath(args.cal_dir) if args.cal_dir else default_cal
    out_dir = os.path.abspath(args.outdir) if args.outdir else os.path.join(sci_dir, "reduced")
    os.makedirs(out_dir, exist_ok=True)

    # --- parse forced-photometry target file ---
    targets = None
    if args.target_file:
        targets = parse_target_file(args.target_file)
        print(f"Forced photometry: {len(targets)} target(s) from {args.target_file}")

    # --- phot-only: skip reduction, re-run photometry on existing red_*.fits ---
    if args.phot_only:
        existing = sorted(glob.glob(os.path.join(out_dir, "red_*.fits")))
        if not existing:
            sys.exit(f"No red_*.fits found in {out_dir}")
        print(f"[phot-only] {len(existing)} file in {out_dir}")
        run_photometry_pass(
            existing, out_dir,
            aperture_radius=args.phot_radius,
            threshold_sigma=args.phot_thresh,
            min_pixels=args.phot_min_pixels,
            catalog_radius_arcmin=args.phot_cat_radius,
            group_tolerance_arcmin=args.phot_group_radius,
            targets=targets,
            verbose=args.phot_verbose,
        )
        print(f"\nDone – photometry in {out_dir}")
        return

    # --- collect files ---
    science_all, biases_sci, flats_sci = collect_files(sci_dir)
    if cal_dir != sci_dir:
        _, biases_cal, flats_cal = collect_files(cal_dir)
        biases_sci.update(biases_cal)
        flats_sci.update(flats_cal)

    biases, flats = biases_sci, flats_sci

    # --- live calibrations: try to download latest from ROS2 server ---
    if args.live:
        biases, flats = _apply_live_calibrations(
            biases, flats, science_all.keys(), cal_dir)

    # --- load master fringe z (if present) ---
    fringe_data = None
    if not args.no_defringe:
        fringe_candidates = glob.glob(os.path.join(cal_dir, "master_fringe*z*.fits"))
        if fringe_candidates:
            fringe_data = fits.getdata(fringe_candidates[0]).astype(np.float64)
            print(f"Master fringe (z): {fringe_candidates[0]}")
        else:
            print("No master_fringe_z found, defringing disabled")

    # precompute fringe high-pass once (avoids re-smoothing per frame)
    fringe_precomp = None
    if fringe_data is not None:
        fringe_precomp = precompute_fringe_hp(fringe_data, background_box=args.fringe_box)

    if not science_all:
        sys.exit(f"No IMG*.fits files found in {sci_dir}")

    print(f"Filters found: {sorted(science_all.keys())}")
    print(f"Bias available: {sorted(biases.keys())}")
    print(f"Flats available: {sorted(flats.keys())}")
    print()

    reduced_files = []   # collect for photometry pass

    # --- reduction ---
    for filt, sci_files in sorted(science_all.items()):
        if filt not in biases:
            print(f"[SKIP] filter {filt}: master_bias missing")
            continue
        if filt not in flats:
            print(f"[SKIP] filter {filt}: master_flat missing")
            continue

        bias_data, _ = load_fits(biases[filt])
        flat_data, _ = load_fits(flats[filt])
        # avoid division by zero
        flat_data[flat_data == 0] = 1.0
        # load BPM for this filter
        bpm = None
        if not args.no_bpm:
            bpm_path = os.path.join(cal_dir, f"bpm_{filt}.fits")
            if os.path.isfile(bpm_path):
                bpm = fits.getdata(bpm_path).astype(bool)
                print(f"  [{filt}] BPM: {np.sum(bpm)} px "
                      f"({np.sum(bpm) / bpm.size * 100:.1f}%)")
            else:
                print(f"  [{filt}] BPM not found: {bpm_path}")

        bias_basename = os.path.basename(biases[filt])
        flat_basename = os.path.basename(flats[filt])
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(
                    _reduce_one_image,
                    sf, bias_data, flat_data, bpm,
                    fringe_data, fringe_precomp,
                    filt, out_dir, bias_basename, flat_basename,
                    args.no_crclean,
                    args.fringe_box, args.fringe_max_shift, args.fringe_shift_step,
                ): sf
                for sf in sci_files
            }
            for fut in as_completed(futures):
                outpath, log_line = fut.result()
                print(log_line)
                reduced_files.append(outpath)

    # --- photometric calibration (grouped by sky position) ---
    if not args.no_phot:
        run_photometry_pass(
            reduced_files, out_dir,
            aperture_radius=args.phot_radius,
            threshold_sigma=args.phot_thresh,
            min_pixels=args.phot_min_pixels,
            catalog_radius_arcmin=args.phot_cat_radius,
            group_tolerance_arcmin=args.phot_group_radius,
            targets=targets,
            verbose=args.phot_verbose,
        )

    print(f"\nDone – reduced files in {out_dir}")


if __name__ == "__main__":
    main()
