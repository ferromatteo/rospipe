#!/usr/bin/env python3
"""
Build a normalised master fringe frame from z-band images.

Steps:
  1. Bias-subtract and flat-divide every input frame
  2. Normalise each frame by its median sky -> sky = 1
  3. Sigma-clipped median stack -> fringe pattern around 1
  4. Subtract 1 -> zero-mean additive fringe pattern

The output pixel values are the *fractional* fringe amplitude
(e.g. +0.05 = fringe adds 5% of sky).

Usage:
    python make_fringe.py <img_dir>
    python make_fringe.py <img_dir> --bias BIAS --flat FLAT -o OUTPUT
"""
import argparse
import glob
import os
import sys
from functools import partial
from multiprocessing import Pool, cpu_count

from astropy.io import fits
from astroscrappy import detect_cosmics
import numpy as np
from scipy.ndimage import gaussian_filter


def odd_box(size):
    size = max(3, int(size))
    return size + 1 if size % 2 == 0 else size


def _process_one_frame(filepath, shape, bias, flat, is_reduced, background_box,
                       bpm=None):
    """Process a single frame: bias/flat/CR + normalise. Returns None on skip."""
    data = fits.getdata(filepath).astype(np.float64)
    if data.shape != shape:
        return None

    if is_reduced:
        reduced = data
    else:
        reduced = (data - bias) / flat
        nanmask = ~np.isfinite(reduced)
        med_fill = np.nanmedian(reduced)
        reduced[nanmask] = med_fill
        hdr_f = fits.getheader(filepath)
        gain = hdr_f.get("GAIN", 1.0)
        rdnoise = hdr_f.get("RONOISE", 4.5) * gain
        crmask, reduced = detect_cosmics(
            reduced, inmask=nanmask,
            sigclip=3.5, sigfrac=0.3, objlim=3.0,
            gain=gain, readnoise=rdnoise, satlevel=65535.0,
            niter=4, sepmed=True, cleantype="meanmask",
            fsmode="median", psfmodel="gauss", psffwhm=2.5, psfsize=7,
            verbose=False,
        )
        reduced[nanmask] = np.nan

    if bpm is not None:
        reduced[bpm] = np.nan

    med = np.nanmedian(reduced)
    if med <= 0:
        return None

    normalised = reduced / med
    if background_box > 0:
        gsigma = odd_box(background_box) / 3.0
        smooth = gaussian_filter(normalised, sigma=gsigma, mode="nearest")
        normalised = normalised - smooth + np.nanmedian(smooth)

    return normalised.astype(np.float32)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_cal = os.path.join(script_dir, "cal")

    ap = argparse.ArgumentParser(description="Build master fringe from z-band images")
    ap.add_argument("img_dir", help="Cartella (ricorsiva) con IMG*.fits z-band")
    ap.add_argument("--bias", default=None,
                    help="Master bias z (default: auto from cal/)")
    ap.add_argument("--flat", default=None,
                    help="Master flat z (default: auto from cal/)")
    ap.add_argument("-o", "--output", default=None,
                    help="Output file (default: cal/master_fringe_z.fits)")
    ap.add_argument("--max", type=int, default=0,
                    help="Max images to use (0 = all)")
    ap.add_argument("--sigma-clip", type=float, default=3.0,
                    help="Sigma clipping threshold (default: 3.0)")
    ap.add_argument("--min-exptime", type=float, default=20.0,
                    help="Min exposure time [s] (default: 20)")
    ap.add_argument("--max-exptime", type=float, default=600.0,
                    help="Max exposure time [s] (default: 600)")
    ap.add_argument("--background-box", type=int, default=129,
                    help="Lato del filtro mediano per rimuovere gradienti lenti (default: 129)")
    ap.add_argument("--reduced", action="store_true",
                    help="Input images are already reduced (skip bias/flat/CR)")
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel workers (0 = all CPUs, 1 = sequential)")
    ap.add_argument("--bpm", default=None,
                    help="Bad pixel mask FITS (default: auto from cal/bpm_z.fits)")
    ap.add_argument("--no-bpm", action="store_true",
                    help="Disable bad pixel masking")
    args = ap.parse_args()

    # --- find calibrations (skip if --reduced) ---
    bias, flat = None, None
    if not args.reduced:
        if args.bias:
            bias_path = args.bias
        else:
            candidates = glob.glob(os.path.join(default_cal, "master_bias*z*.fits"))
            if not candidates:
                sys.exit(f"No z-band master_bias in {default_cal}")
            bias_path = candidates[0]

        if args.flat:
            flat_path = args.flat
        else:
            candidates = glob.glob(os.path.join(default_cal, "master_flat*z*.fits"))
            if not candidates:
                sys.exit(f"No z-band master_flat in {default_cal}")
            flat_path = candidates[0]

        print(f"Bias: {bias_path}")
        print(f"Flat: {flat_path}")

        bias = fits.getdata(bias_path).astype(np.float64)
        flat = fits.getdata(flat_path).astype(np.float64)
        flat[flat == 0] = 1.0

    # --- load BPM ---
    bpm = None
    if not args.no_bpm:
        if args.bpm:
            bpm_path = args.bpm
        else:
            bpm_path = os.path.join(default_cal, "bpm_z.fits")
        if os.path.isfile(bpm_path):
            bpm = fits.getdata(bpm_path).astype(bool)
            print(f"BPM: {bpm_path}  ({np.sum(bpm)} bad pixels, "
                  f"{np.sum(bpm) / bpm.size * 100:.1f}%)")
        else:
            print(f"BPM non trovata: {bpm_path} — proseguo senza")

    # --- collect z-band images ---
    pattern = "red_IMG*.fits" if args.reduced else "IMG*.fits"
    all_fits = sorted(glob.glob(os.path.join(args.img_dir, "**", pattern),
                                recursive=True))
    z_files = []
    for f in all_fits:
        hdr = fits.getheader(f)
        filt = hdr.get("FILTER", "").strip()
        if filt != "z":
            continue
        if not args.reduced:
            exp = hdr.get("EXPTIME", 0)
            if exp < args.min_exptime or exp > args.max_exptime:
                continue
        z_files.append(f)

    if not z_files:
        sys.exit(f"No z-band IMG*.fits (exptime {args.min_exptime}-{args.max_exptime}s) "
                 f"found in {args.img_dir}")
    if args.max > 0:
        z_files = z_files[:args.max]
    print(f"{len(z_files)} z-band images selected\n")

    # --- normalise each frame ---
    shape = fits.getdata(z_files[0]).shape
    nworkers = args.workers if args.workers > 0 else cpu_count()

    process_fn = partial(
        _process_one_frame,
        shape=shape,
        bias=bias,
        flat=flat,
        is_reduced=args.reduced,
        background_box=args.background_box,
        bpm=bpm,
    )

    stack = []
    if nworkers == 1:
        # sequential
        for i, f in enumerate(z_files):
            result = process_fn(f)
            if result is not None:
                stack.append(result)
            if (i + 1) % 10 == 0 or i == len(z_files) - 1:
                print(f"  {i+1}/{len(z_files)} processed")
    else:
        print(f"Using {nworkers} parallel workers")
        with Pool(nworkers) as pool:
            for i, result in enumerate(pool.imap(process_fn, z_files)):
                if result is not None:
                    stack.append(result)
                if (i + 1) % 10 == 0 or i == len(z_files) - 1:
                    print(f"  {i+1}/{len(z_files)} processed")

    if len(stack) < 5:
        sys.exit(f"Only {len(stack)} usable frames, need >= 5")

    # --- sigma-clipped median (3 iterations) ---
    print(f"\nStacking {len(stack)} frames (sigma={args.sigma_clip}) ...")
    cube = np.array(stack)
    for iteration in range(3):
        med_plane = np.nanmedian(cube, axis=0)
        std_plane = np.nanstd(cube, axis=0)
        std_plane[std_plane == 0] = 1.0
        bad = np.abs(cube - med_plane[np.newaxis]) > args.sigma_clip * std_plane[np.newaxis]
        cube = np.where(bad, np.nan, cube)
        nrej = np.sum(bad)
        print(f"  iteration {iteration+1}: rejected {nrej} pixels")
    master = np.nanmedian(cube, axis=0)

    # zero-mean fringe pattern
    master -= 1.0
    master[~np.isfinite(master)] = 0.0

    # count valid frames per pixel — mask where too few contributed
    n_valid = np.sum(np.isfinite(cube), axis=0)
    min_contrib = max(5, len(stack) // 4)
    n_masked = np.sum(n_valid < min_contrib)
    master[n_valid < min_contrib] = 0.0
    print(f"  masked {n_masked} pixels with < {min_contrib} contributors")

    # clip extreme outliers (bad pixels / edge artifacts)
    master = np.clip(master, -0.5, 0.5)

    # --- save ---
    out_path = args.output or os.path.join(default_cal, "master_fringe_z.fits")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    hdr = fits.Header()
    hdr["FILTER"] = "z"
    hdr["NFRAMES"] = (len(stack), "Frames used")
    hdr["SIGCLIP"] = (args.sigma_clip, "Sigma clipping threshold")
    hdr["BKGBOX"] = (args.background_box, "Median-filter box for background flattening")
    hdr["HISTORY"] = "Master fringe (zero-mean, normalised)"
    hdr["HISTORY"] = f"Built from {len(stack)} z-band frames"
    fits.writeto(out_path, master.astype(np.float32), header=hdr, overwrite=True)

    rms = np.std(master)
    peak = np.max(np.abs(master))
    print(f"\nSaved: {out_path}")
    print(f"  RMS  = {rms:.4f}  ({rms*100:.1f}% of sky)")
    print(f"  Peak = {peak:.4f}  ({peak*100:.1f}% of sky)")


if __name__ == "__main__":
    main()
