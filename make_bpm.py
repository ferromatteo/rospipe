#!/usr/bin/env python3
"""
Build a bad-pixel mask (BPM) for each ROSS2 filter/quadrant.

Sources for flagging:
  1. NaN/Inf in master flat  (dead / zero-padded region)
  2. Low flat response       (vignetting, flat < --flat-min)
  3. Hot pixels in bias      (> N sigma above median)
  4. Deviant flat pixels     (> N sigma from median in well-illuminated zone)

Output: one FITS per filter in cal/, pixel = 0 (good) or 1 (bad).

Usage:
    python make_bpm.py
    python make_bpm.py --cal cal/ --flat-min 0.5 --bias-sigma 5 --flat-sigma 7
"""
import argparse
import glob
import os
import sys

from astropy.io import fits
import numpy as np


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_cal = os.path.join(script_dir, "cal")

    ap = argparse.ArgumentParser(description="Build BPM per filter from master bias/flat")
    ap.add_argument("--cal", default=default_cal,
                    help="Cartella calibrazioni (default: cal/)")
    ap.add_argument("--flat-min", type=float, default=0.5,
                    help="Soglia minima flat (vignetting, default: 0.5)")
    ap.add_argument("--bias-sigma", type=float, default=5.0,
                    help="Sigma per hot pixel nel bias (default: 5)")
    ap.add_argument("--flat-sigma", type=float, default=7.0,
                    help="Sigma per pixel devianti nel flat (default: 7)")
    args = ap.parse_args()

    cal_dir = os.path.abspath(args.cal)
    bias_files = sorted(glob.glob(os.path.join(cal_dir, "master_bias*.fits")))
    flat_files = sorted(glob.glob(os.path.join(cal_dir, "master_flat*.fits")))

    if not bias_files or not flat_files:
        sys.exit(f"Servono master_bias e master_flat in {cal_dir}")

    # match bias/flat per filtro
    def get_filter(path):
        return fits.getheader(path).get("FILTER", "").strip()

    bias_by_filt = {get_filter(f): f for f in bias_files}
    flat_by_filt = {get_filter(f): f for f in flat_files}

    filters = sorted(set(bias_by_filt) & set(flat_by_filt))
    if not filters:
        sys.exit("Nessun filtro in comune tra bias e flat")

    print(f"Filtri: {filters}\n")

    for filt in filters:
        bias = fits.getdata(bias_by_filt[filt]).astype(np.float64)
        flat = fits.getdata(flat_by_filt[filt]).astype(np.float64)

        bpm = np.zeros(flat.shape, dtype=np.uint8)

        # 1. NaN/Inf nel flat (zona morta / zero-padded)
        nan_mask = ~np.isfinite(flat)
        bpm[nan_mask] = 1
        n_nan = int(np.sum(nan_mask))

        # 2. Flat basso (vignettatura)
        low_mask = np.isfinite(flat) & (flat < args.flat_min)
        bpm[low_mask] = 1
        n_low = int(np.sum(low_mask))

        # 3. Hot pixel nel bias (nella zona illuminata)
        good_region = np.isfinite(flat) & (flat >= args.flat_min)
        bmed = np.median(bias[good_region])
        bmad = np.median(np.abs(bias[good_region] - bmed))
        bsig = 1.4826 * bmad
        if bsig > 0:
            hot_mask = good_region & (bias > bmed + args.bias_sigma * bsig)
            bpm[hot_mask] = 1
            n_hot = int(np.sum(hot_mask))
        else:
            n_hot = 0

        # 4. Pixel devianti nel flat (zona ben illuminata)
        fmed = np.nanmedian(flat[good_region])
        fmad = np.median(np.abs(flat[good_region] - fmed))
        fsig = 1.4826 * fmad
        if fsig > 0:
            dev_mask = good_region & (np.abs(flat - fmed) > args.flat_sigma * fsig)
            # escludi quelli già flaggati come hot
            dev_mask &= ~bpm.astype(bool)
            bpm[dev_mask] = 1
            n_dev = int(np.sum(dev_mask))
        else:
            n_dev = 0

        total = int(np.sum(bpm))

        # salva
        out_name = f"bpm_{filt}.fits"
        out_path = os.path.join(cal_dir, out_name)
        hdr = fits.Header()
        hdr["FILTER"] = filt
        hdr["BPMNNAN"] = (n_nan, "NaN/Inf in flat")
        hdr["BPMLOW"] = (n_low, f"Flat < {args.flat_min}")
        hdr["BPMHOT"] = (n_hot, f"Bias > {args.bias_sigma} sigma")
        hdr["BPMDEV"] = (n_dev, f"Flat > {args.flat_sigma} sigma")
        hdr["BPMTOT"] = (total, "Total bad pixels")
        hdr["HISTORY"] = f"BPM built from master bias+flat, filter {filt}"
        fits.writeto(out_path, bpm, header=hdr, overwrite=True)

        pct = total / bpm.size * 100
        print(f"[{filt}] {out_name}: {total} bad ({pct:.1f}%)"
              f"  —  NaN={n_nan}  low_flat={n_low}  hot={n_hot}  deviant={n_dev}")

    print(f"\nDone — BPM salvate in {cal_dir}/")


if __name__ == "__main__":
    main()
