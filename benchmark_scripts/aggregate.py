r"""Collect raw results into one view. No reductions.

TEXT/VISION (run_morf_lerf): mean MoRF and mean LeRF curves per method, plus the
  area-between (lerf-morf, trapezoid) shown only as a convenience summary of the
  raw curves — you can ignore it and read the curves.
EQA (run_tgs_tps): TGS / TPS means.

    python aggregate.py --results-dir results
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.results_dir, "*.json")))
    print(f"loaded {len(files)} result files\n")

    curves, eqa = {}, {}
    for path in files:
        try:
            d = json.load(open(path))
        except Exception:
            continue
        task, method = d.get("task"), d.get("method")
        if "records" in d and d["records"]:
            morf = np.array([r["morf"] for r in d["records"]], float).mean(0)
            lerf = np.array([r["lerf"] for r in d["records"]], float).mean(0)
            area = float(np.trapz(lerf, dx=1) - np.trapz(morf, dx=1))
            curves.setdefault(task, {})[method] = (morf, lerf, area, d.get("n"))
        elif "summary" in d:
            s = d["summary"]
            eqa.setdefault(task, {})[method] = (s["tgs"], s["tps"], s.get("n"))

    if curves:
        print("MoRF / LeRF (raw mean curves; area = trapz(lerf)-trapz(morf), a convenience)")
        for task in sorted(curves):
            print(f"\n{task}")
            for m, (morf, lerf, area, n) in sorted(curves[task].items()):
                print(f"  {m:10s} area={area:+.4f}  n={n}")
                print(f"             morf={[round(float(v),3) for v in morf]}")
                print(f"             lerf={[round(float(v),3) for v in lerf]}")
    if eqa:
        print("\nEQA — TGS / TPS")
        for task in sorted(eqa):
            print(f"\n{task}")
            for m, (tgs, tps, n) in sorted(eqa[task].items()):
                print(f"  {m:10s} TGS={tgs:.4f} TPS={tps:.4f}  n={n}")
    if not (curves or eqa):
        print("(no results yet — run the task scripts first)")


if __name__ == "__main__":
    main()
