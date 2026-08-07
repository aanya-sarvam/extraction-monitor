"""
build_snapshot.py -- RUN LOCALLY (needs gcloud + your aanya@sarvam.ai auth).

Bundles a small set of deeds into ONE self-contained file, deeds_snapshot.json,
which the Streamlit viewer reads on Streamlit Community Cloud (where there is no
gcloud and no GCS access). Each deed gets its registry input JSON + its Gemini
extracted rows, so the deployed app needs zero cloud calls.

Usage (from the deed-validator repo root):
  python build_snapshot.py --n 15
  python build_snapshot.py --regnos 910010900711,940640200020   # pin specific deeds
  python build_snapshot.py --n 15 --seed 7                        # different random 15

Then commit deeds_snapshot.json to the repo and deploy.
"""
from __future__ import annotations

import argparse
import datetime
import json
import random
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

CSV_DEFAULT = "data/vertex_batch/grounding_fields.csv"
PREFIX_DEFAULT = "gs://vision-vertex-batch-asia-south1/inputs/pdf"
GCLOUD = shutil.which("gcloud") or "gcloud"

ROW_COLS = ["field_id", "item_index", "attr", "english_value", "found",
            "odia_text", "script", "page", "confidence", "latin_readback",
            "notes", "book_label"]


def fetch(reg, prefix):
    uri = f"{prefix}/{reg}.json"
    try:
        p = subprocess.run([GCLOUD, "storage", "cat", uri],
                           capture_output=True, timeout=60)
    except Exception as e:  # noqa: BLE001
        return None, str(e)[:160]
    if p.returncode != 0:
        return None, p.stderr.decode("utf-8", "replace").strip()[:160]
    try:
        return json.loads(p.stdout.decode("utf-8")), None
    except json.JSONDecodeError as e:
        return None, f"bad json: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_DEFAULT)
    ap.add_argument("--prefix", default=PREFIX_DEFAULT)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--regnos", help="comma-separated reg_nos (overrides --n)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="deeds_snapshot.json")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype=str).fillna("")
    df["reg_no"] = df["reg_no"].str.strip()
    all_regs = df["reg_no"].dropna().unique().tolist()
    have = set(all_regs)

    if args.regnos:
        want = [r.strip() for r in args.regnos.split(",") if r.strip()]
        missing = [r for r in want if r not in have]
        if missing:
            print(f"WARNING: not in CSV, skipping: {missing}")
        sample = [r for r in want if r in have]
    else:
        rng = random.Random(args.seed)
        sample = rng.sample(all_regs, min(args.n, len(all_regs)))

    print(f"Fetching {len(sample)} input JSON(s) from {args.prefix} ...")
    fetched = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(fetch, r, args.prefix): r for r in sample}
        for fut in futs:
            fetched[futs[fut]] = fut.result()

    deeds, ok = {}, 0
    for reg in sample:
        api, err = fetched[reg]
        if err:
            print(f"  SKIP {reg}: {err}")
            continue
        sub = df[df["reg_no"] == reg]
        rows = [{c: r.get(c, "") for c in ROW_COLS} for _, r in sub.iterrows()]
        deeds[reg] = {"input": api, "gemini": rows}
        ok += 1
        print(f"  ok   {reg}  ({len(rows)} field row(s))")

    snap = {"meta": {"built_at": datetime.datetime.now().isoformat(timespec="seconds"),
                     "n": ok, "csv": args.csv, "prefix": args.prefix},
            "deeds": deeds}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=1)
    print(f"\nWrote {args.out}: {ok}/{len(sample)} deed(s) bundled. "
          f"Commit this file and deploy.")


if __name__ == "__main__":
    main()
