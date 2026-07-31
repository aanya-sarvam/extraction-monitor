"""
Ground Truth vs Gemini — Metadata-Ceiling Experiment Viewer
Run locally:  streamlit run ground_truth_metadata_experiment_view.py

DIAGNOSTIC VIEWER — this compares a Gemini run where the EXPERT-CORRECTED
ground truth itself was fed in as the metadata anchor (instead of the DB's
original OCR value), against that same ground truth. It answers "given a
known-good anchor, can Gemini still find/transcribe the matching text on the
page?" — an upper-bound test, not a real-world accuracy measurement. See
grounding_realtime_gemini_ground_truth.py for how this run was produced.

Expects these files in the same folder (or upload from the sidebar):
  ground_truth.csv        (DB export: deed_number, section, label,
                            corrected_english, corrected_odia,
                            gemini_original_english, position)
  realtime_fields_gt.csv  (the ground-truth-as-metadata Gemini run: reg_no,
                            field_id, item_index, attr, english_value, found,
                            odia_text, latin_readback, ...)
"""

import os
import pandas as pd
import streamlit as st

from diff_logic import load_ground_truth, load_realtime, build_comparison

st.set_page_config(page_title="GT-as-Metadata Ceiling Test", layout="wide")

st.title("🧪 Ground Truth vs Gemini — Metadata-Ceiling Experiment")
st.caption(
    "Gemini was given the EXPERT-CORRECTED ground truth itself as the "
    "metadata anchor (instead of the DB's original OCR value) — this tests "
    "an upper bound (\"can Gemini find it at all, given a perfect anchor?\"), "
    "not real-world accuracy. A mismatch here means Gemini got it wrong even "
    "with a known-correct anchor — worth checking the actual page image."
)

with st.sidebar:
    st.header("Data source")
    st.caption("Defaults to the bundled files — upload to override.")
    gt_f = st.file_uploader("ground_truth.csv", type="csv", key="gt")
    rt_f = st.file_uploader("realtime_fields_gt.csv", type="csv", key="rt")

gt = load_ground_truth(gt_f) if gt_f else (
    load_ground_truth("ground_truth.csv") if os.path.exists("ground_truth.csv") else None)
rt = load_realtime(rt_f) if rt_f else (
    load_realtime("realtime_fields_gt.csv") if os.path.exists("realtime_fields_gt.csv")
    else (load_realtime("realtime_fields.csv") if os.path.exists("realtime_fields.csv") else None))

if gt is None or rt is None:
    st.warning(
        "Upload both ground_truth.csv and realtime_fields_gt.csv (sidebar) "
        "to continue.",
        icon="⚠️",
    )
    st.stop()

comparison = build_comparison(gt, rt)

if comparison.empty or "ground_truth_odia" not in comparison.columns:
    gt_deeds = sorted(set(gt["deed_number"])) if gt is not None else []
    rt_deeds = sorted(set(rt["reg_no"])) if rt is not None else []
    overlap = sorted(set(gt_deeds) & set(rt_deeds))
    st.error(
        "No comparable rows were produced. This usually means the two files "
        "don't share any deed numbers.",
        icon="🚫",
    )
    st.write(f"**ground_truth.csv deeds ({len(gt_deeds)}):** {gt_deeds[:20]}")
    st.write(f"**realtime_fields_gt.csv reg_nos ({len(rt_deeds)}):** {rt_deeds[:20]}")
    st.write(f"**Overlapping deeds:** {overlap if overlap else 'NONE — upload matching files'}")
    st.stop()


def _blank(s: str) -> bool:
    return str(s).strip() in ("", "nan")


# Reclassify "both-blank": build_comparison's own both-blank check uses the
# ORIGINAL DB metadata (irrelevant here — this experiment sends a DIFFERENT
# anchor: corrected_english, falling back to corrected_odia). A row should
# only be "nothing to locate" if BOTH corrected_english AND corrected_odia
# were blank (meaning THIS script had nothing to send either); if ground
# truth actually has an Odia value but Gemini still came back empty, that's
# a genuine miss, not "nothing to locate".
def _fix_issue_type(row):
    anchor_was_sent = not (_blank(row["ground_truth_english"]) and _blank(row["ground_truth_odia"]))
    gemini_empty = _blank(row["fresh_gemini_odia"]) and _blank(row["fresh_gemini_readback"])
    if row["issue_type"].startswith("both-blank") and anchor_was_sent and gemini_empty:
        return "content mismatch"
    return row["issue_type"]


comparison["issue_type"] = comparison.apply(_fix_issue_type, axis=1)

all_deeds_expected = sorted(set(gt["deed_number"]))
covered_deeds = sorted(comparison["deed_number"].unique().tolist())
missing_deeds = sorted(set(all_deeds_expected) - set(covered_deeds))
if missing_deeds:
    st.info(
        f"**{len(missing_deeds)} deed(s) produced no results in this run:** "
        f"{missing_deeds} — typically because the expert never filled in "
        f"`corrected_english` for any field on that deed, so there was no "
        f"anchor to send even in this experiment.",
        icon="ℹ️",
    )

n_mismatch = int((comparison["issue_type"] == "content mismatch").sum())
c1, c2 = st.columns(2)
c1.metric("Deeds compared", len(covered_deeds))
c2.metric("Genuine content mismatches", n_mismatch,
          help="Gemini disagreed with ground truth even though it was given "
               "the correct value as the search anchor — a real page-reading "
               "issue, not a missing-metadata issue.")

reg_nos = covered_deeds
choice = st.selectbox("Select deed (reg_no)", reg_nos)
ddf = comparison[comparison["deed_number"] == choice].copy()

view = ddf[[
    "field",
    "ground_truth_odia", "ground_truth_readback",
    "fresh_gemini_odia", "fresh_gemini_readback",
    "issue_type",
]].rename(columns={
    "field": "Field",
    "ground_truth_odia": "Ground Truth / Anchor Sent (Odia)",
    "ground_truth_readback": "Ground Truth Readback (EN)",
    "fresh_gemini_odia": "Gemini Output (Odia)",
    "fresh_gemini_readback": "Gemini Readback (EN)",
    "issue_type": "Status",
})


def hl(row):
    status = row["Status"]
    if status == "content mismatch":
        return ["background-color: #e53935; color: white; font-weight: bold;"] * len(row)
    if status.startswith("both-blank"):
        return ["background-color: #ff9800; color: black; font-weight: bold;"] * len(row)
    return [""] * len(row)


st.dataframe(view.style.apply(hl, axis=1), use_container_width=True, hide_index=True)

st.caption(
    "🔴 red = genuine content mismatch even with a correct anchor (real "
    "page-reading issue — check the scan). 🟠 orange = both blank (the "
    "expert never filled this field in either, so there was no anchor here "
    "in this experiment). No highlight = match, spelling/formatting-only "
    "difference, or deed_type (expected transcription-vs-category divergence)."
)
