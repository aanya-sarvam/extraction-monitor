"""
mismatch_viewer.py -- deploys on Streamlit Community Cloud.

Reads a self-contained deeds_snapshot.json (built locally by build_snapshot.py)
and shows, per deed, three columns side by side:
    ORIGINAL (registry input JSON) | GEMINI (english_value) | GEMINI readback
with GENUINE content mismatches flagged red. Format and spelling differences are
NOT flagged.

No gcloud, no GCS, no credentials -- everything the app needs is in the snapshot,
so it runs anywhere the repo is cloned.

WHAT COUNTS AS A MISMATCH (reused from compute_mismatches.diff_deed + tolerances):
  names -> fuzzy >= 0.80 (SAHU vs SAHOO = match) ; amount -> digits only ;
  khata/plot -> numeric intersection ; dates -> parsed then compared ;
  district/office/deed_type -> fuzzy/containment (spelling variants = match).

Local run:   streamlit run mismatch_viewer.py
Cloud:       point Streamlit Cloud at this file; commit deeds_snapshot.json too.
"""
from __future__ import annotations

import html
import json
import os
import random
import re
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st

# Streamlit Cloud runs from the repo ROOT, not this file's folder, so resolve the
# snapshot next to this script rather than relying on the working directory.
SNAPSHOT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "deeds_snapshot.json")
NAME_THRESHOLD = 0.80
PLACE_THRESHOLD = 0.60
CAT_THRESHOLD = 0.60


# ========================= comparison primitives ==========================
def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().upper())


def digits(s):
    d = "".join(c for c in (s or "") if c.isdigit())
    return int(d) if d else None


def ratio(a, b):
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def names_match(grounding_names, registry_names):
    diffs = []
    reg = list(registry_names)
    for gn in grounding_names:
        best = max((ratio(gn, rn) for rn in reg), default=0.0)
        if best < NAME_THRESHOLD:
            diffs.append(gn)
    extra = len(grounding_names) != len(registry_names)
    return (not diffs and not extra), {"grounding_count": len(grounding_names),
                                       "registry_count": len(registry_names)}


def parse_parties(s):
    names = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        part = re.sub(r"^\d+\s*-\s*", "", part)
        part = part.split("(")[0].strip()
        if part:
            names.append(part)
    return names


def parse_property(s):
    items = []
    for chunk in re.split(r",\s*\d+-\s*", (s or "")):
        if "Village" not in chunk and "Khata" not in chunk:
            continue

        def grab(key):
            m = re.search(key + r"\s*:?\s*([^\n]+?)(?:\s{2,}|Khata|Plot|Area|Total|Kissam|Boundary|$)", chunk)
            return m.group(1).strip() if m else ""
        items.append({"village": grab("Village"), "khata": grab("Khata"),
                      "plot": grab("Plot"), "area": grab("Area")})
    return items


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def _yr(y):
    y = int(y)
    return (y + (2000 if y < 50 else 1900)) if y < 100 else y


def parse_date(s):
    s = (s or "").strip().lower()
    if not s:
        return None
    m = re.search(r"(\d{1,2})\s*(?:st|nd|rd|th)?\D+([a-z]{3,})\D*?(\d{2,4})", s)
    if m and m.group(2)[:3] in _MONTHS:
        return (_yr(m.group(3)), _MONTHS[m.group(2)[:3]], int(m.group(1)))
    m = re.match(r"\D*(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})", s)
    if m:
        return (_yr(m.group(3)), int(m.group(2)), int(m.group(1)))
    return None


def date_match(g, a):
    pg, pa = parse_date(g), parse_date(a)
    if pg and pa:
        return pg == pa
    return norm(g) == norm(a) or ratio(g, a) >= 0.85


# ========================= build "g" from snapshot rows ===================
def g_fields(g, fid, attr=None):
    out = []
    for f in g["fields"]:
        if f["id"] == fid and (attr is None or f["attr"] == attr):
            v = (f["english_value"] or "").strip()
            if v:
                out.append(v)
    return out


def g_scalar(g, fid):
    v = g_fields(g, fid)
    return v[0] if v else ""


def g_readbacks(g, fid, attr=None):
    out = []
    for f in g["fields"]:
        if f["id"] == fid and (attr is None or f["attr"] == attr):
            rb = (f.get("latin_readback") or "").strip()
            if rb:
                out.append(rb)
    return out


def build_g(reg_no, rows):
    fields = []
    book = ""
    for r in rows:
        book = book or r.get("book_label", "")
        fields.append({"id": r.get("field_id", ""), "attr": r.get("attr", ""),
                       "english_value": r.get("english_value", ""),
                       "latin_readback": r.get("latin_readback", ""),
                       "odia_text": r.get("odia_text", ""), "page": r.get("page", ""),
                       "found": r.get("found", "")})
    return {"reg_no": reg_no, "book_label": book, "fields": fields}


# ========================= per-deed comparison ============================
def compare_deed(g, api):
    rows = []

    def scalar_row(label, gid, akey, kind="text"):
        gv = g_scalar(g, gid)
        av = str(api.get(akey) or "")
        rb = " / ".join(g_readbacks(g, gid))
        if not gv or not av:
            verdict = None
        elif kind == "amount":
            verdict = (digits(gv) == digits(av))
        elif kind == "date":
            verdict = date_match(gv, av)
        elif kind == "place":
            verdict = (norm(gv) == norm(av) or ratio(gv, av) >= PLACE_THRESHOLD)
        elif kind == "category":
            verdict = (norm(gv) == norm(av) or norm(gv) in norm(av)
                       or norm(av) in norm(gv) or ratio(gv, av) >= CAT_THRESHOLD)
        else:
            verdict = (norm(gv) == norm(av))
        rows.append((label, av, gv, rb, verdict))

    scalar_row("Deed type", "deed_type", "deedType", "category")
    scalar_row("District", "district", "district", "place")
    scalar_row("Office", "office", "office", "place")
    scalar_row("Registration date", "registration_date", "registrationDate", "date")
    scalar_row("Presentation date", "presentation_date", "presentationDate", "date")
    scalar_row("Consideration amount", "consideration_amount", "considerationAmount", "amount")

    for label, gid, akey in [("Sellers", "seller_details", "sellerDetails"),
                             ("Buyers", "buyer_details", "buyerDetails")]:
        gnames = g_fields(g, gid, attr="name")
        anames = parse_parties(api.get(akey))
        rb = g_readbacks(g, gid, attr="name")
        if not gnames and not anames:
            continue
        ok, info = names_match(gnames, anames)
        note = ""
        if info["grounding_count"] != info["registry_count"]:
            note = f"  (count {info['grounding_count']} vs {info['registry_count']})"
        rows.append((label, "\n".join(anames) or "—",
                     ("\n".join(gnames) or "—") + note, "\n".join(rb), ok))

    aprops = parse_property(api.get("propertyDetails"))
    gkhata = g_fields(g, "property_details", attr="khata")
    gplot = g_fields(g, "property_details", attr="plot")
    if aprops and (gkhata or gplot):
        akhata = {digits(p["khata"]) for p in aprops if digits(p["khata"])}
        aplot = {digits(p["plot"]) for p in aprops if digits(p["plot"])}
        k_ok = (not gkhata) or bool({digits(x) for x in gkhata} & akhata)
        p_ok = (not gplot) or bool({digits(x) for x in gplot} & aplot)
        rows.append(("Khata", ", ".join(str(x) for x in sorted(akhata)) or "—",
                     ", ".join(gkhata) or "—", "", k_ok))
        rows.append(("Plot", ", ".join(str(x) for x in sorted(aplot)) or "—",
                     ", ".join(gplot) or "—", "", p_ok))

    has_mismatch = any(v is False for _, _, _, _, v in rows)
    return rows, has_mismatch


# ========================= data ===========================================
@st.cache_data(show_spinner=False)
def load_snapshot(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ========================= rendering ======================================
def _cell(v):
    return html.escape(str(v)).replace("\n", "<br>")


def render_deed(reg_no, g, rows, has_mismatch):
    badge = ("<span style='background:#c0392b;color:#fff;padding:2px 10px;"
             "border-radius:10px;font-size:0.8em'>CONTENT MISMATCH</span>"
             if has_mismatch else
             "<span style='background:#27ae60;color:#fff;padding:2px 10px;"
             "border-radius:10px;font-size:0.8em'>clean</span>")
    st.markdown(
        f"#### `{reg_no}` &nbsp; {badge} "
        f"<span style='color:#888;font-size:0.85em'>book: {g.get('book_label') or '—'}</span>",
        unsafe_allow_html=True)
    head = ("<tr style='text-align:left;border-bottom:2px solid #999;"
            "background:#fff;color:#111'>"
            "<th style='padding:6px 10px;width:14%;color:#111'>Field</th>"
            "<th style='padding:6px 10px;width:30%;color:#111'>Original (registry)</th>"
            "<th style='padding:6px 10px;width:30%;color:#111'>Gemini</th>"
            "<th style='padding:6px 10px;width:26%;color:#111'>Gemini readback</th></tr>")
    body = ""
    for label, orig, gem, rb, verdict in rows:
        if verdict is False:
            bg, fg = "#fdecea", "#7a1c12"      # red row, dark red text
        elif verdict is True:
            bg, fg = "#eafaf1", "#14512e"      # green row, dark green text
        else:
            bg, fg = "#f0f0f0", "#333333"      # grey row, dark grey text
        body += (f"<tr style='background:{bg};color:{fg};border-bottom:1px solid #ccc;"
                 f"vertical-align:top'>"
                 f"<td style='padding:6px 10px;font-weight:700;color:{fg}'>{_cell(label)}</td>"
                 f"<td style='padding:6px 10px;color:{fg}'>{_cell(orig)}</td>"
                 f"<td style='padding:6px 10px;color:{fg}'>{_cell(gem)}</td>"
                 f"<td style='padding:6px 10px;color:{fg};opacity:0.85'>{_cell(rb)}</td></tr>")
    st.markdown(f"<table style='width:100%;border-collapse:collapse;font-size:0.9em;"
                f"background:#fff'>"
                f"{head}{body}</table>", unsafe_allow_html=True)
    st.write("")


# ========================= app ============================================
st.set_page_config(page_title="Deed mismatch viewer", layout="wide")
st.title("🧾 Deed review — Original vs Gemini vs Readback")
st.caption("Red = genuine content mismatch. Spelling variants and date/amount "
           "formatting are treated as matches, not mismatches.")

with st.sidebar:
    st.header("Data")
    snap_path = st.text_input("Snapshot file", SNAPSHOT_DEFAULT)
    only_mm = st.checkbox("Only deeds with content mismatches", value=False)
    if st.button("🔀 Reshuffle order", use_container_width=True):
        st.session_state["nonce"] = st.session_state.get("nonce", 0) + 1
    specific = st.text_input("Jump to reg_no (optional)").strip()

if not os.path.exists(snap_path):
    st.error(f"No snapshot at `{snap_path}`. Build it locally with "
             f"`python build_snapshot.py --n 15` and commit the file.")
    st.stop()

snap = load_snapshot(snap_path)
deeds = snap.get("deeds", {})
meta = snap.get("meta", {})
st.caption(f"{len(deeds)} deed(s) in snapshot · built {meta.get('built_at', '—')}")

order = list(deeds.keys())
random.Random(st.session_state.get("nonce", 0)).shuffle(order)

if specific:
    order = [specific] if specific in deeds else []
    if not order:
        st.warning(f"`{specific}` not in snapshot.")

shown = 0
for reg in order:
    d = deeds[reg]
    g = build_g(reg, d["gemini"])
    rows, mm = compare_deed(g, d["input"])
    if only_mm and not mm:
        continue
    render_deed(reg, g, rows, mm)
    shown += 1

if only_mm:
    st.caption(f"Showing {shown} deed(s) with genuine content mismatches "
               f"(of {len(deeds)} in snapshot).")
else:
    st.caption(f"Showing {shown} deed(s).")
