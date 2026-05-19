
"""
TikTok Causal Hub — MVP
Upload raw TTAM or SOT data → clean & aggregate → quality check → EDA → configure analysis → export ready-to-use CSV
"""

import io, os, json, warnings, re
import urllib.request
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Compatibility patches for causalimpact 0.2.6 + pandas 2.0+ + Python 3.14
if not hasattr(pd.core.dtypes.common, 'is_datetime_or_timedelta_dtype'):
    pd.core.dtypes.common.is_datetime_or_timedelta_dtype = (
        lambda x: pd.api.types.is_datetime64_any_dtype(x) or pd.api.types.is_timedelta64_dtype(x)
    )

# Patch causalimpact.misc.standardize_all_variables — uses series[0] broken in pandas 2.0+
# Original returns {"data_pre": ..., "data_post": ..., "orig_std_params": (mu, sd)}
# Bug: data_mu[0] / data_sd[0] use label lookup in pandas 2.x, need .iloc[0]
try:
    import causalimpact.misc as _ci_misc
    import causalimpact.analysis as _ci_analysis

    def _patched_standardize(data, pre_period, post_period):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("``data`` must be of type `pandas.DataFrame`")
        data_mu = data.loc[pre_period[0]:pre_period[1], :].mean(skipna=True)
        data_sd = data.loc[pre_period[0]:pre_period[1], :].std(skipna=True, ddof=0)
        data = data - data_mu
        data_sd = data_sd.fillna(1)
        data[data != 0] = data[data != 0] / data_sd
        y_mu = data_mu.iloc[0]
        y_sd = data_sd.iloc[0]
        data_pre  = data.loc[pre_period[0]:pre_period[1], :]
        data_post = data.loc[post_period[0]:post_period[1], :]
        return {"data_pre": data_pre, "data_post": data_post, "orig_std_params": (y_mu, y_sd)}

    _ci_misc.standardize_all_variables = _patched_standardize
    _ci_analysis.standardize_all_variables = _patched_standardize
except Exception as _e:
    print(f"causalimpact patch warning: {_e}")
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
warnings.filterwarnings("ignore")

# ── Palette ────────────────────────────────────────────────────────────────────
C = dict(
    bg="#0B0B18",
    surface="#111126",
    surface2="#1A1A35",
    border="#252545",
    accent="#6666EE",
    accent_d="#5555DD",
    accent_xl="#1A1A3A",
    text="#EAEAFF",
    grey="#8888B8",
    grey_l="#555578",
    red="#FF5C6E",
    amber="#FFB84D",
    green="#40C87A",
    white="#FFFFFF",
    cyan="#6666EE",
    purple="#7B7FFF",
    purple_l="#A0A4FF",
    purple_xl="#1A1A3A",
)
CAT_COLORS = {
    "TopView":    "#7B7FFF",  # bright lilac
    "Brand_ExTV": "#A0A4FF",  # lighter lilac
    "MidFunnel":  "#BB88EE",  # soft purple
    "Performance": "#40C87A", # bright green
    "Other":      "#555578",  # muted
}
# Chart palette using the spec provided
CHART = dict(
    paper_bgcolor=C["surface"], plot_bgcolor=C["surface"],
    font=dict(color=C["text"], family="Inter, Arial, sans-serif", size=12),
    xaxis=dict(gridcolor="#252545", showgrid=True, zeroline=False,
               tickfont=dict(color=C["grey"]), linecolor="#252545"),
    yaxis=dict(gridcolor="#252545", showgrid=True, zeroline=False,
               tickfont=dict(color=C["grey"]), linecolor="#252545"),
    margin=dict(l=50, r=30, t=45, b=45),
    legend=dict(bgcolor=C["surface"], bordercolor=C["border"], borderwidth=1,
                font=dict(color=C["text"])),
    hovermode="x unified",
)

st.set_page_config(page_title="TikTok Causal Hub", page_icon="🎯",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
/* Apply Inter to text elements only — NOT on * so Material icon font ligatures keep working */
body, p, div, span, input, select, textarea, label, li, h1, h2, h3, h4, h5, h6, td, th, a, button {{
    font-family: 'Inter', sans-serif;}}
html, body {{background:{C['bg']};color:{C['text']};}}
.stApp {{background:{C['bg']};}}
.stApp [class*="block-container"] {{background:{C['bg']};}}
section[data-testid="stSidebar"] {{background:{C['surface']};border-right:1px solid {C['border']};box-shadow:4px 0 16px rgba(0,0,0,0.4);}}
section[data-testid="stSidebar"] * {{background:transparent;}}
/* Cards */
.card{{background:{C['surface']};border:1px solid {C['border']};border-radius:16px;padding:20px 24px;margin-bottom:14px;box-shadow:0 2px 8px rgba(0,0,0,0.35);}}
.card-c{{border-left:3px solid {C['purple']};}}
.card-a{{border-left:3px solid {C['amber']};}}
.card-r{{border-left:3px solid {C['red']};}}
.card-g{{border-left:3px solid {C['grey_l']};}}
/* Metric tiles */
.mrow{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;}}
.mtile{{background:{C['surface']};border:1px solid {C['border']};border-radius:16px;padding:14px 18px;flex:1;min-width:120px;box-shadow:0 2px 8px rgba(0,0,0,0.35);}}
.ml{{font-size:11px;color:{C['grey']};text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;}}
.mv{{font-size:20px;font-weight:700;color:{C['text']};}}
.mvc{{color:{C['purple']};}} .mva{{color:{C['amber']};}} .mvr{{color:{C['red']};}}
/* Pills & badges */
.pill{{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:600;margin:2px;}}
.pill-r{{background:rgba(255,92,110,0.12);color:{C['red']};border:1px solid rgba(255,92,110,0.3);}}
.pill-a{{background:rgba(255,184,77,0.12);color:{C['amber']};border:1px solid rgba(255,184,77,0.3);}}
.pill-c{{background:{C['purple_xl']};color:{C['purple']};border:1px solid {C['purple_l']}40;}}
.pill-g{{background:{C['surface2']};color:{C['grey']};border:1px solid {C['border']};}}
.sbadge{{background:{C['purple']};color:#fff;font-weight:700;font-size:11px;padding:3px 10px;border-radius:999px;}}
/* Buttons */
.stButton>button{{background:{C['purple']};color:#fff!important;border:none;border-radius:12px;font-weight:600;transition:all .15s;font-size:13px!important;}}
.stButton>button:hover{{background:{C['accent_d']};color:#fff!important;}}
.stButton>button *, .stButton>button p, .stButton>button span, .stButton>button div {{color:#fff!important;}}
/* Form inputs */
div[data-baseweb="select"]>div{{background:{C['surface']}!important;border-color:{C['border']}!important;color:{C['text']}!important;}}
.stTextArea textarea{{background:{C['surface']}!important;border-color:{C['border']}!important;color:{C['text']}!important;}}
.stTextInput input{{background:{C['surface']}!important;border-color:{C['border']}!important;color:{C['text']}!important;}}
.stMultiSelect [data-baseweb="tag"]{{background:{C['purple_xl']}!important;color:{C['purple_l']}!important;}}
/* Date inputs */
[data-testid="stDateInput"] input{{background:{C['surface2']}!important;border-color:{C['border']}!important;color:{C['text']}!important;}}
/* Tabs */
.stTabs [data-baseweb="tab"]{{color:{C['grey']};font-size:13px;}}
.stTabs [aria-selected="true"]{{color:{C['purple']};font-weight:600;border-bottom:2px solid {C['purple']};}}
hr{{border-color:{C['border']};}}
p, li {{color:{C['text']};font-size:13px;}}
/* Typography scale */
.t-section{{font-size:15px;font-weight:600;color:{C['text']};margin-bottom:8px;}}
.t-label{{font-size:11px;font-weight:600;color:{C['grey']};text-transform:uppercase;letter-spacing:.07em;}}
.t-body{{font-size:13px;color:{C['text']};line-height:1.5;}}
.t-meta{{font-size:12px;color:{C['grey']};line-height:1.5;}}
/* Expander */
[data-testid="stExpander"]{{background:{C['surface']}!important;border-color:{C['border']}!important;}}
/* Hide expander toggle icon that renders as raw icon text */
[data-testid="stExpanderToggleIcon"] {{display:none!important;}}
/* Hide file uploader label */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploaderDropzone"] label {{display:none!important;}}
/* Hide sidebar collapse toggle */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stBaseButton-headerNoPadding"] {{display:none!important;}}
/* Hide Streamlit chrome */
[data-testid="stToolbar"], [data-testid="stDecoration"], [class*="shortcut"] {{display:none!important;}}
</style>""", unsafe_allow_html=True)

# ── Default L4 mapping ─────────────────────────────────────────────────────────
DEFAULT_L4 = {
    "Topview CPM Buy": "TopView", "TopView": "TopView",
    "BA VV 15s Focused View": "Brand_ExTV", "BA VV 6s Focused View": "Brand_ExTV",
    "BA Reach": "Brand_ExTV", "Standard Feed Target Frequency": "Brand_ExTV",
    "Pulse Core Evergreen": "Brand_ExTV", "Branded Mission": "Brand_ExTV",
    "Web Traffic LPV": "MidFunnel", "Web Traffic Deep LPV": "MidFunnel",
    "Spark Ads": "MidFunnel", "In-Feed Ad": "MidFunnel",
    "Catalog Ads Conversions": "Performance", "Web non-Catalog Conversions": "Performance",
    "App Install": "Performance", "Shopping Ads": "Performance", "GMV Max": "Performance",
}


# ── Google Sheets URL helper ──────────────────────────────────────────────────
def gsheets_url_to_csv_url(url: str):
    """Convert any Google Sheets share/edit URL to a direct CSV export URL.
    Returns (csv_url, gid) or (None, None) if not a valid Sheets URL."""
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
    if not m:
        return None, None
    sheet_id = m.group(1)
    gid_m = re.search(r"gid=(\d+)", url)
    gid = gid_m.group(1) if gid_m else "0"
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    return csv_url, gid


def fetch_gsheet_as_filelike(url: str):
    """Fetch a Google Sheet export URL and return (BytesIO, filename) or raise."""
    csv_url, _ = gsheets_url_to_csv_url(url)
    if not csv_url:
        raise ValueError("Not a valid Google Sheets URL.")
    req = urllib.request.Request(csv_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = resp.read()
    return io.BytesIO(data), "gsheet_import.csv"


# ── Use Cases / Hypotheses ────────────────────────────────────────────────────
USE_CASES = [
    dict(
        id="cross_channel",
        name="Cross-Channel DTC Validation",
        icon="🔗",
        tag="GA4 / SOT",
        tag_cls="pill-c",
        challenge="TikTok drives awareness but doesn't get credit in GA4. View-through conversions are invisible and branded search spend keeps climbing.",
        recipe="Multi-channel Causal Analysis on GA4/Shopify data. Prove TikTok's impact on Direct, Google Search, and Organic simultaneously.",
        pitch="TikTok is a discovery engine. Share your GA4 data and we will prove our spend directly caused your Direct sales to spike — and that your Google audience was already primed by TikTok before they searched.",
        data_source="GA4 or Shopify daily export: Direct, Paid Search, Organic, TikTok (sessions + conversions)",
        colour="#2563EB",
    ),
    dict(
        id="marketplace_spillover",
        name="Marketplace / Amazon Scale",
        icon="📦",
        tag="Amazon / 3P",
        tag_cls="pill-a",
        challenge="Client is scaling TikTok Shop but wants to know if it's lifting sales on external marketplaces like Amazon.",
        recipe="Marketplace Spillover using third-party marketplace source-of-truth data.",
        pitch="Scaling TTS expands the pie. Share your Amazon Seller data, and we will prove our TikTok Shop scaling period created a spillover order and revenue spike on Amazon.",
        data_source="Amazon Seller Central daily orders & revenue export",
        colour="#D97706",
    ),
    dict(
        id="tts_full_funnel_brand",
        name="TTS Full-Funnel Halo (Brand → Shop)",
        icon="🏪",
        tag="TikTok Shop",
        tag_cls="pill-c",
        challenge="Client relies solely on GMV Max and doesn't see value in bundling brand formats like TopView or Consideration Ads.",
        recipe="Causal Analysis on TTS GMV measuring a Brand format intervention.",
        pitch="We will run a burst of TopView / Consideration Ads and use Causal Analysis to prove the immediate halo spike it creates on your lower-funnel TTS orders.",
        data_source="TikTok Shop daily GMV + TTAM spend export",
        colour="#84CC16",
    ),
    dict(
        id="ttam_full_funnel_brand",
        name="TTAM Full-Funnel Halo (Brand → Performance)",
        icon="📈",
        tag="TTAM",
        tag_cls="pill-c",
        challenge="Client relies solely on performance formats (web, catalog, Smart+) and doesn't see value in brand formats like TopView or Brand Auction.",
        recipe="Causal Analysis on TTAM data measuring a Brand / Consideration format intervention.",
        pitch="We will run a burst of TopView / Consideration Ads and use Causal Analysis to prove the immediate halo spike it creates on your lower-funnel TTAM conversions.",
        data_source="TTAM daily export (raw L4 tag level) — this is the Chemist Warehouse use case",
        colour="#84CC16",
    ),
    dict(
        id="tts_baseline",
        name="TTS Baseline Proof",
        icon="💰",
        tag="TikTok Shop",
        tag_cls="pill-a",
        challenge="Client uses GMV Max but doesn't know if scaling TTS actually drives incremental sales above organic baseline.",
        recipe="Causal Analysis on TTS GMV to isolate the scaling period lift.",
        pitch="We will use Causal modelling to predict what your shop sales would be without the scale period, isolating all other variables including promotional periods.",
        data_source="TikTok Shop daily GMV + ad spend export",
        colour="#059669",
    ),
    dict(
        id="sot_holdout",
        name="The SOT Holdout",
        icon="🔒",
        tag="Client SOT",
        tag_cls="pill-g",
        challenge="Client refuses to scale based on TTAM or CLS and requires validation using their own internal data source.",
        recipe="Causal Analysis on Client Source-of-Truth data with a TikTok spend surge as the intervention.",
        pitch="Surge TikTok spend by 30%, keep all other channels flat, share your daily internal data, and we will mathematically isolate our impact on your sales.",
        data_source="Client internal SoT — daily sales, orders, or revenue",
        colour="#6B7280",
    ),
]


# ── Hypothesis configuration (data requirements + smart defaults) ─────────────
HYPOTHESIS_CONFIG = {
    "cross_channel": {
        "file_format": "sot",
        "target_keywords": ["Direct_Revenue", "Direct_Conversions", "Revenue", "Orders", "Conversions"],
        "covariate_keywords": ["TikTok", "Paid_Social", "Paid_Search", "Spend", "Cost", "Sessions"],
        "required_channels": ["Direct", "Paid_Social_TikTok"],
        "min_pre_days": 30,
        "intervention_label": "TikTok spend surge",
        "data_tips": "GA4 or Shopify daily export — Direct, Paid Search, Organic & TikTok channels (sessions + conversions).",
    },
    "marketplace_spillover": {
        "file_format": "sot",
        "target_keywords": ["Orders", "Revenue", "Sales", "GMV"],
        "covariate_keywords": ["TikTok", "Spend", "Cost", "GMV"],
        "required_channels": [],
        "min_pre_days": 30,
        "intervention_label": "TikTok Shop scaling period",
        "data_tips": "Amazon Seller Central daily orders & revenue export (or equivalent marketplace SoT).",
    },
    "tts_full_funnel_brand": {
        "file_format": "sot",
        "target_keywords": ["GMV", "Orders", "Revenue", "Sales"],
        "covariate_keywords": ["TopView", "Brand", "Spend", "Cost", "Performance"],
        "required_channels": [],
        "min_pre_days": 30,
        "intervention_label": "Brand format burst (TopView / Consideration)",
        "data_tips": "TikTok Shop daily GMV export + TTAM spend breakdown by format.",
    },
    "ttam_full_funnel_brand": {
        "file_format": "ttam",
        "target_keywords": ["Performance", "PerformanceCost", "Conversions", "Revenue"],
        "covariate_keywords": ["TopView", "Brand_ExTV", "BrandExTVCost", "MidFunnel", "MidFunnelCost"],
        "required_channels": ["TopView", "Performance"],
        "min_pre_days": 30,
        "intervention_label": "Brand format burst (TopView / Consideration)",
        "data_tips": "Raw TTAM export — one row per L4 product tag per day. Must include TopView & Performance rows.",
    },
    "tts_baseline": {
        "file_format": "sot",
        "target_keywords": ["GMV", "Orders", "Revenue", "Sales"],
        "covariate_keywords": ["Spend", "Cost", "GMVMax", "TikTok"],
        "required_channels": [],
        "min_pre_days": 30,
        "intervention_label": "TikTok Shop scaling period",
        "data_tips": "TikTok Shop daily GMV + ad spend export with clear pre-scale and scale periods.",
    },
    "sot_holdout": {
        "file_format": "sot",
        "target_keywords": ["Revenue", "Orders", "Sales", "Conversions", "Sessions"],
        "covariate_keywords": ["TikTok", "Paid_Social", "Spend", "Cost"],
        "required_channels": [],
        "min_pre_days": 30,
        "intervention_label": "TikTok spend surge (+30%)",
        "data_tips": "Client internal SoT — daily sales, orders, or revenue. TikTok spend must have a clear surge period.",
    },
}


def check_hypothesis_compatibility(df, file_type, hypothesis_id):
    """Returns a dict: {status, checks, score} for the given hypothesis + data combo."""
    cfg = HYPOTHESIS_CONFIG.get(hypothesis_id)
    if not cfg:
        return {"status": "unknown", "checks": [], "score": 0.0}

    checks = []
    score = 0.0
    total = 0.0

    # 1. File format match
    total += 1
    expected_fmt = cfg["file_format"]
    fmt_match = (
        (expected_fmt == "ttam" and file_type in ("raw_ttam", "pre_aggregated")) or
        (expected_fmt == "sot"  and file_type in ("sot", "pre_aggregated")) or
        expected_fmt == "any"
    )
    if fmt_match:
        score += 1
        checks.append({"label": "File format", "status": "pass",
                        "detail": f"{file_type} matches {expected_fmt} expectation"})
    else:
        checks.append({"label": "File format", "status": "fail",
                        "detail": f"Expected {expected_fmt} data but got {file_type} — results may be unreliable"})

    # 2. Date coverage
    total += 1
    if "p_date" in df.columns and df["p_date"].notna().any():
        n_days = (df["p_date"].max() - df["p_date"].min()).days + 1
        min_req = cfg["min_pre_days"] * 2
        if n_days >= min_req:
            score += 1
            checks.append({"label": "Date coverage", "status": "pass",
                            "detail": f"{n_days} days total — enough for a clean pre/post split"})
        else:
            checks.append({"label": "Date coverage", "status": "warn",
                            "detail": f"{n_days} days — recommend ≥ {min_req} for a reliable split"})
    else:
        checks.append({"label": "Date coverage", "status": "fail", "detail": "No valid date column detected"})

    # 3. Target column presence
    total += 1
    num_cols = [c for c in df.columns if c != "p_date" and pd.api.types.is_numeric_dtype(df[c])]
    target_kws = cfg["target_keywords"]
    matched_targets = [c for c in num_cols if any(kw.lower() in c.lower() for kw in target_kws)]
    if matched_targets:
        score += 1
        checks.append({"label": "Target columns", "status": "pass",
                        "detail": f"Found: {', '.join(matched_targets[:3])}"})
    else:
        checks.append({"label": "Target columns", "status": "warn",
                        "detail": f"No columns matching {target_kws[:2]} — check column naming or select manually in Step 6"})

    # 4. Required channels (only if specified)
    req_channels = cfg.get("required_channels", [])
    if req_channels:
        total += 1
        found_ch = [ch for ch in req_channels if any(ch.lower() in c.lower() for c in df.columns)]
        if len(found_ch) == len(req_channels):
            score += 1
            checks.append({"label": "Required channels", "status": "pass",
                            "detail": f"All present: {', '.join(req_channels)}"})
        elif found_ch:
            score += 0.5
            missing = [c for c in req_channels if c not in found_ch]
            checks.append({"label": "Required channels", "status": "warn",
                            "detail": f"Partial match — missing: {', '.join(missing)}"})
        else:
            checks.append({"label": "Required channels", "status": "fail",
                            "detail": f"Missing required channels: {', '.join(req_channels)}"})

    pct = score / total if total > 0 else 0.0
    status = "good" if pct >= 0.75 else ("partial" if pct >= 0.5 else "poor")
    return {"status": status, "checks": checks, "score": pct}


def build_eda_summary(df, pre_df, post_df, hypothesis, file_type, channel_roles, target_col, conv_cols_in, sess_cols_in):
    import numpy as np
    lines = []
    hyp_name = next((u["name"] for u in USE_CASES if u["id"] == hypothesis), hypothesis or "Not specified")
    lines.append(f"HYPOTHESIS: {hyp_name}")
    lines.append(f"DATA TYPE: {'SOT/GA4 multi-channel' if file_type == 'sot' else 'TTAM campaign data'}")
    lines.append(f"DATE RANGE: {df['p_date'].min().date()} to {df['p_date'].max().date()} ({len(df)} days)")
    if len(pre_df) and len(post_df):
        lines.append(f"PRE-PERIOD: {pre_df['p_date'].min().date()} to {pre_df['p_date'].max().date()} ({len(pre_df)} days)")
        lines.append(f"POST-PERIOD: {post_df['p_date'].min().date()} to {post_df['p_date'].max().date()} ({len(post_df)} days)")
    lines.append("")
    lines.append("CHANNEL PERFORMANCE (pre vs post):")
    check_cols = list(dict.fromkeys((conv_cols_in or []) + (sess_cols_in or [])))
    check_cols = [c for c in check_cols if c in df.columns][:10]
    for col in check_cols:
        pre_avg  = pre_df[col].mean()  if col in pre_df.columns  and len(pre_df)  else 0
        post_avg = post_df[col].mean() if col in post_df.columns and len(post_df) else 0
        lift = (post_avg - pre_avg) / pre_avg * 100 if pre_avg else 0
        sign = "+" if lift >= 0 else ""
        lines.append(f"  {col}: {pre_avg:,.1f} pre -> {post_avg:,.1f} post ({sign}{lift:.1f}%)")
    if file_type == "sot":
        lines.append("")
        lines.append("CONVERSION RATES (CVR) - conversions / sessions:")
        ch_order = ["Direct","Paid_Social_TikTok","Paid_Search_Google","Paid_Social_Meta","Organic_Search"]
        _CH_ALT_PFX = {
            "Direct": ["direct_"], "Paid_Social_TikTok": ["tiktok_"],
            "Paid_Search_Google": ["google_cpc_"], "Paid_Social_Meta": ["meta_"],
            "Organic_Search": ["organic_"],
        }
        for ch in ch_order:
            _pfx = [ch + "_"] + _CH_ALT_PFX.get(ch, [])
            s_col = next((c for c in df.columns if any(c.startswith(p) for p in _pfx) and "session" in c.lower()), None)
            c_col = next((c for c in df.columns if any(c.startswith(p) for p in _pfx) and "conversion" in c.lower()), None)
            if s_col and c_col and s_col in pre_df.columns and c_col in pre_df.columns:
                pre_cvr  = pre_df[c_col].sum()  / pre_df[s_col].sum()  * 100 if pre_df[s_col].sum()  > 0 else 0
                post_cvr = post_df[c_col].sum() / post_df[s_col].sum() * 100 if len(post_df) > 0 and post_df[s_col].sum() > 0 else 0
                delta = (post_cvr - pre_cvr) / pre_cvr * 100 if pre_cvr else 0
                sign = "+" if delta >= 0 else ""
                lines.append(f"  {ch}: {pre_cvr:.2f}% -> {post_cvr:.2f}% CVR ({sign}{delta:.1f}%)")
    if channel_roles:
        lines.append("")
        lines.append("COVARIATE STABILITY:")
        for col, role in channel_roles.items():
            if role == "Covariate" and col in pre_df.columns:
                pre_avg  = pre_df[col].mean()
                post_avg = post_df[col].mean() if len(post_df) else pre_avg
                chg = abs((post_avg - pre_avg) / pre_avg * 100) if pre_avg else 0
                risk = "LOW" if chg < 15 else "MEDIUM" if chg < 35 else "HIGH"
                lines.append(f"  {col}: {chg:.1f}% change post-period - {risk} endogeneity risk")
    if target_col and target_col in pre_df.columns:
        y = pre_df[target_col].fillna(0).values
        if len(y) > 3:
            slope_pct = np.polyfit(np.arange(len(y)), y, 1)[0] / y.mean() * 100 if y.mean() else 0
            cv = y.std() / y.mean() * 100 if y.mean() else 0
            lines.append("")
            lines.append(f"TARGET VARIABLE PRE-PERIOD ({target_col}):")
            lines.append(f"  Trend: {slope_pct:+.2f}%/day | Volatility (CV): {cv:.0f}%")
    return "\n".join(lines)


GEO_OPTIONS = {
    "Australia (AU)":     ("en-AU", 600,  "AU"),
    "New Zealand (NZ)":   ("en-NZ", 720,  "NZ"),
    "United States (US)": ("en-US", -300, "US"),
    "United Kingdom (UK)":("en-GB", 0,    "GB"),
    "Singapore (SG)":     ("en-SG", 480,  "SG"),
    "Global":             ("en-US", 0,    ""),
}

def fetch_google_trends(keywords, start_date, end_date, geo_label="Australia (AU)"):
    """Fetch Google Trends. Auto-installs pytrends if missing."""
    hl, tz, geo_code = GEO_OPTIONS.get(geo_label, ("en-AU", 600, "AU"))
    try:
        from pytrends.request import TrendReq
    except ImportError:
        try:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pytrends", "--quiet"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            from pytrends.request import TrendReq
        except Exception as ie:
            return None, f"Could not install pytrends: {ie}. Run manually: pip install pytrends"
    try:
        pytrends = TrendReq(hl=hl, tz=tz, timeout=(10, 25))
        tf = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"
        kws = [k.strip() for k in keywords if k.strip()][:5]
        pytrends.build_payload(kws, timeframe=tf, geo=geo_code)
        df_trends = pytrends.interest_over_time()
        if df_trends.empty:
            return None, "No trend data returned — Google Trends may have throttled the request. Wait 60 seconds and try again, or try broader keywords."
        df_trends = df_trends.drop(columns=["isPartial"], errors="ignore")
        df_trends.index = pd.to_datetime(df_trends.index)
        return df_trends, None
    except Exception as e:
        err = str(e)
        if "429" in err or "Too Many Requests" in err:
            return None, "Google Trends rate limit hit — wait 60 seconds and try again."
        return None, f"Trends error: {err}"



# ── CausalImpact Model Functions ──────────────────────────────────────────────

def run_causal_impact(df, target, covariates, pre_start, pre_end, post_start, post_end):
    """Run BSTS CausalImpact model."""
    try:
        from causalimpact import CausalImpact
    except ImportError:
        return None, "causalimpact not installed. Run: pip3 install causalimpact"
    try:
        cols = [target] + [c for c in covariates if c in df.columns and c != target]
        ci_df = df[['p_date'] + cols].copy().set_index('p_date')
        ci_df.index = pd.to_datetime(ci_df.index)
        ci_df = ci_df.sort_index()
        full_idx = pd.date_range(ci_df.index.min(), ci_df.index.max(), freq='D')
        ci_df = ci_df.reindex(full_idx, fill_value=0)
        ci_df.index.name = 'date'
        for col in ci_df.columns:
            ci_df[col] = ci_df[col].astype(float)

        pre  = [str(pd.Timestamp(pre_start).date()),  str(pd.Timestamp(pre_end).date())]
        post = [str(pd.Timestamp(post_start).date()), str(pd.Timestamp(post_end).date())]

        # Try with nseasons first, fall back to no model_args
        try:
            ci = CausalImpact(ci_df, pre, post, model_args={'niter': 1000, 'nseasons': 7})
        except Exception:
            try:
                ci = CausalImpact(ci_df, pre, post, model_args={'niter': 1000})
            except Exception:
                ci = CausalImpact(ci_df, pre, post)

        # Call run() explicitly — required by jamalsenouci/causalimpact v0.2.x
        ci.run()

        # Compute p-value directly from inferences (ci.summary() prints to stdout
        # and returns None in jamalsenouci/causalimpact v0.2.x, so regex parsing fails)
        try:
            import scipy.stats as _st
            post_inf = ci.inferences.loc[post_start:post_end]
            mean_pred  = float(post_inf['point_pred'].mean())
            mean_upper = float(post_inf['point_pred_upper'].mean())
            std_pred   = (mean_upper - mean_pred) / 1.96
            if std_pred > 0:
                z_score = (0 - mean_pred) / std_pred
                ci._extracted_p_value = float(_st.norm.cdf(z_score))
            else:
                ci._extracted_p_value = None
        except Exception as pe:
            print(f"p-value extraction error: {pe}")
            ci._extracted_p_value = None

        if not hasattr(ci, 'inferences') or ci.inferences is None or list(ci.inferences.columns) == list(ci_df.columns):
            return None, "Model ran but produced no predictions. Check terminal for details."

        return ci, None

    except BaseException as e:
        import traceback as tb
        full = tb.format_exc()
        print("=== CausalImpact error ===\n" + full)
        lines = [l.strip() for l in full.strip().split("\n") if l.strip()]
        return None, lines[-1] if lines else str(e)


def run_placebo_tests(df, target, covariates, pre_start, pre_end, fractions=None):
    """Run in-sample placebo tests by creating imaginary interventions within the pre-period.

    For each cut fraction, the pre-period is split into a fake pre/post window. A well-specified
    model should find NO significant effect (p >= 0.05) at any of these cut points.

    Returns list of result dicts.
    """
    import scipy.stats as _st

    pre_s = pd.Timestamp(pre_start)
    pre_e = pd.Timestamp(pre_end)
    total_days = (pre_e - pre_s).days + 1
    MIN_EACH = 14  # minimum days required on each side of the cut

    if fractions is None:
        fractions = [1/3, 1/2, 2/3]

    results = []
    for frac in fractions:
        cut = pre_s + pd.Timedelta(days=max(MIN_EACH, int(round(total_days * frac))))
        placebo_pre_end    = cut - pd.Timedelta(days=1)
        placebo_post_start = cut
        placebo_post_end   = pre_e

        pre_len  = (placebo_pre_end    - pre_s).days + 1
        post_len = (placebo_post_end   - placebo_post_start).days + 1
        if pre_len < MIN_EACH or post_len < MIN_EACH:
            continue

        ci_obj, ci_err = run_causal_impact(
            df, target, covariates,
            str(pre_s.date()), str(placebo_pre_end.date()),
            str(placebo_post_start.date()), str(placebo_post_end.date()),
        )

        if ci_err or ci_obj is None:
            results.append(dict(cut_date=str(cut.date()), frac=frac, pre_days=pre_len,
                                post_days=post_len, rel_effect_pct=None,
                                p_value=None, passed=None, error=ci_err))
            continue

        inf      = ci_obj.inferences
        obs_col  = next((c for c in ['response','y','observed']      if c in inf.columns), None)
        pred_col = next((c for c in ['point_pred','y_pred','predicted'] if c in inf.columns), None)
        try:
            post_inf    = inf[inf.index >= placebo_post_start]
            actual_mean = float(pd.to_numeric(post_inf[obs_col],  errors='coerce').mean())
            pred_mean   = float(pd.to_numeric(post_inf[pred_col], errors='coerce').mean())
            rel_effect  = ((actual_mean - pred_mean) / abs(pred_mean) * 100) if pred_mean else None
        except Exception:
            rel_effect = None

        p_value = getattr(ci_obj, '_extracted_p_value', None)
        passed  = (p_value is not None and p_value >= 0.05)
        results.append(dict(cut_date=str(cut.date()), frac=frac, pre_days=pre_len,
                            post_days=post_len, rel_effect_pct=rel_effect,
                            p_value=p_value, passed=passed, error=None))
    return results


def compute_pre_ape_table(ci, pre_start, pre_end):
    """Daily APE table for pre-period — matches R CALM output exactly."""
    s = getattr(ci, 'inferences', None)
    if s is None:
        return pd.DataFrame({'Date': [], 'Observed': [], 'Predicted': [], 'APE (%)': []}), 0.0
    mask = (s.index >= pd.Timestamp(pre_start)) & (s.index <= pd.Timestamp(pre_end))
    pre  = s[mask].copy()
    # Python causalimpact uses 'y' and 'y_pred'; R used 'response' and 'point.pred'
    obs_col  = next((c for c in ['y', 'response', 'observed', 'actual']       if c in pre.columns), None)
    pred_col = next((c for c in ['y_pred', 'point_pred', 'predicted', 'yhat'] if c in pre.columns), None)
    if obs_col is None or pred_col is None:
        print(f"=== inferences columns: {list(pre.columns)} ===")
        print(f"=== ci attributes: {[a for a in dir(ci) if not a.startswith('_')]} ===")
        for attr in ['results','series','data','posterior','trace','predictions','model']:
            val = getattr(ci, attr, None)
            if val is not None:
                if hasattr(val, 'columns'): print(f"=== ci.{attr} columns: {list(val.columns)} ===")
                elif hasattr(val, 'keys'):  print(f"=== ci.{attr} keys: {list(val.keys())} ===")
                else: print(f"=== ci.{attr}: {type(val).__name__} ===")
        return pd.DataFrame({'Date': [], 'Observed': [], 'Predicted': [], 'APE (%)': []}), 0.0
    obs  = pre[obs_col].values
    pred = pre[pred_col].values
    denom = np.abs(obs); denom[denom == 0] = np.nan
    ape  = np.abs(obs - pred) / denom * 100
    tbl  = pd.DataFrame({
        'Date':      pre.index.strftime('%Y-%m-%d'),
        'Observed':  np.round(obs, 2),
        'Predicted': np.round(pred, 2),
        'APE (%)':   np.round(ape, 2),
    })
    mape = float(np.nanmean(ape))
    return tbl, mape


def build_three_panel_chart(ci, post_start, target_name):
    """Light-mode three-panel CausalImpact chart with dynamic column detection."""
    s = ci.inferences.copy()
    for col in s.columns:
        try: s[col] = pd.to_numeric(s[col], errors='coerce')
        except: pass

    dates_str = [str(d)[:10] for d in s.index]
    post_mask = s.index >= pd.Timestamp(post_start)

    obs_col  = next((c for c in ['y','response','observed','actual']        if c in s.columns), None)
    pred_col = next((c for c in ['y_pred','point_pred','predicted','yhat']  if c in s.columns), None)
    pred_lo  = next((c for c in ['y_pred_lower','point_pred_lower']         if c in s.columns), None)
    pred_hi  = next((c for c in ['y_pred_upper','point_pred_upper']         if c in s.columns), None)
    eff_col  = next((c for c in ['point_effect','y_effect','effect']        if c in s.columns), None)
    eff_lo   = next((c for c in ['point_effect_lower','y_effect_lower']     if c in s.columns), None)
    eff_hi   = next((c for c in ['point_effect_upper','y_effect_upper']     if c in s.columns), None)

    if eff_col is None and obs_col and pred_col:
        s['_effect'] = s[obs_col] - s[pred_col]
        eff_col = '_effect'

    cum_eff = pd.Series(np.nan, index=s.index)
    if eff_col:
        _c = s[eff_col].copy().astype(float); _c[~post_mask] = 0
        _c = _c.cumsum().astype(float); _c[~post_mask] = np.nan
        cum_eff = _c

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["Original", "Pointwise Effect", "Cumulative Effect"],
                        vertical_spacing=0.07)

    if pred_lo and pred_hi:
        # Clip CI bands to ±3x actual data range to prevent huge bands dominating chart
        if obs_col:
            actual_range = float(s[obs_col].abs().max()) * 3
            s[pred_hi] = s[pred_hi].clip(-actual_range, actual_range)
            s[pred_lo] = s[pred_lo].clip(-actual_range, actual_range)
        fig.add_trace(go.Scatter(x=dates_str+dates_str[::-1],
            y=s[pred_hi].tolist()+s[pred_lo].tolist()[::-1],
            fill="toself", fillcolor="rgba(123,127,255,0.20)", line=dict(width=0),
            showlegend=False, hoverinfo="skip"), row=1, col=1)
    if pred_col:
        fig.add_trace(go.Scatter(x=dates_str, y=s[pred_col].tolist(),
            line=dict(color="#84CC16", width=2, dash="dash"), name="Predicted"), row=1, col=1)
    if obs_col:
        fig.add_trace(go.Scatter(x=dates_str, y=s[obs_col].tolist(),
            line=dict(color="#059669", width=2.5), name=f"Actual ({target_name})"), row=1, col=1)

    if eff_lo and eff_hi:
        if eff_col:
            eff_range = float(s[eff_col].abs().max()) * 3 if s[eff_col].abs().max() > 0 else 1000
            s[eff_hi] = s[eff_hi].clip(-eff_range, eff_range)
            s[eff_lo] = s[eff_lo].clip(-eff_range, eff_range)
        fig.add_trace(go.Scatter(x=dates_str+dates_str[::-1],
            y=s[eff_hi].tolist()+s[eff_lo].tolist()[::-1],
            fill="toself", fillcolor="rgba(255,92,110,0.18)", line=dict(width=0),
            showlegend=False, hoverinfo="skip"), row=2, col=1)
    if eff_col:
        fig.add_trace(go.Scatter(x=dates_str, y=s[eff_col].tolist(),
            line=dict(color="#DC2626", width=2), name="Daily effect"), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#9CA3AF", width=1, dash="dot"), row=2, col=1)

    fig.add_trace(go.Scatter(x=dates_str, y=cum_eff.tolist(),
        line=dict(color="#D97706", width=2), name="Cumulative effect"), row=3, col=1)
    fig.add_hline(y=0, line=dict(color="#9CA3AF", width=1, dash="dot"), row=3, col=1)

    ps = str(post_start)[:10]
    for row in [1, 2, 3]:
        fig.add_vline(x=ps, line=dict(color="#059669", width=1.5, dash="dot"), row=row, col=1)

    fig.update_layout(
        paper_bgcolor="#FFFFFF", plot_bgcolor="#F7F7FB",
        font=dict(color="#1A1A2E", family="Inter, Arial, sans-serif", size=12),
        height=660, showlegend=True,
        legend=dict(bgcolor="#FFFFFF", bordercolor="#E2DEEF", borderwidth=1, orientation="h", y=-0.06),
        margin=dict(l=55, r=30, t=50, b=60),
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor="#E2DEEF", showgrid=True, zeroline=False, row=i, col=1)
        fig.update_yaxes(gridcolor="#E2DEEF", showgrid=True, zeroline=False, row=i, col=1)
    return fig

def build_diagnostic_signals(ci, pre_ape_df, mape, p_value, pre_start, pre_end, post_start, post_end):
    """Deterministic signal detection. Returns structured list for LLM synthesis."""
    signals = []

    # 1. MAPE
    if mape > 15:
        signals.append({"type": "high_mape", "severity": "critical", "value": round(mape, 2),
                         "message": f"MAPE is {mape:.1f}% — poor pre-period fit. Predictions are unreliable."})
    elif mape > 10:
        signals.append({"type": "high_mape", "severity": "warning", "value": round(mape, 2),
                         "message": f"MAPE is {mape:.1f}% — borderline fit. Review high-APE dates."})

    # 2. P-value
    if p_value is not None:
        if p_value > 0.1:
            signals.append({"type": "not_significant", "severity": "critical", "value": round(p_value, 4),
                             "message": f"p={p_value:.4f} — result is not statistically significant."})
        elif p_value > 0.05:
            signals.append({"type": "marginal_significance", "severity": "warning", "value": round(p_value, 4),
                             "message": f"p={p_value:.4f} — marginally significant. Interpret with caution."})

    # 3. High-APE date clusters
    if 'APE (%)' in pre_ape_df.columns:
        high = pre_ape_df[pre_ape_df['APE (%)'] > 25].copy()
        if len(high) >= 3:
            worst = high.nlargest(3, 'APE (%)')[['Date', 'APE (%)']].to_dict('records')
            signals.append({"type": "high_ape_cluster", "severity": "warning", "count": len(high),
                             "top_dates": worst,
                             "message": f"{len(high)} days with APE > 25% — possible structural break or anomaly window."})
        elif len(high) > 0:
            worst = high.nlargest(len(high), 'APE (%)')[['Date', 'APE (%)']].to_dict('records')
            signals.append({"type": "high_ape_dates", "severity": "info", "top_dates": worst,
                             "message": f"{len(high)} day(s) with APE > 25%."})

    # 4. Pre-period length
    pre_days  = (pd.Timestamp(pre_end)  - pd.Timestamp(pre_start)).days  + 1
    post_days = (pd.Timestamp(post_end) - pd.Timestamp(post_start)).days + 1
    ratio = pre_days / max(post_days, 1)
    if pre_days < 30:
        signals.append({"type": "short_pre_period", "severity": "warning", "value": pre_days,
                         "message": f"Pre-period is only {pre_days} days — minimum recommended is 30."})
    elif ratio < 2:
        signals.append({"type": "low_pre_post_ratio", "severity": "info", "value": round(ratio, 1),
                         "message": f"Pre/post ratio is {ratio:.1f}× (recommended ≥ 2×)."})

    # 5. CI crossing zero in post-period
    try:
        post_s = ci.inferences[ci.inferences.index >= pd.Timestamp(post_start)]
        lo_col = next((c for c in ['point_effect_lower','y_pred_lower'] if c in post_s.columns), None)
        hi_col = next((c for c in ['point_effect_upper','y_pred_upper'] if c in post_s.columns), None)
        if lo_col and hi_col:
            cross_pct = ((post_s[lo_col] < 0) & (post_s[hi_col] > 0)).mean()
            if cross_pct > 0.4:
                signals.append({"type": "ci_crosses_zero", "severity": "warning",
                                 "value": round(cross_pct * 100, 1),
                                 "message": f"95% CI crosses zero on {cross_pct*100:.0f}% of post-period days — effect direction uncertain."})
    except Exception:
        pass

    return signals

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in dict(
    raw_df=None, daily_df=None, file_type=None,
    causal_context="", parsed_context={},
    l4_map={}, conversion_events=[], target_col=None,
    quality_signals=[], flag_vars=[], smoothing=False, smooth_window=7,
    advertiser="", step=0, hypothesis=None, max_step=0,
    pre_start=None, pre_end=None, post_start=None, post_end=None,
    covariate_selection={},
    sot_targets=[], sot_treatment_start=None, sot_treatment_end=None,
    sot_channel_roles={}, cpa_inputs={}, sot_selected_targets=[], llm_eda_result=None,
    visual_studio_history=[], gt_data=None, gt_keywords=[], summary_report=None,
    ci_result=None, ci_mape=None, ci_ape_df=None, ci_p_value=None,
    ci_signals=None, ci_llm_diagnosis=None, ci_narrator=None,
    ci_summary_data=None, ci_covariates=[], ci_run_count=0,
    ci_placebo_results=None, ci_placebo_extended=False,
    brief_text="", ai_synthesis=None, _kb_saved=False,
).items():
    if k not in st.session_state:
        st.session_state[k] = v



# ── Helpers ────────────────────────────────────────────────────────────────────
def pill(t, c="pill-g"): return f'<span class="pill {c}">{t}</span>'
def badge(t): return f'<span class="sbadge">{t}</span>'
def mtile(lbl, val, cls=""): return f'<div class="mtile"><div class="ml">{lbl}</div><div class="mv {cls}">{val}</div></div>'

def parse_dates(df, col="p_date"):
    import re as _re
    df = df.copy()
    sample = df[col].dropna().astype(str).head(30)
    day_first = any(
        int(_re.split(r"[/\-]", s)[0]) > 12
        for s in sample if _re.match(r"\d{1,2}[/\-]\d{1,2}[/\-]\d{4}", s)
    )
    df[col] = pd.to_datetime(df[col], dayfirst=day_first, errors="coerce")
    return df.sort_values(col).reset_index(drop=True)

SOT_HYPOTHESES = {"cross_channel", "marketplace_spillover", "sot_holdout"}

# ── Master channel + metric vocabulary ────────────────────────────────────────
CH_NAME_MAP = {
    # TikTok
    "tiktok":                  "Paid_Social_TikTok",
    "tiktok ads":              "Paid_Social_TikTok",
    "paid social tiktok":      "Paid_Social_TikTok",
    "paid social - tiktok":    "Paid_Social_TikTok",
    "paid_social_tiktok":      "Paid_Social_TikTok",
    "paid social-tiktok":      "Paid_Social_TikTok",
    "tt":                      "Paid_Social_TikTok",
    # Meta
    "meta":                    "Paid_Social_Meta",
    "meta ads":                "Paid_Social_Meta",
    "facebook":                "Paid_Social_Meta",
    "fb":                      "Paid_Social_Meta",
    "paid social meta":        "Paid_Social_Meta",
    "paid social - meta":      "Paid_Social_Meta",
    "paid_social_meta":        "Paid_Social_Meta",
    "instagram":               "Paid_Social_Meta",
    # Google paid
    "google cpc":              "Paid_Search_Google",
    "google paid":             "Paid_Search_Google",
    "paid search":             "Paid_Search_Google",
    "paid search google":      "Paid_Search_Google",
    "paid search - google":    "Paid_Search_Google",
    "paid_search":             "Paid_Search_Google",
    "paid_search_google":      "Paid_Search_Google",
    "sem":                     "Paid_Search_Google",
    "google sem":              "Paid_Search_Google",
    # Direct
    "direct":                  "Direct",
    # Organic
    "organic":                 "Organic_Search",
    "organic search":          "Organic_Search",
    "organic_search":          "Organic_Search",
    "organic (google)":        "Organic_Search",
    "organic search (google)": "Organic_Search",
    "seo":                     "Organic_Search",
    # Other / catch-all
    "other":                   "Other",
    "others":                  "Other",
}

METRIC_NAME_MAP = {
    "sessions":    "Sessions",
    "session":     "Sessions",
    "visits":      "Sessions",
    "users":       "Sessions",
    "conversions": "Conversions",
    "conversion":  "Conversions",
    "conv":        "Conversions",
    "orders":      "Conversions",
    "purchases":   "Conversions",
    "transactions":"Conversions",
    "revenue":     "Revenue",
    "gmv":         "Revenue",
    "sales":       "Revenue",
    "value":       "Revenue",
    "roas":        "ROAS",
    "cac":         "CaC",
    "cpa":         "CaC",
    "cost per":    "CaC",
}

_CH_KEYS = list(set(CH_NAME_MAP.values()))

def _ch(name):
    """Resolve a raw channel name to canonical key."""
    n = str(name).strip().lower()
    if n in ("","nan","none","channel","date"): return None
    if n in CH_NAME_MAP: return CH_NAME_MAP[n]
    n2 = n.replace("-"," ").replace("_"," ").strip()
    if n2 in CH_NAME_MAP: return CH_NAME_MAP[n2]
    best, best_len = None, 0
    for k, v in CH_NAME_MAP.items():
        kn = k.replace("-"," ").replace("_"," ")
        if kn in n2 and len(kn) > best_len:
            best, best_len = v, len(kn)
    return best

def _met(name):
    """Resolve a raw metric name to canonical key."""
    n = str(name).strip().lower()
    if n in METRIC_NAME_MAP: return METRIC_NAME_MAP[n]
    for k, v in METRIC_NAME_MAP.items():
        if k in n: return v
    return None

def _to_num(series):
    """Safely coerce a Series to numeric. Skip if already typed."""
    if pd.api.types.is_numeric_dtype(series): return series
    if pd.api.types.is_datetime64_any_dtype(series): return series
    try:
        cleaned = series.astype(str).str.replace(r"[$,€£¥]","",regex=True).str.strip()
        num = pd.to_numeric(cleaned, errors="coerce")
        return num if num.notna().sum() > len(series)*0.3 else series
    except Exception:
        return series

def _dedup(cols):
    seen={}; out=[]
    for c in cols:
        if c not in seen: seen[c]=0; out.append(c)
        else: seen[c]+=1; out.append(f"{c}_{seen[c]+1}")
    return out

def _build_cols_from_two_header_rows(row0, row1):
    """
    Given two header rows (lists of strings), build canonical column names.
    row0 = channel names (forward-filled over merged cells)
    row1 = metric names
    Returns list of canonical column names.
    """
    last_ch = None; ch_filled = []
    for v in row0:
        sv = str(v).strip() if v is not None and str(v).lower() not in ("nan","none","") else ""
        if sv:
            canon = _ch(sv)
            # Always update last_ch when we see a non-empty header cell, even if
            # unrecognised — prevents unknown channels from inheriting the previous one
            last_ch = canon if canon else sv.replace(" ", "_").replace("-", "_")
        ch_filled.append(last_ch)

    cols = []
    for i, (ch, mt) in enumerate(zip(ch_filled, row1)):
        mt_s = str(mt).strip() if mt is not None and str(mt).lower() not in ("nan","none","") else ""
        met  = _met(mt_s)
        if i == 0 or mt_s.lower() in ("date","week","day","period",""):
            cols.append("p_date")
        elif ch and met:
            cols.append(f"{ch}_{met}")
        elif ch and mt_s:
            cols.append(f"{ch}_{mt_s}".replace(" ","_").replace("-","_"))
        elif met:
            cols.append(met)
        else:
            cols.append(mt_s.replace(" ","_") or f"col_{i}")
    return _dedup(cols)

def _is_two_header(peek_df):
    """Return True if this looks like a merged-channel-header file."""
    if len(peek_df) < 2: return False
    row0 = [str(v).strip() for v in peek_df.iloc[0]]
    row1 = [str(v).strip() for v in peek_df.iloc[1]]
    ch_hits  = sum(1 for v in row0 if _ch(v) is not None)
    met_hits = sum(1 for v in row1 if _met(v) is not None)
    return ch_hits >= 1 and met_hits >= 2

def normalise_file(file_obj_or_bytes, filename="file.csv"):
    """
    THE single entry point for all file types.
    Tries strategies in order, returns (df_canonical, ftype, warnings[]).
    df_canonical always has:
      - 'p_date' column (datetime)
      - numeric value columns named {Channel}_{Metric} using canonical keys
    """
    import io
    warnings = []

    # ── Strategy 0: detect TTAM ────────────────────────────────────────────────
    def _looks_like_ttam(cols):
        cl = [c.lower() for c in cols]
        return "l4 product tag" in cl and "advertiser name" in cl

    # ── Read raw bytes ─────────────────────────────────────────────────────────
    is_excel = str(filename).lower().endswith((".xlsx",".xls"))
    try:
        if is_excel:
            if hasattr(file_obj_or_bytes, "read"):
                raw_bytes = file_obj_or_bytes.read()
                file_obj_or_bytes.seek(0)
            else:
                raw_bytes = file_obj_or_bytes
            peek  = pd.read_excel(io.BytesIO(raw_bytes), header=None, nrows=4)
            full  = pd.read_excel(io.BytesIO(raw_bytes), header=None)
        else:
            if isinstance(file_obj_or_bytes, bytes):
                text = file_obj_or_bytes.decode("utf-8", errors="replace")
            elif hasattr(file_obj_or_bytes, "read"):
                raw_bytes = file_obj_or_bytes.read()
                file_obj_or_bytes.seek(0)
                text = raw_bytes.decode("utf-8", errors="replace")
            else:
                text = str(file_obj_or_bytes)
            peek = pd.read_csv(io.StringIO(text), header=None, nrows=4)
            full = pd.read_csv(io.StringIO(text), header=None)
    except Exception as e:
        return None, None, [f"Could not read file: {e}"]

    # ── Route: TTAM ────────────────────────────────────────────────────────────
    try:
        if is_excel:
            flat_test = pd.read_excel(io.BytesIO(raw_bytes), nrows=2)
        else:
            flat_test = pd.read_csv(io.StringIO(text), nrows=2)
        if _looks_like_ttam(flat_test.columns):
            df = flat_test.__class__(pd.read_excel(io.BytesIO(raw_bytes)) if is_excel
                                     else pd.read_csv(io.StringIO(text)))
            df.columns = [c.strip() for c in df.columns]
            df = _finalise(df, "raw_ttam")
            return df, "raw_ttam", []
    except Exception:
        pass

    # ── Route: two-header (BabyBoo / Triangl style) ──────────────────────────
    if _is_two_header(peek):
        try:
            row0 = peek.iloc[0].tolist()
            row1 = peek.iloc[1].tolist()
            cols = _build_cols_from_two_header_rows(row0, row1)
            df   = full.iloc[2:].copy().reset_index(drop=True)
            df.columns = cols[:len(df.columns)]
            for c in df.columns:
                if c == "p_date": continue
                df[c] = _to_num(df[c])
            df = _finalise(df, "sot")
            if len(df) < 3:
                warnings.append("Very few rows after parsing — check file format.")
            return df, "sot", warnings
        except Exception as e:
            warnings.append(f"Two-header parse failed ({e}), trying flat parse...")

    # ── Route: flat single-header ─────────────────────────────────────────────
    try:
        if is_excel:
            df = pd.read_excel(io.BytesIO(raw_bytes))
        else:
            df = pd.read_csv(io.StringIO(text))
        df.columns = [str(c).strip() for c in df.columns]

        # Find date column
        date_col = next((c for c in df.columns
                         if any(k in c.lower() for k in ["date","week","day","period"])),
                        df.columns[0])
        df = df.rename(columns={date_col: "p_date"})

        # If the dataset already has ttam_* or flag_* columns it's pre-cleaned —
        # skip renaming entirely so no duplicate/mangled column names appear.
        _is_pre_cleaned = any(
            c.lower().startswith("ttam_") or c.lower().startswith("flag_")
            for c in df.columns
        )
        if not _is_pre_cleaned:
            rename = {}
            for col in df.columns:
                if col == "p_date": continue
                cl = col.lower().replace(" ","_").replace("-","_")
                ch_f = next((v for k,v in CH_NAME_MAP.items()
                             if k.replace(" ","_").replace("-","_") in cl), None)
                met_f = next((v for k,v in METRIC_NAME_MAP.items() if k in cl), None)
                if ch_f and met_f:
                    new = f"{ch_f}_{met_f}"
                    if new not in rename.values(): rename[col] = new
            if rename: df = df.rename(columns=rename)

        for c in df.columns:
            if c == "p_date": continue
            df[c] = _to_num(df[c])

        ftype = detect_type(df)
        df = _finalise(df, ftype)
        return df, ftype, warnings
    except Exception as e:
        return None, None, [f"All parse strategies failed: {e}"]

def _finalise(df, ftype):
    """Common post-parse cleanup."""
    df = df.copy()
    if "p_date" not in df.columns:
        date_cands = [c for c in df.columns if "date" in str(c).lower()]
        if date_cands: df = df.rename(columns={date_cands[0]: "p_date"})
        else: df = df.rename(columns={df.columns[0]: "p_date"})
    import re as _re
    _samp = df["p_date"].dropna().astype(str).head(30)
    _dfirst = any(int(_re.split(r"[/\-]", s)[0]) > 12 for s in _samp if _re.match(r"\d{1,2}[/\-]\d{1,2}[/\-]\d{4}", s))
    df["p_date"] = pd.to_datetime(df["p_date"], dayfirst=_dfirst, errors="coerce")
    df = df.dropna(subset=["p_date"]).sort_values("p_date").reset_index(drop=True)
    return df

def detect_type(df):
    if {"L4 Product Tag","Advertiser Name","Cost (USD)"}.issubset(set(df.columns)):
        return "raw_ttam"
    col_lower = [c.lower() for c in df.columns]
    hits = sum(1 for s in ["direct","organic","paid_social","paid_search","tiktok","meta"]
               if any(s in c for c in col_lower))
    return "sot" if hits >= 2 else "pre_aggregated"

def parse_sot_columns(df):
    """
    Return dict: canonical_channel_key → [list of metric columns].
    Works on any DataFrame that went through normalise_file().
    """
    channels = {}
    for col in df.columns:
        if col == "p_date" or "date" in col.lower(): continue
        matched = False
        # Check canonical prefix first (fast path for normalised files)
        for ck in _CH_KEYS:
            if col.startswith(ck + "_") or col == ck:
                channels.setdefault(ck, []).append(col)
                matched = True; break
        if not matched:
            # Fuzzy fallback for pre-cleaned files — use startswith to avoid
            # "tt" matching "ttam_*" or "tiktok" matching "non_tiktok_*"
            cl = col.lower().replace(" ","_").replace("-","_")
            for k, v in CH_NAME_MAP.items():
                kn = k.replace(" ","_").replace("-","_")
                if cl.startswith(kn + "_") or cl == kn:
                    channels.setdefault(v, []).append(col)
                    matched = True; break
        if not matched:
            channels.setdefault("Other", []).append(col)
    return channels

def aggregate_ttam(df, l4_map, conv_events):
    df = df.copy()
    df["Category"] = df["L4 Product Tag"].map(l4_map).fillna("Other")
    cost_p = df.pivot_table(index="p_date", columns="Category", values="Cost (USD)",
                             aggfunc="sum", fill_value=0).reset_index()
    cost_p.columns = ["p_date" if c=="p_date" else f"{c}Cost" for c in cost_p.columns]
    imp_p = df.pivot_table(index="p_date", columns="Category", values="Impressions",
                            aggfunc="sum", fill_value=0).reset_index()
    imp_p.columns = ["p_date" if c=="p_date" else f"{c}Impressions" for c in imp_p.columns]
    conv_df = df[df["Optimization Event(External Actions)"].isin(conv_events)] if conv_events else df
    conv_agg = conv_df.groupby("p_date")["Conversions"].sum().reset_index()
    conv_agg.columns = ["p_date", "Conversions_Raw"]
    wd = df.groupby("p_date")["Weekdays/Weekends"].first().reset_index()
    wd["IsWeekend"] = (wd["Weekdays/Weekends"] == "Weekends").astype(int)
    out = cost_p.merge(imp_p, on="p_date", how="outer")
    out = out.merge(conv_agg, on="p_date", how="left")
    out = out.merge(wd[["p_date","IsWeekend"]], on="p_date", how="left")
    out["Conversions_Raw"] = out["Conversions_Raw"].fillna(0).astype(int)
    cost_cols = [c for c in out.columns if c.endswith("Cost")]
    out["TotalCost"] = out[cost_cols].sum(axis=1)
    out["CPA"] = out.apply(lambda r: r["TotalCost"]/r["Conversions_Raw"] if r["Conversions_Raw"]>0 else np.nan, axis=1)
    imp_cols = [c for c in out.columns if c.endswith("Impressions")]
    out["TotalImpressions"] = out[imp_cols].sum(axis=1)
    return out

def quality_checks(df):
    sigs = []
    dates = pd.to_datetime(df["p_date"])
    full = pd.date_range(dates.min(), dates.max(), freq="D")
    missing = sorted(set(full)-set(dates))
    if missing:
        sigs.append(dict(sev="high", msg=f"{len(missing)} missing date(s)",
            detail=", ".join(d.strftime("%d %b") for d in missing[:5])+("…" if len(missing)>5 else "")))
    n = len(df)
    if n < 40:
        sigs.append(dict(sev="high", msg=f"Only {n} days — minimum ~40 recommended",
            detail="Extend your TTAM extraction window if possible."))
    elif n < 60:
        sigs.append(dict(sev="medium", msg=f"{n} days — borderline pre-period length",
            detail="60+ days gives more robust counterfactual estimates."))
    conv_c = next((c for c in df.columns if "conversion" in c.lower() or c.lower()=="conversions"), None)
    if conv_c:
        s = pd.to_numeric(df[conv_c], errors="coerce").fillna(0)
        rm = s.rolling(7, center=True, min_periods=3).mean()
        rs = s.rolling(7, center=True, min_periods=3).std().fillna(1)
        z = ((s-rm)/(rs+1e-9)).abs()
        bad = df.loc[z>3.5, "p_date"]
        if len(bad):
            sigs.append(dict(sev="medium", msg=f"{len(bad)} outlier date(s) in {conv_c} (>3.5σ)",
                detail="Consider flagging: "+", ".join(bad.dt.strftime("%d %b").tolist()[:5])))
    for sc in [c for c in df.columns if c.endswith("Cost") and "Total" not in c]:
        s = pd.to_numeric(df[sc], errors="coerce").fillna(0)
        zeros = (s==0).sum()
        if 3 < zeros < len(df)*0.8:
            sigs.append(dict(sev="low", msg=f"{sc}: {zeros} zero-spend days",
                detail="May need a launch flag if spend started mid-window."))
    return sigs

def suggest_flags(df, synthesis=None):
    """Merge data-driven flags with brief-derived flags from AI synthesis."""
    flags = []
    seen_names = set()
    dates = pd.to_datetime(df["p_date"])
    date_min = dates.min().date()
    date_max = dates.max().date()

    # ── Brief-derived flags (highest priority, from synthesis) ─────────────────
    if synthesis:
        for fs in synthesis.get("flag_suggestions", []):
            try:
                s = str(pd.Timestamp(fs["start"]).date())
                e = str(pd.Timestamp(fs["end"]).date())
            except Exception:
                continue
            name = fs.get("name", "Brief_Flag")
            if name in seen_names:
                continue
            seen_names.add(name)
            flags.append(dict(
                name=name, start=s, end=e,
                confirmed=True,
                is_treatment=fs.get("is_treatment", False),
                source="brief",
                rationale=f"[From brief] {fs.get('rationale', '')}",
            ))
        # Also surface promo events not already covered
        for pe in synthesis.get("promo_events", []):
            if not pe.get("flag_recommended"):
                continue
            name = pe.get("name", "Promo_Flag").replace(" ", "_")
            if name in seen_names:
                continue
            try:
                s = str(pd.Timestamp(pe["start"]).date())
                e = str(pd.Timestamp(pe["end"]).date())
            except Exception:
                continue
            seen_names.add(name)
            flags.append(dict(
                name=name, start=s, end=e,
                confirmed=True,
                is_treatment=False,
                source="brief",
                rationale=f"[From brief] {pe.get('rationale', pe.get('name', ''))}",
            ))

    # ── Data-driven flags (TopView spend window) ───────────────────────────────
    if "TopViewCost" in df.columns:
        tv = pd.to_numeric(df["TopViewCost"], errors="coerce").fillna(0)
        active = df.loc[tv>0, "p_date"]
        if len(active):
            name = "TopView_Window"
            if name not in seen_names:
                seen_names.add(name)
                flags.append(dict(
                    name=name,
                    start=str(active.min().date()),
                    end=str(active.max().date()),
                    confirmed=False, is_treatment=True,
                    source="data",
                    rationale=f"[From data] TopView spend detected {active.min().strftime('%d %b')}–{active.max().strftime('%d %b')}. This is your treatment period.",
                ))

    # ── Data-driven flags (channel launch structural breaks) ───────────────────
    for sc in [c for c in df.columns if c.endswith("Cost") and "Total" not in c]:
        s = pd.to_numeric(df[sc], errors="coerce").fillna(0)
        if s.mean() > 0 and (s[:14]<s.mean()*0.1).sum()>4 and s[14:].mean()>s[:14].mean()*3:
            launch = dates[s>s.mean()*0.25].min()
            if pd.notna(launch):
                name = f"{sc.replace('Cost','')}Launch_Flag"
                if name not in seen_names:
                    seen_names.add(name)
                    flags.append(dict(
                        name=name,
                        start=str(date_min),
                        end=str(launch.date()),
                        confirmed=False, is_treatment=False,
                        source="data",
                        rationale=f"[From data] {sc} near-zero until {launch.strftime('%d %b')} — structural break needs a flag.",
                    ))
    return flags

def apply_smooth(df, col, w=7):
    df = df.copy()
    df[f"{col}_Smoothed"] = df[col].rolling(w, center=True, min_periods=3).mean()
    return df

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;padding-bottom:18px;border-bottom:1px solid {C['border']};margin-bottom:18px;">
      <span style="font-size:26px">🎯</span>
      <div><div style="font-size:17px;font-weight:700">Causal Hub</div>
      <div style="font-size:11px;color:{C['grey']}">TikTok Measurement · AUNZ</div></div></div>""", unsafe_allow_html=True)

    # Show confirmed hypothesis in sidebar (set by AI synthesis in step 1)
    if st.session_state.hypothesis:
        uc = next((u for u in USE_CASES if u["id"] == st.session_state.hypothesis), None)
        if uc:
            st.markdown(f"""<div style="background:{uc['colour']}12;border:1px solid {uc['colour']}33;border-radius:8px;padding:10px 12px;margin-bottom:14px;">
<div style="font-size:10px;color:{C['grey']};text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px">HYPOTHESIS</div>
<div style="font-size:12px;font-weight:600;color:{C['text']}">{uc['icon']} {uc['name']}</div>
<div style="margin-top:6px">{pill(uc['tag'], uc['tag_cls'])}</div></div>""", unsafe_allow_html=True)

    # Sidebar shows logical groups mapping to internal steps
    LOGICAL_STEPS = [
        ("1 · Upload",  [0],    0),
        ("2 · Explore", [4],    4),
        ("3 · Model",   [6],    6),
        ("4 · Results", [7],    7),
    ]
    cur = st.session_state.step
    st.session_state.max_step = max(st.session_state.get("max_step", 0), cur)

    st.markdown(f'<style>.step-btn button{{background:transparent!important;border:none!important;box-shadow:none!important;color:{C["grey"]}!important;font-weight:400!important;text-align:left!important;padding:8px 12px!important;border-radius:6px!important;width:100%!important;font-size:13px!important;border-left:2px solid transparent!important;}}.step-btn-active button{{background:#EDE9FE!important;color:{C["purple"]}!important;font-weight:600!important;border-left:2px solid {C["purple"]}!important;}}.step-btn button:hover{{background:#F5F3FF!important;color:{C["purple"]}!important;}}</style>', unsafe_allow_html=True)

    for label, internal_steps, nav_target in LOGICAL_STEPS:
        active = cur in internal_steps
        reachable = any(s <= st.session_state.max_step for s in internal_steps)
        css_class = "step-btn-active" if active else "step-btn"
        if reachable:
            with st.container():
                st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                if st.button(label, key=f"nav_{label}", use_container_width=True):
                    st.session_state.step = nav_target
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="padding:8px 12px;border-radius:6px;margin:2px 0;color:{C["grey_l"]};font-size:13px;opacity:0.45;cursor:default">{label}</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    if st.session_state.daily_df is not None:
        df_ = parse_dates(st.session_state.daily_df)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"""<div style="font-size:11px;color:{C['grey']};margin-bottom:6px">LOADED</div>
<div style="font-size:13px;font-weight:600">{st.session_state.advertiser or "—"}</div>
<div style="font-size:11px;color:{C['grey']}">{df_['p_date'].min().strftime('%d %b')} – {df_['p_date'].max().strftime('%d %b %Y')} · {len(df_)} days</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# AI SYNTHESIS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_synthesis_stats(df, brief_text=""):
    """Compute structured stats summary for LLM synthesis. Python does the maths, LLM writes the narrative."""
    import re as _re
    lines = []
    df = df.copy()
    df["p_date"] = pd.to_datetime(df["p_date"], errors="coerce")
    df = df.dropna(subset=["p_date"]).sort_values("p_date")

    date_min = df["p_date"].min().date()
    date_max = df["p_date"].max().date()
    total_days = (date_max - date_min).days + 1
    lines.append(f"DATE RANGE: {date_min} to {date_max} ({total_days} calendar days, {len(df)} rows)")

    # Attempt to detect intervention window from brief
    interv_start = None
    if brief_text:
        date_hits = _re.findall(r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}", brief_text)
        parsed_hits = []
        for dh in date_hits:
            try:
                parsed_hits.append(pd.to_datetime(dh, dayfirst=True))
            except Exception:
                pass
        if len(parsed_hits) >= 2:
            parsed_hits.sort()
            interv_start = parsed_hits[0].date()
            interv_end   = parsed_hits[-1].date()
            lines.append(f"BRIEF DATES DETECTED: {interv_start} to {interv_end}")
        elif len(parsed_hits) == 1:
            interv_start = parsed_hits[0].date()
            lines.append(f"BRIEF DATE DETECTED: {interv_start}")

    # Detect numeric columns (exclude date)
    num_cols = [c for c in df.columns if c != "p_date" and pd.api.types.is_numeric_dtype(df[c])]

    # Split into pre/post using intervention start or 80/20
    if interv_start and interv_start > date_min:
        pre_df  = df[df["p_date"].dt.date < interv_start]
        post_df = df[df["p_date"].dt.date >= interv_start]
        lines.append(f"SPLIT: pre={len(pre_df)}d (before {interv_start}), post={len(post_df)}d (from {interv_start})")
        ratio = len(pre_df) / len(post_df) if len(post_df) > 0 else 0
        ratio_q = "✅ Good" if ratio >= 2 else "⚠️ Borderline" if ratio >= 1 else "❌ Too short"
        lines.append(f"PRE/POST RATIO: {ratio:.1f}x — {ratio_q}")
    else:
        n = len(df)
        split_idx = int(n * 0.8)
        pre_df  = df.iloc[:split_idx]
        post_df = df.iloc[split_idx:]
        lines.append(f"SPLIT (80/20 auto): pre={len(pre_df)}d, post={len(post_df)}d")

    # Per-channel DoD summary
    lines.append("\nCHANNEL PERFORMANCE (pre avg → post avg, % change):")
    channel_stats = []
    for col in num_cols[:15]:
        pre_avg  = pre_df[col].mean()  if len(pre_df)  else 0
        post_avg = post_df[col].mean() if len(post_df) else 0
        if pre_avg == 0 and post_avg == 0: continue
        lift = (post_avg - pre_avg) / pre_avg * 100 if pre_avg != 0 else 0
        sign = "+" if lift >= 0 else ""
        cv   = pre_df[col].std() / pre_avg * 100 if pre_avg != 0 else 0
        channel_stats.append((col, pre_avg, post_avg, lift, cv))
        lines.append(f"  {col}: {pre_avg:,.1f} → {post_avg:,.1f} ({sign}{lift:.1f}%) | Pre-CV: {cv:.0f}%")

    # Outlier detection: top 3 spike/dip days per key column
    lines.append("\nOUTLIER FLAGS (days > 2.5 SD from rolling mean):")
    for col, *_ in channel_stats[:5]:
        roll = df[col].rolling(7, min_periods=1, center=True).mean()
        roll_std = df[col].rolling(7, min_periods=1, center=True).std().fillna(1)
        z = (df[col] - roll) / roll_std.replace(0, 1)
        spikes = df.loc[z.abs() > 2.5, "p_date"].dt.date.tolist()
        if spikes:
            lines.append(f"  {col}: {spikes[:4]}")

    # Correlation matrix (pre-period only, top pairs)
    lines.append("\nPRE-PERIOD CORRELATIONS (R²) — top covariate candidates:")
    if len(num_cols) > 1 and len(pre_df) > 5:
        corr = pre_df[num_cols].corr()
        pairs = []
        for i, c1 in enumerate(num_cols):
            for c2 in num_cols[i+1:]:
                r = corr.loc[c1, c2]
                if not np.isnan(r):
                    pairs.append((abs(r), c1, c2, r))
        pairs.sort(reverse=True)
        for r_abs, c1, c2, r in pairs[:8]:
            lines.append(f"  {c1} × {c2}: R={r:.2f} (R²={r_abs**2:.2f})")

    return "\n".join(lines)


# ── Brief knowledge base ───────────────────────────────────────────────────────
_KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "briefs_knowledge_base.json")

def _load_knowledge_base():
    if os.path.exists(_KB_PATH):
        try:
            with open(_KB_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_brief_to_knowledge_base(advertiser, hypothesis_type, brief_text, synthesis, outcome_notes=""):
    """Append a completed analysis to the knowledge base for future context."""
    kb = _load_knowledge_base()
    entry = {
        "date": str(pd.Timestamp.now().date()),
        "advertiser": advertiser,
        "hypothesis_type": hypothesis_type,
        "brief_summary": brief_text[:800] if brief_text else "",
        "confounders": [c.get("channel") for c in synthesis.get("cross_channel_confounders", [])],
        "promo_events": [e.get("name") for e in synthesis.get("promo_events", [])],
        "analytical_risk": synthesis.get("analytical_risk", ""),
        "two_model_needed": synthesis.get("two_model_recommendation", {}).get("needed", False),
        "outcome_notes": outcome_notes,
    }
    kb.append(entry)
    kb = kb[-30:]  # keep the 30 most recent
    try:
        with open(_KB_PATH, "w") as f:
            json.dump(kb, f, indent=2)
    except Exception:
        pass

def _build_past_cases_context(hypothesis_type, n=3):
    """Return a short summary of the most relevant past cases for the LLM prompt."""
    kb = _load_knowledge_base()
    if not kb:
        return ""
    same_type = [e for e in kb if e.get("hypothesis_type") == hypothesis_type]
    others = [e for e in kb if e.get("hypothesis_type") != hypothesis_type]
    candidates = (same_type + others)[-n:]
    lines = []
    for e in candidates:
        conf_str = ", ".join(e.get("confounders", [])) or "none noted"
        promo_str = ", ".join(e.get("promo_events", [])) or "none"
        lines.append(
            f"• [{e['date']}] {e['advertiser']} | {e['hypothesis_type']} | Risk: {e['analytical_risk']} "
            f"| Confounders: {conf_str} | Promos: {promo_str} | Two-model: {e.get('two_model_needed','?')}"
            + (f" | Notes: {e['outcome_notes']}" if e.get("outcome_notes") else "")
        )
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — BRIEF + UPLOAD (entry point)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.step == 0:
    st.markdown(f'''<div style="text-align:center;padding:32px 0 24px">
      <div style="font-size:30px;margin-bottom:10px">🎯</div>
      <div style="font-size:26px;font-weight:700;color:{C['text']};margin-bottom:6px">TikTok Causal Hub</div>
      <div style="font-size:14px;color:{C['grey']};max-width:480px;margin:0 auto">Paste the brief for context, upload your data, then explore and model.</div>
    </div>''', unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown(f'<div style="font-size:11px;font-weight:600;color:{C["grey"]};letter-spacing:.07em;margin-bottom:6px">PASTE THE BRIEF</div>', unsafe_allow_html=True)
        brief_in = st.text_area(
            "brief",
            value=st.session_state.brief_text,
            height=220,
            placeholder=(
                'Paste the brief from the sales team here — e.g.\n\n'
                '"Babyboo ran Smart+ Catalog from 1 Mar – 5 May 2026 ($470k). '
                'They want to prove holistic value of Smart+ Catalog and disprove '
                'negative impact on Non-Catalog conversions. KPI: conversions (ROAS & CPA). '
                'Submitted by Nikki Strover-Wood, KA3 Fashion team."'
            ),
            label_visibility="collapsed",
        )
        st.session_state.brief_text = brief_in

        if brief_in:
            st.markdown(f'<div style="font-size:11px;color:{C["grey"]};margin-top:4px">✓ Brief received — upload your data file to continue.</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown(f'<div style="font-size:11px;font-weight:600;color:{C["grey"]};letter-spacing:.07em;margin-bottom:6px">UPLOAD DATA</div>', unsafe_allow_html=True)

        _tab_upload, _tab_gsheet = st.tabs(["📁 Upload CSV / Excel", "🔗 Google Sheet URL"])

        uploaded = None
        _gs_raw = None

        with _tab_upload:
            uploaded = st.file_uploader(
                "upload", type=["csv", "xlsx"], label_visibility="collapsed",
                help="GA4/SOT daily export, raw TTAM export, or pre-aggregated daily CSV"
            )

        with _tab_gsheet:
            st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:8px">Share the sheet as <strong>Anyone with the link → Viewer</strong>, then paste the URL below.</div>', unsafe_allow_html=True)
            _gs_url = st.text_input("Google Sheets URL", placeholder="https://docs.google.com/spreadsheets/d/...", label_visibility="collapsed")
            if _gs_url:
                _csv_url, _ = gsheets_url_to_csv_url(_gs_url)
                if not _csv_url:
                    st.error("That doesn't look like a valid Google Sheets URL.")
                elif st.button("Fetch sheet", use_container_width=True):
                    with st.spinner("Fetching…"):
                        try:
                            _gs_filelike, _gs_fname = fetch_gsheet_as_filelike(_gs_url)
                            st.session_state._gs_filelike = _gs_filelike
                            st.session_state._gs_fname = _gs_fname
                            st.success("Sheet fetched — scroll down to review.")
                        except Exception as _ge:
                            st.error(f"Could not fetch sheet: {_ge}")
            if st.session_state.get("_gs_filelike"):
                _gs_raw = st.session_state._gs_filelike
                _gs_raw.seek(0)

        st.markdown(f"""<div style="font-size:12px;color:{C['grey']};line-height:1.9;margin-top:10px">
<span style="color:{C['purple']};font-weight:600">GA4 / SOT export</span> — multi-channel daily data (sessions, conversions, revenue).<br>
<span style="color:{C['purple']};font-weight:600">Raw TTAM export</span> — one row per L4 product tag per day.<br>
<span style="color:{C['purple']};font-weight:600">Pre-aggregated</span> — already has BrandCostExTV, MidFunnelCost etc.</div>""", unsafe_allow_html=True)

        def _process_raw_file(filelike, fname):
            raw, ftype, parse_warnings = normalise_file(filelike, fname)
            if raw is None or len(raw) == 0:
                st.error(f"Could not read file: {'; '.join(parse_warnings)}")
                return
            for w in parse_warnings:
                st.warning(w)
            file_sig = f"{fname}_{len(raw)}"
            if st.session_state.get("_last_file_sig") != file_sig:
                st.session_state._last_file_sig = file_sig
                for k in ["sot_treatment_start", "sot_treatment_end", "pre_start", "pre_end",
                          "post_start", "post_end", "summary_report", "visual_studio_history",
                          "gt_data", "daily_df", "sot_selected_targets", "sot_channel_roles",
                          "ai_synthesis"]:
                    st.session_state[k] = [] if "history" in k else None
            st.session_state.raw_df = raw
            st.session_state.file_type = ftype
            if ftype == "raw_ttam" and "Advertiser Name" in raw.columns and not st.session_state.advertiser:
                st.session_state.advertiser = str(raw["Advertiser Name"].dropna().iloc[0])
            if ftype == "pre_aggregated":
                st.session_state.daily_df = raw
            date_range_str = (
                f"{raw['p_date'].min().strftime('%d %b %Y')} – {raw['p_date'].max().strftime('%d %b %Y')}"
                if raw["p_date"].notna().any() else "—"
            )
            ch_detected = parse_sot_columns(raw) if ftype == "sot" else {}
            ch_names = [k.replace("Paid_Social_", "").replace("Paid_Search_", "").replace("_", " ")
                        for k in ch_detected if k != "Other"]
            ch_str = " · ".join(ch_names) if ch_names else ""
            ftype_label = {"raw_ttam": "Raw TTAM", "sot": "SOT / GA4", "pre_aggregated": "Pre-aggregated"}.get(ftype, ftype)
            st.markdown(f"""<div style="background:{C['purple']}08;border:1px solid {C['purple']}33;border-radius:8px;padding:14px;margin-top:10px">
<div style="font-weight:600;color:{C['purple']};margin-bottom:4px">✓ {fname}</div>
<div style="font-size:12px;color:{C['grey']}">{len(raw):,} rows · {date_range_str}</div>
{f'<div style="font-size:11px;color:{C["grey"]};margin-top:4px">Channels: {ch_str}</div>' if ch_str else ""}
<div style="margin-top:6px">{pill(ftype_label, "pill-c")}</div>
</div>""", unsafe_allow_html=True)

        if uploaded:
            try:
                uploaded.seek(0)
                _process_raw_file(uploaded, uploaded.name)
            except Exception as e:
                st.error(f"Could not read file: {e}")
        elif _gs_raw:
            try:
                _process_raw_file(_gs_raw, st.session_state.get("_gs_fname", "gsheet_import.csv"))
            except Exception as e:
                st.error(f"Could not parse sheet: {e}")

    # Advertiser name + Analyse button
    if st.session_state.raw_df is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        adv_c, btn_c = st.columns([3, 1])
        with adv_c:
            adv = st.text_input("Advertiser name", value=st.session_state.advertiser or "", placeholder="e.g. Babyboo")
            st.session_state.advertiser = adv
        with btn_c:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Continue to Explore →", use_container_width=True, type="primary"):
                if st.session_state.file_type == "raw_ttam":
                    try:
                        _default_conv = ["Complete Payment", "Purchase", "Checkout", "Submit Form", "App Install"]
                        agg = aggregate_ttam(st.session_state.raw_df, DEFAULT_L4, _default_conv)
                        st.session_state.daily_df = agg
                    except Exception as _agg_e:
                        st.session_state.daily_df = st.session_state.raw_df.copy()
                else:
                    st.session_state.daily_df = st.session_state.raw_df.copy()
                st.session_state.step = 4
                st.session_state.max_step = max(st.session_state.max_step, 4)
                st.rerun()

elif st.session_state.step == 1:
    # ── AI Analysis (Explore — top half) ─────────────────────────────────────
    st.markdown(f'<span style="font-size:20px;font-weight:600">AI Analysis</span>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{C["grey"]};margin-bottom:16px;font-size:13px">AI-detected signals, watchouts and confounders — overlaid on your campaign timeline.</div>', unsafe_allow_html=True)

    df_synth = st.session_state.daily_df
    if df_synth is None and st.session_state.raw_df is not None:
        df_synth = st.session_state.raw_df.copy()

    if df_synth is None:
        st.warning("No data loaded. Go back and upload a file.")
        if st.button("Back"): st.session_state.step = 0; st.rerun()
        st.stop()

    df_synth = parse_dates(df_synth)
    date_min_s = df_synth["p_date"].min().date()
    date_max_s = df_synth["p_date"].max().date()

    syn = st.session_state.ai_synthesis or {}
    if not syn:
        st.info("AI analysis not yet available — return to Configure to run it.")
        if st.button("Back to Configure"): st.session_state.step = 2; st.rerun()
        st.stop()

    # ── Helpers ────────────────────────────────────────────────────────────
    _risk_map  = {"low": C["green"], "medium": C["amber"], "high": C["red"], "very_high": C["red"]}
    _ar  = (syn.get("analytical_risk") or "medium").lower()
    _arc = _risk_map.get(_ar, C["amber"])
    ccf  = syn.get("cross_channel_confounders", [])
    pes  = syn.get("promo_events", [])

    # ── Read-only period reminder strip ────────────────────────────────────
    _pre_s  = st.session_state.pre_start
    _pre_e  = st.session_state.pre_end
    _post_s = st.session_state.post_start
    _post_e = st.session_state.post_end
    _tgt_lbl = st.session_state.get("target_col") or "—"
    _fmt = lambda d: d.strftime("%-d %b %Y") if d else "—"
    try:
        _pre_days_r  = (_pre_e  - _pre_s).days  + 1 if _pre_s and _pre_e  else 0
        _post_days_r = (_post_e - _post_s).days + 1 if _post_s and _post_e else 0
        _ratio_r     = _pre_days_r / _post_days_r if _post_days_r else 0
        _rc_r        = C["green"] if _ratio_r >= 2 else C["amber"] if _ratio_r >= 1 else C["red"]
    except Exception:
        _pre_days_r = _post_days_r = 0; _ratio_r = 0; _rc_r = C["red"]
    st.markdown(
        f'<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:{C["surface"]};'
        f'border:1px solid {C["border"]};border-radius:10px;padding:10px 16px;margin-bottom:16px;font-size:12px">'
        f'<span style="color:{C["grey"]};font-weight:600;letter-spacing:.05em;font-size:11px">ANALYSIS WINDOW</span>'
        f'<span style="color:{C["border"]}">|</span>'
        f'<span style="color:{C["text"]}">Pre: <strong>{_fmt(_pre_s)} → {_fmt(_pre_e)}</strong> ({_pre_days_r}d)</span>'
        f'<span style="color:{C["border"]}">·</span>'
        f'<span style="color:{C["text"]}">Post: <strong>{_fmt(_post_s)} → {_fmt(_post_e)}</strong> ({_post_days_r}d)</span>'
        f'<span style="color:{C["border"]}">·</span>'
        f'<span style="font-weight:700;color:{_rc_r}">Ratio {_ratio_r:.1f}×</span>'
        f'<span style="color:{C["border"]}">|</span>'
        f'<span style="color:{C["grey"]}">Target: <strong style="color:{C["text"]}">{_tgt_lbl}</strong></span>'
        f'<span style="margin-left:auto;font-size:11px;color:{C["grey"]}">← Edit in Configure</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Chart + Signals ─────────────────────────────────────────────────────
    col_chart, col_signals = st.columns([3, 2], gap="large")

    with col_chart:
        # Metric selector for the chart (visual only — doesn't change model target)
        _num_cols = [c for c in df_synth.columns if c != "p_date" and pd.api.types.is_numeric_dtype(df_synth[c])]
        _pref_kws = ["conversion", "conv", "purchase", "order", "revenue", "sales", "session"]
        _default_chart_col = st.session_state.get("target_col") or next(
            (c for kw in _pref_kws for c in _num_cols if kw in c.lower()), _num_cols[0] if _num_cols else None
        )
        _chart_col = _default_chart_col
        if _num_cols and len(_num_cols) > 1:
            _vis_cols = ([_default_chart_col] + [c for c in _num_cols if c != _default_chart_col])[:5]
            _chart_col = st.radio("Chart metric", _vis_cols, horizontal=True,
                key="analysis_chart_col", label_visibility="collapsed")

        # Use periods from session state (set in Configure)
        def _clamp_d(d, lo, hi): return max(lo, min(hi, d)) if d else lo
        _new_pre_s  = _clamp_d(_pre_s  or date_min_s, date_min_s, date_max_s)
        _new_pre_e  = _clamp_d(_pre_e  or date_min_s, date_min_s, date_max_s)
        _new_post_s = _clamp_d(_post_s or date_min_s, date_min_s, date_max_s)
        _new_post_e = _clamp_d(_post_e or date_max_s, date_min_s, date_max_s)

        # Chart — add data trace FIRST so Plotly infers datetime x-axis before vrects
        if _chart_col and _chart_col in df_synth.columns:
            _fig_p = go.Figure()
            # Data line first (establishes x-axis type as datetime)
            _fig_p.add_trace(go.Scatter(
                x=df_synth["p_date"], y=df_synth[_chart_col],
                mode="lines", line=dict(color=C["purple"], width=2.5),
                name=_chart_col, hovertemplate="%{x|%d %b}: %{y:,.0f}<extra></extra>",
                fill="tozeroy", fillcolor="rgba(123,127,255,0.12)",
            ))
            # Pre-period shading
            _fig_p.add_vrect(x0=str(_new_pre_s), x1=str(_new_pre_e),
                fillcolor=C["purple"], opacity=0.07, line_width=0)
            # Post-period shading
            _fig_p.add_vrect(x0=str(_new_post_s), x1=str(_new_post_e),
                fillcolor=C["green"], opacity=0.07, line_width=0)
            # Boundary line — no built-in annotation (added separately below)
            _int_x = pd.Timestamp(_new_post_s).value // 10**6
            _fig_p.add_vline(x=_int_x,
                line=dict(color=C["purple"], width=1.5, dash="dash"))
            _fig_p.add_annotation(
                x=_int_x, xref="x", y=0.99, yref="paper",
                text="Intervention start",
                showarrow=False, xanchor="left", yanchor="top",
                font=dict(size=10, color=C["purple_l"], family="Inter"),
                bgcolor="rgba(17,17,38,0.94)", borderpad=4,
            )
            # Promo events — black dotted lines, staggered annotations
            for _i, _pe in enumerate(pes):
                try:
                    _pe_x = pd.Timestamp(_pe["start"]).value // 10**6
                    _fig_p.add_vline(x=_pe_x,
                        line=dict(color="#FFFFFF", width=1, dash="dot"))
                    _fig_p.add_annotation(
                        x=_pe_x, xref="x",
                        y=0.99 - (_i + 1) * 0.13,  # stagger each label lower
                        yref="paper",
                        text=_pe.get("name", "Event"),
                        showarrow=False, xanchor="left", yanchor="top",
                        font=dict(size=8, color="#FFFFFF", family="Inter"),
                        bgcolor="rgba(17,17,38,0.92)", borderpad=3,
                    )
                except Exception:
                    pass
            # Pre / Post labels — coloured pill at top of each region
            try:
                _mid_pre  = pd.Timestamp(_new_pre_s) + (pd.Timestamp(_new_pre_e) - pd.Timestamp(_new_pre_s)) / 2
                _mid_post = pd.Timestamp(_new_post_s) + (pd.Timestamp(_new_post_e) - pd.Timestamp(_new_post_s)) / 2
                for _mx, _lbl, _bg in [
                    (_mid_pre,  "PRE",  "rgba(123,127,255,0.88)"),
                    (_mid_post, "POST", "rgba(64,200,122,0.88)"),
                ]:
                    _fig_p.add_annotation(
                        x=_mx, xref="x", y=0.97, yref="paper",
                        text=f"<b>{_lbl}</b>",
                        showarrow=False, xanchor="center", yanchor="top",
                        font=dict(size=11, color="#FFFFFF", family="Inter"),
                        bgcolor=_bg, borderpad=5,
                        bordercolor="rgba(0,0,0,0)",
                    )
            except Exception:
                pass
            _fig_p.update_layout(**{**CHART,
                "height": 300,
                "margin": dict(l=40, r=20, t=10, b=30),
                "showlegend": False,
                "paper_bgcolor": C["surface"],
                "plot_bgcolor":  C["surface"],
            })
            st.plotly_chart(_fig_p, use_container_width=True, config={"displayModeBar": False})
            # Colour key
            _key_html = (
                f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:16px;font-size:11px;color:{C["grey"]}">'
                f'<span style="width:10px;height:10px;border-radius:2px;background:{C["purple"]};opacity:0.5;display:inline-block"></span>Pre-period</span>'
                f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:16px;font-size:11px;color:{C["grey"]}">'
                f'<span style="width:10px;height:10px;border-radius:2px;background:{C["green"]};opacity:0.5;display:inline-block"></span>Post-period (intervention)</span>'
                f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:16px;font-size:11px;color:{C["grey"]}">'
                f'<span style="border-top:2px dotted #FFFFFF;width:16px;display:inline-block;margin-bottom:1px"></span>Promotional event</span>'
            )
            st.markdown(f'<div style="margin-top:-6px;margin-bottom:4px">{_key_html}</div>', unsafe_allow_html=True)

    with col_signals:
        # Key watchouts (top 3 only)
        watchouts = (syn.get("watchouts") or [])[:3]
        if watchouts:
            st.markdown(f'<div style="font-size:13px;font-weight:600;color:{C["text"]};margin-bottom:10px">Key watchouts</div>', unsafe_allow_html=True)
            for wo in watchouts:
                st.markdown(f'<div style="border-left:3px solid {C["amber"]};background:{C["amber"]}08;padding:8px 12px;margin-bottom:7px;font-size:13px;color:{C["text"]};border-radius:0 8px 8px 0;line-height:1.5">{wo}</div>', unsafe_allow_html=True)

        # Confounders as compact pills
        if ccf:
            st.markdown(f'<div style="font-size:13px;font-weight:600;color:{C["text"]};margin:14px 0 8px">Cross-channel confounders</div>', unsafe_allow_html=True)
            for cf in ccf:
                cr = (cf.get("confounder_risk") or "medium").lower()
                crc = _risk_map.get(cr, C["amber"])
                must_badge = f'<span style="font-size:11px;font-weight:600;color:{C["red"]}">✓ Must include</span>' if cf.get("must_include_as_covariate") else ""
                st.markdown(f'''<div style="display:flex;justify-content:space-between;align-items:flex-start;padding:9px 12px;border:1px solid {C["border"]};border-radius:8px;margin-bottom:6px">
  <div style="flex:1;min-width:0">
    <span style="font-size:13px;font-weight:600;color:{C["text"]}">{cf.get("channel","")}</span>
    <div style="font-size:12px;color:{C["grey"]};margin-top:2px;line-height:1.4">{cf.get("description") or cf.get("spend_change","")}</div>
  </div>
  <div style="text-align:right;flex-shrink:0;margin-left:10px">{must_badge}<div style="font-size:11px;font-weight:700;color:{crc};margin-top:2px">{cr.upper()}</div></div>
</div>''', unsafe_allow_html=True)

        # Risk callout (if high)
        if _ar in ("high", "very_high") and syn.get("analytical_risk_reason"):
            st.markdown(f'<div style="margin-top:14px;background:{C["red"]}08;border:1px solid {C["red"]}30;border-radius:8px;padding:10px 12px;font-size:13px;color:{C["text"]};line-height:1.5"><span style="font-weight:700;color:{C["red"]}">⚠ High risk —</span> {syn.get("analytical_risk_reason","")}</div>', unsafe_allow_html=True)

    # ── Full analysis (collapsible) ─────────────────────────────────────────
    with st.expander("View full analysis details"):
        ex_l, ex_r = st.columns([3, 2], gap="large")
        with ex_l:
            # What the data shows
            st.markdown(f'<div style="font-size:12px;font-weight:600;color:{C["text"]};margin-bottom:6px">What the data shows</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card" style="font-size:12px">{syn.get("what_data_shows","—")}</div>', unsafe_allow_html=True)
            # Raw DoD
            st.markdown(f'<div style="font-size:12px;font-weight:600;color:{C["text"]};margin:12px 0 6px">Raw difference-in-differences</div>', unsafe_allow_html=True)
            did_lines = syn.get("raw_did", "—").split("\n")
            did_html = "".join(f'<div style="font-size:11px;color:{C["text"]};padding:4px 0;border-bottom:1px solid {C["border"]}">{ln}</div>' for ln in did_lines if ln.strip())
            st.markdown(f'<div class="card">{did_html}</div>', unsafe_allow_html=True)
            # Intervention assessment
            st.markdown(f'<div style="font-size:12px;font-weight:600;color:{C["text"]};margin:12px 0 6px">Intervention window assessment</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card" style="font-size:12px">{syn.get("intervention_assessment","—")}</div>', unsafe_allow_html=True)
            # Promo events
            if pes:
                st.markdown(f'<div style="font-size:12px;font-weight:600;color:{C["text"]};margin:12px 0 6px">Promotional events</div>', unsafe_allow_html=True)
                for pe in pes:
                    pi = (pe.get("impact") or "medium").lower()
                    pic = _risk_map.get(pi, C["amber"])
                    sits = pe.get("sits_in","")
                    flag_t = "Flag recommended" if pe.get("flag_recommended") else ""
                    st.markdown(f'''<div style="border:1px solid {C["border"]};border-radius:8px;padding:8px 12px;margin-bottom:5px;font-size:11px">
<div style="display:flex;justify-content:space-between"><strong style="color:{C["text"]}">{pe.get("name","")}</strong>
<span>{pill(f"In {sits}-period","pill-a" if sits=="post" else "pill-g")} {pill(flag_t,"pill-r") if flag_t else ""}</span></div>
<div style="color:{C["grey"]};margin-top:3px">{pe.get("start","")} → {pe.get("end","")} · Impact: <span style="color:{pic};font-weight:600">{pi.upper()}</span></div>
<div style="color:{C["grey"]};margin-top:3px">{pe.get("rationale","")}</div>
</div>''', unsafe_allow_html=True)
            # Spikes
            if syn.get("spikes_and_anomalies"):
                st.markdown(f'<div style="font-size:12px;font-weight:600;color:{C["text"]};margin:12px 0 6px">Spikes & anomalies</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card card-a" style="font-size:12px">{syn.get("spikes_and_anomalies","")}</div>', unsafe_allow_html=True)
        with ex_r:
            # Covariate recommendations
            covs = syn.get("covariate_recommendations", [])
            if covs:
                st.markdown(f'<div style="font-size:12px;font-weight:600;color:{C["text"]};margin-bottom:6px">Covariate recommendations</div>', unsafe_allow_html=True)
                for cov in covs:
                    role = cov.get("role",""); risk = cov.get("risk","")
                    pill_cls = "pill-c" if role == "Covariate" else "pill-r"
                    risk_cls = "pill-g" if risk in ("None","Low") else "pill-a" if risk == "Medium" else "pill-r"
                    st.markdown(f'''<div style="border:1px solid {C["border"]};border-radius:8px;padding:9px 12px;margin-bottom:5px">
<div style="display:flex;justify-content:space-between;margin-bottom:3px">
  <span style="font-size:12px;font-weight:600;color:{C["text"]}">{cov.get("column","")}</span>
  <span>{pill(role,pill_cls)} {pill(f"Risk: {risk}",risk_cls)}</span>
</div>
<div style="font-size:11px;color:{C["grey"]}">{cov.get("rationale","")}</div>
</div>''', unsafe_allow_html=True)
            # Brief gaps
            bgaps = syn.get("brief_gaps",[])
            if bgaps:
                st.markdown(f'<div style="font-size:12px;font-weight:600;color:{C["text"]};margin:12px 0 6px">Brief gaps</div>', unsafe_allow_html=True)
                for bg in bgaps:
                    st.markdown(f'<div style="border-left:3px solid {C["grey_l"]};padding:6px 10px;margin-bottom:4px;font-size:11px;color:{C["grey"]};border-radius:0 6px 6px 0">{bg}</div>', unsafe_allow_html=True)

    # ── Nav ─────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _an1, _an2, _an3 = st.columns([1, 2, 3])
    with _an1:
        if st.button("Back", key="an_back"):
            st.session_state.step = 2; st.rerun()
    with _an2:
        if st.button("Re-analyse", use_container_width=True, key="an_rerun"):
            st.session_state.ai_synthesis = None
            st.session_state.step = 2; st.rerun()
    with _an3:
        if st.button("Continue to EDA →", use_container_width=True, type="primary", key="an_fwd"):
            st.session_state.step = 4
            st.session_state.max_step = max(st.session_state.max_step, 4)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — CONFIGURE (master date selector + target variable)
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    st.markdown(f'<span style="font-size:20px;font-weight:600">Configure Analysis</span>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{C["grey"]};margin-bottom:20px;font-size:13px">Set your pre/post periods and pick the metric you want to model. These settings control both the AI analysis and all EDA charts.</div>', unsafe_allow_html=True)

    df_cfg = parse_dates(st.session_state.daily_df.copy())
    _d_min = df_cfg["p_date"].min().date()
    _d_max = df_cfg["p_date"].max().date()

    # ── Seed defaults: data-driven fallback (80/20 split) ────────────────────────
    def _cfg_defaults():
        syn_ = st.session_state.ai_synthesis or {}
        rp_  = syn_.get("recommended_periods", {})
        if rp_ and rp_.get("pre_start"):
            try:
                return (pd.Timestamp(rp_["pre_start"]).date(),
                        pd.Timestamp(rp_["pre_end"]).date(),
                        pd.Timestamp(rp_["post_start"]).date(),
                        pd.Timestamp(rp_["post_end"]).date())
            except Exception:
                pass
        # Data-driven: 80/20 split
        split = df_cfg["p_date"].iloc[int(len(df_cfg) * 0.8)].date()
        return _d_min, split, split, _d_max

    _def_ps, _def_pe, _def_pss, _def_pse = _cfg_defaults()

    def _clamp(d, lo, hi): return max(lo, min(hi, d)) if d else lo
    _pre_s_cur  = _clamp(st.session_state.pre_start  or _def_ps,  _d_min, _d_max)
    _pre_e_cur  = _clamp(st.session_state.pre_end    or _def_pe,  _d_min, _d_max)
    _post_s_cur = _clamp(st.session_state.post_start or _def_pss, _d_min, _d_max)
    _post_e_cur = _clamp(st.session_state.post_end   or _def_pse, _d_min, _d_max)

    # ── Show synthesis hypothesis suggestion if available ──────────────────────
    _syn_cfg = st.session_state.ai_synthesis or {}
    _hyp_sug_cfg = _syn_cfg.get("hypothesis_suggestion", "")
    if _hyp_sug_cfg:
        st.markdown(f'<div style="background:{C["purple"]}0D;border:1px solid {C["purple"]}30;border-radius:10px;padding:10px 16px;margin-bottom:16px;font-size:13px;color:{C["text"]};line-height:1.5"><span style="font-size:11px;font-weight:600;color:{C["purple"]};letter-spacing:.07em;display:block;margin-bottom:4px">AI SUGGESTION</span>{_hyp_sug_cfg}</div>', unsafe_allow_html=True)

    # ── Master date pickers ────────────────────────────────────────────────────
    st.markdown(f'<div style="font-size:11px;font-weight:600;color:{C["grey"]};letter-spacing:.07em;margin-bottom:8px">PRE / POST PERIODS</div>', unsafe_allow_html=True)
    _pc1, _pc2, _pc3, _pc4 = st.columns(4)
    with _pc1:
        st.markdown(f'<div style="font-size:10px;color:{C["grey"]};font-weight:600;letter-spacing:.06em;margin-bottom:3px">PRE START</div>', unsafe_allow_html=True)
        _cfg_pre_s = st.date_input("cfg_ps", value=_pre_s_cur, min_value=_d_min, max_value=_d_max, label_visibility="collapsed", key="cfg_pre_s")
    with _pc2:
        st.markdown(f'<div style="font-size:10px;color:{C["grey"]};font-weight:600;letter-spacing:.06em;margin-bottom:3px">PRE END</div>', unsafe_allow_html=True)
        _cfg_pre_e = st.date_input("cfg_pe", value=_pre_e_cur, min_value=_d_min, max_value=_d_max, label_visibility="collapsed", key="cfg_pre_e")
    with _pc3:
        st.markdown(f'<div style="font-size:10px;color:{C["grey"]};font-weight:600;letter-spacing:.06em;margin-bottom:3px">POST START</div>', unsafe_allow_html=True)
        _cfg_post_s = st.date_input("cfg_pss", value=_post_s_cur, min_value=_d_min, max_value=_d_max, label_visibility="collapsed", key="cfg_post_s")
    with _pc4:
        st.markdown(f'<div style="font-size:10px;color:{C["grey"]};font-weight:600;letter-spacing:.06em;margin-bottom:3px">POST END</div>', unsafe_allow_html=True)
        _cfg_post_e = st.date_input("cfg_pse", value=_post_e_cur, min_value=_d_min, max_value=_d_max, label_visibility="collapsed", key="cfg_post_e")

    # Pre/post ratio badge
    try:
        _cfg_pre_days  = (_cfg_pre_e  - _cfg_pre_s).days  + 1
        _cfg_post_days = (_cfg_post_e - _cfg_post_s).days + 1
        _cfg_ratio     = _cfg_pre_days / _cfg_post_days if _cfg_post_days > 0 else 0
        _cfg_ratio_c   = C["green"] if _cfg_ratio >= 2 else C["amber"] if _cfg_ratio >= 1 else C["red"]
        _cfg_ratio_note = "✓ Good" if _cfg_ratio >= 2 else "⚠ Borderline" if _cfg_ratio >= 1 else "✗ Too short"
    except Exception:
        _cfg_pre_days = _cfg_post_days = 0; _cfg_ratio = 0; _cfg_ratio_c = C["red"]; _cfg_ratio_note = "—"

    st.markdown(f'''<div style="display:flex;gap:10px;margin:10px 0 20px;flex-wrap:wrap">
  <div style="background:{C["purple"]}15;border:1px solid {C["purple"]}30;border-radius:8px;padding:5px 12px;font-size:12px;color:{C["purple"]}"><strong>Pre:</strong> {_cfg_pre_days}d</div>
  <div style="background:{C["surface2"]};border:1px solid {C["border"]};border-radius:8px;padding:5px 12px;font-size:12px;color:{C["text"]}"><strong>Post:</strong> {_cfg_post_days}d</div>
  <div style="background:{_cfg_ratio_c}15;border:1px solid {_cfg_ratio_c}40;border-radius:8px;padding:5px 12px;font-size:12px;font-weight:700;color:{_cfg_ratio_c}">Ratio {_cfg_ratio:.1f}× — {_cfg_ratio_note}</div>
</div>''', unsafe_allow_html=True)

    # ── Analysis target ────────────────────────────────────────────────────────
    st.markdown(f'<div style="font-size:11px;font-weight:600;color:{C["grey"]};letter-spacing:.07em;margin-bottom:8px">ANALYSIS TARGET</div>', unsafe_allow_html=True)
    _num_cols_cfg = [c for c in df_cfg.columns if c != "p_date" and pd.api.types.is_numeric_dtype(df_cfg[c])]
    _pref_kws_cfg = ["conversion", "conv", "purchase", "order", "revenue", "sales", "session"]
    _sorted_cols_cfg = sorted(_num_cols_cfg, key=lambda c: next((i for i,kw in enumerate(_pref_kws_cfg) if kw in c.lower()), len(_pref_kws_cfg)))
    _default_target_cfg = st.session_state.target_col if st.session_state.target_col in _sorted_cols_cfg else (_sorted_cols_cfg[0] if _sorted_cols_cfg else None)
    if _sorted_cols_cfg:
        _cfg_target = st.selectbox(
            "Primary variable to model (Y)",
            options=_sorted_cols_cfg,
            index=_sorted_cols_cfg.index(_default_target_cfg) if _default_target_cfg in _sorted_cols_cfg else 0,
            format_func=lambda c: c.replace("_", " "),
            label_visibility="visible",
            key="cfg_target_sel",
        )
        st.markdown(f'<div style="font-size:11px;color:{C["grey"]};margin-top:4px">Any numeric column in your dataset. The model will predict a counterfactual for this variable during the post-period.</div>', unsafe_allow_html=True)
    else:
        _cfg_target = None
        st.warning("No numeric columns found in dataset.")

    # ── For TTAM: optional conversion event reconfiguration ───────────────────
    if st.session_state.file_type == "raw_ttam" and st.session_state.raw_df is not None:
        with st.expander("Advanced: TTAM conversion event & L4 mapping"):
            _raw_tt2 = st.session_state.raw_df
            _opt_evts = sorted(_raw_tt2["Optimization Event(External Actions)"].dropna().unique().tolist()) if "Optimization Event(External Actions)" in _raw_tt2.columns else []
            if _opt_evts:
                st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:8px">Select which conversion event defines the dependent variable. Changing this re-aggregates the dataset.</div>', unsafe_allow_html=True)
                _cur_evt = st.session_state.get("target_event") or _opt_evts[0]
                _evt_sel = st.selectbox("Target conversion event", _opt_evts,
                    index=_opt_evts.index(_cur_evt) if _cur_evt in _opt_evts else 0, key="cfg_evt")
                if st.button("Re-aggregate with this event", key="cfg_reagg"):
                    try:
                        _reagg_map = {**DEFAULT_L4, **st.session_state.get("role_overrides", {})}
                        _agg2 = aggregate_ttam(_raw_tt2, _reagg_map, [_evt_sel])
                        st.session_state.daily_df = parse_dates(_agg2)
                        st.session_state.target_event = _evt_sel
                        df_cfg = parse_dates(st.session_state.daily_df.copy())
                        st.success("Re-aggregated.")
                        st.rerun()
                    except Exception as _re2:
                        st.error(f"Re-aggregation failed: {_re2}")

    # ── Nav ────────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _cfg_b1, _cfg_b2 = st.columns([1, 5])
    with _cfg_b1:
        if st.button("Back", key="cfg_back"):
            st.session_state.step = 0; st.rerun()
    with _cfg_b2:
        if st.button("Confirm & Continue to Explore →", use_container_width=True, type="primary", key="cfg_fwd"):
            st.session_state.pre_start  = _cfg_pre_s
            st.session_state.pre_end    = _cfg_pre_e
            st.session_state.post_start = _cfg_post_s
            st.session_state.post_end   = _cfg_post_e
            st.session_state.target_col = _cfg_target
            # For SOT: set sot_treatment_start/end to post period for EDA compatibility
            st.session_state.sot_treatment_start = _cfg_post_s
            st.session_state.sot_treatment_end   = _cfg_post_e
            st.session_state.step = 1
            st.session_state.max_step = max(st.session_state.max_step, 1)
            st.rerun()
# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — QUALITY CHECK
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    st.markdown(f'<span style="font-size:20px;font-weight:600">Data Quality</span>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{C["grey"]};margin-bottom:16px;font-size:13px">Review the data before analysis.</div>', unsafe_allow_html=True)

    df = parse_dates(st.session_state.daily_df.copy())

    # ── TTAM: optional aggregation reconfiguration ─────────────────────────────
    if st.session_state.file_type == "raw_ttam" and st.session_state.raw_df is not None:
        with st.expander("Advanced: Reconfigure L4 aggregation (optional — defaults are pre-applied)"):
            st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:10px">The data was automatically aggregated using standard L4 mappings. Expand to review or customise before proceeding.</div>', unsafe_allow_html=True)
            _raw_tt = st.session_state.raw_df
            _l4_opts = sorted(_raw_tt["L4 Product Tag"].dropna().unique().tolist()) if "L4 Product Tag" in _raw_tt.columns else []
            _cur_map = {**DEFAULT_L4}
            _cat_keys = ["TopView","Brand_ExTV","MidFunnel","Performance","Other"]
            _tag_by_cat = {}
            for _tag in _l4_opts:
                _cat = _cur_map.get(_tag, "Other")
                _tag_by_cat.setdefault(_cat, []).append(_tag)
            _changed = False
            for _cat in _cat_keys:
                _tags_in = _tag_by_cat.get(_cat, [])
                st.markdown(f'<div style="font-size:12px;font-weight:600;color:{C["text"]};margin:8px 0 4px">{_cat}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:11px;color:{C["grey"]};margin-bottom:4px">{", ".join(_tags_in) if _tags_in else "—"}</div>', unsafe_allow_html=True)
            if st.button("Re-aggregate with default mapping", key="re_agg_btn"):
                try:
                    _default_events = st.session_state.get("conversion_events") or ["Complete Payment","Purchase","Checkout","Submit Form","App Install"]
                    _agg_new = aggregate_ttam(st.session_state.raw_df, DEFAULT_L4, _default_events)
                    st.session_state.daily_df = _agg_new
                    df = parse_dates(_agg_new.copy())
                    st.success("Re-aggregated successfully.")
                    st.rerun()
                except Exception as _ae:
                    st.error(f"Aggregation error: {_ae}")

    sigs = quality_checks(df)
    st.session_state.quality_signals = sigs

    n = len(df); dates = df["p_date"]
    conv_c = next((c for c in df.columns if "conversion" in c.lower() or c.lower()=="conversions"), None)
    total_c = int(df[conv_c].sum()) if conv_c else 0
    highs = [s for s in sigs if s["sev"]=="high"]

    st.markdown(f'<div class="mrow">{mtile("Date Range",f"{dates.min().strftime(chr(37)+"d "+chr(37)+"b")}–{dates.max().strftime(chr(37)+"d "+chr(37)+"b %Y")}")}{mtile("Days",str(n),"mvc" if n>=60 else "mva" if n>=40 else "mvr")}{mtile("Total Conversions",f"{total_c:,}")}{mtile("High Issues",str(len(highs)),"mvr" if highs else "mvc")}</div>', unsafe_allow_html=True)

    if sigs:
        for s in sigs:
            cls = {"high":"card-r","medium":"card-a","low":"card-g"}.get(s["sev"],"card-g")
            pc_ = {"high":"pill-r","medium":"pill-a","low":"pill-g"}.get(s["sev"],"pill-g")
            st.markdown(f'<div class="card {cls}">{pill(s["sev"].upper(),pc_)} <strong style="margin-left:8px">{s["msg"]}</strong><div style="font-size:12px;color:{C["grey"]};margin-top:5px">{s.get("detail","")}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="card card-c"><strong style="color:{C["purple"]}">✓ No issues detected</strong><div style="font-size:12px;color:{C["grey"]};margin-top:4px">Data looks clean.</div></div>', unsafe_allow_html=True)

    if conv_c:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["p_date"], y=df[conv_c], mode="lines",
            line=dict(color=C["purple"],width=2), fill="tozeroy", fillcolor="rgba(123,127,255,0.16)", name=conv_c))
        fig.update_layout(**CHART, title=dict(text="Conversion series overview",font=dict(size=14)))
        st.plotly_chart(fig, use_container_width=True)

    # ── Detailed Checks (spreadsheet methodology) ──────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:15px;font-weight:600;margin-bottom:12px">Detailed Checks</div>', unsafe_allow_html=True)

    tc1, tc2, tc3, tc4 = st.tabs(["📅 Calendar & Gaps", "📈 Weekday Trend", "🔍 Outliers", "🔗 Covariate Validation"])

    _post_s = pd.to_datetime(st.session_state.post_start) if st.session_state.post_start else None
    _post_e = pd.to_datetime(st.session_state.post_end)   if st.session_state.post_end   else None

    with tc1:
        st.markdown("Missing date check — are there gaps in the daily series that need imputing?")
        full_range = pd.date_range(df["p_date"].min(), df["p_date"].max(), freq="D")
        missing_dates = sorted([d for d in full_range if d not in df["p_date"].values])
        if missing_dates:
            st.markdown(f'<div class="card card-r">{pill("GAPS DETECTED","pill-r")} <strong style="margin-left:8px">{len(missing_dates)} missing date(s)</strong><div style="font-size:12px;color:{C["grey"]};margin-top:4px">{", ".join(d.strftime("%d %b") for d in missing_dates[:12])}{"…" if len(missing_dates)>12 else ""}</div></div>', unsafe_allow_html=True)
            imp = st.radio("Imputation method", ["Neighbour average (interpolate)", "Weekday cyclical average"],
                horizontal=True, key="imp_radio",
                help="Neighbour average: fills gaps using linear interpolation between surrounding values — good for random missing days. Weekday cyclical average: fills with the average for that day of week — better when demand has strong weekly patterns.")
            if st.button("Apply imputation", key="imp_btn"):
                _df_imp = parse_dates(st.session_state.daily_df.copy()).set_index("p_date").reindex(full_range)
                _df_imp.index.name = "p_date"
                _num = _df_imp.select_dtypes(include="number").columns
                if imp.startswith("Neighbour"):
                    _df_imp[_num] = _df_imp[_num].interpolate(method="linear", limit_direction="both")
                else:
                    _df_imp["_dow"] = pd.to_datetime(_df_imp.index).dayofweek
                    for _c in _num:
                        _df_imp[_c] = _df_imp[_c].fillna(_df_imp.groupby("_dow")[_c].transform("mean"))
                    _df_imp = _df_imp.drop(columns=["_dow"])
                st.session_state.daily_df = _df_imp.reset_index()
                st.success(f"Imputed {len(missing_dates)} gap(s). Dataset updated.")
                st.rerun()
        else:
            st.markdown(f'<div class="card card-c">✓ No date gaps — series is complete ({len(df)} days)</div>', unsafe_allow_html=True)

        if _post_s:
            _pre_n  = len(df[df["p_date"] < _post_s])
            _post_n = len(df[(df["p_date"] >= _post_s) & (df["p_date"] <= _post_e)]) if _post_e else len(df[df["p_date"] >= _post_s])
            if _post_n > 0:
                _ratio = _pre_n / _post_n
                _cls = "card-c" if _ratio >= 2 else "card-a" if _ratio >= 1.5 else "card-r"
                st.markdown(f'<div class="card {_cls}" style="margin-top:10px"><strong>Pre/post ratio: {_ratio:.1f}x</strong> ({_pre_n} pre · {_post_n} post days)<div style="font-size:12px;color:{C["grey"]};margin-top:4px">Minimum recommended: 2× post-period length. Below 2x the model has less data to establish a reliable baseline, widening confidence intervals.</div></div>', unsafe_allow_html=True)

    with tc2:
        st.markdown("Weekday cyclicality — does demand vary significantly by day of week in the pre-period?")
        st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:10px">A strong weekday pattern (>20% deviation from the weekly average) suggests you should add a weekend flag or day-of-week dummy as a covariate so the model does not mistake the pattern for intervention effect.</div>', unsafe_allow_html=True)
        if conv_c and conv_c in df.columns:
            _pre_tc = df[df["p_date"] < _post_s].copy() if _post_s else df.copy()
            _pre_tc["_dow"] = _pre_tc["p_date"].dt.day_name()
            _dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            _dow_avg = _pre_tc.groupby("_dow")[conv_c].mean().reindex(_dow_order)
            _overall = _dow_avg.mean()
            _pct_dev = ((_dow_avg - _overall) / _overall * 100).round(1)
            _bar_cols = [C["purple"] if abs(v) > 20 else C["grey_l"] for v in _pct_dev]
            _fig_dow = go.Figure(go.Bar(x=_dow_order, y=_dow_avg.values, marker_color=_bar_cols,
                customdata=_pct_dev.values, hovertemplate="%{x}: %{y:,.0f} avg (%{customdata:+.1f}% vs weekly avg)<extra></extra>"))
            _fig_dow.add_hline(y=_overall, line_dash="dot", line_color=C["amber"], annotation_text="Weekly avg")
            _fig_dow.update_layout(**CHART, height=260, title=dict(text=f"Avg {conv_c} by weekday (pre-period)", font=dict(size=13)))
            st.plotly_chart(_fig_dow, use_container_width=True)
            _strong = [d for d, v in _pct_dev.items() if abs(v) > 20]
            if _strong:
                st.markdown(f'<div class="card card-a">{pill("CYCLICALITY DETECTED","pill-a")} <strong style="margin-left:8px">{", ".join(_strong)}</strong> deviate >20% from weekly avg — consider adding a weekend or day-of-week flag as a covariate.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="card card-c">✓ No strong weekday cyclicality — no day-of-week flag needed</div>', unsafe_allow_html=True)
        else:
            st.info("No conversion column detected.")

    with tc3:
        st.markdown("Outlier detection — three methods to flag abnormal days in the pre-period")
        st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:10px">Not all flagged days need treatment. If the spike/dip has a clear business explanation (promotion, product launch, etc.) keep it as-is and add a dummy flag. Only winsorise or remove if it is unexplained noise.</div>', unsafe_allow_html=True)
        if conv_c and conv_c in df.columns:
            _pre_out = df[df["p_date"] < _post_s].copy() if _post_s else df.copy()
            _c1, _c2 = st.columns(2)
            with _c1:
                _dod_t = st.slider("Day-over-day threshold (%)", 50, 300, 150, 25, key="dod_t",
                    help="Flags any day where the metric changes by more than this % vs the previous day. 150% is a reasonable starting point — lower it if your data is stable, raise it if you have legitimately volatile days.")
            with _c2:
                _z_t = st.slider("Z-score threshold", 1.5, 4.0, 2.5, 0.5, key="z_t",
                    help="Flags days whose value is more than this many standard deviations from the pre-period mean. 2.5 is standard — lower = more sensitive, higher = only extreme outliers.")

            _dod_chg = _pre_out[conv_c].pct_change().abs() * 100
            _dod_f = _pre_out[_dod_chg > _dod_t]["p_date"].dt.strftime("%d %b").tolist()
            _m = _pre_out[conv_c].mean(); _s = _pre_out[conv_c].std()
            _z_f = _pre_out[abs((_pre_out[conv_c] - _m) / _s) > _z_t]["p_date"].dt.strftime("%d %b").tolist() if _s > 0 else []
            _q1, _q3 = _pre_out[conv_c].quantile(0.25), _pre_out[conv_c].quantile(0.75)
            _iqr_f = _pre_out[(_pre_out[conv_c] < _q1-1.5*(_q3-_q1)) | (_pre_out[conv_c] > _q3+1.5*(_q3-_q1))]["p_date"].dt.strftime("%d %b").tolist()

            _cm1, _cm2, _cm3 = st.columns(3)
            with _cm1: st.markdown(f'<div class="card {"card-r" if _dod_f else "card-c"}"><strong>Day-over-day</strong> <span style="font-size:11px;color:{C["grey"]}">({_dod_t}% threshold)</span><br><span style="font-size:22px;font-weight:700">{len(_dod_f)}</span> flags<br><div style="font-size:11px;color:{C["grey"]}">{", ".join(_dod_f[:4]) if _dod_f else "None"}</div></div>', unsafe_allow_html=True)
            with _cm2: st.markdown(f'<div class="card {"card-r" if _z_f else "card-c"}"><strong>Z-score</strong> <span style="font-size:11px;color:{C["grey"]}">({_z_t}σ threshold)</span><br><span style="font-size:22px;font-weight:700">{len(_z_f)}</span> flags<br><div style="font-size:11px;color:{C["grey"]}">{", ".join(_z_f[:4]) if _z_f else "None"}</div></div>', unsafe_allow_html=True)
            with _cm3: st.markdown(f'<div class="card {"card-r" if _iqr_f else "card-c"}"><strong>IQR</strong> <span style="font-size:11px;color:{C["grey"]}">1.5× fence</span><br><span style="font-size:22px;font-weight:700">{len(_iqr_f)}</span> flags<br><div style="font-size:11px;color:{C["grey"]}">{", ".join(_iqr_f[:4]) if _iqr_f else "None"}</div></div>', unsafe_allow_html=True)

            _all_f = sorted(set(_dod_f + _z_f + _iqr_f))
            if _all_f:
                st.markdown(f'<div class="card card-a" style="margin-top:10px"><strong>Flagged pre-period dates:</strong> {", ".join(_all_f)}<div style="font-size:12px;color:{C["grey"]};margin-top:6px">Review each flagged date. If business-driven: keep and add a dummy flag in the Flag Variables step. If unexplained noise: winsorise using the option below.</div></div>', unsafe_allow_html=True)
                with st.expander("Winsorisation — cap extreme values at 95th percentile"):
                    st.markdown(f'<div style="font-size:12px;color:{C["grey"]}">Winsorisation replaces extreme outliers with the 95th percentile value, preserving the shape of the series without removing data points. Only use this for values confirmed as unexplained noise — not business events.</div>', unsafe_allow_html=True)
                    _p95 = _pre_out[conv_c].quantile(0.95); _p05 = _pre_out[conv_c].quantile(0.05)
                    st.markdown(f"Pre-period 5th pct: {_p05:,.0f} · 95th pct: {_p95:,.0f}")
                    if st.button(f"Winsorise {conv_c} at 95th percentile", key="winsor_btn"):
                        _df_w = parse_dates(st.session_state.daily_df.copy())
                        _df_w[conv_c] = _df_w[conv_c].clip(lower=_p05, upper=_p95)
                        st.session_state.daily_df = _df_w
                        st.success(f"Winsorised {conv_c}. Values capped between {_p05:,.0f} and {_p95:,.0f}.")
                        st.rerun()
        else:
            st.info("No conversion column detected.")

    with tc4:
        st.markdown("Covariate validation — checks whether your covariates are safe to use as controls")
        st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:10px">Three checks: (1) Pre/post average change — if a covariate changed >20% after the intervention it may itself have been affected, making it a bad control. (2) Extrapolation risk — if post-period values fall outside the pre-period range, the model must extrapolate beyond what it learned. (3) Scale check — if covariates have very different scales, log-standardisation can improve model fit.</div>', unsafe_allow_html=True)
        if _post_s and st.session_state.daily_df is not None:
            _df_cv = parse_dates(st.session_state.daily_df.copy())
            _cov_list = [c for c in _df_cv.columns if c not in ["p_date","prepost_1","prepost"]
                         and _df_cv[c].dtype in ["float64","int64"]
                         and not any(k in c.lower() for k in ["cost","topview","flag","spend"])]
            if _cov_list:
                _pre_cv  = _df_cv[_df_cv["p_date"] < _post_s]
                _post_cv = _df_cv[_df_cv["p_date"] >= _post_s] if not _post_e else _df_cv[(_df_cv["p_date"] >= _post_s) & (_df_cv["p_date"] <= _post_e)]
                _cv_rows = []
                for _c in _cov_list[:10]:
                    _pa = _pre_cv[_c].mean(); _pb = _post_cv[_c].mean()
                    if _pa and _pa != 0:
                        _chg = (_pb - _pa) / abs(_pa) * 100
                        _p10 = _pre_cv[_c].quantile(0.1); _p90 = _pre_cv[_c].quantile(0.9)
                        _extrap_low  = int((_post_cv[_c] < _p10).sum())
                        _extrap_high = int((_post_cv[_c] > _p90).sum())
                        _extrap_pct  = (_extrap_low + _extrap_high) / len(_post_cv) * 100 if len(_post_cv) > 0 else 0
                        _std = _pre_cv[_c].std()
                        _z_post_max = abs((_post_cv[_c] - _pa) / _std).max() if _std > 0 else 0
                        _cv_rows.append({
                            "Covariate": _c,
                            "Pre avg": f"{_pa:,.0f}",
                            "Post avg": f"{_pb:,.0f}",
                            "Δ pre→post": f"{_chg:+.1f}%",
                            "Endogeneity": "⚠️ >20%" if abs(_chg) > 20 else "✓",
                            "Extrapolation": f"⚠️ {_extrap_pct:.0f}% out of range" if _extrap_pct > 20 else "✓",
                            "Max Z-post": f"{'⚠️ ' if _z_post_max > 3 else ''}{_z_post_max:.1f}σ"
                        })
                if _cv_rows:
                    st.dataframe(pd.DataFrame(_cv_rows), use_container_width=True, hide_index=True)
                    _risky = [r["Covariate"] for r in _cv_rows if "⚠️" in r["Endogeneity"]]
                    _extrap_risky = [r["Covariate"] for r in _cv_rows if "⚠️" in r["Extrapolation"]]
                    if _risky:
                        st.markdown(f'<div class="card card-a">{pill("ENDOGENEITY RISK","pill-a")} <strong style="margin-left:8px">{", ".join(_risky)}</strong> changed >20% pre→post. These may have been affected by the intervention itself — reconsider including them as covariates.</div>', unsafe_allow_html=True)
                    if _extrap_risky:
                        st.markdown(f'<div class="card card-a">{pill("EXTRAPOLATION RISK","pill-a")} <strong style="margin-left:8px">{", ".join(_extrap_risky)}</strong> have post-period values outside the pre-period P10–P90 range. The model must extrapolate — revalidate these covariates or adjust your hypothesis.</div>', unsafe_allow_html=True)

                with st.expander("Scale adjustment — log transform + standardise"):
                    st.markdown(f'<div style="font-size:12px;color:{C["grey"]}">When covariates have very different scales or high volatility, log transformation followed by standardisation (z-score) can improve model stability. Only apply to variables with right-skewed distributions and large absolute values.</div>', unsafe_allow_html=True)
                    _scale_cols = st.multiselect("Select columns to log-standardise", options=_cov_list, key="scale_cols")
                    if _scale_cols and st.button("Apply log-standardisation", key="scale_btn"):
                        _df_sc = parse_dates(st.session_state.daily_df.copy())
                        for _sc in _scale_cols:
                            if (_df_sc[_sc] > 0).all():
                                _log_vals = np.log(_df_sc[_sc])
                                _df_sc[f"{_sc}_logstd"] = (_log_vals - _log_vals.mean()) / _log_vals.std()
                            else:
                                st.warning(f"{_sc} has zero/negative values — log transform skipped.")
                        st.session_state.daily_df = _df_sc
                        st.success(f"Added log-standardised columns: {', '.join(f'{c}_logstd' for c in _scale_cols)}")
                        st.rerun()
            else:
                st.info("No numeric covariate columns detected.")
        else:
            st.info("Set pre/post dates in Analysis Setup first to enable covariate validation.")

    st.markdown("<br>", unsafe_allow_html=True)
    cb1, cb2 = st.columns([1,5])
    with cb1:
        if st.button("Back", key="q_back"): st.session_state.step=0; st.rerun()
    with cb2:
        if st.button("Continue to Configure →", use_container_width=True, key="q_fwd"): st.session_state.step=2; st.session_state.max_step=max(st.session_state.max_step,2); st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    st.markdown(f'<span style="font-size:20px;font-weight:600">Explore</span>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{C["grey"]};margin-bottom:16px;font-size:13px">Set your analysis window, choose a target variable, and explore the data before modelling.</div>', unsafe_allow_html=True)

    df = parse_dates(st.session_state.daily_df.copy())

    # ── CONTROL BAR ────────────────────────────────────────────────────────────
    _date_min = df["p_date"].min().date()
    _date_max = df["p_date"].max().date()
    _total_days = (_date_max - _date_min).days + 1
    _split = int(_total_days * 0.6)
    if st.session_state.pre_start  is None: st.session_state.pre_start  = _date_min
    if st.session_state.pre_end    is None: st.session_state.pre_end    = _date_min + pd.Timedelta(days=_split - 1)
    if st.session_state.post_start is None: st.session_state.post_start = _date_min + pd.Timedelta(days=_split)
    if st.session_state.post_end   is None: st.session_state.post_end   = _date_max

    _num_cols_e = [c for c in df.columns if c != "p_date" and pd.api.types.is_numeric_dtype(df[c])]
    _pref_kws_e = ["conversion","conv","purchase","order","revenue","sales","session"]
    _sorted_cols_e = sorted(_num_cols_e, key=lambda c: next((i for i,kw in enumerate(_pref_kws_e) if kw in c.lower()), len(_pref_kws_e)))

    _eb1, _eb2, _eb3, _eb4, _eb5 = st.columns([2, 2, 2, 2, 3])
    with _eb1: _pre_s  = st.date_input("Pre start",  value=st.session_state.pre_start,  min_value=_date_min, max_value=_date_max, key="exp_pre_s")
    with _eb2: _pre_e  = st.date_input("Pre end",    value=st.session_state.pre_end,    min_value=_date_min, max_value=_date_max, key="exp_pre_e")
    with _eb3: _post_s = st.date_input("Post start", value=st.session_state.post_start, min_value=_date_min, max_value=_date_max, key="exp_post_s")
    with _eb4: _post_e = st.date_input("Post end",   value=st.session_state.post_end,   min_value=_date_min, max_value=_date_max, key="exp_post_e")
    with _eb5:
        _tgt_def_e = st.session_state.target_col if st.session_state.target_col in _sorted_cols_e else (_sorted_cols_e[0] if _sorted_cols_e else None)
        _tgt_e = st.selectbox("Target variable", _sorted_cols_e,
            index=_sorted_cols_e.index(_tgt_def_e) if _tgt_def_e in _sorted_cols_e else 0,
            format_func=lambda c: c.replace("_"," "), key="exp_target")

    st.session_state.pre_start  = _pre_s
    st.session_state.pre_end    = _pre_e
    st.session_state.post_start = _post_s
    st.session_state.post_end   = _post_e
    st.session_state.target_col = _tgt_e

    try:
        _pre_days_e  = (_pre_e  - _pre_s).days  + 1
        _post_days_e = (_post_e - _post_s).days + 1
        _ratio_e     = _pre_days_e / _post_days_e if _post_days_e > 0 else 0
        _rc_e = C["green"] if _ratio_e >= 2 else C["amber"] if _ratio_e >= 1 else C["red"]
        _rn_e = "✓ Good" if _ratio_e >= 2 else "⚠ Borderline" if _ratio_e >= 1 else "✗ Too short"
    except Exception:
        _pre_days_e = _post_days_e = 0; _ratio_e = 0; _rc_e = C["red"]; _rn_e = "—"
    st.markdown(
        f'<div style="display:flex;gap:10px;margin:8px 0 20px;flex-wrap:wrap">'
        f'<div style="background:{C["purple"]}15;border:1px solid {C["purple"]}30;border-radius:8px;padding:5px 12px;font-size:12px;color:{C["purple"]}"><strong>Pre:</strong> {_pre_days_e}d</div>'
        f'<div style="background:{C["surface2"]};border:1px solid {C["border"]};border-radius:8px;padding:5px 12px;font-size:12px;color:{C["text"]}"><strong>Post:</strong> {_post_days_e}d</div>'
        f'<div style="background:{_rc_e}15;border:1px solid {_rc_e}40;border-radius:8px;padding:5px 12px;font-size:12px;font-weight:700;color:{_rc_e}">Ratio {_ratio_e:.1f}× — {_rn_e}</div>'
        f'</div>', unsafe_allow_html=True)

    t_start = pd.Timestamp(_post_s) if _post_s else None
    t_end   = pd.Timestamp(_post_e) if _post_e else None
    pre_df  = df[df["p_date"] < t_start]  if t_start is not None else df.copy()
    post_df = df[(df["p_date"] >= t_start) & (df["p_date"] <= t_end)] if t_start is not None else pd.DataFrame()

    if st.session_state.file_type in ("sot", "pre_aggregated"):
        # Shared channel/metric column lists
        conv_cols_sot = [c for c in df.columns if "conversions" in c.lower() and not c.lower().startswith("ttam_")]
        sess_cols_sot = [c for c in df.columns if "sessions"    in c.lower() and not c.lower().startswith("ttam_")]
        rev_cols_sot  = [c for c in df.columns if "revenue"     in c.lower() and not c.lower().startswith("ttam_")]

        CH_ORDER  = ["Direct","Paid_Social_TikTok","Paid_Search_Google","Paid_Social_Meta","Organic_Search"]
        CH_COLORS = {"Direct":C["purple"],"Paid_Social_TikTok":"#00D8FF","Paid_Search_Google":"#FF7070","Paid_Social_Meta":"#5BA8FF","Organic_Search":"#4AE882"}
        def ch_col(ch): return CH_COLORS.get(ch, C["grey"])
        def ch_lbl(ch): return ch.replace("Paid_Social_","").replace("Paid_Search_","").replace("_Search","").replace("_"," ")

        _CH_PREFIXES = [
            ("Direct_","Direct"),("direct_","Direct"),
            ("Paid_Social_TikTok_","Paid_Social_TikTok"),("tiktok_","Paid_Social_TikTok"),
            ("Paid_Search_Google_","Paid_Search_Google"),("google_cpc_","Paid_Search_Google"),
            ("Paid_Social_Meta_","Paid_Social_Meta"),("meta_","Paid_Social_Meta"),
            ("Organic_Search_","Organic_Search"),("organic_","Organic_Search"),
        ]
        channels = {}
        for _col in df.columns:
            if _col == "p_date": continue
            for _pfx, _ch in _CH_PREFIXES:
                if _col.startswith(_pfx):
                    channels.setdefault(_ch, []).append(_col); break

        avail_chs = [ch for ch in CH_ORDER if ch in channels]

        def add_treatment(fig):
            if t_start is not None:
                fig.add_vrect(x0=str(t_start.date()), x1=str(t_end.date()), fillcolor="rgba(123,127,255,0.14)", line_width=0)
                fig.add_shape(type="line", x0=str(t_start.date()), x1=str(t_start.date()), y0=0, y1=1, yref="paper", line=dict(color=C["purple"],width=2,dash="dash"))
                fig.add_annotation(x=str(t_start.date()), y=1, yref="paper", text="Campaign start", showarrow=False, yanchor="bottom", font=dict(color=C["purple"],size=10), xshift=5)
            return fig

        # ── CONV col for quality checks ────────────────────────────────────────
        conv_c = _tgt_e if _tgt_e in df.columns else (conv_cols_sot[0] if conv_cols_sot else None)

        # ── 4 TABS ─────────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Channels", "🔗 Covariate Checks", "🚩 Flags"])

        # ── TAB 1: OVERVIEW ────────────────────────────────────────────────────
        with tab1:
            # Target time series with pre/post shading + confirmed flags
            _fig_ov1 = go.Figure()
            if conv_c and conv_c in df.columns:
                _fig_ov1.add_trace(go.Scatter(
                    x=df["p_date"], y=df[conv_c], mode="lines",
                    line=dict(color=C["purple"], width=2), name=conv_c.replace("_"," ")))
            if t_start is not None and t_end is not None:
                if pre_df is not None and len(pre_df):
                    _fig_ov1.add_vrect(x0=str(pre_df["p_date"].min().date()), x1=str(t_start.date()),
                        fillcolor="rgba(123,127,255,0.10)", line_width=0, annotation_text="Pre", annotation_position="top left",
                        annotation_font=dict(color=C["purple_l"],size=10))
                _fig_ov1.add_vrect(x0=str(t_start.date()), x1=str(t_end.date()),
                    fillcolor="rgba(64,200,122,0.10)", line_width=0, annotation_text="Post", annotation_position="top left",
                    annotation_font=dict(color=C["green"],size=10))
                _fig_ov1.add_shape(type="line", x0=str(t_start.date()), x1=str(t_start.date()), y0=0, y1=1, yref="paper",
                    line=dict(color=C["purple"],width=2,dash="dash"))
            _confirmed_ov1 = [f for f in st.session_state.flag_vars if f.get("confirmed")]
            for _fv1 in _confirmed_ov1:
                _fc1 = C["amber"]
                _fig_ov1.add_vrect(x0=_fv1["start"], x1=_fv1["end"], fillcolor=_fc1, opacity=0.08,
                    annotation_text=_fv1["name"], annotation_font=dict(color=_fc1,size=9))
            _fig_ov1.update_layout(**CHART, title=dict(text=f"{conv_c.replace('_',' ') if conv_c else 'Target'} — full series", font=dict(size=14)))
            st.plotly_chart(_fig_ov1, use_container_width=True)

            # Pre/post summary tiles
            if len(pre_df) and len(post_df) and conv_c and conv_c in df.columns:
                _pm = pre_df[conv_c].mean(); _qm = post_df[conv_c].mean()
                _pct = (_qm - _pm) / _pm * 100 if _pm else 0
                _mc_pct = "mvc" if _pct > 0 else "mvr"
                st.markdown(
                    f'<div class="mrow">'
                    f'{mtile("Pre avg/day", f"{_pm:,.0f}")}'
                    f'{mtile("Post avg/day", f"{_qm:,.0f}")}'
                    f'{mtile("Raw change", f"{_pct:+.1f}%", _mc_pct)}'
                    f'{mtile("Pre days", str(len(pre_df)))}'
                    f'{mtile("Post days", str(len(post_df)))}'
                    f'</div>', unsafe_allow_html=True)

            # Quality pills (compact, non-blocking)
            _q_sigs = quality_checks(df)
            if _q_sigs:
                _pill_html = '<div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:12px">'
                for _qs in _q_sigs:
                    _qc = {"high":C["red"],"medium":C["amber"],"low":C["green"]}.get(_qs["sev"],C["grey"])
                    _pill_html += f'<span style="background:{_qc}18;border:1px solid {_qc}40;border-radius:20px;padding:3px 10px;font-size:11px;color:{_qc}">{_qs["msg"]}</span>'
                st.markdown(_pill_html + "</div>", unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="font-size:12px;color:{C["green"]};margin-top:10px">✓ No data quality issues detected</div>', unsafe_allow_html=True)

        # ── TAB 2: CHANNELS ────────────────────────────────────────────────────
        with tab2:
            # Channel snapshot cards
            _snap_channels = parse_sot_columns(df)
            _snap_cards = []
            for _ch_k, _ch_cols in _snap_channels.items():
                if _ch_k == "Other": continue
                for _col in _ch_cols:
                    if not any(k in _col.lower() for k in ["conversions","sessions","revenue"]): continue
                    _pre_avg  = pre_df[_col].mean()  if _col in pre_df.columns  and len(pre_df)  else 0
                    _post_avg = post_df[_col].mean() if _col in post_df.columns and len(post_df) else 0
                    _lift_s   = (_post_avg - _pre_avg) / _pre_avg * 100 if _pre_avg else 0
                    _lbl = _col.replace("Paid_Social_TikTok_","TikTok ").replace("Paid_Social_Meta_","Meta ").replace("Paid_Search_Google_","Google ").replace("Organic_Search_","Organic ").replace("Direct_","Direct ").replace("_"," ")
                    _snap_cards.append((_lbl, _pre_avg, _post_avg, _lift_s, "tiktok" in _col.lower()))
            if _snap_cards:
                st.markdown(f'<div style="font-size:11px;font-weight:600;color:{C["grey"]};letter-spacing:.06em;margin-bottom:8px">CHANNEL SNAPSHOT — Pre vs Post Lift</div>', unsafe_allow_html=True)
                _sn_cols = st.columns(min(len(_snap_cards), 4))
                for _si, (_lbl, _pa, _pb, _lft, _is_tt) in enumerate(_snap_cards):
                    with _sn_cols[_si % 4]:
                        _lc = C["purple"] if _lft > 0 else C["red"]
                        _border_c = "#00D8FF" if _is_tt else C["border"]
                        _sign = "+" if _lft >= 0 else ""
                        st.markdown(
                            f'<div style="border:1px solid {_border_c};border-radius:8px;padding:10px 14px;margin-bottom:10px">'
                            f'<div style="font-size:10px;color:{C["grey"]};margin-bottom:3px">{_lbl}</div>'
                            f'<div style="font-size:18px;font-weight:700;color:{_lc}">{_sign}{_lft:.1f}%</div>'
                            f'<div style="font-size:11px;color:{C["grey"]};margin-top:2px">{_pa:,.0f} → {_pb:,.0f} avg/day</div>'
                            f'</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Data overview selectors
            st.markdown(f'<div style="font-size:11px;font-weight:600;color:{C["grey"]};letter-spacing:.06em;margin-bottom:10px">DATA OVERVIEW</div>', unsafe_allow_html=True)
            ov_c1, ov_c2, ov_c3 = st.columns([2, 3, 1])
            with ov_c1: ov_metric   = st.selectbox("Metric", ["Conversions","Sessions","Revenue"], key="ov_metric")
            with ov_c2: ov_channels = st.multiselect("Channels", options=avail_chs, default=avail_chs, key="ov_channels")
            with ov_c3: ov_smooth   = st.selectbox("Smooth", [1,3,7,14], index=2, key="ov_smooth")
            ov_view = st.radio("View", ["Time Series","Conversion Rate","Pre vs Post","Channel Mix","Lift Overview"], horizontal=True, key="ov_view")

            col_lookup_ov = {"Conversions": conv_cols_sot, "Sessions": sess_cols_sot, "Revenue": rev_cols_sot}
            fig_ov = go.Figure()
            cvr_rows_ov = []

            if ov_view == "Time Series":
                for ch in ov_channels:
                    col = next((c for c in channels.get(ch,[]) if ov_metric.lower() in c.lower()), None)
                    if col and col in df.columns:
                        series = df[col].rolling(ov_smooth, center=True, min_periods=1).mean()
                        fig_ov.add_trace(go.Scatter(x=df["p_date"], y=series, mode="lines",
                            line=dict(color=ch_col(ch), width=2.5), name=ch_lbl(ch) + (f" ({ov_smooth}d avg)" if ov_smooth > 1 else "")))
                fig_ov.update_layout(**CHART, title=dict(text=f"{ov_metric} by Channel", font=dict(size=14)))
                add_treatment(fig_ov)

            elif ov_view == "Conversion Rate":
                for ch in ov_channels:
                    s_col = next((c for c in channels.get(ch,[]) if "session" in c.lower()), None)
                    c_col = next((c for c in channels.get(ch,[]) if "conversion" in c.lower()), None)
                    if not s_col or not c_col: continue
                    cvr = (df[c_col] / df[s_col].replace(0, np.nan) * 100).rolling(ov_smooth, center=True, min_periods=1).mean()
                    fig_ov.add_trace(go.Scatter(x=df["p_date"], y=cvr, mode="lines",
                        line=dict(color=ch_col(ch), width=2.5), name=ch_lbl(ch)))
                    pre_cvr  = pre_df[c_col].sum()  / pre_df[s_col].sum()  * 100 if len(pre_df)  and pre_df[s_col].sum()  > 0 else 0
                    post_cvr = post_df[c_col].sum() / post_df[s_col].sum() * 100 if len(post_df) and post_df[s_col].sum() > 0 else 0
                    delta = (post_cvr - pre_cvr) / pre_cvr * 100 if pre_cvr else 0
                    cvr_rows_ov.append(dict(ch=ch_lbl(ch), pre=pre_cvr, post=post_cvr, delta=delta, color=ch_col(ch)))
                fig_ov.update_layout(**CHART, title=dict(text=f"Conversion Rate (%) — {ov_smooth}d smoothed", font=dict(size=14)))
                add_treatment(fig_ov)

            elif ov_view == "Pre vs Post":
                pv_ov, ptv_ov, lbs_ov = [], [], []
                for ch in ov_channels:
                    col = next((c for c in channels.get(ch,[]) if ov_metric.lower() in c.lower()), None)
                    if col and col in df.columns:
                        pv_ov.append(pre_df[col].mean() if len(pre_df) else 0)
                        ptv_ov.append(post_df[col].mean() if len(post_df) else 0)
                        lbs_ov.append(ch_lbl(ch))
                fig_ov.add_trace(go.Bar(name="Pre-Period",  x=lbs_ov, y=pv_ov,  marker_color=C["grey_l"], opacity=0.75))
                fig_ov.add_trace(go.Bar(name="Post-Period", x=lbs_ov, y=ptv_ov, marker_color=C["purple"]))
                fig_ov.update_layout(**CHART, barmode="group", title=dict(text=f"Daily Avg {ov_metric} — Pre vs Post", font=dict(size=14)))

            elif ov_view == "Channel Mix":
                df_w = df.copy()
                df_w["week"] = df_w["p_date"].dt.to_period("W").apply(lambda r: r.start_time)
                df_grp = df_w.groupby("week").sum(numeric_only=True).reset_index().rename(columns={"week":"p_date"})
                for ch in ov_channels:
                    col = next((c for c in channels.get(ch,[]) if ov_metric.lower() in c.lower()), None)
                    if col and col in df_grp.columns:
                        fig_ov.add_trace(go.Bar(x=df_grp["p_date"], y=df_grp[col], name=ch_lbl(ch),
                            marker_color=ch_col(ch), hovertemplate="%{x|%d %b}: %{y:,.0f}<extra></extra>"))
                if t_start is not None:
                    fig_ov.add_vline(x=pd.Timestamp(t_start).value // 10**6,
                        line=dict(color=C["purple_l"],width=2,dash="dash"),
                        annotation_text="Campaign start", annotation_position="top right",
                        annotation_font=dict(color=C["purple_l"],size=10))
                fig_ov.update_layout(**CHART, barmode="stack", bargap=0.1,
                    title=dict(text=f"Weekly {ov_metric} by Channel (Stacked)", font=dict(size=14)))

            elif ov_view == "Lift Overview":
                lift_rows = []
                for ch in ov_channels:
                    col = next((c for c in channels.get(ch,[]) if ov_metric.lower() in c.lower()), None)
                    if col and col in df.columns and len(pre_df) and len(post_df):
                        pre_avg  = pre_df[col].mean(); post_avg = post_df[col].mean()
                        abs_lift = post_avg - pre_avg
                        pct_lift = abs_lift / pre_avg * 100 if pre_avg else 0
                        lift_rows.append({"ch":ch_lbl(ch),"abs":abs_lift,"pct":pct_lift,"pre":pre_avg,"post":post_avg,"color":ch_col(ch)})
                if lift_rows:
                    lift_rows.sort(key=lambda x: x["abs"])
                    _max_abs = max(abs(r["abs"]) for r in lift_rows); _min_val = min(r["abs"] for r in lift_rows)
                    fig_ov.add_trace(go.Bar(y=[r["ch"] for r in lift_rows], x=[r["abs"] for r in lift_rows], orientation="h",
                        marker_color=[C["green"] if r["abs"] >= 0 else C["red"] for r in lift_rows],
                        text=[f"{r['pct']:+.1f}%  ({r['pre']:,.0f} → {r['post']:,.0f})" if r["abs"] >= 0 else "" for r in lift_rows],
                        textposition="outside", textfont=dict(color=C["text"],size=11), cliponaxis=False))
                    for _nr in lift_rows:
                        if _nr["abs"] < 0:
                            fig_ov.add_annotation(x=_max_abs*0.02, y=_nr["ch"],
                                text=f"{_nr['pct']:+.1f}%  ({_nr['pre']:,.0f} → {_nr['post']:,.0f})",
                                showarrow=False, xanchor="left", yanchor="middle",
                                font=dict(size=11,color=C["text"],family="Inter"))
                    fig_ov.add_vline(x=0, line=dict(color=C["border"],width=1))
                    _left_pad = min(_min_val*1.8,-_max_abs*0.05) if _min_val < 0 else -_max_abs*0.02
                    fig_ov.update_layout(**CHART,
                        title=dict(text=f"Daily Avg {ov_metric} — Pre vs Post Lift",font=dict(size=14)),
                        xaxis_title="Daily average change", height=max(300,len(lift_rows)*65),
                        xaxis_range=[_left_pad,_max_abs*1.1], margin=dict(l=160,r=240,t=45,b=45))

            st.plotly_chart(fig_ov, use_container_width=True)

            if ov_view == "Conversion Rate" and cvr_rows_ov:
                html_cvr = '<div class="mrow">'
                for r in sorted(cvr_rows_ov, key=lambda x: -abs(x["delta"])):
                    sign = "+"; mc = "mvc"
                    if r["delta"] < 0: sign = ""; mc = "mvr"
                    html_cvr += (f'<div class="mtile" style="border-left:3px solid {r["color"]}">'
                                 f'<div class="ml">{r["ch"]}</div><div class="mv {mc}">{sign}{r["delta"]:.1f}%</div>'
                                 f'<div style="font-size:11px;color:{C["grey"]}">{r["pre"]:.2f}% → {r["post"]:.2f}%</div></div>')
                st.markdown(html_cvr + "</div>", unsafe_allow_html=True)

            stab_entries = []
            for ch in (ov_channels or avail_chs):
                col_s = next((c for c in channels.get(ch,[]) if ov_metric.lower() in c.lower()), None)
                if col_s and col_s in pre_df.columns:
                    ps = pre_df[col_s].dropna()
                    if len(ps) < 3: continue
                    cv = ps.std() / ps.mean() * 100 if ps.mean() != 0 else 0
                    slope = np.polyfit(np.arange(len(ps)), ps.values, 1)[0] / ps.mean() * 100 if ps.mean() != 0 else 0
                    mc = "mvc" if cv < 20 else "mva" if cv < 40 else "mvr"
                    stab_entries.append((ch, cv, slope, mc))
            if stab_entries:
                st.markdown(f'<div style="font-size:11px;font-weight:600;color:{C["grey"]};letter-spacing:.06em;margin:10px 0 6px">PRE-PERIOD STABILITY</div>', unsafe_allow_html=True)
                html_stab = '<div class="mrow">'
                for ch_s, cv_s, slope_s, mc_s in stab_entries:
                    html_stab += (f'<div class="mtile" style="border-left:3px solid {ch_col(ch_s)}">'
                                  f'<div class="ml">{ch_lbl(ch_s)}</div><div class="mv {mc_s}">{cv_s:.0f}% CV</div>'
                                  f'<div style="font-size:11px;color:{C["grey"]}">{slope_s:+.1f}%/day</div></div>')
                st.markdown(html_stab + "</div>", unsafe_allow_html=True)

            # Google Trends overlay
            with st.expander("🔍 Google Search Trends Overlay"):
                st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:12px">Overlay category and brand search volume to contextualise channel lifts. Relative index (0–100).</div>', unsafe_allow_html=True)
                gt_c1, gt_c2, gt_c3, gt_c4 = st.columns([3, 3, 2, 1])
                with gt_c1: gt_brand    = st.text_input("Brand keyword",    placeholder="e.g. Babyboo", key="gt_brand")
                with gt_c2: gt_category = st.text_input("Category keyword", placeholder="e.g. baby fashion", key="gt_cat")
                with gt_c3: gt_geo      = st.selectbox("Market", list(GEO_OPTIONS.keys()), index=0, key="gt_geo")
                with gt_c4:
                    st.markdown("<br>", unsafe_allow_html=True)
                    gt_run = st.button("Fetch Trends", use_container_width=True, key="gt_run")
                if gt_run and (gt_brand or gt_category):
                    _kw_list = [k for k in [gt_brand, gt_category] if k.strip()]
                    with st.spinner(f"Fetching Google Trends ({gt_geo})..."):
                        _df_tr, _gt_err = fetch_google_trends(_kw_list, df["p_date"].min().date(), df["p_date"].max().date(), gt_geo)
                    if _gt_err:
                        st.markdown(f'<div class="card card-a">Google Trends error: {_gt_err}</div>', unsafe_allow_html=True)
                    elif _df_tr is not None:
                        st.session_state.gt_data = _df_tr; st.session_state.gt_keywords = _kw_list
                if st.session_state.get("gt_data") is not None:
                    df_gt = st.session_state.gt_data; gt_kws = st.session_state.get("gt_keywords",[])
                    overlay_options = conv_cols_sot + sess_cols_sot
                    gt_overlay_col = st.selectbox("Overlay channel metric", overlay_options, key="gt_overlay")
                    fig_gt = go.Figure()
                    if gt_overlay_col and gt_overlay_col in df.columns:
                        fig_gt.add_trace(go.Scatter(x=df["p_date"], y=df[gt_overlay_col].rolling(7,center=True,min_periods=1).mean(),
                            mode="lines", name=gt_overlay_col.replace("_"," ")+" (7d avg)", line=dict(color=C["purple"],width=2.5), yaxis="y1"))
                    TREND_COLORS = ["#EA4335","#34A853","#FBBC05","#4285F4"]
                    for i, kw in enumerate(gt_kws):
                        if kw in df_gt.columns:
                            td = df_gt[kw].reindex(pd.date_range(df_gt.index.min(), df_gt.index.max(), freq="D")).interpolate()
                            fig_gt.add_trace(go.Scatter(x=td.index, y=td.values, mode="lines", name=f"Search: {kw}",
                                line=dict(color=TREND_COLORS[i%len(TREND_COLORS)],width=2,dash="dot"), yaxis="y2", opacity=0.8))
                    add_treatment(fig_gt)
                    _gt_layout = {k: v for k, v in CHART.items() if k not in ("yaxis","yaxis2","legend")}
                    fig_gt.update_layout(**_gt_layout,
                        title=dict(text=f"Channel Metric vs Google Search Interest ({gt_geo})", font=dict(size=14)),
                        yaxis=dict(title="Channel metric", gridcolor="#E5E7EB", showgrid=True),
                        yaxis2=dict(title="Search interest (0–100)", overlaying="y", side="right", showgrid=False, range=[0,110], tickfont=dict(color=C["grey"],size=10)),
                        legend=dict(orientation="h", y=-0.2))
                    st.plotly_chart(fig_gt, use_container_width=True)
                    if gt_overlay_col and gt_overlay_col in df.columns:
                        corr_rows = []
                        for kw in gt_kws:
                            if kw in df_gt.columns:
                                td = df_gt[kw].reindex(pd.date_range(df_gt.index.min(),df_gt.index.max(),freq="D")).interpolate()
                                merged = pd.merge(df[["p_date",gt_overlay_col]].rename(columns={"p_date":"date"}),
                                    td.rename("trend").reset_index().rename(columns={"index":"date"}), on="date", how="inner")
                                if len(merged) > 5:
                                    corr_rows.append((kw, merged[gt_overlay_col].corr(merged["trend"])))
                        if corr_rows:
                            html_corr = '<div class="mrow">'
                            for kw, r in corr_rows:
                                mc = "mvc" if abs(r)>0.6 else "mva" if abs(r)>0.3 else "mvr"
                                html_corr += f'<div class="mtile"><div class="ml">"{kw}" vs channel</div><div class="mv {mc}">{r:.2f} R</div><div style="font-size:11px;color:{C["grey"]}">{"Strong" if abs(r)>0.6 else "Moderate" if abs(r)>0.3 else "Weak"} {"positive" if r>0 else "negative"}</div></div>'
                            st.markdown(html_corr + "</div>", unsafe_allow_html=True)

        # ── TAB 3: COVARIATE CHECKS ────────────────────────────────────────────
        with tab3:
            tc2, tc3, tc4 = st.tabs(["📈 Weekday Trend", "🔍 Outliers", "🔗 Covariate Validation"])

            with tc2:
                st.markdown("Weekday cyclicality — does demand vary significantly by day of week in the pre-period?")
                st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:10px">A strong weekday pattern (>20% deviation from the weekly average) suggests you should add a weekend flag or day-of-week dummy as a covariate so the model does not mistake the pattern for intervention effect.</div>', unsafe_allow_html=True)
                if conv_c and conv_c in df.columns:
                    _pre_tc = pre_df.copy() if len(pre_df) else df.copy()
                    _pre_tc["_dow"] = _pre_tc["p_date"].dt.day_name()
                    _dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                    _dow_avg = _pre_tc.groupby("_dow")[conv_c].mean().reindex(_dow_order)
                    _overall = _dow_avg.mean()
                    _pct_dev = ((_dow_avg - _overall) / _overall * 100).round(1)
                    _bar_cols = [C["purple"] if abs(v) > 20 else C["grey_l"] for v in _pct_dev]
                    _fig_dow = go.Figure(go.Bar(x=_dow_order, y=_dow_avg.values, marker_color=_bar_cols,
                        customdata=_pct_dev.values, hovertemplate="%{x}: %{y:,.0f} avg (%{customdata:+.1f}% vs weekly avg)<extra></extra>"))
                    _fig_dow.add_hline(y=_overall, line_dash="dot", line_color=C["amber"], annotation_text="Weekly avg")
                    _fig_dow.update_layout(**CHART, height=260, title=dict(text=f"Avg {conv_c.replace('_',' ')} by weekday (pre-period)", font=dict(size=13)))
                    st.plotly_chart(_fig_dow, use_container_width=True)
                    _strong = [d for d, v in _pct_dev.items() if abs(v) > 20]
                    if _strong:
                        st.markdown(f'<div class="card card-a">{pill("CYCLICALITY DETECTED","pill-a")} <strong style="margin-left:8px">{", ".join(_strong)}</strong> deviate >20% from weekly avg — consider adding a weekend or day-of-week flag as a covariate.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="card card-c">✓ No strong weekday cyclicality — no day-of-week flag needed</div>', unsafe_allow_html=True)
                else:
                    st.info("No target column detected.")

            with tc3:
                st.markdown("Outlier detection — three methods to flag abnormal days in the pre-period")
                st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:10px">Not all flagged days need treatment. If the spike/dip has a clear business explanation keep it as-is and add a dummy flag. Only winsorise if it is unexplained noise.</div>', unsafe_allow_html=True)
                if conv_c and conv_c in df.columns:
                    _pre_out = pre_df.copy() if len(pre_df) else df.copy()
                    _c1, _c2 = st.columns(2)
                    with _c1: _dod_t = st.slider("Day-over-day threshold (%)", 50, 300, 150, 25, key="dod_t")
                    with _c2: _z_t   = st.slider("Z-score threshold", 1.5, 4.0, 2.5, 0.5, key="z_t")
                    _dod_chg = _pre_out[conv_c].pct_change().abs() * 100
                    _dod_f = _pre_out[_dod_chg > _dod_t]["p_date"].dt.strftime("%d %b").tolist()
                    _m = _pre_out[conv_c].mean(); _s = _pre_out[conv_c].std()
                    _z_f = _pre_out[abs((_pre_out[conv_c]-_m)/_s) > _z_t]["p_date"].dt.strftime("%d %b").tolist() if _s > 0 else []
                    _q1, _q3 = _pre_out[conv_c].quantile(0.25), _pre_out[conv_c].quantile(0.75)
                    _iqr_f = _pre_out[(_pre_out[conv_c]<_q1-1.5*(_q3-_q1))|(_pre_out[conv_c]>_q3+1.5*(_q3-_q1))]["p_date"].dt.strftime("%d %b").tolist()
                    _cm1, _cm2, _cm3 = st.columns(3)
                    with _cm1: st.markdown(f'<div class="card {"card-r" if _dod_f else "card-c"}"><strong>Day-over-day</strong><br><span style="font-size:22px;font-weight:700">{len(_dod_f)}</span> flags<br><div style="font-size:11px;color:{C["grey"]}">{", ".join(_dod_f[:4]) if _dod_f else "None"}</div></div>', unsafe_allow_html=True)
                    with _cm2: st.markdown(f'<div class="card {"card-r" if _z_f else "card-c"}"><strong>Z-score</strong><br><span style="font-size:22px;font-weight:700">{len(_z_f)}</span> flags<br><div style="font-size:11px;color:{C["grey"]}">{", ".join(_z_f[:4]) if _z_f else "None"}</div></div>', unsafe_allow_html=True)
                    with _cm3: st.markdown(f'<div class="card {"card-r" if _iqr_f else "card-c"}"><strong>IQR fence</strong><br><span style="font-size:22px;font-weight:700">{len(_iqr_f)}</span> flags<br><div style="font-size:11px;color:{C["grey"]}">{", ".join(_iqr_f[:4]) if _iqr_f else "None"}</div></div>', unsafe_allow_html=True)
                    _all_f = sorted(set(_dod_f + _z_f + _iqr_f))
                    if _all_f:
                        st.markdown(f'<div class="card card-a" style="margin-top:10px"><strong>Flagged pre-period dates:</strong> {", ".join(_all_f)}<div style="font-size:12px;color:{C["grey"]};margin-top:6px">If business-driven: keep and add a flag in the Flags tab. If unexplained noise: winsorise below.</div></div>', unsafe_allow_html=True)
                        with st.expander("Winsorisation — cap extreme values at 95th percentile"):
                            _p95 = _pre_out[conv_c].quantile(0.95); _p05 = _pre_out[conv_c].quantile(0.05)
                            st.markdown(f"Pre-period 5th pct: {_p05:,.0f} · 95th pct: {_p95:,.0f}")
                            if st.button(f"Winsorise {conv_c} at 95th percentile", key="winsor_btn"):
                                _df_w = parse_dates(st.session_state.daily_df.copy())
                                _df_w[conv_c] = _df_w[conv_c].clip(lower=_p05, upper=_p95)
                                st.session_state.daily_df = _df_w
                                st.success(f"Winsorised {conv_c}. Values capped between {_p05:,.0f} and {_p95:,.0f}.")
                                st.rerun()
                else:
                    st.info("No target column detected.")

            with tc4:
                st.markdown("Covariate validation — checks whether your covariates are safe to use as controls")
                st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:10px">Three checks: (1) Pre/post average change — if a covariate changed >20% after the intervention it may itself have been affected. (2) Extrapolation risk — if post-period values fall outside the pre-period range. (3) Scale — if covariates have very different scales, log-standardisation can help.</div>', unsafe_allow_html=True)
                if t_start is not None and st.session_state.daily_df is not None:
                    _df_cv = parse_dates(st.session_state.daily_df.copy())
                    _cov_list = [c for c in _df_cv.columns if c not in ["p_date","prepost_1","prepost"]
                                 and _df_cv[c].dtype in ["float64","int64"]
                                 and not any(k in c.lower() for k in ["cost","topview","flag","spend","pre_period","post_period"])]
                    if _cov_list:
                        _pre_cv  = _df_cv[_df_cv["p_date"] < t_start]
                        _post_cv = _df_cv[(_df_cv["p_date"] >= t_start) & (_df_cv["p_date"] <= t_end)] if t_end else _df_cv[_df_cv["p_date"] >= t_start]
                        _cv_rows = []
                        for _c in _cov_list[:15]:
                            _pa = _pre_cv[_c].mean(); _pb = _post_cv[_c].mean()
                            if _pa and _pa != 0:
                                _chg = (_pb - _pa) / abs(_pa) * 100
                                _p10 = _pre_cv[_c].quantile(0.1); _p90 = _pre_cv[_c].quantile(0.9)
                                _extrap_pct = ((_post_cv[_c] < _p10).sum() + (_post_cv[_c] > _p90).sum()) / len(_post_cv) * 100 if len(_post_cv) > 0 else 0
                                _std = _pre_cv[_c].std()
                                _z_post_max = abs((_post_cv[_c] - _pa) / _std).max() if _std > 0 else 0
                                _cv_rows.append({
                                    "Covariate": _c, "Pre avg": f"{_pa:,.0f}", "Post avg": f"{_pb:,.0f}",
                                    "Δ pre→post": f"{_chg:+.1f}%",
                                    "Endogeneity": "⚠️ >20%" if abs(_chg) > 20 else "✓",
                                    "Extrapolation": f"⚠️ {_extrap_pct:.0f}% out of range" if _extrap_pct > 20 else "✓",
                                    "Max Z-post": f"{'⚠️ ' if _z_post_max > 3 else ''}{_z_post_max:.1f}σ"
                                })
                        if _cv_rows:
                            st.dataframe(pd.DataFrame(_cv_rows), use_container_width=True, hide_index=True)
                            _risky = [r["Covariate"] for r in _cv_rows if "⚠️" in r["Endogeneity"]]
                            _extrap_risky = [r["Covariate"] for r in _cv_rows if "⚠️" in r["Extrapolation"]]
                            if _risky:
                                st.markdown(f'<div class="card card-a">{pill("ENDOGENEITY RISK","pill-a")} <strong style="margin-left:8px">{", ".join(_risky)}</strong> changed >20% pre→post — may have been affected by the intervention itself.</div>', unsafe_allow_html=True)
                            if _extrap_risky:
                                st.markdown(f'<div class="card card-a">{pill("EXTRAPOLATION RISK","pill-a")} <strong style="margin-left:8px">{", ".join(_extrap_risky)}</strong> have post-period values outside the pre-period P10–P90 range.</div>', unsafe_allow_html=True)
                        with st.expander("Scale adjustment — log transform + standardise"):
                            st.markdown(f'<div style="font-size:12px;color:{C["grey"]}">Log transformation + z-score standardisation can improve model stability for right-skewed covariates with large absolute values.</div>', unsafe_allow_html=True)
                            _scale_cols = st.multiselect("Select columns to log-standardise", options=_cov_list, key="scale_cols")
                            if _scale_cols and st.button("Apply log-standardisation", key="scale_btn"):
                                _df_sc = parse_dates(st.session_state.daily_df.copy())
                                for _sc in _scale_cols:
                                    if (_df_sc[_sc] > 0).all():
                                        _log_vals = np.log(_df_sc[_sc])
                                        _df_sc[f"{_sc}_logstd"] = (_log_vals - _log_vals.mean()) / _log_vals.std()
                                    else:
                                        st.warning(f"{_sc} has zero/negative values — log transform skipped.")
                                st.session_state.daily_df = _df_sc
                                st.success(f"Added log-standardised columns: {', '.join(f'{c}_logstd' for c in _scale_cols)}")
                                st.rerun()
                    else:
                        st.info("No numeric covariate columns detected.")
                else:
                    st.info("Set post-period start date to enable covariate validation.")

        # ── TAB 4: FLAGS ───────────────────────────────────────────────────────
        with tab4:
            _df_flags = df.copy()
            if not st.session_state.flag_vars:
                st.session_state.flag_vars = suggest_flags(_df_flags, synthesis=None)
            _brief_flags = [f for f in st.session_state.flag_vars if f.get("source") == "brief"]
            _data_flags  = [f for f in st.session_state.flag_vars if f.get("source") != "brief"]
            if _brief_flags:
                st.markdown(f'<div class="t-meta" style="color:{C["purple"]};margin-bottom:10px">✓ {len(_brief_flags)} flag(s) detected from brief · {len(_data_flags)} detected from data</div>', unsafe_allow_html=True)
            for i, fv in enumerate(st.session_state.flag_vars):
                style = "card-c" if fv.get("is_treatment") else "card-a"
                label = "TREATMENT WINDOW" if fv.get("is_treatment") else "COVARIATE FLAG"
                lp = "pill-c" if fv.get("is_treatment") else "pill-a"
                src_badge = pill("from brief","pill-c") if fv.get("source")=="brief" else pill("from data","pill-g")
                st.markdown(f'<div class="card {style}">{pill(label,lp)} {src_badge} <strong style="margin-left:8px;font-size:14px">{fv["name"]}</strong><div class="t-meta" style="margin-top:5px">{fv["rationale"]}</div></div>', unsafe_allow_html=True)
                cn, cs, ce, cc = st.columns([3, 2, 2, 1])
                with cn:
                    st.markdown(f'<div class="t-label" style="margin-bottom:3px">Name</div>', unsafe_allow_html=True)
                    nn = st.text_input("Name", value=fv["name"], key=f"fn{i}", label_visibility="collapsed")
                    st.session_state.flag_vars[i]["name"] = nn
                with cs:
                    st.markdown(f'<div class="t-label" style="margin-bottom:3px">Start</div>', unsafe_allow_html=True)
                    ns = st.date_input("Start", value=pd.to_datetime(fv["start"]).date(),
                        min_value=_df_flags["p_date"].min().date(), max_value=_df_flags["p_date"].max().date(),
                        key=f"fs{i}", label_visibility="collapsed")
                    st.session_state.flag_vars[i]["start"] = str(ns)
                with ce:
                    st.markdown(f'<div class="t-label" style="margin-bottom:3px">End</div>', unsafe_allow_html=True)
                    ne = st.date_input("End", value=pd.to_datetime(fv["end"]).date(),
                        min_value=_df_flags["p_date"].min().date(), max_value=_df_flags["p_date"].max().date(),
                        key=f"fe{i}", label_visibility="collapsed")
                    st.session_state.flag_vars[i]["end"] = str(ne)
                with cc:
                    st.markdown(f'<div class="t-label" style="margin-bottom:3px">Include</div>', unsafe_allow_html=True)
                    chk = st.checkbox("Include", value=fv.get("confirmed",False), key=f"fc{i}", label_visibility="collapsed")
                    st.session_state.flag_vars[i]["confirmed"] = chk
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("Add custom flag"):
                xa, xb, xc, xd = st.columns([2,2,2,2])
                with xa: fn = st.text_input("Flag name", placeholder="e.g. FootyFinals_WarmupFlag")
                with xb: fs = st.date_input("Start date", min_value=_df_flags["p_date"].min().date(), max_value=_df_flags["p_date"].max().date(), key="nfs")
                with xc: fe = st.date_input("End date",   min_value=_df_flags["p_date"].min().date(), max_value=_df_flags["p_date"].max().date(), key="nfe")
                with xd: fr = st.text_input("Rationale", placeholder="Why flag this period?")
                if st.button("Add flag"):
                    if fn:
                        st.session_state.flag_vars.append(dict(name=fn,start=str(fs),end=str(fe),rationale=fr,confirmed=True,is_treatment=False))
                        st.rerun()
            _confirmed_flags = [f for f in st.session_state.flag_vars if f.get("confirmed")]
            if _confirmed_flags:
                _tgt_flag = st.session_state.target_col or conv_c
                if _tgt_flag and _tgt_flag in _df_flags.columns:
                    st.markdown("<br>", unsafe_allow_html=True)
                    fig_fl = go.Figure()
                    fig_fl.add_trace(go.Scatter(x=_df_flags["p_date"], y=_df_flags[_tgt_flag], mode="lines",
                        line=dict(color=C["purple"],width=2), name=_tgt_flag.replace("_"," ")))
                    for fv in _confirmed_flags:
                        _fc = C["purple"] if fv.get("is_treatment") else C["amber"]
                        fig_fl.add_vrect(x0=fv["start"],x1=fv["end"],fillcolor=_fc,opacity=0.1,
                            annotation_text=fv["name"],annotation=dict(font=dict(color=_fc,size=10),yanchor="top"))
                    fig_fl.update_layout(**CHART, title=dict(text="Confirmed flags on target series",font=dict(size=14)))
                    st.plotly_chart(fig_fl, use_container_width=True)

        # ── NAVIGATION ─────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        _exp_b1, _exp_b2 = st.columns([1, 5])
        with _exp_b1:
            if st.button("Back", key="exp_back"): st.session_state.step=0; st.rerun()
        with _exp_b2:
            if st.button("Continue to Model →", use_container_width=True, type="primary", key="exp_fwd"):
                st.session_state.step=6; st.session_state.max_step=max(st.session_state.max_step,6); st.rerun()
        st.stop()

    conv_cols = [c for c in df.columns if "conversion" in c.lower() or c.lower()=="conversions"]
    spend_cats = [c for c in df.columns if c.endswith("Cost") and "Total" not in c and "CPA" not in c]
    date_min = df["p_date"].min().date()
    date_max = df["p_date"].max().date()

    # Read target and periods from session state (set in Configure)
    target = st.session_state.get("target_col")
    if not target or target not in df.columns:
        target = conv_cols[0] if conv_cols else None

    pre_s  = st.session_state.pre_start  or date_min
    pre_e  = st.session_state.pre_end    or date_min
    post_s = st.session_state.post_start or date_min
    post_e = st.session_state.post_end   or date_max

    # Period reminder strip (read-only)
    try:
        pre_days  = (pd.Timestamp(pre_e)  - pd.Timestamp(pre_s)).days  + 1
        post_days = (pd.Timestamp(post_e) - pd.Timestamp(post_s)).days + 1
        ratio     = pre_days / post_days if post_days > 0 else 0
        overlap   = pd.Timestamp(post_s) <= pd.Timestamp(pre_e)
    except Exception:
        pre_days = post_days = 0; ratio = 0; overlap = False
    _rc_eda = C["green"] if ratio >= 2 else C["amber"] if ratio >= 1 else C["red"]
    st.markdown(
        f'<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:{C["surface"]};'
        f'border:1px solid {C["border"]};border-radius:10px;padding:10px 16px;margin-bottom:16px;font-size:12px">'
        f'<span style="color:{C["grey"]};font-weight:600;letter-spacing:.05em;font-size:11px">PERIODS</span>'
        f'<span style="color:{C["border"]}">|</span>'
        f'<span style="color:{C["text"]}">Pre: <strong>{pre_s} → {pre_e}</strong> ({pre_days}d)</span>'
        f'<span style="color:{C["border"]}">·</span>'
        f'<span style="color:{C["text"]}">Post: <strong>{post_s} → {post_e}</strong> ({post_days}d)</span>'
        f'<span style="color:{C["border"]}">·</span>'
        f'<span style="font-weight:700;color:{_rc_eda}">Ratio {ratio:.1f}×</span>'
        f'<span style="color:{C["border"]}">|</span>'
        f'<span style="color:{C["grey"]}">Target: <strong style="color:{C["text"]}">{target or "—"}</strong></span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    ratio_ok   = ratio >= 2.0
    ratio_warn = 1.0 <= ratio < 2.0

    if overlap:
        st.error("Pre-period end must be before post-period start.")
    elif target:
        # ── Compute period slices for charts ─────────────────────────────────
        pre_df  = df[(df["p_date"].dt.date >= pre_s)  & (df["p_date"].dt.date <= pre_e)]
        post_df = df[(df["p_date"].dt.date >= post_s) & (df["p_date"].dt.date <= post_e)]
        gap_df  = df[(df["p_date"].dt.date > pre_e)   & (df["p_date"].dt.date < post_s)]

        pre_avg  = pre_df[target].mean()  if len(pre_df)  else 0
        post_avg = post_df[target].mean() if len(post_df) else 0
        raw_lift = (post_avg - pre_avg) / pre_avg * 100 if pre_avg > 0 else 0

        def add_period_vrects(fig):
            """Add shaded period regions to any chart."""
            fig.add_vrect(x0=str(pre_s), x1=str(pre_e),
                fillcolor="rgba(123,127,255,0.09)", line_width=0,
                annotation_text="Pre", annotation_position="top left",
                annotation=dict(font=dict(color=C["purple"],size=10)))
            fig.add_vrect(x0=str(post_s), x1=str(post_e),
                fillcolor="rgba(5,150,105,0.07)", line_width=0,
                annotation_text="Post", annotation_position="top left",
                annotation=dict(font=dict(color="#059669",size=10)))
            # Intervention line — use add_shape (add_vline breaks on date axes)
            fig.add_shape(type="line",
                x0=str(post_s), x1=str(post_s), y0=0, y1=1, yref="paper",
                line=dict(color=C["purple"], width=2, dash="dash"), opacity=0.7)
            fig.add_annotation(x=str(post_s), y=1, yref="paper",
                text="Intervention", showarrow=False, yanchor="bottom",
                font=dict(color=C["purple"], size=10), xshift=4)
            return fig

        tq, t1, t2, t3, t4, t5 = st.tabs(["📋 Data Quality","📈 Time Series","💰 Spend Breakdown","🔗 Correlations","📊 Period Comparison","🧮 Covariate Selection"])

        with tq:
            _dq_sigs = quality_checks(df)
            _dq_n = len(df); _dq_dates = df["p_date"]
            _dq_conv = next((c for c in df.columns if "conversion" in c.lower() or c.lower()=="conversions"), target)
            _dq_total = int(df[_dq_conv].sum()) if _dq_conv and _dq_conv in df.columns else 0
            _dq_highs = [s for s in _dq_sigs if s["sev"]=="high"]
            _dq_dr = f"{_dq_dates.min().strftime('%d %b')}–{_dq_dates.max().strftime('%d %b %Y')}"
            st.markdown(
                f'<div class="mrow">'
                f'{mtile("Date Range", _dq_dr)}'
                f'{mtile("Days", str(_dq_n), "mvc" if _dq_n>=60 else "mva" if _dq_n>=40 else "mvr")}'
                f'{mtile("Target Total", f"{_dq_total:,}")}'
                f'{mtile("Issues", str(len(_dq_highs)), "mvr" if _dq_highs else "mvc")}'
                f'</div>', unsafe_allow_html=True)
            for _sq in _dq_sigs:
                _cls = {"high":"card-r","medium":"card-a","low":"card-g"}.get(_sq["sev"],"card-g")
                _pc = {"high":"pill-r","medium":"pill-a","low":"pill-g"}.get(_sq["sev"],"pill-g")
                st.markdown(f'<div class="card {_cls}">{pill(_sq["sev"].upper(),_pc)} <strong style="margin-left:8px">{_sq["msg"]}</strong><div style="font-size:12px;color:{C["grey"]};margin-top:5px">{_sq.get("detail","")}</div></div>', unsafe_allow_html=True)
            if not _dq_sigs:
                st.markdown(f'<div class="card card-c">✓ No issues detected — data looks clean.</div>', unsafe_allow_html=True)
            # Target series with configured period shading
            if target and target in df.columns:
                _dq_fig = go.Figure()
                _dq_fig.add_trace(go.Scatter(x=df["p_date"], y=df[target], mode="lines",
                    line=dict(color=C["purple"],width=2), fill="tozeroy", fillcolor="rgba(123,127,255,0.14)", name=target))
                if pre_s and pre_e:
                    _dq_fig.add_vrect(x0=str(pre_s), x1=str(pre_e), fillcolor=C["purple"], opacity=0.06, line_width=0,
                        annotation_text="Pre", annotation=dict(font=dict(color=C["purple"],size=10),yanchor="top"))
                if post_s and post_e:
                    _dq_fig.add_vrect(x0=str(post_s), x1=str(post_e), fillcolor=C["green"], opacity=0.06, line_width=0,
                        annotation_text="Post", annotation=dict(font=dict(color=C["green"],size=10),yanchor="top"))
                    _dq_fig.add_vline(x=pd.Timestamp(post_s).value//10**6,
                        line=dict(color=C["purple"],width=1.5,dash="dash"))
                _dq_fig.update_layout(**CHART, title=dict(text=f"{target} — with configured analysis periods",font=dict(size=14)))
                st.plotly_chart(_dq_fig, use_container_width=True)
            # Detailed checks as expanders (period-aware)
            _dq_post_s = pd.to_datetime(post_s) if post_s else None
            _dq_post_e = pd.to_datetime(post_e) if post_e else None
            with st.expander("📅 Calendar & Gaps"):
                _dq_full = pd.date_range(df["p_date"].min(), df["p_date"].max(), freq="D")
                _dq_missing = sorted([d for d in _dq_full if d not in df["p_date"].values])
                if _dq_missing:
                    st.markdown(f'<div class="card card-r">{pill("GAPS DETECTED","pill-r")} <strong style="margin-left:8px">{len(_dq_missing)} missing date(s)</strong><div style="font-size:12px;color:{C["grey"]};margin-top:4px">{", ".join(d.strftime("%d %b") for d in _dq_missing[:12])}{"…" if len(_dq_missing)>12 else ""}</div></div>', unsafe_allow_html=True)
                    _dq_imp = st.radio("Imputation", ["Neighbour average (interpolate)","Weekday cyclical average"], horizontal=True, key="dq_imp")
                    if st.button("Apply imputation", key="dq_imp_btn"):
                        _dfi = parse_dates(st.session_state.daily_df.copy()).set_index("p_date").reindex(_dq_full); _dfi.index.name="p_date"
                        _num_i = _dfi.select_dtypes(include="number").columns
                        if _dq_imp.startswith("Neighbour"): _dfi[_num_i] = _dfi[_num_i].interpolate(method="linear",limit_direction="both")
                        else:
                            _dfi["_dow"] = pd.to_datetime(_dfi.index).dayofweek
                            for _ci in _num_i: _dfi[_ci] = _dfi[_ci].fillna(_dfi.groupby("_dow")[_ci].transform("mean"))
                            _dfi = _dfi.drop(columns=["_dow"])
                        st.session_state.daily_df = _dfi.reset_index(); st.success(f"Imputed {len(_dq_missing)} gap(s)."); st.rerun()
                else:
                    st.markdown(f'<div class="card card-c">✓ No date gaps — series is complete ({_dq_n} days)</div>', unsafe_allow_html=True)
                if _dq_post_s:
                    _pre_n_dq = len(df[df["p_date"] < _dq_post_s])
                    _post_n_dq = len(df[(df["p_date"] >= _dq_post_s) & (df["p_date"] <= _dq_post_e)]) if _dq_post_e else len(df[df["p_date"] >= _dq_post_s])
                    if _post_n_dq > 0:
                        _rat_dq = _pre_n_dq / _post_n_dq
                        _cls_dq = "card-c" if _rat_dq >= 2 else "card-a" if _rat_dq >= 1.5 else "card-r"
                        st.markdown(f'<div class="card {_cls_dq}" style="margin-top:8px"><strong>Pre/post ratio: {_rat_dq:.1f}×</strong> ({_pre_n_dq} pre · {_post_n_dq} post days)<div style="font-size:12px;color:{C["grey"]};margin-top:4px">Recommended ≥ 2×. Below 2× the model has less data to establish a reliable baseline.</div></div>', unsafe_allow_html=True)
            with st.expander("📈 Weekday Trend"):
                if _dq_conv and _dq_conv in df.columns:
                    _pre_tc_dq = df[df["p_date"] < _dq_post_s].copy() if _dq_post_s else df.copy()
                    _pre_tc_dq["_dow"] = _pre_tc_dq["p_date"].dt.day_name()
                    _dow_ord = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                    _dow_avg_dq = _pre_tc_dq.groupby("_dow")[_dq_conv].mean().reindex(_dow_ord)
                    _overall_dq = _dow_avg_dq.mean()
                    _pct_dq = ((_dow_avg_dq - _overall_dq) / _overall_dq * 100).round(1)
                    _bar_c_dq = [C["purple"] if abs(v) > 20 else C["grey_l"] for v in _pct_dq]
                    _fig_dq = go.Figure(go.Bar(x=_dow_ord, y=_dow_avg_dq.values, marker_color=_bar_c_dq,
                        customdata=_pct_dq.values, hovertemplate="%{x}: %{y:,.0f} avg (%{customdata:+.1f}% vs weekly avg)<extra></extra>"))
                    _fig_dq.add_hline(y=_overall_dq, line_dash="dot", line_color=C["amber"], annotation_text="Weekly avg")
                    _fig_dq.update_layout(**CHART, height=240, title=dict(text=f"Avg {_dq_conv} by weekday (pre-period)",font=dict(size=13)))
                    st.plotly_chart(_fig_dq, use_container_width=True)
                    _strong_dq = [d for d, v in _pct_dq.items() if abs(v) > 20]
                    if _strong_dq:
                        st.markdown(f'<div class="card card-a">{pill("CYCLICALITY","pill-a")} <strong style="margin-left:8px">{", ".join(_strong_dq)}</strong> deviate >20% — consider a weekend flag covariate.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="card card-c">✓ No strong weekday cyclicality detected.</div>', unsafe_allow_html=True)
            with st.expander("🔍 Outlier Detection"):
                if _dq_conv and _dq_conv in df.columns:
                    _pre_out_dq = df[df["p_date"] < _dq_post_s].copy() if _dq_post_s else df.copy()
                    _oc1, _oc2 = st.columns(2)
                    with _oc1: _dod_t_dq = st.slider("Day-over-day threshold (%)", 50, 300, 150, 25, key="dq_dod_t")
                    with _oc2: _z_t_dq = st.slider("Z-score threshold", 1.5, 4.0, 2.5, 0.5, key="dq_z_t")
                    _dod_f_dq = _pre_out_dq[_pre_out_dq[_dq_conv].pct_change().abs()*100 > _dod_t_dq]["p_date"].dt.strftime("%d %b").tolist()
                    _m_dq = _pre_out_dq[_dq_conv].mean(); _s_dq = _pre_out_dq[_dq_conv].std()
                    _z_f_dq = _pre_out_dq[abs((_pre_out_dq[_dq_conv]-_m_dq)/_s_dq) > _z_t_dq]["p_date"].dt.strftime("%d %b").tolist() if _s_dq > 0 else []
                    _q1_dq, _q3_dq = _pre_out_dq[_dq_conv].quantile(0.25), _pre_out_dq[_dq_conv].quantile(0.75)
                    _iqr_f_dq = _pre_out_dq[(_pre_out_dq[_dq_conv]<_q1_dq-1.5*(_q3_dq-_q1_dq))|(_pre_out_dq[_dq_conv]>_q3_dq+1.5*(_q3_dq-_q1_dq))]["p_date"].dt.strftime("%d %b").tolist()
                    _om1, _om2, _om3 = st.columns(3)
                    with _om1: st.markdown(f'<div class="card {"card-r" if _dod_f_dq else "card-c"}"><strong>Day-over-day</strong><br><span style="font-size:20px;font-weight:700">{len(_dod_f_dq)}</span> flags<br><div style="font-size:11px;color:{C["grey"]}">{", ".join(_dod_f_dq[:4]) if _dod_f_dq else "None"}</div></div>', unsafe_allow_html=True)
                    with _om2: st.markdown(f'<div class="card {"card-r" if _z_f_dq else "card-c"}"><strong>Z-score</strong><br><span style="font-size:20px;font-weight:700">{len(_z_f_dq)}</span> flags<br><div style="font-size:11px;color:{C["grey"]}">{", ".join(_z_f_dq[:4]) if _z_f_dq else "None"}</div></div>', unsafe_allow_html=True)
                    with _om3: st.markdown(f'<div class="card {"card-r" if _iqr_f_dq else "card-c"}"><strong>IQR</strong><br><span style="font-size:20px;font-weight:700">{len(_iqr_f_dq)}</span> flags<br><div style="font-size:11px;color:{C["grey"]}">{", ".join(_iqr_f_dq[:4]) if _iqr_f_dq else "None"}</div></div>', unsafe_allow_html=True)
                    _all_f_dq = sorted(set(_dod_f_dq + _z_f_dq + _iqr_f_dq))
                    if _all_f_dq:
                        with st.expander(f"Winsorise {_dq_conv} at 95th percentile"):
                            _p95_dq = _pre_out_dq[_dq_conv].quantile(0.95); _p05_dq = _pre_out_dq[_dq_conv].quantile(0.05)
                            st.markdown(f"Pre 5th pct: {_p05_dq:,.0f} · 95th pct: {_p95_dq:,.0f}")
                            if st.button(f"Apply winsorisation", key="dq_winsor"):
                                _df_w_dq = parse_dates(st.session_state.daily_df.copy())
                                _df_w_dq[_dq_conv] = _df_w_dq[_dq_conv].clip(_p05_dq, _p95_dq)
                                st.session_state.daily_df = _df_w_dq; st.success("Winsorised."); st.rerun()
            with st.expander("🔗 Covariate Validation"):
                if _dq_post_s:
                    _df_cv_dq = df.copy()
                    _cov_l_dq = [c for c in _df_cv_dq.columns if c not in ["p_date","prepost_1","prepost"]
                                 and _df_cv_dq[c].dtype in ["float64","int64"]
                                 and not any(k in c.lower() for k in ["cost","topview","flag","spend"])]
                    _pre_cv_dq  = _df_cv_dq[_df_cv_dq["p_date"] < _dq_post_s]
                    _post_cv_dq = _df_cv_dq[(_df_cv_dq["p_date"] >= _dq_post_s) & (_df_cv_dq["p_date"] <= _dq_post_e)] if _dq_post_e else _df_cv_dq[_df_cv_dq["p_date"] >= _dq_post_s]
                    _cv_rows_dq = []
                    for _cc in _cov_l_dq[:10]:
                        _pa_dq = _pre_cv_dq[_cc].mean(); _pb_dq = _post_cv_dq[_cc].mean() if len(_post_cv_dq) else _pa_dq
                        if _pa_dq and _pa_dq != 0:
                            _chg_dq = (_pb_dq - _pa_dq) / abs(_pa_dq) * 100
                            _p10_dq = _pre_cv_dq[_cc].quantile(0.1); _p90_dq = _pre_cv_dq[_cc].quantile(0.9)
                            _ext_dq = ((_post_cv_dq[_cc] < _p10_dq).sum() + (_post_cv_dq[_cc] > _p90_dq).sum()) / len(_post_cv_dq) * 100 if len(_post_cv_dq) else 0
                            _cv_rows_dq.append({"Covariate":_cc,"Pre avg":f"{_pa_dq:,.0f}","Post avg":f"{_pb_dq:,.0f}","Δ pre→post":f"{_chg_dq:+.1f}%","Endogeneity":"⚠️ >20%" if abs(_chg_dq)>20 else "✓","Extrapolation":f"⚠️ {_ext_dq:.0f}% out of range" if _ext_dq>20 else "✓"})
                    if _cv_rows_dq:
                        st.dataframe(pd.DataFrame(_cv_rows_dq), use_container_width=True, hide_index=True)
                else:
                    st.info("Configure pre/post periods first to enable covariate validation.")

        with t1:
            fig = go.Figure()
            s = df[target]
            fig.add_trace(go.Scatter(x=df["p_date"], y=s, mode="lines",
                line=dict(color=C["purple"],width=1.2,dash="dot"), name="Daily", opacity=0.45))
            fig.add_trace(go.Scatter(x=df["p_date"], y=s.rolling(7,center=True,min_periods=3).mean(),
                mode="lines", line=dict(color=C["purple"],width=2.5), name="7-day avg"))
            for fv in [f for f in st.session_state.flag_vars if f.get("confirmed")]:
                col_ = C["amber"]
                fig.add_vrect(x0=fv["start"], x1=fv["end"], fillcolor=col_, opacity=0.07,
                    annotation_text=fv["name"], annotation=dict(font=dict(color=col_,size=10),yanchor="top"))
            fig = add_period_vrects(fig)
            fig.update_layout(**CHART, title=dict(text=f"{target} — pre / post periods highlighted",font=dict(size=14)))
            st.plotly_chart(fig, use_container_width=True)

        with t2:
            if spend_cats:
                fig2 = go.Figure()
                for sc in spend_cats:
                    fig2.add_trace(go.Bar(x=df["p_date"], y=df[sc], name=sc.replace("Cost",""),
                        marker_color=CAT_COLORS.get(sc.replace("Cost",""),C["grey"])))
                fig2 = add_period_vrects(fig2)
                fig2.update_layout(**CHART, barmode="stack", title=dict(text="Daily spend by category",font=dict(size=14)))
                st.plotly_chart(fig2, use_container_width=True)
                total_sp = sum(df[c].sum() for c in spend_cats)
                html = '<div class="mrow">'
                for sc in spend_cats:
                    pre_sp  = pre_df[sc].sum()  if sc in pre_df.columns  else 0
                    post_sp = post_df[sc].sum()  if sc in post_df.columns else 0
                    html += f'<div class="mtile"><div class="ml">{sc.replace("Cost","")}</div><div class="mv" style="font-size:15px">${df[sc].sum():,.0f}</div><div style="font-size:10px;color:{C["grey"]}">Pre avg ${pre_sp/pre_days if pre_days else 0:,.0f}/d · Post avg ${post_sp/post_days if post_days else 0:,.0f}/d</div></div>'
                st.markdown(html+"</div>", unsafe_allow_html=True)

        with t3:
            # Correlations computed on PRE-PERIOD ONLY (analysis uses only pre-period for covariate selection)
            pre_num = pre_df.select_dtypes("number")
            pre_num = pre_num.drop(columns=[c for c in pre_num.columns if any(x in c.lower() for x in ["flag","weekend","impressions"])], errors="ignore")
            if target in pre_num.columns and len(pre_num.columns) > 1:
                corr = pre_num.corr()[target].drop(target).sort_values(ascending=False)
                fig3 = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h",
                    marker_color=[C["purple"] if v>=0 else "#2563EB" for v in corr.values]))
                fig3.update_layout(**CHART, title=dict(text=f"Pearson R with {target} — PRE-PERIOD ONLY",font=dict(size=14)),
                    height=max(280, len(corr)*30+80))
                fig3.update_xaxes(range=[-1,1], gridcolor=C["border"])
                st.plotly_chart(fig3, use_container_width=True)
                st.markdown(f'<div style="font-size:12px;color:{C["grey"]}">Correlations computed on pre-period data only ({pre_days} days) — this is what the model will see. R > 0.5 = strong covariate candidate.</div>', unsafe_allow_html=True)

        with t4:
            # Period comparison — driven by the date selectors above
            lift_color = C["purple"] if raw_lift > 0 else C["red"]
            lift_label = f"+{raw_lift:.1f}%" if raw_lift > 0 else f"{raw_lift:.1f}%"

            st.markdown(f'''<div class="card card-c">
  <div style="font-size:10px;color:{C["purple"]};font-weight:700;letter-spacing:.1em;margin-bottom:12px">
    PRE {pre_s.strftime("%d %b")}–{pre_e.strftime("%d %b %Y")} ({pre_days}d)  →  POST {post_s.strftime("%d %b")}–{post_e.strftime("%d %b %Y")} ({post_days}d)
  </div>
  <div class="mrow">
    {mtile("Pre-period avg", f"{pre_avg:,.1f}", "")}
    {mtile("Post-period avg", f"{post_avg:,.1f}", "")}
    {mtile("Raw lift (unadjusted)", lift_label, "mvc" if raw_lift > 5 else "mva" if raw_lift > 0 else "mvr")}
    {mtile("Pre:post ratio", f"{ratio:.1f}x", "mvc" if ratio_ok else "mva" if ratio_warn else "mvr")}
  </div>
  <div style="font-size:11px;color:{C["grey"]}">
    Raw lift is unadjusted — the external analysis controls for covariates to isolate the true incremental effect.
    {" ⚠ Only " + str(post_days) + " post-period days — BSTS needs at least 7, ideally 14+" if post_days < 7 else ""}
  </div>
</div>''', unsafe_allow_html=True)

            # Bar chart: pre vs post
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(
                x=["Pre-period avg", "Post-period avg"],
                y=[pre_avg, post_avg],
                marker_color=[C["grey"], C["purple"]],
                text=[f"{pre_avg:,.1f}", f"{post_avg:,.1f}"],
                textposition="outside",
                textfont=dict(color=C["text"]),
                width=0.5,
            ))
            fig4.update_layout(**CHART, showlegend=False,
                title=dict(text=f"Avg daily {target}: pre vs post", font=dict(size=14)))
            fig4.update_yaxes(range=[0, max(pre_avg, post_avg) * 1.25])
            st.plotly_chart(fig4, use_container_width=True)

            # Daily line within each period for context
            fig5 = go.Figure()
            if len(pre_df):
                fig5.add_trace(go.Scatter(x=pre_df["p_date"], y=pre_df[target],
                    mode="lines+markers", name="Pre-period",
                    line=dict(color=C["grey"], width=1.5),
                    marker=dict(size=4, color=C["grey"])))
            if len(gap_df):
                fig5.add_trace(go.Scatter(x=gap_df["p_date"], y=gap_df[target],
                    mode="lines+markers", name="Gap (excluded)",
                    line=dict(color=C["grey"], width=1, dash="dot"),
                    marker=dict(size=3, color=C["grey"]), opacity=0.4))
            if len(post_df):
                fig5.add_trace(go.Scatter(x=post_df["p_date"], y=post_df[target],
                    mode="lines+markers", name="Post-period",
                    line=dict(color=C["purple"], width=2),
                    marker=dict(size=5, color=C["purple"])))
                # Post-period mean line
                fig5.add_hline(y=pre_avg, line_dash="dash", line_color=C["grey"],
                    opacity=0.6, annotation_text=f"Pre avg {pre_avg:.0f}",
                    annotation_position="left", annotation=dict(font=dict(color=C["grey"],size=10)))
            fig5.update_layout(**CHART, title=dict(text=f"Daily {target} — pre vs post detail",font=dict(size=13)))
            st.plotly_chart(fig5, use_container_width=True)

        with t5:
            # ── Covariate selection engine ──────────────────────────────────
            st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:16px;line-height:1.7">Covariates help the model predict what conversions <em>would have been</em> without the intervention. Good covariates have high pre-period correlation with your target AND don\'t change behaviour in the post-period (endogeneity risk).</div>', unsafe_allow_html=True)

            cov_candidates = [c for c in df.select_dtypes("number").columns
                              if c != target and "Impressions" not in c and c != "CPA"]

            if len(pre_df) > 5 and len(post_df) > 0 and cov_candidates:
                rows_cov = []
                for col in cov_candidates:
                    pre_vals  = pre_df[col].dropna()
                    post_vals = post_df[col].dropna()
                    if pre_vals.std() == 0:
                        continue

                    # Pre-period Pearson R with target
                    pre_r = pre_df[[col, target]].dropna().corr().iloc[0, 1] if len(pre_df) > 3 else 0
                    r2 = pre_r ** 2

                    # Post/pre mean change — endogeneity signal
                    pre_mean  = pre_vals.mean()
                    post_mean = post_vals.mean() if len(post_vals) else pre_mean
                    pct_change = abs((post_mean - pre_mean) / pre_mean * 100) if pre_mean != 0 else 0

                    # Classify
                    if r2 < 0.03:
                        r_label, r_color = "Weak", C["grey"]
                    elif r2 < 0.15:
                        r_label, r_color = "Moderate", C["amber"]
                    else:
                        r_label, r_color = "Strong", C["purple"]

                    if pct_change < 20:
                        endo_label, endo_color = "Low risk", "#059669"
                    elif pct_change < 50:
                        endo_label, endo_color = "Medium risk", C["amber"]
                    else:
                        endo_label, endo_color = "High risk", C["red"]

                    # Auto-recommendation
                    if r2 >= 0.15 and pct_change < 20:
                        rec = "Include"
                        rec_color = "#059669"
                        rec_reason = f"Strong pre-period correlation (R²={r2:.2f}) with low endogeneity risk."
                    elif r2 >= 0.05 and pct_change < 35:
                        rec = "Consider"
                        rec_color = C["amber"]
                        rec_reason = f"Moderate correlation (R²={r2:.2f}). Include if no better alternative."
                    elif pct_change >= 50:
                        rec = "Caution"
                        rec_color = C["red"]
                        rec_reason = f"Changes {pct_change:.0f}% in post-period — may distort counterfactual prediction."
                    elif "weekend" in col.lower() or "flag" in col.lower():
                        rec = "Include"
                        rec_color = "#059669"
                        rec_reason = "Structural variable — no endogeneity risk. Captures weekly/event seasonality."
                    else:
                        rec = "Skip"
                        rec_color = C["grey"]
                        rec_reason = f"Weak correlation (R²={r2:.2f}) — unlikely to improve analysis fit."

                    # Override for structural vars
                    if "weekend" in col.lower():
                        r_label, r_color = "Structural", C["purple"]
                        endo_label, endo_color = "None", "#059669"

                    rows_cov.append(dict(
                        col=col, pre_r=pre_r, r2=r2, r_label=r_label, r_color=r_color,
                        pre_mean=pre_mean, post_mean=post_mean, pct_change=pct_change,
                        endo_label=endo_label, endo_color=endo_color,
                        rec=rec, rec_color=rec_color, rec_reason=rec_reason,
                    ))

                # Sort: Include first, then Consider, then Caution, then Skip
                rank = {"Include": 0, "Consider": 1, "Caution": 2, "Skip": 3}
                rows_cov.sort(key=lambda r: (rank.get(r["rec"], 9), -r["r2"]))

                # Init selection from recommendations
                if not st.session_state.covariate_selection:
                    st.session_state.covariate_selection = {
                        r["col"]: r["rec"] in ("Include",) for r in rows_cov
                    }

                # Render covariate cards
                for r in rows_cov:
                    is_selected = st.session_state.covariate_selection.get(r["col"], False)
                    border = f"2px solid {r['rec_color']}" if is_selected else f"1px solid {C['border']}"
                    bg = f"{r['rec_color']}08" if is_selected else C["surface"]

                    c_left, c_right = st.columns([5, 1])
                    with c_left:
                        st.markdown(f'''<div style="border:{border};background:{bg};border-radius:10px;padding:14px 18px;margin-bottom:8px">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;flex-wrap:wrap">
    <span style="font-size:14px;font-weight:700;color:{C["text"]}">{r["col"]}</span>
    <span style="font-size:11px;font-weight:600;color:{r["r_color"]};background:{r["r_color"]}12;padding:2px 8px;border-radius:4px">R²={r["r2"]:.2f} · {r["r_label"]}</span>
    <span style="font-size:11px;font-weight:600;color:{r["endo_color"]};background:{r["endo_color"]}12;padding:2px 8px;border-radius:4px">Post change +{r["pct_change"]:.0f}% · {r["endo_label"]}</span>
    <span style="font-size:11px;font-weight:700;color:{r["rec_color"]}">{r["rec"]}</span>
  </div>
  <div style="font-size:12px;color:{C["grey"]}">{r["rec_reason"]}</div>
  <div style="display:flex;gap:16px;margin-top:8px;font-size:11px;color:{C["grey"]}">
    <span>Pre avg: <strong style="color:{C["text"]}">{r["pre_mean"]:,.1f}</strong></span>
    <span>Post avg: <strong style="color:{C["text"]}">{r["post_mean"]:,.1f}</strong></span>
    <span>Pre-period R: <strong style="color:{C["text"]}">{r["pre_r"]:+.3f}</strong></span>
  </div>
</div>''', unsafe_allow_html=True)
                    with c_right:
                        st.markdown("<br>", unsafe_allow_html=True)
                        checked = st.checkbox(
                            "Use", value=is_selected,
                            key=f"cov_{r['col']}"
                        )
                        st.session_state.covariate_selection[r["col"]] = checked

                selected_covs = [k for k, v in st.session_state.covariate_selection.items() if v]
                st.markdown("<br>", unsafe_allow_html=True)
                if selected_covs:
                    cov_str = ", ".join(f"{c}" for c in selected_covs)
                    st.markdown(f'<div style="background:{C["purple"]}0A;border:1px solid {C["purple"]}33;border-radius:8px;padding:12px 16px"><div style="font-size:11px;color:{C["purple"]};font-weight:700;margin-bottom:6px">SELECTED COVARIATES ({len(selected_covs)})</div><div style="font-size:13px;color:{C["text"]}">{", ".join(selected_covs)}</div><div style="font-size:11px;color:{C["grey"]};margin-top:6px">These will appear in your config summary and export notes. Enter them as covariates when you run the analysis in the external tool.</div></div>', unsafe_allow_html=True)
                else:
                    st.warning("No covariates selected — the analysis will run as univariate (less accurate counterfactual).")
            else:
                st.info("Define your pre and post periods in the Period Comparison tab first — covariate analysis requires both periods to be set.")

    cb1, cb2 = st.columns([1,5])
    with cb1:
        if st.button("Back", key="eda_back"): st.session_state.step=1; st.rerun()
    with cb2:
        if st.button("Continue to Model →", use_container_width=True, type="primary", key="eda_fwd"):
            st.session_state.step=6; st.session_state.max_step=max(st.session_state.max_step,6); st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — FLAG VARIABLES
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 5:
    st.markdown(f'{badge("Step 5")} <span style="font-size:20px;font-weight:600;margin-left:10px">Flag Variables</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="t-meta" style="margin-bottom:18px">Confirmed flags are added as binary columns to your export. Include them as covariates when running the analysis.</div>', unsafe_allow_html=True)

    df = parse_dates(st.session_state.daily_df.copy())

    if not st.session_state.flag_vars:
        st.session_state.flag_vars = suggest_flags(df, synthesis=st.session_state.ai_synthesis)

    # Source summary
    _brief_flags = [f for f in st.session_state.flag_vars if f.get("source") == "brief"]
    _data_flags  = [f for f in st.session_state.flag_vars if f.get("source") != "brief"]
    if _brief_flags:
        st.markdown(f'<div class="t-meta" style="color:{C["purple"]};margin-bottom:10px">✓ {len(_brief_flags)} flag(s) pre-loaded from brief · {len(_data_flags)} detected from data</div>', unsafe_allow_html=True)

    for i, fv in enumerate(st.session_state.flag_vars):
        style = "card-c" if fv.get("is_treatment") else "card-a"
        label = "TREATMENT WINDOW" if fv.get("is_treatment") else "COVARIATE FLAG"
        lp = "pill-c" if fv.get("is_treatment") else "pill-a"
        src_badge = pill("from brief", "pill-c") if fv.get("source") == "brief" else pill("from data", "pill-g")
        st.markdown(f'<div class="card {style}">{pill(label,lp)} {src_badge} <strong style="margin-left:8px;font-size:14px">{fv["name"]}</strong><div class="t-meta" style="margin-top:5px">{fv["rationale"]}</div></div>', unsafe_allow_html=True)
        cn, cs, ce, cc = st.columns([3, 2, 2, 1])
        with cn:
            st.markdown(f'<div class="t-label" style="margin-bottom:3px">Name</div>', unsafe_allow_html=True)
            nn = st.text_input("Name", value=fv["name"], key=f"fn{i}", label_visibility="collapsed")
            st.session_state.flag_vars[i]["name"] = nn
        with cs:
            st.markdown(f'<div class="t-label" style="margin-bottom:3px">Start</div>', unsafe_allow_html=True)
            ns = st.date_input("Start", value=pd.to_datetime(fv["start"]).date(),
                min_value=df["p_date"].min().date(), max_value=df["p_date"].max().date(),
                key=f"fs{i}", label_visibility="collapsed")
            st.session_state.flag_vars[i]["start"] = str(ns)
        with ce:
            st.markdown(f'<div class="t-label" style="margin-bottom:3px">End</div>', unsafe_allow_html=True)
            ne = st.date_input("End", value=pd.to_datetime(fv["end"]).date(),
                min_value=df["p_date"].min().date(), max_value=df["p_date"].max().date(),
                key=f"fe{i}", label_visibility="collapsed")
            st.session_state.flag_vars[i]["end"] = str(ne)
        with cc:
            st.markdown(f'<div class="t-label" style="margin-bottom:3px">Include</div>', unsafe_allow_html=True)
            chk = st.checkbox("Include", value=fv.get("confirmed",False), key=f"fc{i}", label_visibility="collapsed")
            st.session_state.flag_vars[i]["confirmed"] = chk

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Add custom flag"):
        xa, xb, xc, xd = st.columns([2,2,2,2])
        with xa: fn = st.text_input("Flag name", placeholder="e.g. FootyFinals_WarmupFlag")
        with xb: fs = st.date_input("Start date", min_value=df["p_date"].min().date(), max_value=df["p_date"].max().date(), key="nfs")
        with xc: fe = st.date_input("End date", min_value=df["p_date"].min().date(), max_value=df["p_date"].max().date(), key="nfe")
        with xd: fr = st.text_input("Rationale", placeholder="Why flag this period?")
        if st.button("Add flag"):
            if fn:
                st.session_state.flag_vars.append(dict(name=fn,start=str(fs),end=str(fe),rationale=fr,confirmed=True,is_treatment=False))
                st.rerun()

    confirmed = [f for f in st.session_state.flag_vars if f.get("confirmed")]
    if confirmed:
        tgt = st.session_state.target_col or next((c for c in df.columns if "conversion" in c.lower()),None)
        if tgt and tgt in df.columns:
            st.markdown("<br>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["p_date"], y=df[tgt], mode="lines",
                line=dict(color=C["purple"],width=2), name=tgt))
            for fv in confirmed:
                col_ = C["purple"] if fv.get("is_treatment") else C["amber"]
                fig.add_vrect(x0=fv["start"],x1=fv["end"],fillcolor=col_,opacity=0.1,
                    annotation_text=fv["name"],annotation=dict(font=dict(color=col_,size=10),yanchor="top"))
            fig.update_layout(**CHART, title=dict(text="Confirmed flags on conversion series",font=dict(size=14)))
            st.plotly_chart(fig, use_container_width=True)

    cb1, cb2 = st.columns([1,5])
    with cb1:
        if st.button("Back", key="f5_back"): st.session_state.step=4; st.rerun()
    with cb2:
        if st.button("Continue to Model →", use_container_width=True, type="primary", key="f5_fwd"):
            st.session_state.step=6; st.session_state.max_step=max(st.session_state.max_step,6); st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — MODEL SETUP
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 6:
    st.markdown(f'<span style="font-size:20px;font-weight:600">Model Setup</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="t-meta" style="margin-bottom:18px">Configure the CausalImpact model. Parameters are pre-filled from your earlier selections — adjust if needed before running.</div>', unsafe_allow_html=True)

    df = parse_dates(st.session_state.daily_df.copy())
    confirmed = [f for f in st.session_state.flag_vars if f.get("confirmed")]
    is_sot = st.session_state.file_type == "sot"

    # ── Compact flag variables section ─────────────────────────────────────────
    with st.expander(f"Flag Variables ({len(confirmed)} confirmed)" + (" — edit before running" if not confirmed else ""), expanded=not confirmed):
        st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:10px">Confirmed flags are added as binary covariates to the model. Use the full editor via <strong>5 · Model → Flags</strong> in the sidebar.</div>', unsafe_allow_html=True)
        _df_flags = parse_dates(st.session_state.daily_df.copy())
        if not st.session_state.flag_vars:
            st.session_state.flag_vars = suggest_flags(_df_flags, synthesis=None)
        for _fi, _fv in enumerate(st.session_state.flag_vars):
            _fc1, _fc2, _fc3, _fc4 = st.columns([3, 2, 2, 1])
            with _fc1:
                st.markdown(f'<div style="font-size:13px;font-weight:600;color:{C["text"]};padding-top:28px">{_fv["name"]}</div>', unsafe_allow_html=True)
            with _fc2:
                _ns = st.date_input("Start", value=pd.to_datetime(_fv["start"]).date(),
                    min_value=_df_flags["p_date"].min().date(), max_value=_df_flags["p_date"].max().date(),
                    key=f"m6_fs{_fi}", label_visibility="visible")
                st.session_state.flag_vars[_fi]["start"] = str(_ns)
            with _fc3:
                _ne = st.date_input("End", value=pd.to_datetime(_fv["end"]).date(),
                    min_value=_df_flags["p_date"].min().date(), max_value=_df_flags["p_date"].max().date(),
                    key=f"m6_fe{_fi}", label_visibility="visible")
                st.session_state.flag_vars[_fi]["end"] = str(_ne)
            with _fc4:
                _chk = st.checkbox("Use", value=_fv.get("confirmed", False), key=f"m6_fc{_fi}")
                st.session_state.flag_vars[_fi]["confirmed"] = _chk
        _fa1, _fa2 = st.columns(2)
        with _fa1:
            _new_flag_name = st.text_input("Add flag name", placeholder="e.g. Easter_2024", key="m6_fn")
        with _fa2:
            _new_flag_dates = st.date_input("Date range", value=(_df_flags["p_date"].min().date(), _df_flags["p_date"].min().date()), key="m6_fd")
        if st.button("Add flag", key="m6_fadd") and _new_flag_name:
            _s, _e = (_new_flag_dates if len(_new_flag_dates) == 2 else (_new_flag_dates[0], _new_flag_dates[0]))
            st.session_state.flag_vars.append(dict(name=_new_flag_name, start=str(_s), end=str(_e), rationale="", confirmed=True, is_treatment=False))
            st.rerun()
        confirmed = [f for f in st.session_state.flag_vars if f.get("confirmed")]

    for fv in confirmed:
        df[fv["name"]] = ((df["p_date"] >= pd.to_datetime(fv["start"])) & (df["p_date"] <= pd.to_datetime(fv["end"]))).astype(int)

    # Smoothing toggle (TTAM only)
    smooth = False
    w = 7
    if not is_sot:
        cs, cw = st.columns([3, 1])
        target_base = st.session_state.target_col or next((c for c in df.columns if "conversion" in c.lower()), None)
        with cs:
            smooth = st.toggle(f"Apply rolling smoothing to target variable", value=st.session_state.smoothing)
            st.session_state.smoothing = smooth
        with cw:
            w = st.number_input("Window", value=st.session_state.smooth_window, min_value=3, max_value=14, step=1) if smooth else 7
            st.session_state.smooth_window = w
        if smooth and target_base:
            df[f"{target_base}_Smoothed"] = df[target_base].rolling(w, center=True, min_periods=3).mean()
            fig_sm = go.Figure()
            fig_sm.add_trace(go.Scatter(x=df["p_date"], y=df[target_base], mode="lines",
                line=dict(color=C["grey"], width=1, dash="dot"), name="Raw", opacity=0.4))
            fig_sm.add_trace(go.Scatter(x=df["p_date"], y=df[f"{target_base}_Smoothed"], mode="lines",
                line=dict(color=C["purple"], width=2.5), name=f"{w}-day smoothed"))
            fig_sm.update_layout(**CHART, title=dict(text="Raw vs smoothed", font=dict(size=14)))
            st.plotly_chart(fig_sm, use_container_width=True)

    num_cols = [c for c in df.columns if c != "p_date" and pd.api.types.is_numeric_dtype(df[c])]

    # Hypothesis-aware column scoring
    _hyp_cfg = HYPOTHESIS_CONFIG.get(st.session_state.hypothesis or "", {})
    _tgt_kws = _hyp_cfg.get("target_keywords", [])
    _cov_kws = _hyp_cfg.get("covariate_keywords", [])

    def _col_target_score(c):
        cl = c.lower()
        for i, kw in enumerate(_tgt_kws):
            if kw.lower() in cl:
                return i
        return len(_tgt_kws) + 1

    def _col_cov_score(c):
        cl = c.lower()
        for i, kw in enumerate(_cov_kws):
            if kw.lower() in cl:
                return i
        return len(_cov_kws) + 1

    num_cols_sorted_target = sorted(num_cols, key=_col_target_score)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="t-section" style="margin-bottom:6px">Model Configuration</div>', unsafe_allow_html=True)
    if _hyp_cfg:
        _hyp_name = next((u["name"] for u in USE_CASES if u["id"] == st.session_state.hypothesis), "")
        _int_label = _hyp_cfg.get("intervention_label", "")
        st.markdown(f'<div class="t-meta" style="margin-bottom:14px">Hypothesis: <strong>{_hyp_name}</strong> · Intervention: {_int_label}</div>', unsafe_allow_html=True)

    mc1, mc2 = st.columns(2)
    with mc1:
        default_target = st.session_state.target_col or (num_cols_sorted_target[0] if num_cols_sorted_target else None)
        if smooth and default_target:
            smoothed_name = f"{default_target}_Smoothed"
            target_options = [smoothed_name] + [c for c in num_cols_sorted_target if c != smoothed_name]
        else:
            target_options = num_cols_sorted_target
        _target_idx = target_options.index(default_target) if default_target in target_options else 0
        target_sel = st.selectbox("Dependent variable (Y)", target_options, index=_target_idx, key="model_target")
        if target_sel != st.session_state.target_col:
            st.session_state.ci_placebo_results = None
            st.session_state.ci_placebo_extended = False
        st.session_state.target_col = target_sel

        flag_cols  = [f["name"] for f in confirmed]
        spend_cols = [c for c in num_cols if c.endswith("Cost") and c != "TotalCost"]
        # hypothesis-aware covariate defaults: prefer columns matching covariate keywords
        hyp_cov_matches = [c for c in num_cols if _col_cov_score(c) < len(_cov_kws) + 1 and c != target_sel]
        default_covs = list(dict.fromkeys(hyp_cov_matches + [c for c in spend_cols + flag_cols if c in num_cols and c != target_sel]))
        default_covs = [c for c in default_covs if c != target_sel]
        # Only pre-select if user hasn't made a manual choice yet
        if st.session_state.ci_covariates:
            default_covs = [c for c in st.session_state.ci_covariates if c in num_cols and c != target_sel]
        cov_options  = [c for c in num_cols if c != target_sel]
        covariates_sel = st.multiselect("Covariates", cov_options, default=default_covs, key="model_covs")
        if set(covariates_sel) != set(st.session_state.ci_covariates or []):
            st.session_state.ci_placebo_results = None
            st.session_state.ci_placebo_extended = False
        st.session_state.ci_covariates = covariates_sel

    with mc2:
        date_min = df["p_date"].min().date()
        date_max = df["p_date"].max().date()
        pre_s_def  = st.session_state.pre_start  or date_min
        pre_e_def  = st.session_state.pre_end    or date_min
        post_s_def = st.session_state.post_start or date_min
        post_e_def = st.session_state.post_end   or date_max

        pre_range  = st.date_input("Pre-period",  value=(pre_s_def,  pre_e_def),  min_value=date_min, max_value=date_max, key="model_pre")
        post_range = st.date_input("Post-period", value=(post_s_def, post_e_def), min_value=date_min, max_value=date_max, key="model_post")
        if len(pre_range) == 2:
            if (pre_range[0], pre_range[1]) != (st.session_state.pre_start, st.session_state.pre_end):
                st.session_state.ci_placebo_results = None
                st.session_state.ci_placebo_extended = False
            st.session_state.pre_start, st.session_state.pre_end = pre_range
        if len(post_range) == 2: st.session_state.post_start, st.session_state.post_end = post_range

    # Validation warnings
    pre_days_m  = (pd.Timestamp(st.session_state.pre_end)  - pd.Timestamp(st.session_state.pre_start)).days  + 1 if st.session_state.pre_start and st.session_state.pre_end else 0
    post_days_m = (pd.Timestamp(st.session_state.post_end) - pd.Timestamp(st.session_state.post_start)).days + 1 if st.session_state.post_start and st.session_state.post_end else 0
    w_html = ""
    if pre_days_m < 30:
        w_html += f'<div class="t-meta" style="color:{C["amber"]};margin-bottom:4px">⚠ Pre-period is {pre_days_m} days — minimum recommended is 30.</div>'
    if pre_days_m > 0 and post_days_m > 0 and pre_days_m < 2 * post_days_m:
        w_html += f'<div class="t-meta" style="color:{C["amber"]};margin-bottom:4px">⚠ Pre/post ratio is {pre_days_m/post_days_m:.1f}× — recommended ≥ 2×.</div>'
    if w_html:
        st.markdown(f'<div style="background:{C["amber"]}11;border:1px solid {C["amber"]}44;border-radius:8px;padding:10px 14px;margin:12px 0">{w_html}</div>', unsafe_allow_html=True)

    cov_str = ", ".join(covariates_sel) if covariates_sel else "None (univariate)"
    st.markdown(f"""<div class="card card-c" style="margin-top:16px">
<div class="t-label" style="margin-bottom:10px">MODEL SPEC</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px 24px">
  <div><div class="t-label" style="margin-bottom:3px">Target (Y)</div><strong class="t-body" style="color:{C['green']}">{target_sel}</strong></div>
  <div><div class="t-label" style="margin-bottom:3px">Pre-period</div><strong class="t-body">{str(st.session_state.pre_start)} → {str(st.session_state.pre_end)} ({pre_days_m}d)</strong></div>
  <div><div class="t-label" style="margin-bottom:3px">Post-period</div><strong class="t-body">{str(st.session_state.post_start)} → {str(st.session_state.post_end)} ({post_days_m}d)</strong></div>
  <div style="grid-column:span 3"><div class="t-label" style="margin-bottom:3px">Covariates</div><span class="t-body">{cov_str}</span></div>
</div>
<div class="t-meta" style="margin-top:10px">BSTS · niter=1000 · nseasons=7 · weekly seasonality</div>
</div>""", unsafe_allow_html=True)

    # ── Placebo Tests ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="t-section" style="margin-bottom:6px">Placebo Tests</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="t-meta" style="margin-bottom:14px">Creates imaginary interventions at equally spaced cut-points inside the pre-period. A stable model should find <strong>no significant effect</strong> (p ≥ 0.05) at every cut. Mixed results may indicate noise or a structural trend — run additional iterations to diagnose.</div>', unsafe_allow_html=True)

    can_placebo = (pre_days_m >= 30)
    if not can_placebo:
        st.markdown(f'<div style="color:{C["grey_l"]};font-size:12px;padding:8px 12px;background:{C["surface2"]};border-radius:8px;border:1px solid {C["border"]}">Pre-period must be ≥ 30 days to run placebo tests ({pre_days_m} days selected).</div>', unsafe_allow_html=True)
    else:
        pb1, pb2 = st.columns([1, 3])
        with pb1:
            run_placebo = st.button("▶ Run Placebo Tests", use_container_width=True, key="btn_placebo")
        with pb2:
            if st.session_state.ci_placebo_results:
                passed_n = sum(1 for r in st.session_state.ci_placebo_results if r['passed'] is True)
                failed_n = sum(1 for r in st.session_state.ci_placebo_results if r['passed'] is False)
                total_n  = len(st.session_state.ci_placebo_results)
                if failed_n == 0:
                    verdict_html = f'<span style="color:{C["green"]};font-weight:700">✓ ROBUST</span> — all {total_n} placebo tests passed. Pre-period is stable.'
                elif passed_n == 0:
                    verdict_html = f'<span style="color:{C["red"]};font-weight:700">✗ NOT STABLE</span> — all {total_n} tests detected a spurious effect. Check for a structural break in the pre-period.'
                else:
                    verdict_html = f'<span style="color:{C["amber"]};font-weight:700">⚠ MIXED</span> — {passed_n}/{total_n} tests passed. May be noise; run additional iterations to confirm.'
                st.markdown(f'<div style="font-size:13px;padding:8px 0">{verdict_html}</div>', unsafe_allow_html=True)

        if run_placebo:
            st.session_state.ci_placebo_extended = False
            with st.spinner("Running placebo tests… (3 cuts × ~15 s each)"):
                st.session_state.ci_placebo_results = run_placebo_tests(
                    df, target_sel, covariates_sel,
                    st.session_state.pre_start, st.session_state.pre_end,
                )
            st.rerun()

        if st.session_state.ci_placebo_results:
            rows = st.session_state.ci_placebo_results
            passed_n = sum(1 for r in rows if r['passed'] is True)
            failed_n = sum(1 for r in rows if r['passed'] is False)

            # Results table
            hdr = f"""<div style="display:grid;grid-template-columns:110px 80px 80px 110px 90px 80px;gap:0;background:{C['surface2']};border:1px solid {C['border']};border-radius:12px 12px 0 0;padding:8px 14px;">
  <span style="font-size:10px;font-weight:700;color:{C['grey']};letter-spacing:.06em">CUT DATE</span>
  <span style="font-size:10px;font-weight:700;color:{C['grey']};letter-spacing:.06em">PRE DAYS</span>
  <span style="font-size:10px;font-weight:700;color:{C['grey']};letter-spacing:.06em">POST DAYS</span>
  <span style="font-size:10px;font-weight:700;color:{C['grey']};letter-spacing:.06em">REL. EFFECT</span>
  <span style="font-size:10px;font-weight:700;color:{C['grey']};letter-spacing:.06em">P-VALUE</span>
  <span style="font-size:10px;font-weight:700;color:{C['grey']};letter-spacing:.06em">RESULT</span>
</div>"""
            body_rows = ""
            for i, r in enumerate(rows):
                is_last = (i == len(rows) - 1)
                br = "0 0 12px 12px" if is_last else "0"
                rel_s  = f"{r['rel_effect_pct']:+.1f}%" if r['rel_effect_pct'] is not None else "—"
                pv_s   = f"{r['p_value']:.3f}" if r['p_value'] is not None else "—"
                if r['passed'] is True:
                    badge_s = f'<span style="background:{C["green"]}22;color:{C["green"]};font-weight:700;font-size:11px;padding:2px 10px;border-radius:999px">PASS</span>'
                elif r['passed'] is False:
                    badge_s = f'<span style="background:{C["red"]}22;color:{C["red"]};font-weight:700;font-size:11px;padding:2px 10px;border-radius:999px">FAIL</span>'
                else:
                    badge_s = f'<span style="background:{C["grey_l"]}22;color:{C["grey_l"]};font-weight:700;font-size:11px;padding:2px 10px;border-radius:999px">ERROR</span>'
                if r['error']:
                    rel_s = pv_s = "—"
                body_rows += f"""<div style="display:grid;grid-template-columns:110px 80px 80px 110px 90px 80px;gap:0;border:1px solid {C['border']};border-top:none;border-radius:{br};padding:10px 14px;background:{C['surface']};font-size:12px;align-items:center">
  <span style="font-weight:600">{r['cut_date']}</span>
  <span style="color:{C['grey']}">{r['pre_days']}</span>
  <span style="color:{C['grey']}">{r['post_days']}</span>
  <span style="color:{C['text']}">{rel_s}</span>
  <span style="color:{C['text']}">{pv_s}</span>
  {badge_s}
</div>"""
            st.markdown(hdr + body_rows, unsafe_allow_html=True)

            # Additional tests if mixed
            mixed = passed_n > 0 and failed_n > 0
            if mixed and not st.session_state.ci_placebo_extended:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:12px;color:{C["amber"]};margin-bottom:8px">Mixed results detected. Run additional iterations at different cut-points to determine whether this is random noise or a fundamental trend shift.</div>', unsafe_allow_html=True)
                if st.button("▶ Run Additional Placebo Tests", key="btn_placebo_ext"):
                    with st.spinner("Running additional placebo tests…"):
                        extra = run_placebo_tests(
                            df, target_sel, covariates_sel,
                            st.session_state.pre_start, st.session_state.pre_end,
                            fractions=[0.25, 7/12, 0.75],
                        )
                    st.session_state.ci_placebo_results = rows + extra
                    st.session_state.ci_placebo_extended = True
                    st.rerun()

    # ── Run Model Buttons ──────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    cb1, cb2, cb3 = st.columns([1, 2, 2])
    with cb1:
        if st.button("Back", key="m6_back"): st.session_state.step = 4; st.rerun()
    with cb2:
        run_model = st.button("▶ Run CausalImpact Model", type="primary", use_container_width=True)
    with cb3:
        out_csv = df.copy(); out_csv["p_date"] = out_csv["p_date"].dt.strftime("%Y-%m-%d")
        safe_n = (st.session_state.advertiser or "advertiser").replace(" ", "_")
        fname_csv = f"{safe_n}_AnalysisInput_{datetime.today().strftime('%Y%m%d')}.csv"
        st.download_button("↓ Export CSV instead", data=out_csv.to_csv(index=False).encode("utf-8"),
                           file_name=fname_csv, mime="text/csv", use_container_width=True)

    if run_model:
        if not (st.session_state.pre_start and st.session_state.pre_end and
                st.session_state.post_start and st.session_state.post_end):
            st.error("Please set pre and post periods before running.")
        else:
            with st.spinner("Running BSTS model… (~15–30 seconds)"):
                ci_obj, ci_err = run_causal_impact(
                    df, target_sel, covariates_sel,
                    st.session_state.pre_start, st.session_state.pre_end,
                    st.session_state.post_start, st.session_state.post_end
                )
            if ci_err:
                st.error(f"Model error: {ci_err}")
                if "not installed" in ci_err:
                    st.code("pip3 install causalimpact")
            else:
                ape_tbl, mape_val = compute_pre_ape_table(ci_obj, st.session_state.pre_start, st.session_state.pre_end)
                pv = getattr(ci_obj, '_extracted_p_value', None) or getattr(ci_obj, 'p_value', None)
                sigs = build_diagnostic_signals(
                    ci_obj, ape_tbl, mape_val, pv,
                    st.session_state.pre_start, st.session_state.pre_end,
                    st.session_state.post_start, st.session_state.post_end
                )
                try:
                    sd = ci_obj.summary_data
                    avg_d = sd.get('average', {}) if isinstance(sd, dict) else {}
                    cum_d = sd.get('cumulative', {}) if isinstance(sd, dict) else {}
                except Exception:
                    avg_d, cum_d = {}, {}

                st.session_state.ci_result       = ci_obj
                st.session_state.ci_mape          = mape_val
                st.session_state.ci_ape_df        = ape_tbl
                st.session_state.ci_p_value       = pv
                st.session_state.ci_signals       = sigs
                st.session_state.ci_summary_data  = {'average': avg_d, 'cumulative': cum_d}
                st.session_state.ci_llm_diagnosis = None
                st.session_state.ci_narrator      = None
                st.session_state.ci_run_count    += 1
                st.session_state.step = 7
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — RESULTS & DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 7:
    st.markdown(f'{badge("Step 7")} <span style="font-size:20px;font-weight:600;margin-left:10px">Results & Diagnostics</span>', unsafe_allow_html=True)

    ci_r   = st.session_state.ci_result
    mape_r = st.session_state.ci_mape
    pval_r = st.session_state.ci_p_value
    ape_r  = st.session_state.ci_ape_df
    sigs_r = st.session_state.ci_signals or []

    if ci_r is None:
        st.warning("No model result found. Go back and run the model.")
        if st.button("← Back to Model Setup"): st.session_state.step = 6; st.rerun()
    else:
        tgt_name = st.session_state.target_col or "target"

        # ── Save brief to knowledge base (once per session, on first results view) ──
        if st.session_state.brief_text and not st.session_state.get("_kb_saved"):
            try:
                save_brief_to_knowledge_base(
                    advertiser=st.session_state.advertiser or "Unknown",
                    hypothesis_type="incrementality",
                    brief_text=st.session_state.brief_text or "",
                    synthesis={},
                    outcome_notes=f"MAPE={st.session_state.ci_mape:.1f}%, p={st.session_state.ci_p_value:.4f}" if st.session_state.ci_mape and st.session_state.ci_p_value else "",
                )
            except Exception:
                pass
            st.session_state["_kb_saved"] = True

        # ── Summary tiles ──
        try:
            sd_r  = st.session_state.ci_summary_data or {}
            avg_r = sd_r.get('average', {})
            cum_r = sd_r.get('cumulative', {})
            actual_avg  = avg_r.get('actual', None)
            pred_avg    = avg_r.get('predicted', None)
            abs_eff_avg = avg_r.get('abs_effect', None)
            rel_eff     = avg_r.get('rel_effect', None)
            cum_effect  = cum_r.get('abs_effect', None)
            # Fallback: compute directly from inferences if summary_data empty
            if actual_avg is None and ci_r is not None:
                inf = ci_r.inferences
                obs  = next((c for c in ['y','response','observed','actual']       if c in inf.columns), None)
                pred = next((c for c in ['y_pred','point_pred','predicted','yhat'] if c in inf.columns), None)
                post_inf = inf[inf.index >= pd.Timestamp(st.session_state.post_start)]
                if obs:  actual_avg  = float(pd.to_numeric(post_inf[obs],  errors='coerce').mean())
                if pred: pred_avg    = float(pd.to_numeric(post_inf[pred], errors='coerce').mean())
                if obs and pred:
                    abs_eff_avg = actual_avg - pred_avg if actual_avg and pred_avg else None
                    rel_eff     = (abs_eff_avg / pred_avg) if pred_avg and pred_avg != 0 else None
                    cum_effect  = float(pd.to_numeric(post_inf[obs] - post_inf[pred], errors='coerce').sum())
        except Exception:
            actual_avg = pred_avg = abs_eff_avg = rel_eff = cum_effect = None

        def fmt_v(v, dec=1, pct=False, sign=False):
            if v is None: return "—"
            if pct: return f"{v*100:+.{dec}f}%" if sign else f"{v*100:.{dec}f}%"
            return f"{v:+,.{dec}f}" if sign else f"{v:,.{dec}f}"

        sig_c  = C["green"]  if (pval_r is not None and pval_r <= 0.05) else C["amber"]
        mape_c = C["green"]  if (mape_r is not None and mape_r <= 10)  else C["amber"] if (mape_r is not None and mape_r <= 15) else C["red"]
        rel_c  = C["green"]  if rel_eff is not None and rel_eff > 0    else C["red"]

        st.markdown(f"""<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:20px">
  <div class="card" style="text-align:center;padding:14px 8px"><div style="font-size:10px;color:{C['grey']};text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px">Avg Actual</div><div style="font-size:22px;font-weight:700;color:{C['text']}">{fmt_v(actual_avg)}</div></div>
  <div class="card" style="text-align:center;padding:14px 8px"><div style="font-size:10px;color:{C['grey']};text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px">Avg Predicted</div><div style="font-size:22px;font-weight:700;color:{C['text']}">{fmt_v(pred_avg)}</div></div>
  <div class="card" style="text-align:center;padding:14px 8px"><div style="font-size:10px;color:{C['grey']};text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px">Avg Daily Lift</div><div style="font-size:22px;font-weight:700;color:{rel_c}">{fmt_v(abs_eff_avg, sign=True)}</div></div>
  <div class="card" style="text-align:center;padding:14px 8px"><div style="font-size:10px;color:{C['grey']};text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px">Relative Lift</div><div style="font-size:22px;font-weight:700;color:{rel_c}">{fmt_v(rel_eff, pct=True, sign=True)}</div></div>
  <div class="card" style="text-align:center;padding:14px 8px"><div style="font-size:10px;color:{C['grey']};text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px">p-value</div><div style="font-size:22px;font-weight:700;color:{sig_c}">{f'{pval_r:.4f}' if pval_r is not None else '—'}</div></div>
  <div class="card" style="text-align:center;padding:14px 8px"><div style="font-size:10px;color:{C['grey']};text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px">Pre MAPE</div><div style="font-size:22px;font-weight:700;color:{mape_c}">{f'{mape_r:.1f}%' if mape_r is not None else '—'}</div></div>
</div>""", unsafe_allow_html=True)

        # ── Three-panel chart ──
        try:
            fig_ci = build_three_panel_chart(ci_r, st.session_state.post_start, tgt_name)
            st.plotly_chart(fig_ci, use_container_width=True)
        except Exception as chart_err:
            st.error(f"Chart error: {chart_err}")

        # ── APE table + LLM diagnostics ──
        col_l, col_r = st.columns([1, 1])

        with col_l:
            st.markdown(f'<div style="font-size:13px;font-weight:600;color:{C["text"]};margin-bottom:8px">Pre-period Daily APE</div>', unsafe_allow_html=True)
            if ape_r is not None and not ape_r.empty:
                def _ape_style(val):
                    try:
                        v = float(val)
                        if v > 25: return f'color: {C["red"]}'
                        if v > 15: return f'color: {C["amber"]}'
                        return f'color: {C["green"]}'
                    except: return ''
                st.dataframe(ape_r.style.map(_ape_style, subset=['APE (%)']),
                             use_container_width=True, height=340, hide_index=True)
                st.markdown(f'<div style="font-size:12px;color:{mape_c};margin-top:4px;font-weight:600">Overall MAPE: {mape_r:.2f}%</div>', unsafe_allow_html=True)

        with col_r:
            st.markdown(f'<div style="font-size:13px;font-weight:600;color:{C["text"]};margin-bottom:8px">AI Diagnostic Layer</div>', unsafe_allow_html=True)

            critical_s = [s for s in sigs_r if s.get('severity') == 'critical']
            warn_s     = [s for s in sigs_r if s.get('severity') == 'warning']
            all_issues = critical_s + warn_s

            if not all_issues:
                st.markdown(f'<div style="background:{C["green"]}11;border:1px solid {C["green"]}33;border-radius:8px;padding:14px 16px"><div style="font-size:13px;font-weight:600;color:{C["green"]};margin-bottom:4px">✓ Model looks clean</div><div style="font-size:12px;color:{C["grey"]}">No critical issues detected. MAPE and significance are within acceptable thresholds.</div></div>', unsafe_allow_html=True)
            else:
                for sig in all_issues:
                    sc = C["red"] if sig.get('severity') == 'critical' else C["amber"]
                    st.markdown(f'<div style="border-left:3px solid {sc};padding:8px 12px;margin-bottom:6px;background:{sc}11;border-radius:0 6px 6px 0"><div style="font-size:11px;color:{sc};font-weight:600;text-transform:uppercase;letter-spacing:.06em">{sig.get("type","").replace("_"," ")}</div><div style="font-size:12px;color:{C["text"]};margin-top:2px">{sig.get("message","")}</div></div>', unsafe_allow_html=True)

        # ── Bottom nav ──
        st.markdown("<br>", unsafe_allow_html=True)
        bn1, bn2, bn3 = st.columns([1, 2, 2])
        with bn1:
            if st.button("Back"): st.session_state.step = 6; st.rerun()
        with bn2:
            if st.button("↺ Re-run with different config", use_container_width=True):
                st.session_state.ci_result = None; st.session_state.step = 6; st.rerun()
        with bn3:
            df_exp = parse_dates(st.session_state.daily_df.copy())
            for fv in [f for f in st.session_state.flag_vars if f.get("confirmed")]:
                df_exp[fv["name"]] = ((df_exp["p_date"] >= pd.to_datetime(fv["start"])) & (df_exp["p_date"] <= pd.to_datetime(fv["end"]))).astype(int)
            df_exp["p_date"] = df_exp["p_date"].dt.strftime("%Y-%m-%d")
            safe_e = (st.session_state.advertiser or "advertiser").replace(" ", "_")
            fname_e = f"{safe_e}_CausalInput_{datetime.today().strftime('%Y%m%d')}.csv"
            st.download_button("↓ Export clean CSV", data=df_exp.to_csv(index=False).encode("utf-8"),
                               file_name=fname_e, mime="text/csv", use_container_width=True)
