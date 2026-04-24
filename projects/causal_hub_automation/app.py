"""
TikTok Causal Hub — MVP
Upload raw TTAM or SOT data → clean & aggregate → quality check → EDA → configure analysis → export ready-to-use CSV
"""

import io, os, json, warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI

warnings.filterwarnings("ignore")

ARK_BASE_URL = "https://ark.ap-southeast.byteplusapi.com/api/v3/"

_ARK_KEY_FROM_SECRETS = ""
_ARK_EP_FROM_SECRETS = ""
try:
    _ARK_KEY_FROM_SECRETS = st.secrets.get("ARK_API_KEY", "")
    _ARK_EP_FROM_SECRETS = st.secrets.get("ARK_ENDPOINT_ID", "")
except Exception:
    pass

# ── Palette ────────────────────────────────────────────────────────────────────
C = dict(
    bg="#F7F7FB",        # light lavender-grey page background
    surface="#FFFFFF",   # white cards
    surface2="#F0EEF8",  # very light purple tint for inputs
    border="#E2DEEF",    # soft purple-grey border
    purple="#7C3AED",    # primary accent — strong purple
    purple_l="#A78BFA",  # lighter purple for chart predicted line
    purple_xl="#EDE9FE", # very light purple for fills/badges
    text="#1A1A2E",      # near-black with purple tint
    grey="#6B7280",      # mid grey for labels
    grey_l="#9CA3AF",    # light grey
    red="#DC2626",       # errors
    amber="#D97706",     # warnings
    green="#059669",     # success
    white="#FFFFFF",
)
CAT_COLORS = {
    "TopView":    "#7C3AED",  # purple
    "Brand_ExTV": "#A78BFA",  # light purple
    "MidFunnel":  "#2563EB",  # blue
    "Performance":"#059669",  # green
    "Other":      "#9CA3AF",  # grey
}
# Chart palette using the spec provided
CHART = dict(
    paper_bgcolor=C["surface"], plot_bgcolor=C["surface"],
    font=dict(color=C["text"], family="Inter, Arial, sans-serif", size=12),
    xaxis=dict(gridcolor="#E5E7EB", showgrid=True, zeroline=False,
               tickfont=dict(color=C["grey"]), linecolor="#E2DEEF"),
    yaxis=dict(gridcolor="#E5E7EB", showgrid=True, zeroline=False,
               tickfont=dict(color=C["grey"]), linecolor="#E2DEEF"),
    margin=dict(l=50, r=30, t=45, b=45),
    legend=dict(bgcolor=C["surface"], bordercolor=C["border"], borderwidth=1),
    hovermode="x unified",
)

st.set_page_config(page_title="TikTok Causal Hub", page_icon="🎯",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{{font-family:'Inter',sans-serif;background:{C['bg']};color:{C['text']};}}
.stApp{{background:{C['bg']};}}
section[data-testid="stSidebar"]{{background:{C['surface']};border-right:1px solid {C['border']};box-shadow:2px 0 8px rgba(124,58,237,0.06);}}
.card{{background:{C['surface']};border:1px solid {C['border']};border-radius:12px;padding:20px 24px;margin-bottom:14px;box-shadow:0 1px 4px rgba(124,58,237,0.07);}}
.card-c{{border-left:3px solid {C['purple']};}}
.card-a{{border-left:3px solid {C['amber']};}}
.card-r{{border-left:3px solid {C['red']};}}
.card-g{{border-left:3px solid {C['grey_l']};}}
.mrow{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;}}
.mtile{{background:{C['surface']};border:1px solid {C['border']};border-radius:8px;padding:14px 18px;flex:1;min-width:120px;box-shadow:0 1px 3px rgba(124,58,237,0.06);}}
.ml{{font-size:10px;color:{C['grey']};text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;}}
.mv{{font-size:22px;font-weight:700;color:{C['text']};}}
.mvc{{color:{C['purple']};}} .mva{{color:{C['amber']};}} .mvr{{color:{C['red']};}}
.pill{{display:inline-block;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:600;margin:2px;}}
.pill-r{{background:#FEE2E2;color:{C['red']};border:1px solid #FECACA;}}
.pill-a{{background:#FEF3C7;color:{C['amber']};border:1px solid #FDE68A;}}
.pill-c{{background:{C['purple_xl']};color:{C['purple']};border:1px solid #C4B5FD;}}
.pill-g{{background:#F3F4F6;color:{C['grey']};border:1px solid #E5E7EB;}}
.sbadge{{background:{C['purple']};color:#fff;font-weight:700;font-size:11px;padding:3px 10px;border-radius:20px;}}
.stButton>button{{background:{C['purple']};color:#fff;border:none;border-radius:8px;font-weight:600;transition:all .15s;}}
.stButton>button:hover{{background:#6D28D9;color:#fff;}}
div[data-baseweb="select"]>div{{background:{C['surface']}!important;border-color:{C['border']}!important;color:{C['text']}!important;}}
.stTextArea textarea{{background:{C['surface']}!important;border-color:{C['border']}!important;color:{C['text']}!important;}}
.stTextInput input{{background:{C['surface']}!important;border-color:{C['border']}!important;color:{C['text']}!important;}}
.stMultiSelect [data-baseweb="tag"]{{background:{C['purple_xl']}!important;color:{C['purple']}!important;}}
.stTabs [data-baseweb="tab"]{{color:{C['grey']};}}
.stTabs [aria-selected="true"]{{color:{C['purple']};font-weight:600;border-bottom:2px solid {C['purple']};}}
hr{{border-color:{C['border']};}}
label{{color:{C['text']}!important;}}
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
        colour="#7C3AED",
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
        colour="#7C3AED",
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
        for ch in ch_order:
            s_col = next((c for c in df.columns if c.startswith(ch) and "Sessions" in c), None)
            c_col = next((c for c in df.columns if c.startswith(ch) and "Conversions" in c), None)
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


def call_ark_eda(api_key, endpoint_id, summary_text, hypothesis, advertiser):
    if not api_key:
        return None, "No Ark API key in sidebar."
    if not endpoint_id:
        return None, "No endpoint ID in sidebar."
    try:
        client = OpenAI(base_url=ARK_BASE_URL, api_key=api_key)
        hyp_name = next((u["name"] for u in USE_CASES if u["id"] == hypothesis), hypothesis or "unspecified")
        prompt = (
            "You are a senior TikTok measurement analyst reviewing EDA for a causal study.\n\n"
            f"ADVERTISER: {advertiser or 'Unknown'}\nHYPOTHESIS: {hyp_name}\n\nEDA SUMMARY:\n{summary_text}\n\n"
            "Provide analysis in this exact structure:\n\n"
            "**What the data shows**\n2-3 sentences on key signals. Be specific — name channels and percentages.\n\n"
            "**Strongest evidence for TikTok impact**\nThe 1-2 most compelling signals.\n\n"
            "**Watch outs before modelling**\nUp to 3 specific risks visible in the data. Be direct.\n\n"
            "**Recommendation**\nOne sentence: ready to model, or needs attention first? If attention needed, say exactly what."
        )
        response = client.chat.completions.create(
            model=endpoint_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=800
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, str(e)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in dict(
    raw_df=None, daily_df=None, file_type=None,
    causal_context="", parsed_context={},
    l4_map={}, conversion_events=[], target_col=None,
    quality_signals=[], flag_vars=[], smoothing=False, smooth_window=7,
    advertiser="", step=0, ark_key=_ARK_KEY_FROM_SECRETS, ark_endpoint=_ARK_EP_FROM_SECRETS, hypothesis=None, max_step=0,
    pre_start=None, pre_end=None, post_start=None, post_end=None,
    covariate_selection={},
    sot_targets=[], sot_treatment_start=None, sot_treatment_end=None,
    sot_channel_roles={}, cpa_inputs={}, sot_selected_targets=[], llm_eda_result=None,
).items():
    if k not in st.session_state:
        st.session_state[k] = v



# ── Helpers ────────────────────────────────────────────────────────────────────
def pill(t, c="pill-g"): return f'<span class="pill {c}">{t}</span>'
def badge(t): return f'<span class="sbadge">{t}</span>'
def mtile(lbl, val, cls=""): return f'<div class="mtile"><div class="ml">{lbl}</div><div class="mv {cls}">{val}</div></div>'

def parse_dates(df, col="p_date"):
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.sort_values(col).reset_index(drop=True)

SOT_HYPOTHESES = {"cross_channel", "marketplace_spillover", "sot_holdout"}

def detect_type(df):
    if {"L4 Product Tag","Advertiser Name","Cost (USD)"}.issubset(set(df.columns)):
        return "raw_ttam"
    # SOT detection: has multiple channel-style columns or p_date + non-TTAM metrics
    col_lower = [c.lower() for c in df.columns]
    sot_signals = ["direct", "organic", "paid search", "paid_search", "sessions", "tiktok"]
    sot_hits = sum(1 for s in sot_signals if any(s in c for c in col_lower))
    if sot_hits >= 3 and "l4 product tag" not in col_lower:
        return "sot"
    return "pre_aggregated"

def parse_sot_columns(df):
    """Return a dict mapping channel name -> list of metric columns."""
    channels = {}
    for col in df.columns:
        cl = col.lower()
        if "date" in cl or cl == "p_date": continue
        for ch in ["direct", "paid search", "paid_search", "organic search", "organic_search",
                   "tiktok", "meta", "total", "facebook"]:
            if ch.replace("_", " ") in cl.replace("_", " "):
                key = ch.replace(" ", "_").title()
                channels.setdefault(key, []).append(col)
                break
        else:
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

def suggest_flags(df):
    flags = []
    dates = pd.to_datetime(df["p_date"])
    if "TopViewCost" in df.columns:
        tv = pd.to_numeric(df["TopViewCost"], errors="coerce").fillna(0)
        active = df.loc[tv>0, "p_date"]
        if len(active):
            flags.append(dict(name="TopView_Window", start=str(active.min().date()),
                end=str(active.max().date()), confirmed=False, is_treatment=True,
                rationale=f"TopView spend detected {active.min().strftime('%d %b')}–{active.max().strftime('%d %b')}. This is your treatment period."))
    for sc in [c for c in df.columns if c.endswith("Cost") and "Total" not in c]:
        s = pd.to_numeric(df[sc], errors="coerce").fillna(0)
        if s.mean() > 0 and (s[:14]<s.mean()*0.1).sum()>4 and s[14:].mean()>s[:14].mean()*3:
            launch = dates[s>s.mean()*0.25].min()
            if pd.notna(launch):
                flags.append(dict(name=f"{sc.replace('Cost','')}Launch_Flag",
                    start=str(dates.min().date()), end=str(launch.date()),
                    confirmed=False, is_treatment=False,
                    rationale=f"{sc} near-zero until {launch.strftime('%d %b')} — structural break needs a flag."))
    return flags

def call_ark(api_key, endpoint_id, text):
    if not api_key or not endpoint_id: return {}
    try:
        client = OpenAI(base_url=ARK_BASE_URL, api_key=api_key)
        prompt = f"""Parse this causal analysis brief. Return ONLY a JSON object, no markdown.
Brief: "{text}"
JSON: {{"advertiser":"","hypothesis_type":"topview_conversion|spend_scaling|tts_gmv|full_funnel|third_party|unknown","target_kpi":"","campaign_format":"","intervention_notes":"","complexity_flags":[],"summary":""}}"""
        r = client.chat.completions.create(
            model=endpoint_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=400
        )
        t = r.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        return json.loads(t)
    except Exception as e:
        return {"error": str(e)}
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

    # Show selected hypothesis in sidebar
    if st.session_state.hypothesis:
        uc = next((u for u in USE_CASES if u["id"] == st.session_state.hypothesis), None)
        if uc:
            st.markdown(f"""<div style="background:{uc['colour']}12;border:1px solid {uc['colour']}33;border-radius:8px;padding:10px 12px;margin-bottom:14px;">
<div style="font-size:10px;color:{C['grey']};text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px">HYPOTHESIS</div>
<div style="font-size:12px;font-weight:600;color:{C['text']}">{uc['icon']} {uc['name']}</div>
<div style="margin-top:6px">{pill(uc['tag'], uc['tag_cls'])}</div></div>""", unsafe_allow_html=True)
            if st.button("↩ Change hypothesis", use_container_width=True, key="change_hyp"):
                st.session_state.hypothesis = None
                st.session_state.step = 0
                st.session_state.max_step = 0
                st.rerun()

    STEPS = ["1 · Context & Upload","2 · Analysis Setup","3 · Quality Check",
             "4 · EDA","5 · Flag Variables","6 · Smooth & Export"]

    # Steps that are accessible (only allow going back, not skipping ahead)
    st.session_state.max_step = max(st.session_state.get("max_step", 0), st.session_state.step)

    st.markdown(f'<style>.step-btn button{{background:transparent!important;border:none!important;box-shadow:none!important;color:{C["grey"]}!important;font-weight:400!important;text-align:left!important;padding:8px 12px!important;border-radius:6px!important;width:100%!important;font-size:13px!important;border-left:2px solid transparent!important;}}.step-btn-active button{{background:#EDE9FE!important;color:{C["purple"]}!important;font-weight:600!important;border-left:2px solid {C["purple"]}!important;}}.step-btn button:hover{{background:#F5F3FF!important;color:{C["purple"]}!important;}}</style>', unsafe_allow_html=True)

    for i, s in enumerate(STEPS, 1):
        active = st.session_state.step == i
        reachable = i <= st.session_state.max_step
        css_class = "step-btn-active" if active else "step-btn"
        if reachable:
            with st.container():
                st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                if st.button(s, key=f"nav_{i}", use_container_width=True):
                    st.session_state.step = i
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="padding:8px 12px;border-radius:6px;margin:2px 0;color:{C["grey_l"]};font-size:13px;opacity:0.45;cursor:default">{s}</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:11px;color:{C["grey"]};margin-bottom:4px">Ark API Key</div>', unsafe_allow_html=True)
    gk = st.text_input("ark_key_input", value=st.session_state.ark_key,
                        type="password", placeholder="Paste API key...", label_visibility="collapsed")
    if gk != st.session_state.ark_key:
        st.session_state.ark_key = gk
    st.markdown(f'<div style="font-size:11px;color:{C["grey"]};margin-top:8px;margin-bottom:4px">Endpoint ID</div>', unsafe_allow_html=True)
    ep = st.text_input("ark_ep_input", value=st.session_state.ark_endpoint,
                        placeholder="ep-XXXXXXXXXX", label_visibility="collapsed")
    if ep != st.session_state.ark_endpoint:
        st.session_state.ark_endpoint = ep
    if st.session_state.ark_key and st.session_state.ark_endpoint:
        st.markdown(pill("✓ Ark connected","pill-c"), unsafe_allow_html=True)
    elif st.session_state.ark_key:
        st.markdown(f'<div style="font-size:11px;color:{C["amber"]};margin-top:4px">Add endpoint ID to enable AI</div>', unsafe_allow_html=True)

    if st.session_state.daily_df is not None:
        df_ = parse_dates(st.session_state.daily_df)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"""<div style="font-size:11px;color:{C['grey']};margin-bottom:6px">LOADED</div>
<div style="font-size:13px;font-weight:600">{st.session_state.advertiser or "—"}</div>
<div style="font-size:11px;color:{C['grey']}">{df_['p_date'].min().strftime('%d %b')} – {df_['p_date'].max().strftime('%d %b %Y')} · {len(df_)} days</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — HYPOTHESIS SELECTION
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.step == 0:
    st.markdown(f'''<div style="text-align:center;padding:40px 0 32px">
      <div style="font-size:32px;margin-bottom:12px">🎯</div>
      <div style="font-size:28px;font-weight:700;color:{C['text']};margin-bottom:8px">What are you trying to prove?</div>
      <div style="font-size:15px;color:{C['grey']};max-width:520px;margin:0 auto">Select the use case that matches your client brief. This shapes how the data is cleaned, aggregated, and interpreted.</div>
    </div>''', unsafe_allow_html=True)

    # 2-column card grid
    col_a, col_b = st.columns(2, gap="large")
    for idx, uc in enumerate(USE_CASES):
        col = col_a if idx % 2 == 0 else col_b
        with col:
            selected = st.session_state.hypothesis == uc["id"]
            border_style = f"2px solid {uc['colour']}" if selected else f"1px solid {C['border']}"
            bg_style = f"{uc['colour']}08" if selected else C["surface"]
            check = "✓ " if selected else ""
            st.markdown(f'''<div style="background:{bg_style};border:{border_style};border-radius:14px;padding:22px 24px;margin-bottom:14px;cursor:pointer;transition:all .15s;">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:10px;">
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="font-size:22px">{uc['icon']}</span>
      <div style="font-size:14px;font-weight:700;color:{C['text']}">{check}{uc['name']}</div>
    </div>
    <span class="pill {uc['tag_cls']}" style="white-space:nowrap">{uc['tag']}</span>
  </div>
  <div style="font-size:12px;color:{C['grey']};margin-bottom:10px;line-height:1.6"><strong style="color:{C['text']}">Challenge:</strong> {uc['challenge']}</div>
  <div style="font-size:12px;color:{C['grey']};margin-bottom:10px;line-height:1.6"><strong style="color:{C['text']}">The pitch:</strong> {uc['pitch']}</div>
  <div style="font-size:11px;background:{C['bg']};border:1px solid {C['border']};border-radius:6px;padding:6px 10px;color:{C['grey']}">📁 {uc['data_source']}</div>
</div>''', unsafe_allow_html=True)
            if st.button(f"{'✓ Selected' if selected else 'Select'} — {uc['name']}", key=f"uc_{uc['id']}", use_container_width=True):
                st.session_state.hypothesis = uc["id"]
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    # Only show continue if a hypothesis is selected
    if st.session_state.hypothesis:
        uc_sel = next(u for u in USE_CASES if u["id"] == st.session_state.hypothesis)
        st.markdown(f'''<div style="background:{uc_sel['colour']}10;border:1px solid {uc_sel['colour']}44;border-radius:10px;padding:14px 20px;margin-bottom:20px;display:flex;align-items:center;gap:14px;">
  <span style="font-size:24px">{uc_sel['icon']}</span>
  <div>
    <div style="font-weight:600;color:{C['text']}">{uc_sel['name']}</div>
    <div style="font-size:12px;color:{C['grey']}">Ready to continue with this hypothesis</div>
  </div>
</div>''', unsafe_allow_html=True)
        _, btn_col, _ = st.columns([1,2,1])
        with btn_col:
            if st.button("Continue →  Upload Data", use_container_width=True, key="hyp_continue"):
                st.session_state.step = 1
                st.session_state.max_step = 1
                st.rerun()
    else:
        st.markdown(f'<div style="text-align:center;font-size:13px;color:{C['grey']}">Select a use case above to continue.</div>', unsafe_allow_html=True)

elif st.session_state.step == 1:
    st.markdown(f'{badge("Step 1")} <span style="font-size:20px;font-weight:600;margin-left:10px">Context & Upload</span>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{C["grey"]};margin-bottom:24px">Describe the brief you received, then upload your TTAM data export.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3,2], gap="large")

    with c1:
        st.markdown('<div style="font-size:12px;font-weight:600;color:#aaa;margin-bottom:6px;letter-spacing:.05em">CAUSAL CONTEXT BRIEF</div>', unsafe_allow_html=True)
        ctx = st.text_area("ctx", value=st.session_state.causal_context, height=170,
            placeholder='e.g. "The CSM says Chemist Warehouse ran a TopView on Sept 18 for Footy Finals. They want to know if it drove incremental TikTok-attributed conversions above their existing mid-funnel baseline. There was also a brand campaign that launched in August which may complicate the pre-period."',
            label_visibility="collapsed")
        st.session_state.causal_context = ctx

        if ctx and st.session_state.ark_key and st.session_state.ark_endpoint:
            if st.button("🤖 Parse brief with AI"):
                with st.spinner("Parsing…"):
                    res = call_ark(st.session_state.ark_key, st.session_state.ark_endpoint, ctx)
                    st.session_state.parsed_context = res
                    if res.get("advertiser") and not st.session_state.advertiser:
                        st.session_state.advertiser = res["advertiser"]
        elif ctx and not (st.session_state.ark_key and st.session_state.ark_endpoint):
            st.markdown(f'<div style="font-size:11px;color:{C["grey"]};margin-top:6px">Add your Ark API key and endpoint ID to auto-parse this brief.</div>', unsafe_allow_html=True)

        pc = st.session_state.parsed_context
        if pc and pc.get("summary"):
            st.markdown(f"""<div style="background:#00F2EA09;border:1px solid {C['purple']}33;border-radius:10px;padding:14px;margin-top:10px">
<div style="font-size:10px;color:{C['purple']};font-weight:600;letter-spacing:.08em;margin-bottom:8px">AI INTERPRETATION</div>
<div style="font-size:13px;margin-bottom:10px">{pc['summary']}</div>
<div>{pill(pc.get('target_kpi',''),'pill-c') if pc.get('target_kpi') else ''} {pill(pc.get('campaign_format',''),'pill-c') if pc.get('campaign_format') else ''} {' '.join(pill(f,'pill-a') for f in pc.get('complexity_flags',[]))}</div>
</div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div style="font-size:12px;font-weight:600;color:#aaa;margin-bottom:6px;letter-spacing:.05em">UPLOAD DATA</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("upload", type=["csv","xlsx"], label_visibility="collapsed",
            help="Raw TTAM export (multi-row/day with L4 tags) or pre-aggregated daily CSV")
        st.markdown(f"""<div style="font-size:12px;color:{C['grey']};line-height:1.9;margin-top:10px">
<span style="color:{C['purple']};font-weight:600">Raw TTAM export</span> — one row per L4 product tag per day.<br>
<span style="color:{C['purple']};font-weight:600">Pre-aggregated</span> — already has BrandCostExTV, MidFunnelCost etc. as columns.</div>""", unsafe_allow_html=True)

        if uploaded:
            try:
                # ── Smart file reading ─────────────────────────────────────
                if uploaded.name.endswith(".xlsx"):
                    raw_peek = pd.read_excel(uploaded, header=None, nrows=3)
                    uploaded.seek(0)
                    # Detect multi-level header (Triangl-style): first row has
                    # merged channel names, second row has metric names
                    first_row_vals = raw_peek.iloc[0].dropna().tolist()
                    second_row_vals = raw_peek.iloc[1].dropna().tolist() if len(raw_peek) > 1 else []
                    is_multilevel = (
                        len(first_row_vals) >= 2 and
                        any(str(v).lower() in ("date","week","channel") for v in first_row_vals[:2]) and
                        any(str(v).lower() in ("sessions","conversions","revenue","cost") for v in second_row_vals)
                    )
                    if is_multilevel:
                        # Parse multi-level: fill merged cells, flatten to col names
                        raw_all = pd.read_excel(uploaded, header=None)
                        channels_row = raw_all.iloc[0].tolist()
                        metrics_row  = raw_all.iloc[1].tolist()
                        filled, cur = [], None
                        for c in channels_row:
                            if pd.notna(c) and str(c).strip() not in ("","nan"): cur = str(c).strip()
                            filled.append(cur)
                        flat_cols = []
                        for i, (ch, mt) in enumerate(zip(filled, metrics_row)):
                            mt_str = str(mt).strip() if pd.notna(mt) else ""
                            ch_str = str(ch).strip() if ch else ""
                            if mt_str.lower() in ("date","week","channel","") or i == 0:
                                flat_cols.append("p_date")
                            else:
                                flat_cols.append(f"{ch_str}_{mt_str}".replace(" - ","_").replace(" ","_"))
                        raw = raw_all.iloc[2:].copy()
                        raw.columns = flat_cols
                        raw = raw[raw["p_date"].notna()]
                    else:
                        raw = pd.read_excel(uploaded)
                else:
                    raw = pd.read_csv(uploaded)

                # ── Normalise date column to p_date ────────────────────────
                if "p_date" not in raw.columns:
                    date_candidates = [c for c in raw.columns
                                       if any(kw in str(c).lower() for kw in ["date","week","day","period"])]
                    if date_candidates:
                        raw = raw.rename(columns={date_candidates[0]: "p_date"})
                    else:
                        # Last resort: use first column
                        raw = raw.rename(columns={raw.columns[0]: "p_date"})

                # ── Strip currency/comma formatting from numeric cols ───────
                for col in raw.columns:
                    if col == "p_date": continue
                    try:
                        cleaned = raw[col].astype(str).str.replace(r"[$,]","",regex=True)
                        raw[col] = pd.to_numeric(cleaned, errors="ignore")
                    except Exception:
                        pass

                raw = parse_dates(raw)
                ftype = detect_type(raw)
                st.session_state.raw_df = raw
                st.session_state.file_type = ftype
                if ftype == "raw_ttam" and "Advertiser Name" in raw.columns and not st.session_state.advertiser:
                    st.session_state.advertiser = raw["Advertiser Name"].dropna().iloc[0]
                if ftype == "pre_aggregated":
                    st.session_state.daily_df = raw

                date_range_str = f"{raw['p_date'].min().strftime('%d %b %Y')} – {raw['p_date'].max().strftime('%d %b %Y')}" if raw["p_date"].notna().any() else "—"
                st.markdown(f"""<div style="background:{C['purple']}08;border:1px solid {C['purple']}33;border-radius:8px;padding:14px;margin-top:10px">
<div style="font-weight:600;color:{C['purple']};margin-bottom:4px">✓ {uploaded.name}</div>
<div style="font-size:12px;color:{C['grey']}">{len(raw):,} rows · {date_range_str}</div>
<div style="margin-top:6px">
{pill("Raw TTAM — step 2 will aggregate","pill-a") if ftype=="raw_ttam" else pill("SOT / GA4 multi-channel — step 2 will assign channel roles","pill-c") if ftype=="sot" else pill("Pre-aggregated — skip to step 3","pill-g")}
</div></div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Could not read file: {e}")

    if st.session_state.raw_df is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        ca, cb = st.columns([3,1])
        with ca:
            adv = st.text_input("Advertiser name", value=st.session_state.advertiser, placeholder="e.g. Chemist Warehouse")
            st.session_state.advertiser = adv
        with cb:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Continue →", use_container_width=True):
                st.session_state.step = 2 if st.session_state.file_type in ("raw_ttam", "sot") else 3
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — MAP & AGGREGATE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    hyp = st.session_state.hypothesis or ""

    # ── Route: SOT vs TTAM ──────────────────────────────────────────────────
    if st.session_state.file_type == "sot":
        # ── SOT Step 2: campaign period + channel snapshot ────────────────────────
        df = st.session_state.raw_df.copy()
        date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
        df = df.rename(columns={date_col: "p_date"})
        df["p_date"] = pd.to_datetime(df["p_date"], errors="coerce")
        df = df.dropna(subset=["p_date"]).sort_values("p_date").reset_index(drop=True)
        for c in df.columns:
            if c != "p_date":
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"[$,]","",regex=True), errors="coerce")

        date_min = df["p_date"].min().date()
        date_max = df["p_date"].max().date()

        st.markdown(f'<div style="font-size:12px;font-weight:600;color:#aaa;margin-bottom:8px;letter-spacing:.05em">CAMPAIGN / INTERVENTION PERIOD</div>', unsafe_allow_html=True)
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown(f'<div style="font-size:11px;color:{C["grey"]};font-weight:600;margin-bottom:4px">CAMPAIGN START</div>', unsafe_allow_html=True)
            t_start = st.date_input("tstart", value=st.session_state.sot_treatment_start or date_min,
                min_value=date_min, max_value=date_max, label_visibility="collapsed", key="sot_tstart")
            st.session_state.sot_treatment_start = t_start
        with tc2:
            st.markdown(f'<div style="font-size:11px;color:{C["grey"]};font-weight:600;margin-bottom:4px">CAMPAIGN END</div>', unsafe_allow_html=True)
            t_end = st.date_input("tend", value=st.session_state.sot_treatment_end or date_max,
                min_value=date_min, max_value=date_max, label_visibility="collapsed", key="sot_tend")
            st.session_state.sot_treatment_end = t_end

        pre_df_sot  = df[df["p_date"].dt.date < t_start]
        post_df_sot = df[(df["p_date"].dt.date >= t_start) & (df["p_date"].dt.date <= t_end)]
        pre_days  = len(pre_df_sot)
        post_days = len(post_df_sot)
        ratio     = pre_days / post_days if post_days else 0
        rc = "mvc" if ratio >= 2 else "mva" if ratio >= 1 else "mvr"
        st.markdown(f'<div class="mrow" style="margin-top:10px">{mtile("Pre-period",f"{pre_days}d")}{mtile("Post-period",f"{post_days}d")}{mtile("Ratio",f"{ratio:.1f}x",rc)}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:12px;font-weight:600;color:#aaa;margin-bottom:4px;letter-spacing:.05em">CHANNEL SNAPSHOT</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:12px">Pre vs post lift for each channel. Select the ones you want to investigate — these become your analysis targets.</div>', unsafe_allow_html=True)

        metric_cols = [c for c in df.columns if c != "p_date" and
                       any(k in c.lower() for k in ["conversion","order","purchase","gmv","sessions"])
                       and df[c].dtype in ["float64","int64"] and df[c].sum() > 0]
        non_tiktok = [c for c in metric_cols if "tiktok" not in c.lower()]
        tiktok_cols = [c for c in metric_cols if "tiktok" in c.lower()]

        if not st.session_state.sot_selected_targets:
            st.session_state.sot_selected_targets = [c for c in non_tiktok if "conversion" in c.lower()][:2]

        selected = list(st.session_state.sot_selected_targets)
        card_cols = st.columns(2)
        for i, col in enumerate(non_tiktok):
            with card_cols[i % 2]:
                pre_avg  = pre_df_sot[col].mean()  if col in pre_df_sot.columns  else 0
                post_avg = post_df_sot[col].mean() if col in post_df_sot.columns and len(post_df_sot) else 0
                lift = (post_avg - pre_avg) / pre_avg * 100 if pre_avg else 0
                is_sel = col in selected
                lift_color = C["purple"] if lift > 0 else C["red"]
                border = f"2px solid {C['purple']}" if is_sel else f"1px solid {C['border']}"
                bg     = f"{C['purple']}08" if is_sel else C["surface"]
                sign   = "+" if lift >= 0 else ""
                st.markdown(f'<div style="border:{border};background:{bg};border-radius:8px;padding:12px 14px;margin-bottom:8px"><div style="display:flex;justify-content:space-between;align-items:center"><div style="font-size:12px;font-weight:600;color:{C["text"]}">{col.replace("_"," ")}</div><div style="font-size:16px;font-weight:700;color:{lift_color}">{sign}{lift:.1f}%</div></div><div style="font-size:11px;color:{C["grey"]};margin-top:3px">{pre_avg:,.0f} pre to {post_avg:,.0f} post avg/day</div></div>', unsafe_allow_html=True)
                checked = st.checkbox("Investigate", value=is_sel, key=f"sot_sel_{col}")
                if checked and col not in selected: selected.append(col)
                elif not checked and col in selected: selected.remove(col)

        st.session_state.sot_selected_targets = selected

        if tiktok_cols:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:11px;font-weight:600;color:#0891B2;margin-bottom:6px;letter-spacing:.05em">TIKTOK (INTERVENTION)</div>', unsafe_allow_html=True)
            tt_html = '<div class="mrow">'
            for col in tiktok_cols[:4]:
                pre_avg  = pre_df_sot[col].mean()  if col in pre_df_sot.columns else 0
                post_avg = post_df_sot[col].mean() if col in post_df_sot.columns and len(post_df_sot) else 0
                lift = (post_avg - pre_avg) / pre_avg * 100 if pre_avg else 0
                sign = "+" if lift >= 0 else ""
                lc = "mvc" if lift > 0 else "mvr"
                tt_html += f'<div class="mtile" style="border-left:3px solid #0891B2"><div class="ml">{col.replace("Paid_Social_TikTok_","").replace("_"," ")}</div><div class="mv {lc}" style="font-size:15px">{sign}{lift:.1f}%</div><div style="font-size:10px;color:{C["grey"]}">{pre_avg:,.0f} to {post_avg:,.0f}</div></div>'
            st.markdown(tt_html + "</div>", unsafe_allow_html=True)

        # Auto-assign roles
        auto_roles = {}
        for col in df.columns:
            if col == "p_date": continue
            if "tiktok" in col.lower(): auto_roles[col] = "Exclude"
            elif col in selected:
                auto_roles[col] = "Primary Target" if col == selected[0] else "Secondary Target"
            elif "organic" in col.lower() and "conversion" in col.lower(): auto_roles[col] = "Covariate"
            else: auto_roles[col] = "Exclude"
        st.session_state.sot_channel_roles = auto_roles
        st.session_state.sot_targets = selected

        if selected:
            cov_auto = [c for c,r in auto_roles.items() if r == "Covariate"]
            st.markdown("<br>", unsafe_allow_html=True)
            s = f'<div style="background:{C["purple"]}08;border:1px solid {C["purple"]}33;border-radius:10px;padding:14px 18px">'
            s += f'<div style="font-size:11px;color:{C["purple"]};font-weight:700;margin-bottom:10px">ANALYSIS SUMMARY</div>'
            for i, tgt in enumerate(selected):
                label = "Run 1 - Primary" if i == 0 else f"Run {i+1} - Secondary"
                s += f'<div style="font-size:12px;margin-bottom:4px"><span style="color:{C["purple"]};font-weight:600">{label}:</span> <strong>{tgt}</strong></div>'
            s += f'<div style="font-size:12px;margin-top:6px;color:{C["grey"]}">Covariate: {", ".join(cov_auto) if cov_auto else "none (univariate)"}</div>'
            s += '</div>'
            st.markdown(s, unsafe_allow_html=True)

        sot_b1, sot_b2 = st.columns([1, 5])
        with sot_b1:
            if st.button("Back", key="s2_back_sot"): st.session_state.step=1; st.rerun()
        with sot_b2:
            if selected:
                if st.button("Confirm & continue", use_container_width=True, key="s2_fwd_sot"):
                    st.session_state.daily_df = df
                    st.session_state.pre_start  = date_min
                    st.session_state.pre_end    = (pd.Timestamp(t_start) - pd.Timedelta(1)).date()
                    st.session_state.post_start = t_start
                    st.session_state.post_end   = t_end
                    st.session_state.target_col = selected[0] if selected else None
                    st.session_state.step = 3
                    st.rerun()
            else:
                st.warning("Select at least one channel.")

    else:
        # ════════════════════════════════════════════════════════════════════
        # TTAM Step 2 (existing flow)
        # ════════════════════════════════════════════════════════════════════
        st.markdown(f'{badge("Step 2")} <span style="font-size:20px;font-weight:600;margin-left:10px">Analysis Setup</span>', unsafe_allow_html=True)
        st.markdown(f'<div style="color:{C["grey"]};margin-bottom:20px">Based on your hypothesis, we have automatically grouped your campaign data. Review and confirm before exporting.</div>', unsafe_allow_html=True)
    
        df = st.session_state.raw_df
        hyp = st.session_state.hypothesis or "ttam_full_funnel_brand"
    
        # Hypothesis-aware recommended conversion events
        HYP_CONV_EVENTS = {
            "ttam_full_funnel_brand": ["Complete Payment", "Purchase", "Checkout"],
            "cross_channel":          ["Complete Payment", "Purchase", "Landing Page View"],
            "marketplace_spillover":  ["Complete Payment", "Purchase"],
            "tts_full_funnel_brand":  ["Complete Payment", "Purchase", "Add to Cart"],
            "tts_baseline":           ["Complete Payment", "Purchase"],
            "sot_holdout":            ["Complete Payment", "Purchase", "Session Start"],
        }
    
        def auto_assign_role(l1, l2, l3, l4):
            l4l = str(l4).lower(); l3l = str(l3).lower()
            l2l = str(l2).lower(); l1l = str(l1).lower()
            if "topview" in l4l or "top view" in l4l or ("openscreen" in l3l and "premium" in l2l):
                return "TopView", "INTERVENTION", C["purple"], "This IS the treatment — defines your pre/post split."
            if "upper funnel" in l1l:
                return "Brand_ExTV", "COVARIATE", "#2563EB", "Upper-funnel brand spend — controls awareness effects."
            if "brand consideration" in l2l or "brand auction" in l3l:
                return "Brand_ExTV", "COVARIATE", "#2563EB", "Brand Auction — key covariate for brand activity."
            if "middle funnel" in l1l and ("web traffic" in l2l or "brand traffic" in l2l):
                return "MidFunnel", "COVARIATE", "#0891B2", "Mid-funnel traffic — controls consideration activity."
            if "lower funnel" in l1l and "web sales" in l2l:
                return "Performance", "COVARIATE", "#059669", "Performance web spend — controls direct-response."
            if "app promotion" in l2l or "app prospecting" in l3l or "app retargeting" in l3l:
                return "App", "COVARIATE", "#7C3AED", "App campaign spend — separate funnel, include as control."
            if "gmv max" in l4l or ("shop" in l4l and "tts" in l4l):
                return "Performance", "COVARIATE", "#059669", "TikTok Shop spend."
            return "Other", "REVIEW", C["grey"], "Could not auto-classify — assign manually below."
    
        hier_df = df.groupby(["L1 Product Tag","L2 Product Tag","L3 Product Tag","L4 Product Tag"]).agg(
            spend=("Cost (USD)","sum"), conversions=("Conversions","sum"), days=("p_date","nunique")
        ).reset_index()
    
        if "role_overrides" not in st.session_state:
            st.session_state.role_overrides = {}
        if "target_event" not in st.session_state:
            st.session_state.target_event = None
    
        CATS = ["TopView","Brand_ExTV","MidFunnel","Performance","App","Other","EXCLUDE"]
        rows = []
        for _, r in hier_df.iterrows():
            auto_cat, auto_label, auto_color, rationale = auto_assign_role(
                r["L1 Product Tag"], r["L2 Product Tag"], r["L3 Product Tag"], r["L4 Product Tag"]
            )
            final_cat = st.session_state.role_overrides.get(r["L4 Product Tag"], auto_cat)
            rows.append(dict(
                l4=r["L4 Product Tag"], l1=r["L1 Product Tag"], l2=r["L2 Product Tag"],
                auto_cat=auto_cat, auto_label=auto_label, auto_color=auto_color,
                final_cat=final_cat, rationale=rationale,
                spend=r["spend"], conversions=r["conversions"], days=int(r["days"])
            ))
    
        intervention_rows = [r for r in rows if r["final_cat"] == "TopView"]
        covariate_rows    = [r for r in rows if r["final_cat"] not in ("TopView","EXCLUDE","Other")]
    
        # ── Analysis structure visual
        st.markdown(f'<div style="font-size:12px;font-weight:600;color:#aaa;margin-bottom:12px;letter-spacing:.05em">AUTO-DETECTED ANALYSIS STRUCTURE</div>', unsafe_allow_html=True)
        opt_events = sorted(df["Optimization Event(External Actions)"].dropna().unique())
        event_conv = df.groupby("Optimization Event(External Actions)")["Conversions"].sum().to_dict()
        rec_events = HYP_CONV_EVENTS.get(hyp, ["Complete Payment"])
        auto_target = next((x for x in opt_events if any(r.lower() in x.lower() for r in rec_events)), opt_events[0] if opt_events else None)
        if not st.session_state.target_event:
            st.session_state.target_event = auto_target
    
        ms1, ms2, ms3 = st.columns([5,1,5])
        with ms1:
            if intervention_rows:
                r0 = intervention_rows[0]
                st.markdown(f'<div style="background:{C["purple"]}0D;border:2px solid {C["purple"]};border-radius:12px;padding:18px;text-align:center"><div style="font-size:10px;color:{C["purple"]};font-weight:700;letter-spacing:.1em;margin-bottom:8px">INTERVENTION (TREATMENT)</div><div style="font-size:14px;font-weight:700;color:{C["text"]}">{r0["l4"]}</div><div style="font-size:11px;color:{C["grey"]};margin-top:4px">${r0["spend"]:,.0f} total spend</div><div style="font-size:11px;color:{C["grey"]}">{r0["days"]} days active</div><div style="font-size:11px;color:{C["purple"]};margin-top:6px;font-style:italic">{r0["rationale"]}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="background:#F5F3FF;border:2px dashed {C["purple"]};border-radius:12px;padding:18px;text-align:center"><div style="font-size:12px;color:{C["grey"]}">No intervention detected — check overrides below</div></div>', unsafe_allow_html=True)
        with ms2:
            st.markdown(f'<div style="display:flex;align-items:center;justify-content:center;height:100%;font-size:24px;color:{C["grey"]}">&#8594;</div>', unsafe_allow_html=True)
        with ms3:
            tgt_ev = st.session_state.target_event or "—"
            tgt_conv = event_conv.get(tgt_ev, 0)
            st.markdown(f'<div style="background:#05996910;border:2px solid #059669;border-radius:12px;padding:18px;text-align:center"><div style="font-size:10px;color:#059669;font-weight:700;letter-spacing:.1em;margin-bottom:8px">TARGET KPI (DEPENDENT VARIABLE)</div><div style="font-size:14px;font-weight:700;color:{C["text"]}">{tgt_ev}</div><div style="font-size:11px;color:{C["grey"]};margin-top:4px">{tgt_conv:,} total conversions</div><div style="font-size:11px;color:#059669;margin-top:6px;font-style:italic">Aggregated from selected conversion event</div></div>', unsafe_allow_html=True)
    
        # Covariates strip
        if covariate_rows:
            cov_html = f'<div style="margin-top:14px;padding:14px 16px;background:{C["bg"]};border:1px solid {C["border"]};border-radius:8px"><div style="font-size:10px;color:{C["grey"]};font-weight:600;letter-spacing:.1em;margin-bottom:10px">COVARIATES (control variables)</div>'
            for r in covariate_rows:
                cov_html += f'<div style="display:inline-block;background:{r["auto_color"]}12;border:1px solid {r["auto_color"]}44;border-radius:6px;padding:5px 12px;margin:3px;font-size:12px"><span style="font-weight:600;color:{r["auto_color"]}">{r["final_cat"]}</span><span style="color:{C["grey"]}"> · {r["l4"]} · ${r["spend"]:,.0f}</span></div>'
            st.markdown(cov_html + '</div>', unsafe_allow_html=True)
    
        # ── Conversion event selector
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:12px;font-weight:600;color:#aaa;margin-bottom:8px;letter-spacing:.05em">SELECT TARGET CONVERSION EVENT</div>', unsafe_allow_html=True)
    
        ev_cols = st.columns(min(len(opt_events), 4))
        for i, ev in enumerate(sorted(opt_events)):
            with ev_cols[i % 4]:
                is_rec = auto_target and ev == auto_target
                is_sel = st.session_state.target_event == ev
                border = f"2px solid {C['purple']}" if is_sel else f"1px solid {C['border']}"
                bg = f"{C['purple']}0A" if is_sel else C["surface"]
                rec_badge = f'<div style="font-size:9px;color:{C['purple']};font-weight:700;margin-bottom:3px">RECOMMENDED</div>' if is_rec else ""
                st.markdown(f'<div style="border:{border};background:{bg};border-radius:8px;padding:10px;margin-bottom:6px">{rec_badge}<div style="font-size:12px;font-weight:600;color:{C["text"]}">{ev}</div><div style="font-size:11px;color:{C["grey"]}">{event_conv.get(ev,0):,} conversions</div></div>', unsafe_allow_html=True)
                if st.button("Select" if not is_sel else "Selected", key=f"ev_{i}_{ev}", use_container_width=True):
                    st.session_state.target_event = ev
                    st.rerun()
    
        # ── Override panel
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Override campaign category assignments"):
            st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:10px">Auto-assignments use the L1/L2/L3 taxonomy. Override only if misclassified for this account.</div>', unsafe_allow_html=True)
            ov_cols = st.columns(2)
            new_overrides = {}
            for i, r in enumerate(rows):
                with ov_cols[i % 2]:
                    current = st.session_state.role_overrides.get(r["l4"], r["auto_cat"])
                    idx_cat = CATS.index(current) if current in CATS else 4
                    chosen = st.selectbox(f"{r['l4']}  (${r['spend']:,.0f})", CATS, index=idx_cat, key=f"ov_{i}_{r['l4']}")
                    if chosen != r["auto_cat"]:
                        new_overrides[r["l4"]] = chosen
            st.session_state.role_overrides = new_overrides
            if new_overrides:
                st.markdown(f'<div style="font-size:11px;color:{C["amber"]};margin-top:8px">{len(new_overrides)} override(s) active.</div>', unsafe_allow_html=True)
    
        # ── Preview & confirm
        st.markdown("<br>", unsafe_allow_html=True)
        final_map = {r["l4"]: st.session_state.role_overrides.get(r["l4"], r["auto_cat"]) for r in rows if st.session_state.role_overrides.get(r["l4"], r["auto_cat"]) != "EXCLUDE"}
        sel_events = [st.session_state.target_event] if st.session_state.target_event else []
    
        if sel_events:
            with st.spinner("Building preview…"):
                prev = aggregate_ttam(df, final_map, sel_events)
                prev = parse_dates(prev)
            days = len(prev); total_c = int(prev["Conversions_Raw"].sum()); total_s = prev["TotalCost"].sum()
            st.markdown(f'<div class="mrow">{mtile("Days",str(days))}{mtile("Target KPI Conversions",f"{total_c:,}")}{mtile("Total Spend",f"${total_s:,.0f}")}{mtile("Avg Daily Conv.",f"{total_c//days if days else 0:,}")}</div>', unsafe_allow_html=True)
            st.dataframe(prev.head(6).assign(p_date=lambda d: d["p_date"].dt.strftime("%Y-%m-%d")), use_container_width=True, height=200)
    
        cb1, cb2 = st.columns([1,5])
        with cb1:
            if st.button("Back"): st.session_state.step=1; st.rerun()
        with cb2:
            if sel_events:
                if st.button("✓ Confirm setup & continue →", use_container_width=True):
                    st.session_state.l4_map = final_map
                    st.session_state.conversion_events = sel_events
                    st.session_state.daily_df = parse_dates(aggregate_ttam(df, final_map, sel_events))
                    st.session_state.step=3; st.rerun()
            else:
                st.warning("Select a target conversion event to continue.")
# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — QUALITY CHECK
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    st.markdown(f'{badge("Step 3")} <span style="font-size:20px;font-weight:600;margin-left:10px">Data Quality Check</span>', unsafe_allow_html=True)

    df = parse_dates(st.session_state.daily_df.copy())
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
            line=dict(color=C["purple"],width=2), fill="tozeroy", fillcolor="rgba(124,58,237,0.1)", name=conv_c))
        fig.update_layout(**CHART, title=dict(text="Conversion series overview",font=dict(size=14)))
        st.plotly_chart(fig, use_container_width=True)

    cb1, cb2 = st.columns([1,5])
    with cb1:
        if st.button("← Back"): st.session_state.step=2 if st.session_state.file_type in ("raw_ttam","sot") else 1; st.rerun()
    with cb2:
        if st.button("Continue to EDA →", use_container_width=True): st.session_state.step=4; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    st.markdown(f'{badge("Step 4")} <span style="font-size:20px;font-weight:600;margin-left:10px">Exploratory Analysis</span>', unsafe_allow_html=True)

    df = parse_dates(st.session_state.daily_df.copy())

    # SOT EDA — early exit before TTAM code
    if st.session_state.file_type == "sot" and st.session_state.daily_df is not None:
        t_start = pd.Timestamp(st.session_state.post_start) if st.session_state.post_start else None
        t_end   = pd.Timestamp(st.session_state.post_end)   if st.session_state.post_end   else None
        pre_df  = df[df["p_date"] < t_start]  if t_start is not None else df.copy()
        post_df = df[(df["p_date"] >= t_start) & (df["p_date"] <= t_end)] if t_start is not None else pd.DataFrame()

        conv_cols_sot = [c for c in df.columns if "Conversions" in c]
        sess_cols_sot = [c for c in df.columns if "Sessions" in c]
        rev_cols_sot  = [c for c in df.columns if "Revenue" in c]

        CH_ORDER  = ["Direct","Paid_Social_TikTok","Paid_Search_Google","Paid_Social_Meta","Organic_Search"]
        CH_COLORS = {"Direct":C["purple"],"Paid_Social_TikTok":"#0891B2","Paid_Search_Google":"#059669","Paid_Social_Meta":"#D97706","Organic_Search":"#6B7280"}
        def ch_col(ch): return CH_COLORS.get(ch, C["grey"])
        def ch_lbl(ch): return ch.replace("Paid_Social_","").replace("Paid_Search_","").replace("_Search","").replace("_"," ")

        channels = {}
        for col in df.columns:
            if col == "p_date": continue
            for ch in CH_ORDER:
                if col.startswith(ch):
                    channels.setdefault(ch, []).append(col); break

        def add_treatment(fig):
            if t_start is not None:
                fig.add_vrect(x0=str(t_start.date()), x1=str(t_end.date()), fillcolor="rgba(124,58,237,0.07)", line_width=0)
                fig.add_shape(type="line", x0=str(t_start.date()), x1=str(t_start.date()), y0=0, y1=1, yref="paper", line=dict(color=C["purple"],width=2,dash="dash"))
                fig.add_annotation(x=str(t_start.date()), y=1, yref="paper", text="Campaign start", showarrow=False, yanchor="bottom", font=dict(color=C["purple"],size=10), xshift=5)
            return fig

        sot_tabs = st.tabs(["All Channels","Conversion Rates","Lagged Correlations","Spend Proxy & ROAS","Pre-Period Stability"])

        with sot_tabs[0]:
            st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:10px">All channels on one chart — look for which ones move when the campaign runs.</div>', unsafe_allow_html=True)
            metric_type = st.radio("Metric", ["Conversions","Sessions","Revenue"], horizontal=True, key="sot_mt")
            col_lookup  = {"Conversions":conv_cols_sot,"Sessions":sess_cols_sot,"Revenue":rev_cols_sot}
            fig = go.Figure()
            for col in col_lookup[metric_type]:
                ch = next((c for c in CH_ORDER if col.startswith(c)), None)
                fig.add_trace(go.Scatter(x=df["p_date"], y=df[col], mode="lines",
                    line=dict(color=ch_col(ch) if ch else C["grey"], width=2), name=ch_lbl(ch) if ch else col))
            add_treatment(fig)
            fig.update_layout(**CHART, title=dict(text=f"All channels — {metric_type}",font=dict(size=14)))
            st.plotly_chart(fig, use_container_width=True)

        with sot_tabs[1]:
            st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:10px">Conversions / sessions per channel. CVR rising without session growth = brand awareness priming the audience.</div>', unsafe_allow_html=True)
            fig_cvr = go.Figure(); cvr_rows = []
            for ch in CH_ORDER:
                if ch not in channels: continue
                s_col = next((c for c in channels[ch] if "Sessions" in c), None)
                c_col = next((c for c in channels[ch] if "Conversions" in c), None)
                if not s_col or not c_col: continue
                cvr = (df[c_col]/df[s_col].replace(0,np.nan)*100).rolling(5,center=True,min_periods=2).mean()
                fig_cvr.add_trace(go.Scatter(x=df["p_date"],y=cvr,mode="lines",line=dict(color=ch_col(ch),width=2),name=ch_lbl(ch)))
                pre_cvr  = pre_df[c_col].sum() /pre_df[s_col].sum() *100 if len(pre_df)  and pre_df[s_col].sum()>0  else 0
                post_cvr = post_df[c_col].sum()/post_df[s_col].sum()*100 if len(post_df) and post_df[s_col].sum()>0 else 0
                delta = (post_cvr-pre_cvr)/pre_cvr*100 if pre_cvr else 0
                cvr_rows.append(dict(ch=ch_lbl(ch),pre=pre_cvr,post=post_cvr,delta=delta,color=ch_col(ch)))
            add_treatment(fig_cvr)
            fig_cvr.update_layout(**CHART,title=dict(text="Conversion rate (%) by channel — 5-day smoothed",font=dict(size=14)))
            st.plotly_chart(fig_cvr,use_container_width=True)
            if cvr_rows:
                html = '<div class="mrow">'
                for r in sorted(cvr_rows,key=lambda x:-abs(x["delta"])):
                    sign="+"; mc="mvc"
                    if r["delta"]<0: sign=""; mc="mvr"
                    html+=f'<div class="mtile" style="border-left:3px solid {r["color"]}"><div class="ml">{r["ch"]}</div><div class="mv {mc}">{sign}{r["delta"]:.1f}%</div><div style="font-size:11px;color:{C["grey"]}">{r["pre"]:.2f}% to {r["post"]:.2f}%</div></div>'
                st.markdown(html+"</div>",unsafe_allow_html=True)

        with sot_tabs[2]:
            st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:10px">Peak correlation at 5-7 day lag = TikTok drives awareness that surfaces as search or direct traffic about a week later.</div>', unsafe_allow_html=True)
            tt_sess = next((c for c in sess_cols_sot if "TikTok" in c),None)
            if tt_sess:
                lag_c1,lag_c2 = st.columns([3,1])
                with lag_c1:
                    lag_target = st.selectbox("Compare TikTok sessions against:", [c for c in sess_cols_sot+conv_cols_sot if "TikTok" not in c], key="lag_tgt")
                with lag_c2:
                    max_lag = st.slider("Max lag (days)",3,21,14,key="lag_max")
                tt_roll  = df[tt_sess].rolling(3,min_periods=1).mean()
                tgt_roll = df[lag_target].rolling(3,min_periods=1).mean()
                lags = list(range(0,max_lag+1))
                rs   = [tt_roll.corr(tgt_roll.shift(-l)) for l in lags]
                peak_lag=lags[rs.index(max(rs))]; peak_r=max(rs)
                fig_lag=go.Figure()
                bar_colors=[C["purple"] if r==peak_r else ("#059669" if r>0.4 else C["grey"]) for r in rs]
                fig_lag.add_trace(go.Bar(x=lags,y=rs,marker_color=bar_colors,text=[f"{r:.2f}" for r in rs],textposition="outside",textfont=dict(color=C["text"],size=10),width=0.6))
                fig_lag.add_hline(y=0.5,line_dash="dash",line_color=C["amber"])
                fig_lag.update_layout(**CHART,title=dict(text=f"TikTok Sessions vs {lag_target.replace('_',' ')} (R by lag)",font=dict(size=14)))
                fig_lag.update_xaxes(title="Days after TikTok activity",dtick=1)
                fig_lag.update_yaxes(range=[-0.2,1.05])
                st.plotly_chart(fig_lag,use_container_width=True)
                if peak_r>0.5:
                    st.markdown(f'<div class="card card-c">Strong signal — TikTok activity predicts {lag_target.replace("_"," ")} approximately <strong>{peak_lag} day(s) later</strong> (R={peak_r:.2f}). Use this as evidence that TikTok is driving downstream activity.</div>',unsafe_allow_html=True)
                elif peak_r>0.3:
                    st.markdown(f'<div class="card card-a">Moderate signal at {peak_lag}-day lag (R={peak_r:.2f}). Directional but treat with caution.</div>',unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="card card-g">Weak signal (peak R={peak_r:.2f}). Try a different target channel.</div>',unsafe_allow_html=True)
            else:
                st.info("No TikTok Sessions column detected.")

        with sot_tabs[3]:
            st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:10px">Estimating channel spend using CaC x Conversions. Blended ROAS = total revenue / estimated total spend.</div>', unsafe_allow_html=True)
            cac_map={}
            for ch in CH_ORDER:
                if ch not in channels: continue
                cac_col  = next((c for c in channels[ch] if any(x in c for x in ["Cac","CAC","CaC","cac"])),None)
                conv_col = next((c for c in channels[ch] if "Conversions" in c),None)
                if cac_col and conv_col:
                    df[f"Est_{ch}_Spend"]=df[cac_col]*df[conv_col]; cac_map[ch]=f"Est_{ch}_Spend"
            if cac_map:
                df["Est_Total_Spend"]=df[list(cac_map.values())].sum(axis=1)
                df["Total_Revenue"]  =df[rev_cols_sot].sum(axis=1) if rev_cols_sot else 0
                df["Blended_ROAS"]   =df["Total_Revenue"]/df["Est_Total_Spend"].replace(0,np.nan)
                fig_sp=go.Figure()
                for ch,col in cac_map.items():
                    fig_sp.add_trace(go.Bar(x=df["p_date"],y=df[col],name=ch_lbl(ch),marker_color=ch_col(ch)))
                add_treatment(fig_sp)
                fig_sp.update_layout(**CHART,barmode="stack",title=dict(text="Estimated daily spend (CaC x Conversions)",font=dict(size=14)))
                st.plotly_chart(fig_sp,use_container_width=True)
                fig_br=go.Figure()
                fig_br.add_trace(go.Scatter(x=df["p_date"],y=df["Blended_ROAS"].rolling(5,center=True,min_periods=2).mean(),mode="lines",line=dict(color=C["purple"],width=2.5),name="Blended ROAS (5-day avg)"))
                add_treatment(fig_br)
                fig_br.update_layout(**CHART,title=dict(text="Blended ROAS — total revenue / estimated total spend",font=dict(size=14)))
                st.plotly_chart(fig_br,use_container_width=True)
                if len(post_df) and "Est_Total_Spend" in df.columns:
                    pre_r = df[df["p_date"] < t_start]["Total_Revenue"].sum() / df[df["p_date"] < t_start]["Est_Total_Spend"].sum() if df[df["p_date"] < t_start]["Est_Total_Spend"].sum() else 0
                    post_mask = (df["p_date"] >= t_start) & (df["p_date"] <= t_end)
                    pos_r = df[post_mask]["Total_Revenue"].sum() / df[post_mask]["Est_Total_Spend"].sum() if df[post_mask]["Est_Total_Spend"].sum() else 0
                    mc="mvc" if pos_r>pre_r else "mvr"
                    st.markdown(f'<div class="mrow">{mtile("Pre blended ROAS",f"{pre_r:.1f}x")}{mtile("Post blended ROAS",f"{pos_r:.1f}x",mc)}{mtile("Change",f"{(pos_r/pre_r-1)*100:+.1f}%" if pre_r else "n/a",mc)}</div>', unsafe_allow_html=True)
            else:
                st.info("No CaC columns detected.")

        with sot_tabs[4]:
            st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:10px">Was the pre-period stable? A volatile pre-period makes the counterfactual less reliable.</div>', unsafe_allow_html=True)
            all_trend = [c for c in conv_cols_sot+sess_cols_sot if c in pre_df.columns]
            if all_trend:
                trend_col = st.selectbox("Metric to check", all_trend, key="trend_col")
                if len(pre_df) > 7:
                    y = pre_df[trend_col].fillna(0).values
                    y_mean = y.mean(); coeffs=np.polyfit(np.arange(len(y)),y,1)
                    slope_pct=coeffs[0]/y_mean*100 if y_mean else 0
                    cv=y.std()/y_mean*100 if y_mean else 0
                    rolling=pd.Series(y).rolling(7,center=True,min_periods=3).mean()
                    fig_tr=go.Figure()
                    fig_tr.add_trace(go.Scatter(x=pre_df["p_date"],y=y,mode="lines",line=dict(color=C["grey"],width=1.2,dash="dot"),name="Daily",opacity=0.5))
                    fig_tr.add_trace(go.Scatter(x=pre_df["p_date"],y=rolling,mode="lines",line=dict(color=C["purple"],width=2.5),name="7-day average"))
                    fig_tr.add_hline(y=y_mean,line_dash="dash",line_color=C["grey"],opacity=0.5)
                    if len(post_df) and trend_col in post_df.columns:
                        fig_tr.add_trace(go.Scatter(x=post_df["p_date"],y=post_df[trend_col],mode="lines",line=dict(color="#059669",width=2.5),name="Post-period"))
                    add_treatment(fig_tr)
                    fig_tr.update_layout(**CHART,title=dict(text=f"{trend_col.replace('_',' ')} — pre-period stability",font=dict(size=14)))
                    st.plotly_chart(fig_tr,use_container_width=True)
                    if abs(slope_pct)>1.5:
                        direction="rising" if coeffs[0]>0 else "falling"
                        st.markdown(f'<div class="card card-a">Pre-period was <strong>{direction}</strong> at {abs(slope_pct):.1f}%/day (volatility CV: {cv:.0f}%). The analysis tool will account for this trend internally.</div>',unsafe_allow_html=True)
                    elif cv>30:
                        st.markdown(f'<div class="card card-a">Pre-period is volatile (CV: {cv:.0f}%). High day-to-day variance makes the counterfactual harder to estimate. Consider smoothing.</div>',unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="card card-c">Pre-period looks stable (CV: {cv:.0f}%, slope: {slope_pct:+.1f}%/day). Clean baseline.</div>',unsafe_allow_html=True)
            else:
                st.info("No conversion or session columns found.")

        # LLM Insights (SOT)
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown(f'<div style="background:linear-gradient(135deg,{C["purple"]}08,{C["surface"]});border:1px solid {C["purple"]}33;border-radius:12px;padding:18px 20px">',unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:13px;font-weight:600;color:{C["text"]};margin-bottom:4px">AI EDA Interpretation</div>',unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:12px">AI reads your EDA data and gives a plain-English interpretation — what the signals mean for your hypothesis and what to watch out for before running the analysis.</div>',unsafe_allow_html=True)
        if not (st.session_state.ark_key and st.session_state.ark_endpoint):
            st.markdown(f'<div style="font-size:12px;color:{C["amber"]}">Add your Ark API key and endpoint ID in the sidebar to enable this.</div>',unsafe_allow_html=True)
        else:
            if st.button("Generate EDA insights", key="llm_sot"):
                with st.spinner("Analysing your data..."):
                    summary = build_eda_summary(df, pre_df, post_df, st.session_state.hypothesis, "sot",
                        st.session_state.sot_channel_roles, st.session_state.target_col, conv_cols_sot, sess_cols_sot)
                    result, err = call_ark_eda(st.session_state.ark_key, st.session_state.ark_endpoint, summary, st.session_state.hypothesis, st.session_state.advertiser)
                    if result: st.session_state.llm_eda_result = result
                    else: st.error(f"Ark error: {err}")
            if st.session_state.get("llm_eda_result"):
                st.markdown(st.session_state.llm_eda_result)
        st.markdown("</div>",unsafe_allow_html=True)

        sot_b1,sot_b2 = st.columns([1,5])
        with sot_b1:
            if st.button("Back",key="sot_eda_back"): st.session_state.step=3; st.rerun()
        with sot_b2:
            if st.button("Continue to Flag Variables", use_container_width=True, key="sot_eda_fwd"):
                st.session_state.step=5; st.rerun()
        st.stop()

    conv_cols = [c for c in df.columns if "conversion" in c.lower() or c.lower()=="conversions"]
    spend_cats = [c for c in df.columns if c.endswith("Cost") and "Total" not in c and "CPA" not in c]
    date_min = df["p_date"].min().date()
    date_max = df["p_date"].max().date()

    target = st.selectbox("Target variable (the metric you want to measure — becomes dependent variable in the analysis)",
                           conv_cols, index=0) if conv_cols else None
    if target:
        st.session_state.target_col = target

    # ── Auto-suggest periods on first load ──────────────────────────────────
    def auto_suggest_periods(df):
        """Suggest pre/post periods from TopView detection and hypothesis."""
        # Find intervention point
        if "TopViewCost" in df.columns:
            tv = df[df["TopViewCost"] > 0]["p_date"]
            if len(tv):
                # TopView = 1 day (billing may show 2) — use first day as treatment start
                intervention_date = tv.min()
                pre_end_auto   = (intervention_date - pd.Timedelta(days=1)).date()
                post_start_auto = intervention_date.date()
                # Default post-period: 14 days (MP will adjust)
                post_end_auto  = min((intervention_date + pd.Timedelta(days=13)).date(),
                                     df["p_date"].max().date())
                pre_start_auto = df["p_date"].min().date()
                return pre_start_auto, pre_end_auto, post_start_auto, post_end_auto
        # No TopView — split at 80/20
        n = len(df)
        split = df["p_date"].iloc[int(n * 0.8)].date()
        return date_min, split, split, date_max

    if st.session_state.pre_start is None:
        ps, pe, pss, pse = auto_suggest_periods(df)
        st.session_state.pre_start  = ps
        st.session_state.pre_end    = pe
        st.session_state.post_start = pss
        st.session_state.post_end   = pse

    # ── Period selector UI ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:12px;font-weight:600;color:#aaa;margin-bottom:12px;letter-spacing:.05em">PERIOD CONFIGURATION</div>', unsafe_allow_html=True)

    # Period quality calc (live)
    def period_quality(pre_s, pre_e, post_s, post_e):
        pre_days  = (pd.Timestamp(pre_e)  - pd.Timestamp(pre_s)).days + 1
        post_days = (pd.Timestamp(post_e) - pd.Timestamp(post_s)).days + 1
        ratio     = pre_days / post_days if post_days > 0 else 0
        overlap   = pd.Timestamp(post_s) <= pd.Timestamp(pre_e)
        return pre_days, post_days, ratio, overlap

    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        st.markdown(f'<div style="font-size:11px;color:{C["grey"]};margin-bottom:4px;font-weight:600">PRE-PERIOD START</div>', unsafe_allow_html=True)
        pre_s = st.date_input("pre_s", value=st.session_state.pre_start,
            min_value=date_min, max_value=date_max, label_visibility="collapsed", key="ps_input")
    with pcol2:
        st.markdown(f'<div style="font-size:11px;color:{C["grey"]};margin-bottom:4px;font-weight:600">PRE-PERIOD END</div>', unsafe_allow_html=True)
        pre_e = st.date_input("pre_e", value=st.session_state.pre_end,
            min_value=date_min, max_value=date_max, label_visibility="collapsed", key="pe_input")
    with pcol3:
        st.markdown(f'<div style="font-size:11px;color:{C["grey"]};margin-bottom:4px;font-weight:600">POST-PERIOD START</div>', unsafe_allow_html=True)
        post_s = st.date_input("post_s", value=st.session_state.post_start,
            min_value=date_min, max_value=date_max, label_visibility="collapsed", key="pss_input")
    with pcol4:
        st.markdown(f'<div style="font-size:11px;color:{C["grey"]};margin-bottom:4px;font-weight:600">POST-PERIOD END</div>', unsafe_allow_html=True)
        post_e = st.date_input("post_e", value=st.session_state.post_end,
            min_value=date_min, max_value=date_max, label_visibility="collapsed", key="pse_input")

    # Persist selections
    st.session_state.pre_start  = pre_s
    st.session_state.pre_end    = pre_e
    st.session_state.post_start = post_s
    st.session_state.post_end   = post_e

    # Live quality bar
    pre_days, post_days, ratio, overlap = period_quality(pre_s, pre_e, post_s, post_e)
    ratio_ok  = ratio >= 2.0
    ratio_warn = 1.0 <= ratio < 2.0
    pre_ok    = pre_days >= 40
    pre_warn  = 28 <= pre_days < 40

    ratio_color = C["purple"] if ratio_ok else C["amber"] if ratio_warn else C["red"]
    pre_color   = C["purple"] if pre_ok   else C["amber"] if pre_warn  else C["red"]

    q_html = f'''<div style="display:flex;gap:10px;flex-wrap:wrap;margin:10px 0 18px">
      <div class="mtile" style="flex:none;min-width:100px"><div class="ml">Pre-period</div><div class="mv" style="color:{pre_color}">{pre_days}d</div></div>
      <div class="mtile" style="flex:none;min-width:100px"><div class="ml">Post-period</div><div class="mv">{post_days}d</div></div>
      <div class="mtile" style="flex:none;min-width:100px"><div class="ml">Ratio (pre:post)</div><div class="mv" style="color:{ratio_color}">{ratio:.1f}x</div></div>
      <div class="mtile" style="flex:1;min-width:200px"><div class="ml">Guidance</div><div style="font-size:12px;padding-top:4px">
        {f'<span style="color:{C["red"]}">⚠ Periods overlap — check dates</span>' if overlap else
         f'<span style="color:{C["red"]}">⚠ Pre-period too short (min 30 days)</span>' if pre_days < 30 else
         f'<span style="color:{C["amber"]}">Pre-period on the short side — extend if possible</span>' if pre_warn else
         f'<span style="color:{C["amber"]}">Pre:post ratio below 2x — analysis may be less stable</span>' if ratio_warn else
         f'<span style="color:{C["purple"]}">✓ Period configuration looks solid</span>'}
      </div></div>
    </div>'''
    st.markdown(q_html, unsafe_allow_html=True)

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
                fillcolor="rgba(124,58,237,0.05)", line_width=0,
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

        t1, t2, t3, t4, t5 = st.tabs(["📈 Time Series","💰 Spend Breakdown","🔗 Correlations","📊 Period Comparison","🧮 Covariate Selection"])

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
    PRE {pre_s.strftime("%d %b")}–{pre_e.strftime("%d %b %Y")} ({pre_days}d) &nbsp;→&nbsp; POST {post_s.strftime("%d %b")}–{post_e.strftime("%d %b %Y")} ({post_days}d)
  </div>
  <div class="mrow">
    {mtile("Pre-period avg", f"{pre_avg:,.1f}", "")}
    {mtile("Post-period avg", f"{post_avg:,.1f}", "")}
    {mtile("Raw lift (unadjusted)", lift_label, "mvc" if raw_lift > 5 else "mva" if raw_lift > 0 else "mvr")}
    {mtile("Pre:post ratio", f"{ratio:.1f}x", "mvc" if ratio_ok else "mva" if ratio_warn else "mvr")}
  </div>
  <div style="font-size:11px;color:{C["grey"]}">
    Raw lift is unadjusted — the external analysis controls for covariates to isolate the true incremental effect.
    {"&nbsp;⚠ Only " + str(post_days) + " post-period days — BSTS needs at least 7, ideally 14+" if post_days < 7 else ""}
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
                    cov_str = ", ".join(f"`{c}`" for c in selected_covs)
                    st.markdown(f'<div style="background:{C["purple"]}0A;border:1px solid {C["purple"]}33;border-radius:8px;padding:12px 16px"><div style="font-size:11px;color:{C["purple"]};font-weight:700;margin-bottom:6px">SELECTED COVARIATES ({len(selected_covs)})</div><div style="font-size:13px;color:{C["text"]}">{", ".join(selected_covs)}</div><div style="font-size:11px;color:{C["grey"]};margin-top:6px">These will appear in your config summary and export notes. Enter them as covariates when you run the analysis in the external tool.</div></div>', unsafe_allow_html=True)
                else:
                    st.warning("No covariates selected — the analysis will run as univariate (less accurate counterfactual).")
            else:
                st.info("Define your pre and post periods in the Period Comparison tab first — covariate analysis requires both periods to be set.")

    # LLM EDA Insights
    st.markdown("<br>",unsafe_allow_html=True)
    with st.container():
        st.markdown(f'<div style="background:linear-gradient(135deg,{C["purple"]}08,{C["surface"]});border:1px solid {C["purple"]}33;border-radius:12px;padding:18px 20px">',unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:13px;font-weight:600;color:{C["text"]};margin-bottom:4px">AI EDA Interpretation</div>',unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:12px">AI reads your pre/post data and returns a structured interpretation — key signals, evidence for TikTok impact, risks before modelling.</div>',unsafe_allow_html=True)
        if not (st.session_state.ark_key and st.session_state.ark_endpoint):
            st.markdown(f'<div style="font-size:12px;color:{C["amber"]}">Add your Ark API key and endpoint ID in the sidebar to enable this.</div>',unsafe_allow_html=True)
        else:
            if st.button("Generate EDA insights", key="llm_ttam"):
                with st.spinner("Analysing your data..."):
                    pre_s = st.session_state.pre_start; post_s = st.session_state.post_start; post_e = st.session_state.post_end
                    pre_df_llm  = df[df["p_date"].dt.date < pre_s] if pre_s else df
                    post_df_llm = df[(df["p_date"].dt.date >= post_s) & (df["p_date"].dt.date <= post_e)] if post_s and post_e else pd.DataFrame()
                    summary = build_eda_summary(df, pre_df_llm, post_df_llm, st.session_state.hypothesis,
                        st.session_state.file_type, st.session_state.sot_channel_roles,
                        st.session_state.target_col, conv_cols, spend_cats)
                    result, err = call_ark_eda(st.session_state.ark_key, st.session_state.ark_endpoint, summary, st.session_state.hypothesis, st.session_state.advertiser)
                    if result: st.session_state.llm_eda_result = result
                    else: st.error(f"Ark error: {err}")
            if st.session_state.get("llm_eda_result"):
                st.markdown(st.session_state.llm_eda_result)
        st.markdown("</div>",unsafe_allow_html=True)

    cb1, cb2 = st.columns([1,5])
    with cb1:
        if st.button("← Back"): st.session_state.step=3; st.rerun()
    with cb2:
        if st.button("Continue to Flag Variables →", use_container_width=True):
            st.session_state.step=5; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — FLAG VARIABLES
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 5:
    st.markdown(f'{badge("Step 5")} <span style="font-size:20px;font-weight:600;margin-left:10px">Flag Variables</span>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{C["grey"]};margin-bottom:18px">Confirmed flags are added as binary columns to your export. Include them as covariates when running the analysis.</div>', unsafe_allow_html=True)

    df = parse_dates(st.session_state.daily_df.copy())

    if not st.session_state.flag_vars:
        st.session_state.flag_vars = suggest_flags(df)

    for i, fv in enumerate(st.session_state.flag_vars):
        style = "card-c" if fv.get("is_treatment") else "card-a"
        label = "TREATMENT WINDOW" if fv.get("is_treatment") else "COVARIATE FLAG"
        lp = "pill-c" if fv.get("is_treatment") else "pill-a"
        st.markdown(f'<div class="card {style}">{pill(label,lp)} <strong style="margin-left:8px;font-size:15px">{fv["name"]}</strong><div style="font-size:12px;color:{C["grey"]};margin-top:5px">{fv["rationale"]}</div></div>', unsafe_allow_html=True)
        cn, cs, ce, cc = st.columns([2,2,2,1])
        with cn:
            nn = st.text_input("Name", value=fv["name"], key=f"fn{i}", label_visibility="collapsed")
            st.session_state.flag_vars[i]["name"] = nn
        with cs:
            ns = st.date_input("Start", value=pd.to_datetime(fv["start"]).date(),
                min_value=df["p_date"].min().date(), max_value=df["p_date"].max().date(),
                key=f"fs{i}", label_visibility="collapsed")
            st.session_state.flag_vars[i]["start"] = str(ns)
        with ce:
            ne = st.date_input("End", value=pd.to_datetime(fv["end"]).date(),
                min_value=df["p_date"].min().date(), max_value=df["p_date"].max().date(),
                key=f"fe{i}", label_visibility="collapsed")
            st.session_state.flag_vars[i]["end"] = str(ne)
        with cc:
            chk = st.checkbox("Include", value=fv.get("confirmed",False), key=f"fc{i}")
            st.session_state.flag_vars[i]["confirmed"] = chk

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("➕ Add custom flag"):
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
        if st.button("← Back"): st.session_state.step=4; st.rerun()
    with cb2:
        if st.button("Continue to Export →", use_container_width=True): st.session_state.step=6; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — SMOOTH & EXPORT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 6:
    st.markdown(f'{badge("Step 6")} <span style="font-size:20px;font-weight:600;margin-left:10px">Smooth & Export</span>', unsafe_allow_html=True)

    df = parse_dates(st.session_state.daily_df.copy())
    target = st.session_state.target_col or next((c for c in df.columns if "conversion" in c.lower()), None)
    confirmed = [f for f in st.session_state.flag_vars if f.get("confirmed")]
    is_sot = st.session_state.file_type == "sot"

    for fv in confirmed:
        df[fv["name"]] = ((df["p_date"] >= pd.to_datetime(fv["start"])) & (df["p_date"] <= pd.to_datetime(fv["end"]))).astype(int)

    # Smoothing (TTAM only)
    smooth = False
    if not is_sot:
        cs, cw = st.columns([3,1])
        with cs:
            smooth = st.toggle(f"Apply rolling average smoothing to {target}", value=st.session_state.smoothing)
            st.session_state.smoothing = smooth
        with cw:
            w = st.number_input("Window", value=st.session_state.smooth_window, min_value=3, max_value=14, step=1) if smooth else 7
            st.session_state.smooth_window = w
        if smooth and target:
            df[f"{target}_Smoothed"] = df[target].rolling(w, center=True, min_periods=3).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["p_date"],y=df[target],mode="lines",line=dict(color=C["grey"],width=1,dash="dot"),name="Raw",opacity=0.4))
            fig.add_trace(go.Scatter(x=df["p_date"],y=df[f"{target}_Smoothed"],mode="lines",line=dict(color=C["purple"],width=2.5),name=f"{w}-day smoothed"))
            fig.update_layout(**CHART,title=dict(text="Raw vs smoothed",font=dict(size=14)))
            st.plotly_chart(fig,use_container_width=True)

    dep_var = f"{target}_Smoothed" if smooth and target else (target or "—")

    # Build display values
    adv = st.session_state.advertiser or "—"
    pc  = st.session_state.parsed_context
    flag_names    = ", ".join(f["name"] for f in confirmed) or "None"
    spend_cats_str = ", ".join(c.replace("Cost","") for c in df.columns if c.endswith("Cost") and "Total" not in c) or "—"
    smooth_note   = f"{w if smooth else 7}-day rolling avg on {target}" if smooth else "None"
    cov_display   = ", ".join(c for c,r in st.session_state.sot_channel_roles.items() if r=="Covariate") if is_sot and st.session_state.sot_channel_roles else ", ".join(k for k,v in st.session_state.covariate_selection.items() if v) or "None selected"

    # Build run instructions
    if is_sot and st.session_state.sot_targets:
        cov_sot = [c for c,r in st.session_state.sot_channel_roles.items() if r=="Covariate"]
        runs_html = ""
        for i, tgt in enumerate(st.session_state.sot_targets):
            label = f"Run {i+1} — {'Primary' if i==0 else 'Secondary'}"
            color = C["purple"] if i==0 else "#0891B2"
            runs_html += f'<div style="border-left:3px solid {color};padding:7px 12px;margin:4px 0;background:{color}08;border-radius:0 6px 6px 0"><span style="font-size:11px;font-weight:700;color:{color}">{label}</span>  <span style="font-size:12px;color:{C["text"]}"><strong>{tgt}</strong></span>  <span style="font-size:11px;color:{C["grey"]}">· covariates: {", ".join(cov_sot) if cov_sot else "none"}</span></div>'
    else:
        runs_html = f'<div style="border-left:3px solid {C["purple"]};padding:7px 12px;margin:4px 0;background:{C["purple"]}08;border-radius:0 6px 6px 0"><span style="font-size:11px;font-weight:700;color:{C["purple"]}">Run 1</span>  <span style="font-size:12px;color:{C["text"]}"><strong>{dep_var}</strong></span>  <span style="font-size:11px;color:{C["grey"]}">· covariates: {cov_display}</span></div>'

    # Condensed config + run guide in one block
    st.markdown(f"""<div class="card card-c">
<div style="font-size:12px;font-weight:600;color:{C['purple']};margin-bottom:14px;letter-spacing:.05em">ANALYSIS SUMMARY & EXPORT</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 32px;font-size:13px;margin-bottom:14px">
  <div><span style="color:{C['grey']}">Advertiser</span><br><strong>{adv}</strong></div>
  <div><span style="color:{C['grey']}">Data range</span><br><strong>{df['p_date'].min().strftime('%d %b %Y')} – {df['p_date'].max().strftime('%d %b %Y')}</strong></div>
  <div><span style="color:{C['purple']};font-weight:600">Pre-period</span><br><strong>{str(st.session_state.pre_start) if st.session_state.pre_start else '—'} to {str(st.session_state.pre_end) if st.session_state.pre_end else '—'}</strong></div>
  <div><span style="color:#059669;font-weight:600">Post-period</span><br><strong>{str(st.session_state.post_start) if st.session_state.post_start else '—'} to {str(st.session_state.post_end) if st.session_state.post_end else '—'}</strong></div>
  <div><span style="color:{C['grey']}">Covariates</span><br><strong>{cov_display}</strong></div>
  <div><span style="color:{C['grey']}">Flag variables</span><br><strong>{flag_names}</strong></div>
</div>
<div style="font-size:11px;font-weight:700;color:{C['grey']};letter-spacing:.08em;margin-bottom:6px">LOAD INTO ANALYSIS TOOL</div>
{runs_html}
</div>""", unsafe_allow_html=True)

    # Download + reset
    out = df.copy(); out["p_date"] = out["p_date"].dt.strftime("%Y-%m-%d")
    safe  = (st.session_state.advertiser or "advertiser").replace(" ","_")
    fname = f"{safe}_AnalysisInput_{datetime.today().strftime('%Y%m%d')}.csv"
    csv_bytes = out.to_csv(index=False).encode("utf-8")

    dl_col, rs_col = st.columns([3,1])
    with dl_col:
        st.download_button(f"Download CSV — {fname}", data=csv_bytes, file_name=fname, mime="text/csv", use_container_width=True)
    with rs_col:
        if st.button("Start new analysis", use_container_width=True):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

    with st.expander(f"Preview dataset ({len(df)} rows · {len(df.columns)} columns)"):
        prev = df.copy(); prev["p_date"] = prev["p_date"].dt.strftime("%Y-%m-%d")
        st.dataframe(prev.head(15), use_container_width=True, height=280)

    if st.button("← Back"): st.session_state.step=5; st.rerun()
