
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

ARK_BASE_URL = "https://api.groq.com/openai/v1/"
OPENAI_DEFAULT_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
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
    if not api_key: return None, "No Ark API key in sidebar."
    if not endpoint_id: return None, "No endpoint ID in sidebar."
    try:
        client = OpenAI(base_url=ARK_BASE_URL, api_key=api_key)
        hyp_name = next((u["name"] for u in USE_CASES if u["id"] == hypothesis), hypothesis or "unspecified")
        prompt = (
            "You are a senior TikTok measurement analyst reviewing EDA for a causal study.\n\n"
            f"ADVERTISER: {advertiser or 'Unknown'}\nHYPOTHESIS: {hyp_name}\n\nEDA SUMMARY:\n{summary_text}\n\n"
            "Provide analysis in this exact structure:\n\n"
            "What the data shows\n2-3 sentences on key signals. Be specific — name channels and percentages.\n\n"
            "Strongest evidence for TikTok impact\nThe 1-2 most compelling signals.\n\n"
            "Watch outs before modelling\nUp to 3 specific risks. Be direct.\n\n"
            "Recommendation\nOne sentence: ready to model, or needs attention first?"
        )
        r = client.chat.completions.create(model=endpoint_id, messages=[{"role":"user","content":prompt}], temperature=0.3, max_tokens=800)
        return r.choices[0].message.content, None
    except Exception as e:
        import traceback
        return None, f"{type(e).name}: {str(e)}\n\n{traceback.format_exc()}"


def call_ark_summary(api_key, endpoint_id, context):
    """Generate a structured pre-modelling summary report."""
    if not api_key or not endpoint_id:
        return None, "No API key configured."
    prompt = f"""You are a senior TikTok Measurement Partner preparing a pre-modelling analysis brief.

DATA CONTEXT:
{context}

Write a structured pre-modelling summary report for an internal audience (Measurement Partner presenting to a client team). 
Use this exact structure with these exact headers:

## Data Overview
One paragraph: date range, total days, pre/post period lengths and ratio quality (2:1 minimum is good, below 1.5:1 is risky), data source.

## Channel Performance Summary
For each channel with data, one bullet: pre-period average → post-period average, % change, and one-sentence interpretation. Bold the most important finding.

## Pre-Period Stability Assessment
One paragraph assessing whether the pre-period is clean enough to train a reliable counterfactual. Mention CV% for key channels. Flag any channels with CV > 30% or slope > 1.5%/day as risks.

## Signals Supporting TikTok Incrementality
2-3 bullets of the strongest evidence from the data that TikTok drove the observed lift (e.g. TikTok conversions rose while other channels held flat, CVR improved post-campaign, direct channel didn't move).

## Risks and Watch-Outs
2-3 bullets of specific risks that could undermine the causal claim (e.g. high pre-period volatility, channels moving together suggesting external demand, short post-period).

## Covariate Recommendation
One paragraph: which channels to include as BSTS covariates and why. Reference correlation with TikTok in the pre-period. Name specific columns.

## Readiness Rating
One of: ✅ Ready to model | ⚠️ Proceed with caution | ❌ Needs attention first
One sentence explaining the rating.

Be specific — name channels, quote percentages, reference actual dates. Write in confident, direct language suitable for a senior analytical audience."""

    try:
        client = OpenAI(base_url=ARK_BASE_URL, api_key=api_key)
        r = client.chat.completions.create(
            model=endpoint_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=1200
        )
        return r.choices[0].message.content, None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)}"


# ── Visual Studio helpers ──────────────────────────────────────────────────────

def build_visual_context(df, pre_df, post_df, hypothesis, conv_cols_sot, sess_cols_sot, rev_cols_sot, channels):
    """Build a compact context dict for the LLM Visual Studio system prompt."""
    CH_ORDER_VS = ["Direct","Paid_Social_TikTok","Paid_Search_Google","Paid_Social_Meta","Organic_Search"]
    ch_summaries = {}
    for ch in CH_ORDER_VS:
        if ch not in channels:
            continue
        cols = channels[ch]
        c_col = next((c for c in cols if "Conversions" in c), None)
        s_col = next((c for c in cols if "Sessions" in c), None)
        entry = {}
        if c_col and c_col in pre_df.columns:
            pre_avg  = pre_df[c_col].mean() if len(pre_df) else 0
            post_avg = post_df[c_col].mean() if len(post_df) and c_col in post_df.columns else 0
            lift = (post_avg - pre_avg) / pre_avg * 100 if pre_avg else 0
            entry["conversions"] = {"pre_avg": round(pre_avg,1), "post_avg": round(post_avg,1), "lift_pct": round(lift,1)}
        if s_col and s_col in pre_df.columns:
            entry["sessions_pre_avg"] = round(pre_df[s_col].mean(), 1) if len(pre_df) else 0
        if entry:
            ch_summaries[ch] = entry
    hyp_name = next((u["name"] for u in USE_CASES if u["id"] == hypothesis), hypothesis or "Not specified")
    return {
        "hypothesis": hyp_name,
        "date_range": f"{df['p_date'].min().date()} to {df['p_date'].max().date()}",
        "pre_period":  f"{pre_df['p_date'].min().date()} to {pre_df['p_date'].max().date()}" if len(pre_df) else None,
        "post_period": f"{post_df['p_date'].min().date()} to {post_df['p_date'].max().date()}" if len(post_df) else None,
        "available_channels": [ch for ch in CH_ORDER_VS if ch in channels],
        "has_revenue": bool(rev_cols_sot),
        "channel_performance": ch_summaries,
    }


def call_ark_visual_studio(api_key, endpoint_id, user_message, context_dict, chat_history):
    """
    LLM returns a JSON chart spec — NOT code. We render it ourselves.
    This is safer, faster, and guarantees consistent branding.
    """
    if not api_key or not endpoint_id:
        return None, "No Ark API key or endpoint in sidebar."

    system_prompt = f"""You are a data visualisation assistant for TikTok causal impact analysis.
You help Measurement Partners explore their GA4/SOT data by creating charts.

DATA CONTEXT:
{json.dumps(context_dict, indent=2)}

AVAILABLE CHART TYPES — you must choose exactly one:

"time_series"        — line chart of a metric over time, pre+post shaded, with period avg labels
fields: metric ("Conversions"|"Sessions"|"Revenue"), channels (list), smoothing (1|3|7|14), title, insight
"cvr"                — conversion rate (%) by channel over time
fields: channels (list), smoothing (1|3|5|7), title, insight
"pre_post_bar"       — grouped bar: daily average per channel, pre vs post, with % change labels
fields: metric ("Conversions"|"Sessions"|"Revenue"), channels (list), title, insight
"stability"          — pre-period stability for one channel with 7-day avg and post overlay
fields: metric, channels (single-item list), title, insight
"scatter"            — scatter plot two columns, coloured by period
fields: x_col, y_col (exact column names), title, insight
"elasticity_scatter" — TikTok sessions vs conversions scatter, regression lines pre/post, efficiency lift annotation
fields: title, insight (use when asked about elasticity, efficiency shift, ROAS, conversion rate shift)
"correlation_heatmap"— pairwise correlation matrix of all channels for a given period
fields: period ("pre"|"post"|"full"), title, insight
"counterfactual"     — naive linear counterfactual: pre-period trend extrapolated into post, vs actual, with lift annotation
fields: metric ("Conversions"|"Sessions"|"Revenue"), channels (single-item list — pick the primary channel), title, insight
"efficiency_trend"   — dual-axis: sessions as bars, CVR%/revenue-per-session as line, pre+post avg. Use for ROAS, efficiency, CVR evolution questions
fields: channels (single-item list), title, insight
"weekly_composition" — stacked weekly session bars by channel + total conversions/revenue line. Use for spend mix, weekly volume questions
fields: metric ("Conversions"|"Sessions"|"Revenue"), channels (list), title, insight
"text"               — when the request cannot be answered with a chart
fields: message (plain English explanation)
STRICT RULES:
Respond ONLY with valid JSON. No markdown, no prose outside the JSON object.
channels must only contain values from available_channels in the context.
insight is required for chart types 1-5: one sentence plain-English finding from the data context.
If channels are not specified in the request, default to all available_channels.
Never invent column names. Only use what is in available_channels.
EXAMPLE RESPONSE:
{{"chart_type":"time_series","metric":"Conversions","channels":["Direct","Paid_Social_TikTok"],"smoothing":7,"title":"7-day rolling conversions — TikTok vs Direct","insight":"TikTok conversions rose 34% post-campaign while Direct held flat, suggesting direct attribution rather than halo."}}"""

    messages = [{"role": "system", "content": system_prompt}]
    for turn in chat_history[-6:]:
        role = turn["role"]
        content = turn["content"] if role == "user" else turn.get("raw_response", "")
        if content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})

    try:
        client = OpenAI(base_url=ARK_BASE_URL, api_key=api_key)
        r = client.chat.completions.create(
            model=endpoint_id, messages=messages, temperature=0.15, max_tokens=500
        )
        raw = r.choices[0].message.content.strip().replace("json","").replace("","").strip()
        spec = json.loads(raw)
        return spec, None
    except json.JSONDecodeError as je:
        return {"chart_type": "text", "message": f"I couldn't parse the response into a chart spec ({je}). Try rephrasing."}, None
    except Exception as e:
        import traceback
        return None, f"{type(e).name}: {str(e)}\n\n{traceback.format_exc()}"


def render_chart_from_spec(spec, df, pre_df, post_df, channels, ch_col_fn, ch_lbl_fn, t_start, t_end, CHART_LAYOUT, C_pal):
    """Render a Plotly figure from a JSON chart spec. No exec() — safe, branded, consistent."""
    def add_iv(fig):
        if t_start and t_end:
            fig.add_vrect(x0=str(t_start.date()), x1=str(t_end.date()),
                          fillcolor="rgba(124,58,237,0.07)", line_width=0)
            fig.add_shape(type="line", x0=str(t_start.date()), x1=str(t_start.date()),
                          y0=0, y1=1, yref="paper",
                          line=dict(color=C_pal["purple"], width=2, dash="dash"))
            fig.add_annotation(x=str(t_start.date()), y=1.02, yref="paper",
                               text="▼ Campaign start", showarrow=False, yanchor="bottom",
                               font=dict(color=C_pal["purple"], size=10, family="Inter"), xshift=4)
        return fig

    def annotate_period_avgs(fig, col, pre_df, post_df, color, t_start, t_end, df):
        """Add pre/post average labels to a time series trace."""
        if len(pre_df) and col in pre_df.columns:
            pre_avg = pre_df[col].mean()
            fig.add_shape(type="line",
                x0=str(df["p_date"].min().date()), x1=str(t_start.date()) if t_start else str(df["p_date"].max().date()),
                y0=pre_avg, y1=pre_avg,
                line=dict(color=color, width=1, dash="dot"), opacity=0.4)
            mid_pre = df["p_date"].iloc[len(pre_df)//2]
            fig.add_annotation(x=str(mid_pre.date()), y=pre_avg,
                text=f"Pre avg: {pre_avg:,.0f}",
                showarrow=False, yanchor="bottom", xanchor="center",
                font=dict(color=color, size=9, family="Inter"),
                bgcolor="rgba(255,255,255,0.8)", borderpad=2)
        if len(post_df) and col in post_df.columns and t_start:
            post_avg = post_df[col].mean()
            mid_post = post_df["p_date"].iloc[len(post_df)//2]
            fig.add_annotation(x=str(mid_post.date()), y=post_avg,
                text=f"Post avg: {post_avg:,.0f}",
                showarrow=False, yanchor="bottom", xanchor="center",
                font=dict(color=color, size=9, family="Inter"),
                bgcolor="rgba(255,255,255,0.8)", borderpad=2)
        return fig

    METRIC_KW = {"Conversions": "Conversions", "Sessions": "Sessions", "Revenue": "Revenue"}
    chart_type   = spec.get("chart_type", "text")
    title        = spec.get("title", "")
    selected_chs = spec.get("channels", list(channels.keys()))
    smoothing    = max(1, int(spec.get("smoothing", 1)))
    metric       = spec.get("metric", "Conversions")
    metric_kw    = METRIC_KW.get(metric, "Conversions")
    show_labels  = spec.get("show_labels", True)

    if chart_type == "text":
        return None

    fig = go.Figure()

    # ── Time Series ──────────────────────────────────────────────────────────
    if chart_type == "time_series":
        first_col = None
        first_color = None
        for ch in selected_chs:
            if ch not in channels: continue
            col = next((c for c in channels[ch] if metric_kw in c), None)
            if not col or col not in df.columns: continue
            series = df[col].rolling(smoothing, center=True, min_periods=1).mean()
            nm = ch_lbl_fn(ch) + (f" ({smoothing}d avg)" if smoothing > 1 else "")
            fig.add_trace(go.Scatter(x=df["p_date"], y=series, mode="lines",
                                     line=dict(color=ch_col_fn(ch), width=2.5), name=nm))
            if first_col is None:
                first_col = col; first_color = ch_col_fn(ch)
        # Period average annotations on the primary trace
        if show_labels and first_col and t_start:
            annotate_period_avgs(fig, first_col, pre_df, post_df, first_color, t_start, t_end, df)
        fig.update_layout(**CHART_LAYOUT, title=dict(text=title or f"{metric} by Channel", font=dict(size=14)))
        add_iv(fig)

    # ── CVR ──────────────────────────────────────────────────────────────────
    elif chart_type == "cvr":
        for ch in selected_chs:
            if ch not in channels: continue
            s_col = next((c for c in channels[ch] if "Sessions" in c), None)
            c_col = next((c for c in channels[ch] if "Conversions" in c), None)
            if not s_col or not c_col: continue
            cvr = (df[c_col] / df[s_col].replace(0, np.nan) * 100).rolling(smoothing, center=True, min_periods=1).mean()
            fig.add_trace(go.Scatter(x=df["p_date"], y=cvr, mode="lines",
                                     line=dict(color=ch_col_fn(ch), width=2.5), name=ch_lbl_fn(ch)))
        fig.update_layout(**CHART_LAYOUT, title=dict(text=title or "Conversion Rate (%) by Channel", font=dict(size=14)))
        add_iv(fig)

    # ── Pre vs Post Bar ───────────────────────────────────────────────────────
    elif chart_type == "pre_post_bar":
        pv, ptv, lbs, deltas = [], [], [], []
        for ch in selected_chs:
            if ch not in channels: continue
            col = next((c for c in channels[ch] if metric_kw in c), None)
            if not col or col not in df.columns: continue
            pre_v  = pre_df[col].mean()  if len(pre_df)  else 0
            post_v = post_df[col].mean() if len(post_df) else 0
            pv.append(pre_v); ptv.append(post_v)
            lbs.append(ch_lbl_fn(ch))
            deltas.append((post_v - pre_v) / pre_v * 100 if pre_v else 0)
        fig.add_trace(go.Bar(name="Pre-Period", x=lbs, y=pv,
                              marker_color=C_pal["grey_l"], opacity=0.8,
                              text=[f"{v:,.0f}" for v in pv], textposition="outside",
                              textfont=dict(size=10, color=C_pal["grey"])))
        fig.add_trace(go.Bar(name="Post-Period", x=lbs, y=ptv,
                              marker_color=C_pal["purple"],
                              text=[f"{v:,.0f}<br><b>{d:+.1f}%</b>" for v, d in zip(ptv, deltas)],
                              textposition="outside",
                              textfont=dict(size=10, color=C_pal["purple"])))
        fig.update_layout(**CHART_LAYOUT, barmode="group",
                           title=dict(text=title or f"Daily Avg {metric} — Pre vs Post", font=dict(size=14)))

    # ── Stability ─────────────────────────────────────────────────────────────
    elif chart_type == "stability":
        ch = selected_chs[0] if selected_chs else None
        col = next((c for c in channels.get(ch, []) if metric_kw in c), None) if ch else None
        if col and col in pre_df.columns:
            y = pre_df[col].fillna(0).values
            rolling = pd.Series(y).rolling(7, center=True, min_periods=3).mean()
            fig.add_trace(go.Scatter(x=pre_df["p_date"], y=y, mode="lines",
                                     line=dict(color=C_pal["grey_l"], width=1.2, dash="dot"),
                                     name="Daily", opacity=0.55))
            fig.add_trace(go.Scatter(x=pre_df["p_date"], y=rolling, mode="lines",
                                     line=dict(color=C_pal["purple"], width=2.5), name="7-day avg"))
            if len(post_df) and col in post_df.columns:
                fig.add_trace(go.Scatter(x=post_df["p_date"], y=post_df[col], mode="lines",
                                         line=dict(color=C_pal["green"], width=2.5), name="Post-period"))
            pre_avg = float(y.mean())
            fig.add_hline(y=pre_avg, line_dash="dash", line_color=C_pal["grey"], opacity=0.5,
                          annotation_text=f"Pre avg: {pre_avg:,.0f}",
                          annotation_font=dict(size=10, color=C_pal["grey"]),
                          annotation_position="right")
            add_iv(fig)
        fig.update_layout(**CHART_LAYOUT, title=dict(text=title or f"{metric} — Pre-period Stability", font=dict(size=14)))

    # ── Scatter ───────────────────────────────────────────────────────────────
    elif chart_type == "scatter":
        x_col = spec.get("x_col"); y_col = spec.get("y_col")
        if x_col and y_col and x_col in df.columns and y_col in df.columns:
            pre_mask  = df["p_date"] < t_start  if t_start else pd.Series([True]*len(df), index=df.index)
            post_mask = ((df["p_date"] >= t_start) & (df["p_date"] <= t_end)) if t_start and t_end else pd.Series([False]*len(df), index=df.index)
            fig.add_trace(go.Scatter(x=df[pre_mask][x_col], y=df[pre_mask][y_col],
                                     mode="markers", name="Pre-period (BAU)",
                                     marker=dict(color=C_pal["grey_l"], size=8, opacity=0.7)))
            fig.add_trace(go.Scatter(x=df[post_mask][x_col], y=df[post_mask][y_col],
                                     mode="markers", name="Post-period",
                                     marker=dict(color=C_pal["purple"], size=10)))
        fig.update_layout(**CHART_LAYOUT, title=dict(text=title or f"{spec.get('y_col','')} vs {spec.get('x_col','')}", font=dict(size=14)))

    # ── Elasticity Scatter ────────────────────────────────────────────────────
    elif chart_type == "elasticity_scatter":
        tt_ch = "Paid_Social_TikTok"
        s_col = next((c for c in channels.get(tt_ch, []) if "Sessions" in c), None)
        c_col = next((c for c in channels.get(tt_ch, []) if "Conversions" in c), None)
        if s_col and c_col and s_col in df.columns and c_col in df.columns:
            pre_mask  = df["p_date"] < t_start  if t_start else pd.Series([True]*len(df), index=df.index)
            post_mask = ((df["p_date"] >= t_start) & (df["p_date"] <= t_end)) if t_start and t_end else pd.Series([False]*len(df), index=df.index)
            pre_x  = df[pre_mask][s_col].values;  pre_y  = df[pre_mask][c_col].values
            post_x = df[post_mask][s_col].values; post_y = df[post_mask][c_col].values
            fig.add_trace(go.Scatter(x=pre_x, y=pre_y, mode="markers", name="Pre-period (BAU)",
                                     marker=dict(color=C_pal["grey_l"], size=8, opacity=0.75)))
            fig.add_trace(go.Scatter(x=post_x, y=post_y, mode="markers", name="Post-period",
                                     marker=dict(color=C_pal["purple"], size=10)))
            all_x = np.concatenate([pre_x, post_x]) if len(post_x) else pre_x
            x_range = np.linspace(all_x.min(), all_x.max(), 100) if len(all_x) > 1 else np.array([])
            if len(pre_x) > 2:
                z = np.polyfit(pre_x, pre_y, 1); p = np.poly1d(z)
                fig.add_trace(go.Scatter(x=x_range, y=p(x_range), mode="lines", name="BAU trend",
                                         line=dict(color=C_pal["grey"], dash="dash", width=1.5)))
            if len(post_x) > 2:
                z2 = np.polyfit(post_x, post_y, 1); p2 = np.poly1d(z2)
                fig.add_trace(go.Scatter(x=x_range, y=p2(x_range), mode="lines", name="Post-campaign trend",
                                         line=dict(color=C_pal["purple"], width=2.5)))
                # Efficiency lift annotation at midpoint spend
                if len(pre_x) > 2:
                    mid_spend = np.median(pre_x)
                    bau_conv   = p(mid_spend)
                    post_conv  = p2(mid_spend)
                    lift_pct   = (post_conv - bau_conv) / bau_conv * 100 if bau_conv else 0
                    direction  = "more" if lift_pct >= 0 else "fewer"
                    lift_color = C_pal["purple"] if lift_pct >= 0 else C_pal["red"]
                    fig.add_annotation(x=mid_spend, y=post_conv,
                        text=f"<b>{abs(lift_pct):.0f}% {direction} conversions<br>at same spend level</b>",
                        showarrow=True, arrowhead=2, arrowcolor=lift_color,
                        font=dict(color=lift_color, size=11, family="Inter"),
                        bgcolor="white", bordercolor=lift_color, borderwidth=1, borderpad=6)
            fig.update_xaxes(title="TikTok Sessions")
            fig.update_yaxes(title="TikTok Conversions")
        fig.update_layout(**CHART_LAYOUT, title=dict(text=title or "Elasticity Shift — Sessions vs Conversions (Pre vs Post)", font=dict(size=14)))

    # ── Correlation Heatmap ───────────────────────────────────────────────────
    elif chart_type == "correlation_heatmap":
        period = spec.get("period", "pre")
        data_period = pre_df if period == "pre" else (post_df if period == "post" else df)
        conv_sess_cols = [c for c in df.columns
                          if any(k in c for k in ["Conversions","Sessions"])
                          and c in data_period.columns
                          and data_period[c].std() > 0][:8]
        if len(conv_sess_cols) >= 2:
            corr = data_period[conv_sess_cols].corr()
            short_labels = [c.replace("Paid_Social_","").replace("Paid_Search_","").replace("_Conversions","Conv").replace("_Sessions","Sess").replace("_"," ") for c in conv_sess_cols]
            z_text = [[f"{corr.iloc[i,j]:.2f}" for j in range(len(conv_sess_cols))] for i in range(len(conv_sess_cols))]
            fig.add_trace(go.Heatmap(
                z=corr.values, x=short_labels, y=short_labels,
                text=z_text, texttemplate="%{text}", textfont=dict(size=10),
                colorscale=[[0,"#DC2626"],[0.5,"#F3F4F6"],[1,"#7C3AED"]],
                zmid=0, showscale=True,
                colorbar=dict(title="R", thickness=12, len=0.8)
            ))
        fig.update_layout(**CHART_LAYOUT)
        fig.update_layout(title=dict(text=title or f"Channel Correlation — {period.title()} Period", font=dict(size=14)), xaxis=dict(tickangle=-30))

    # ── Naive Counterfactual ──────────────────────────────────────────────────
    elif chart_type == "counterfactual":
        ch = selected_chs[0] if selected_chs else (list(channels.keys())[0] if channels else None)
        col = next((c for c in channels.get(ch, []) if metric_kw in c), None) if ch else None
        if col and col in pre_df.columns and t_start and t_end and len(post_df) > 0:
            pre_y = pre_df[col].fillna(0).values
            pre_x = np.arange(len(pre_y))
            coeffs = np.polyfit(pre_x, pre_y, 1)
            trend_fn = np.poly1d(coeffs)
            post_len = len(post_df)
            post_x_idx = np.arange(len(pre_y), len(pre_y) + post_len)
            cf_values = trend_fn(post_x_idx)
            # Plot pre-period
            fig.add_trace(go.Scatter(x=pre_df["p_date"], y=pre_y, mode="lines",
                name="Pre-period actual", line=dict(color=C_pal["grey"], width=2)))
            # Counterfactual (dashed)
            fig.add_trace(go.Scatter(x=post_df["p_date"], y=cf_values, mode="lines",
                name="Expected (no campaign)", line=dict(color=C_pal["grey_l"], width=2, dash="dash")))
            # Actual post
            if col in post_df.columns:
                post_actual = post_df[col].fillna(0).values
                fig.add_trace(go.Scatter(x=post_df["p_date"], y=post_actual, mode="lines",
                    name="Post-period actual", line=dict(color=C_pal["purple"], width=2.5)))
                # Shaded lift area
                fig.add_trace(go.Scatter(
                    x=list(post_df["p_date"]) + list(reversed(list(post_df["p_date"]))),
                    y=list(post_actual) + list(reversed(cf_values.tolist())),
                    fill="toself", fillcolor="rgba(124,58,237,0.12)",
                    line=dict(color="rgba(0,0,0,0)"), name="Lift gap", showlegend=True))
                # Lift annotation
                actual_sum = post_actual.sum(); cf_sum = cf_values.sum()
                lift_pct = (actual_sum - cf_sum) / cf_sum * 100 if cf_sum else 0
                lift_abs = actual_sum - cf_sum
                mid_idx = len(post_df) // 2
                fig.add_annotation(
                    x=str(post_df["p_date"].iloc[mid_idx].date()),
                    y=float(post_actual[mid_idx]),
                    text=f"<b>{lift_pct:+.1f}% above expected<br>({lift_abs:+,.0f} total)</b>",
                    showarrow=True, arrowhead=2, arrowcolor=C_pal["purple"],
                    font=dict(color=C_pal["purple"], size=11, family="Inter"),
                    bgcolor="white", bordercolor=C_pal["purple"], borderwidth=1, borderpad=6,
                    ay=-50)
            add_iv(fig)
        fig.update_layout(**CHART_LAYOUT,
                           title=dict(text=title or f"Naive Counterfactual — {ch_lbl_fn(ch) if ch else metric}", font=dict(size=14)))

    # ── ROAS / Efficiency Trend ───────────────────────────────────────────────
    elif chart_type == "efficiency_trend":
        # Dual-axis: sessions (volume proxy) left, CVR or revenue/session (efficiency) right
        # Mirrors the ROAS evolution charts from TikTok QBR decks
        ch = selected_chs[0] if selected_chs else "Paid_Social_TikTok"
        s_col = next((c for c in channels.get(ch, []) if "Sessions" in c), None)
        c_col = next((c for c in channels.get(ch, []) if "Conversions" in c), None)
        r_col = next((c for c in channels.get(ch, []) if "Revenue" in c), None)

        if s_col and (c_col or r_col) and s_col in df.columns:
            # Sessions as volume bars
            fig.add_trace(go.Bar(
                x=df["p_date"], y=df[s_col].rolling(7, min_periods=1).mean(),
                name=f"{ch_lbl_fn(ch)} Sessions (7d avg)",
                marker_color=ch_col_fn(ch), opacity=0.3, yaxis="y1"
            ))
            # Efficiency metric as line on right axis
            if r_col and r_col in df.columns:
                eff = (df[r_col] / df[s_col].replace(0, np.nan)).rolling(7, min_periods=1).mean()
                eff_label = "Revenue per Session (7d avg)"
            elif c_col and c_col in df.columns:
                eff = (df[c_col] / df[s_col].replace(0, np.nan) * 100).rolling(7, min_periods=1).mean()
                eff_label = "CVR % (7d avg)"
            else:
                eff = None; eff_label = ""

            if eff is not None:
                fig.add_trace(go.Scatter(
                    x=df["p_date"], y=eff, mode="lines",
                    name=eff_label,
                    line=dict(color=C_pal["purple"], width=2.5), yaxis="y2"
                ))
                # Pre/post average lines on efficiency axis
                if t_start and len(pre_df) and s_col in pre_df.columns:
                    pre_eff_avg = eff[df["p_date"] < t_start].mean()
                    post_eff_avg = eff[(df["p_date"] >= t_start) & (df["p_date"] <= t_end)].mean() if t_end else None
                    fig.add_hline(y=float(pre_eff_avg), line_dash="dash",
                                  line_color=C_pal["grey"], opacity=0.7,
                                  annotation_text=f"Pre avg: {pre_eff_avg:.2f}",
                                  annotation_font=dict(size=10, color=C_pal["grey"]),
                                  annotation_position="left", yref="y2")
                    if post_eff_avg is not None:
                        fig.add_hline(y=float(post_eff_avg), line_dash="dash",
                                      line_color=C_pal["purple"], opacity=0.9,
                                      annotation_text=f"Post avg: {post_eff_avg:.2f}",
                                      annotation_font=dict(size=10, color=C_pal["purple"]),
                                      annotation_position="right", yref="y2")

            add_iv(fig)
            _dual = {k: v for k, v in CHART_LAYOUT.items() if k not in ("yaxis","yaxis2")}
            fig.update_layout(
                **_dual,
                title=dict(text=title or f"{ch_lbl_fn(ch)} — Sessions vs Efficiency", font=dict(size=14)),
                yaxis=dict(title="Sessions (7d rolling)", gridcolor="#E5E7EB", showgrid=True),
                yaxis2=dict(title=eff_label, overlaying="y", side="right",
                            showgrid=False, tickfont=dict(color=C_pal["purple"], size=10))
            )

    # ── Weekly Spend/Volume Composition + Outcome Line ─────────────────────────
    elif chart_type == "weekly_composition":
        # Stacked bars: each channel's sessions by week
        # Line overlay: total conversions or revenue by week
        # Mirrors the "Weekly Spend Composition vs GMV Evolution" chart
        outcome_cols = [c for c in df.columns if metric_kw in c]
        df_weekly = df.copy()
        df_weekly["week"] = df_weekly["p_date"].dt.to_period("W").dt.start_time

        fig_data = df_weekly.groupby("week").sum(numeric_only=True).reset_index()

        # Stacked session bars per channel
        for ch in selected_chs:
            if ch not in channels: continue
            s_col = next((c for c in channels[ch] if "Sessions" in c), None)
            if not s_col or s_col not in fig_data.columns: continue
            fig.add_trace(go.Bar(
                x=fig_data["week"], y=fig_data[s_col],
                name=ch_lbl_fn(ch), marker_color=ch_col_fn(ch), opacity=0.8, yaxis="y1"
            ))

        # Total outcome line (conversions or revenue) on right axis
        total_outcome = None
        for col in outcome_cols:
            if col in fig_data.columns:
                if total_outcome is None:
                    total_outcome = fig_data[col].copy()
                else:
                    total_outcome = total_outcome + fig_data[col]

        if total_outcome is not None:
            fig.add_trace(go.Scatter(
                x=fig_data["week"], y=total_outcome,
                mode="lines+markers", name=f"Total {metric}",
                line=dict(color=C_pal["purple"], width=3),
                marker=dict(color=C_pal["purple"], size=8),
                yaxis="y2"
            ))

        # Campaign line
        if t_start:
            fig.add_shape(type="line",
                x0=str(t_start.date()), x1=str(t_start.date()),
                y0=0, y1=1, yref="paper",
                line=dict(color=C_pal["purple"], width=2, dash="dash"))
            fig.add_annotation(x=str(t_start.date()), y=1.02, yref="paper",
                text="▼ Campaign", showarrow=False,
                font=dict(color=C_pal["purple"], size=10))

        _dual2 = {k: v for k, v in CHART_LAYOUT.items() if k not in ("yaxis","yaxis2")}
        fig.update_layout(
            **_dual2,
            barmode="stack",
            title=dict(text=title or f"Weekly Sessions vs Total {metric}", font=dict(size=14)),
            yaxis=dict(title="Weekly Sessions", gridcolor="#E5E7EB", showgrid=True),
            yaxis2=dict(title=f"Weekly {metric}", overlaying="y", side="right",
                        showgrid=False, tickfont=dict(color=C_pal["purple"], size=10)),
            legend=dict(orientation="h", y=-0.2)
        )

    return fig


def get_suggested_prompts(hypothesis, conv_cols_sot, sess_cols_sot, channels):
    """Context-aware prompt suggestions. Returns two lists: standard and advanced."""
    CH_ORDER_VS = ["Direct","Paid_Social_TikTok","Paid_Search_Google","Paid_Social_Meta","Organic_Search"]
    avail = [ch for ch in CH_ORDER_VS if ch in channels]
    has_tiktok  = "Paid_Social_TikTok" in avail
    has_direct  = "Direct" in avail
    has_organic = "Organic_Search" in avail
    has_revenue = bool([c for c in conv_cols_sot+sess_cols_sot if "Revenue" in c])

    standard = []
    if has_tiktok and has_direct:
        standard.append("Show TikTok vs Direct conversions — 7-day average")
    standard.append("Show 7-day rolling conversions for all channels pre vs post")
    standard.append("Compare conversion rates across all channels")
    if has_organic:
        standard.append("Did organic search lift after the campaign started?")
    standard.append("Show a pre vs post bar chart of daily average conversions by channel")

    advanced = [
        "Show TikTok efficiency trend — sessions vs CVR over time",
        "Show weekly channel session composition vs total conversions",
        "Show me the elasticity shift — did we get more conversions per session post-campaign?",
        "Show a naive counterfactual — what would conversions have looked like without the campaign?",
        "Show the channel correlation heatmap for the pre-period",
    ]
    if has_revenue:
        advanced.append("Show TikTok revenue per session — did efficiency improve post-campaign?")
    if has_tiktok:
        advanced.append("Show TikTok conversion rate stability in the pre-period")

    def dedup(lst):
        seen = set(); out = []
        for p in lst:
            if p not in seen: seen.add(p); out.append(p)
        return out

    return dedup(standard)[:4], dedup(advanced)[:5]


def fetch_google_trends(keywords, start_date, end_date):
    """Fetch Google Trends. Auto-installs pytrends if missing."""
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
        pytrends = TrendReq(hl="en-AU", tz=600, timeout=(10, 25))
        tf = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"
        kws = [k.strip() for k in keywords if k.strip()][:5]
        pytrends.build_payload(kws, timeframe=tf, geo="AU")
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



# ── Session state ──────────────────────────────────────────────────────────────
for k, v in dict(
    raw_df=None, daily_df=None, file_type=None,
    causal_context="", parsed_context={},
    l4_map={}, conversion_events=[], target_col=None,
    quality_signals=[], flag_vars=[], smoothing=False, smooth_window=7,
    advertiser="", step=0, ark_key=GROQ_API_KEY, ark_endpoint=OPENAI_DEFAULT_MODEL, hypothesis=None, max_step=0,
    pre_start=None, pre_end=None, post_start=None, post_end=None,
    covariate_selection={},
    sot_targets=[], sot_treatment_start=None, sot_treatment_end=None,
    sot_channel_roles={}, cpa_inputs={}, sot_selected_targets=[], llm_eda_result=None,
    visual_studio_history=[], gt_data=None, gt_keywords=[], summary_report=None,
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

# ── Master channel + metric vocabulary ────────────────────────────────────────
CH_NAME_MAP = {
    # TikTok
    "tiktok":                  "Paid_Social_TikTok",
    "paid social tiktok":      "Paid_Social_TikTok",
    "paid social - tiktok":    "Paid_Social_TikTok",
    "paid_social_tiktok":      "Paid_Social_TikTok",
    "paid social-tiktok":      "Paid_Social_TikTok",
    "tt":                      "Paid_Social_TikTok",
    # Meta
    "meta":                    "Paid_Social_Meta",
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
            if canon: last_ch = canon
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

        # Rename flat columns to canonical
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
    df["p_date"] = pd.to_datetime(df["p_date"], errors="coerce")
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
            # Fuzzy fallback for non-normalised files
            cl = col.lower().replace(" ","_").replace("-","_")
            for k, v in CH_NAME_MAP.items():
                if k.replace(" ","_").replace("-","_") in cl:
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
        r = client.chat.completions.create(model=endpoint_id, messages=[{"role":"user","content":prompt}], temperature=0.1, max_tokens=400)
        t = r.choices[0].message.content.strip().replace("json","").replace("","").strip()
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

    # Sidebar shows 3 logical groups mapping to internal steps 1-6
    LOGICAL_STEPS = [
        ("1 · Setup", [1, 2], 1),
        ("2 · Data Treatment", [3], 3),
        ("3 · EDA + Export", [4, 6], 4),
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
    # LLM hardcoded to Groq for POC
    st.session_state.ark_key      = GROQ_API_KEY
    st.session_state.ark_endpoint = OPENAI_DEFAULT_MODEL
    globals()["ARK_BASE_URL"]     = "https://api.groq.com/openai/v1/"
    st.markdown(pill("✓ Groq AI connected", "pill-c"), unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:10px;color:{C["grey_l"]};margin-top:3px">llama-3.3-70b-versatile</div>', unsafe_allow_html=True)

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
                # ── Single unified parser ──────────────────────────────────
                uploaded.seek(0)
                raw, ftype, parse_warnings = normalise_file(uploaded, uploaded.name)

                if raw is None or len(raw) == 0:
                    st.error(f"Could not read file: {'; '.join(parse_warnings)}")
                else:
                    for w in parse_warnings:
                        st.warning(w)

                    # Reset stale state on new file
                    file_sig = f"{uploaded.name}_{uploaded.size}"
                    if st.session_state.get("_last_file_sig") != file_sig:
                        st.session_state._last_file_sig = file_sig
                        for k in ["sot_treatment_start","sot_treatment_end","pre_start","pre_end",
                                  "post_start","post_end","summary_report","visual_studio_history",
                                  "gt_data","daily_df","sot_selected_targets","sot_channel_roles"]:
                            st.session_state[k] = [] if "history" in k else (None if k != "gt_data" else None)

                    st.session_state.raw_df = raw
                    st.session_state.file_type = ftype
                    if ftype == "raw_ttam" and "Advertiser Name" in raw.columns and not st.session_state.advertiser:
                        st.session_state.advertiser = str(raw["Advertiser Name"].dropna().iloc[0])
                    if ftype == "pre_aggregated":
                        st.session_state.daily_df = raw

                    date_range_str = (f"{raw['p_date'].min().strftime('%d %b %Y')} – "
                                      f"{raw['p_date'].max().strftime('%d %b %Y')}"
                                      if raw["p_date"].notna().any() else "—")
                    ch_detected = parse_sot_columns(raw) if ftype == "sot" else {}
                    ch_names = [k.replace("Paid_Social_","").replace("Paid_Search_","").replace("_"," ")
                                for k in ch_detected if k != "Other"]
                    ch_str = " · ".join(ch_names) if ch_names else ""
                    st.markdown(f"""<div style="background:{C['purple']}08;border:1px solid {C['purple']}33;border-radius:8px;padding:14px;margin-top:10px">
<div style="font-weight:600;color:{C['purple']};margin-bottom:4px">✓ {uploaded.name}</div>
<div style="font-size:12px;color:{C['grey']}">{len(raw):,} rows · {date_range_str}</div>
{f'<div style="font-size:11px;color:{C["grey"]};margin-top:4px">Channels: {ch_str}</div>' if ch_str else ""}
<div style="margin-top:6px">
{pill("Raw TTAM — step 2 will aggregate","pill-a") if ftype=="raw_ttam" else pill("SOT / GA4 multi-channel","pill-c") if ftype=="sot" else pill("Pre-aggregated","pill-g")}
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
            _lbl = "Auto-aggregate & Continue →" if st.session_state.file_type == "raw_ttam" else "Continue →"
            if st.button(_lbl, use_container_width=True):
                if st.session_state.file_type == "raw_ttam":
                    try:
                        _raw = st.session_state.raw_df
                        agg = aggregate_ttam(_raw, L4_MAP, CONV_EVENTS)
                        st.session_state.daily_df = agg
                        st.session_state.step = 3
                    except Exception:
                        st.session_state.step = 2  # fall back to manual mapping
                elif st.session_state.file_type == "sot":
                    st.session_state.step = 2
                else:
                    st.session_state.step = 3
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — MAP & AGGREGATE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    hyp = st.session_state.hypothesis or ""

    # ── Route: SOT vs TTAM ──────────────────────────────────────────────────
    if st.session_state.file_type == "sot":
        # ── SOT Step 2: campaign period + channel snapshot ────────────────────────
        # Re-run normalise_file from raw bytes to guarantee canonical columns
        df = st.session_state.raw_df.copy()
        # Ensure date column is clean
        if "p_date" not in df.columns:
            dc = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
            df = df.rename(columns={dc: "p_date"})
        df["p_date"] = pd.to_datetime(df["p_date"], errors="coerce")
        df = df.dropna(subset=["p_date"]).sort_values("p_date").reset_index(drop=True)
        # Safely convert any remaining object columns to numeric
        for c in df.columns:
            if c == "p_date": continue
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = _to_num(df[c])

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
        st.markdown(f'<div style="font-size:11px;font-weight:600;color:#9CA3AF;margin-bottom:4px;letter-spacing:.05em">CHANNEL SNAPSHOT</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:12px">Pre vs post lift for each channel. Select the ones you want to investigate — these become your analysis targets.</div>', unsafe_allow_html=True)

        # Use parse_sot_columns — works for all normalised file formats
        channels_det = parse_sot_columns(df)
        tiktok_cols  = channels_det.get("Paid_Social_TikTok", [])
        non_tt_chs   = {k: v for k, v in channels_det.items()
                        if k not in ("Paid_Social_TikTok","Other")}

        # For each non-TikTok channel, prefer Conversions > Sessions > Revenue
        def _best(cols):
            for kw in ["Conversions","Sessions","Revenue"]:
                m = [c for c in cols if kw in c]
                if m: return m
            return cols

        # non_tiktok = list of best metric cols, one per channel
        non_tiktok = []
        for ch_k, ch_cols in non_tt_chs.items():
            non_tiktok.extend(_best(ch_cols)[:1])

        # all_non_tt = all conversion/session columns for full card display
        all_non_tt = [c for cols in non_tt_chs.values() for c in cols
                      if any(k in c for k in ["Conversions","Sessions","Revenue","ROAS","CaC"])]

        if not st.session_state.sot_selected_targets:
            st.session_state.sot_selected_targets = [c for c in non_tiktok if "Conversions" in c][:2] or non_tiktok[:2]

        selected = list(st.session_state.sot_selected_targets)
        card_cols = st.columns(2)
        for i, col in enumerate(all_non_tt if all_non_tt else non_tiktok):
            with card_cols[i % 2]:
                pre_s  = pre_df_sot[col] if col in pre_df_sot.columns else pd.Series(dtype=float)
                post_s = post_df_sot[col] if col in post_df_sot.columns else pd.Series(dtype=float)
                pre_avg  = float(pre_s.mean())  if len(pre_s)  else 0
                post_avg = float(post_s.mean()) if len(post_s) else 0
                lift = (post_avg - pre_avg) / pre_avg * 100 if pre_avg else 0
                is_sel = col in selected
                lift_color = C["purple"] if lift > 0 else C["red"]
                border = f"2px solid {C['purple']}" if is_sel else f"1px solid {C['border']}"
                bg     = f"{C['purple']}08" if is_sel else C["surface"]
                sign   = "+" if lift >= 0 else ""
                # Clean label: strip canonical channel prefix
                lbl = col
                for pfx in ["Paid_Social_TikTok_","Paid_Social_Meta_","Paid_Search_Google_",
                             "Organic_Search_","Direct_"]:
                    lbl = lbl.replace(pfx,"")
                ch_lbl = col.replace("Paid_Social_","").replace("Paid_Search_","").replace("_Conversions","").replace("_Sessions","").replace("_Revenue","").replace("_"," ")
                st.markdown(
                    f'<div style="border:{border};background:{bg};border-radius:8px;padding:12px 14px;margin-bottom:8px">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center">'
                    f'<div><div style="font-size:10px;color:{C["grey"]};margin-bottom:2px">{ch_lbl}</div>'
                    f'<div style="font-size:12px;font-weight:600;color:{C["text"]}">{lbl.replace("_"," ")}</div></div>'
                    f'<div style="font-size:16px;font-weight:700;color:{lift_color}">{sign}{lift:.1f}%</div></div>'
                    f'<div style="font-size:11px;color:{C["grey"]};margin-top:3px">{pre_avg:,.0f} pre → {post_avg:,.0f} post avg/day</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
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
            st.markdown(f'<div style="display:flex;align-items:center;justify-content:center;height:100%;font-size:24px;color:{C["grey"]}">→</div>', unsafe_allow_html=True)
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
                with st.expander("🔧 Winsorisation — cap extreme values at 95th percentile"):
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

                with st.expander("📐 Scale adjustment — log transform + standardise"):
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
        CH_COLORS = {"Direct":C["purple"],"Paid_Social_TikTok":"#00C2FF","Paid_Search_Google":"#EA4335","Paid_Social_Meta":"#1877F2","Organic_Search":"#34A853"}
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

        # ── CONSOLIDATED DATA OVERVIEW ─────────────────────────────────────────────
        st.markdown(f'<div style="font-size:11px;font-weight:600;color:{C["grey"]};letter-spacing:.06em;margin-bottom:10px">DATA OVERVIEW</div>', unsafe_allow_html=True)

        ov_c1, ov_c2, ov_c3 = st.columns([2, 3, 1])
        with ov_c1:
            ov_metric = st.selectbox("Metric", ["Conversions","Sessions","Revenue"], key="ov_metric")
        with ov_c2:
            avail_chs = [ch for ch in CH_ORDER if ch in channels]
            ov_channels = st.multiselect("Channels", options=avail_chs, default=avail_chs, key="ov_channels")
        with ov_c3:
            ov_smooth = st.selectbox("Smooth", [1,3,7,14], index=2, key="ov_smooth")

        ov_view = st.radio("View", ["Time Series","Conversion Rate","Pre vs Post"], horizontal=True, key="ov_view")

        col_lookup_ov = {"Conversions": conv_cols_sot, "Sessions": sess_cols_sot, "Revenue": rev_cols_sot}
        fig_ov = go.Figure()
        cvr_rows_ov = []

        if ov_view == "Time Series":
            for ch in ov_channels:
                col_list = col_lookup_ov.get(ov_metric, [])
                col = next((c for c in col_list if c.startswith(ch)), None)
                if col and col in df.columns:
                    series = df[col].rolling(ov_smooth, center=True, min_periods=1).mean()
                    nm = ch_lbl(ch) + (f" ({ov_smooth}d avg)" if ov_smooth > 1 else "")
                    fig_ov.add_trace(go.Scatter(x=df["p_date"], y=series, mode="lines",
                        line=dict(color=ch_col(ch), width=2.5), name=nm))
            fig_ov.update_layout(**CHART, title=dict(text=f"{ov_metric} by Channel", font=dict(size=14)))
            add_treatment(fig_ov)

        elif ov_view == "Conversion Rate":
            for ch in ov_channels:
                if ch not in channels: continue
                s_col = next((c for c in channels[ch] if "Sessions" in c), None)
                c_col = next((c for c in channels[ch] if "Conversions" in c), None)
                if not s_col or not c_col: continue
                cvr = (df[c_col] / df[s_col].replace(0, np.nan) * 100).rolling(ov_smooth, center=True, min_periods=1).mean()
                fig_ov.add_trace(go.Scatter(x=df["p_date"], y=cvr, mode="lines",
                    line=dict(color=ch_col(ch), width=2.5), name=ch_lbl(ch)))
                pre_cvr  = pre_df[c_col].sum() / pre_df[s_col].sum()  * 100 if len(pre_df)  and pre_df[s_col].sum()  > 0 else 0
                post_cvr = post_df[c_col].sum() / post_df[s_col].sum() * 100 if len(post_df) and post_df[s_col].sum() > 0 else 0
                delta = (post_cvr - pre_cvr) / pre_cvr * 100 if pre_cvr else 0
                cvr_rows_ov.append(dict(ch=ch_lbl(ch), pre=pre_cvr, post=post_cvr, delta=delta, color=ch_col(ch)))
            fig_ov.update_layout(**CHART, title=dict(text=f"Conversion Rate (%) — {ov_smooth}d smoothed", font=dict(size=14)))
            add_treatment(fig_ov)

        elif ov_view == "Pre vs Post":
            pv_ov, ptv_ov, lbs_ov = [], [], []
            for ch in ov_channels:
                col_list = col_lookup_ov.get(ov_metric, [])
                col = next((c for c in col_list if c.startswith(ch)), None)
                if col and col in df.columns:
                    pv_ov.append(pre_df[col].mean() if len(pre_df) else 0)
                    ptv_ov.append(post_df[col].mean() if len(post_df) else 0)
                    lbs_ov.append(ch_lbl(ch))
            fig_ov.add_trace(go.Bar(name="Pre-Period", x=lbs_ov, y=pv_ov,
                                    marker_color=C["grey_l"], opacity=0.75))
            fig_ov.add_trace(go.Bar(name="Post-Period", x=lbs_ov, y=ptv_ov,
                                    marker_color=C["purple"]))
            fig_ov.update_layout(**CHART, barmode="group",
                                  title=dict(text=f"Daily Avg {ov_metric} — Pre vs Post", font=dict(size=14)))

        st.plotly_chart(fig_ov, use_container_width=True)

        # CVR delta tiles (visible only in CVR mode)
        if ov_view == "Conversion Rate" and cvr_rows_ov:
            html_cvr = '<div class="mrow">'
            for r in sorted(cvr_rows_ov, key=lambda x: -abs(x["delta"])):
                sign = "+"; mc = "mvc"
                if r["delta"] < 0: sign = ""; mc = "mvr"
                html_cvr += (f'<div class="mtile" style="border-left:3px solid {r["color"]}">'
                             f'<div class="ml">{r["ch"]}</div>'
                             f'<div class="mv {mc}">{sign}{r["delta"]:.1f}%</div>'
                             f'<div style="font-size:11px;color:{C["grey"]}">{r["pre"]:.2f}% → {r["post"]:.2f}%</div></div>')
            st.markdown(html_cvr + "</div>", unsafe_allow_html=True)

        # Pre-period stability row (always visible below the chart)
        col_lookup_flat = col_lookup_ov.get(ov_metric, [])
        stab_entries = []
        for ch in (ov_channels or avail_chs):
            col_s = next((c for c in col_lookup_flat if c.startswith(ch)), None)
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
                              f'<div class="ml">{ch_lbl(ch_s)}</div>'
                              f'<div class="mv {mc_s}">{cv_s:.0f}% CV</div>'
                              f'<div style="font-size:11px;color:{C["grey"]}">{slope_s:+.1f}%/day</div></div>')
            st.markdown(html_stab + "</div>", unsafe_allow_html=True)

        # ── GOOGLE TRENDS OVERLAY ──────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div style="height:1px;background:{C["border"]};margin:0 0 20px"></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:13px;font-weight:700;color:{C["text"]};margin-bottom:4px">🔍 Google Search Trends Overlay</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:12px">Overlay category and brand search volume to contextualise whether organic/direct channel lifts were driven by TikTok or external demand. Relative index (0–100) — useful for narrative, not model input.</div>', unsafe_allow_html=True)

        gt_c1, gt_c2, gt_c3 = st.columns([3, 3, 1])
        with gt_c1:
            gt_brand = st.text_input("Brand keyword", placeholder="e.g. Chemist Warehouse", key="gt_brand")
        with gt_c2:
            gt_category = st.text_input("Category keyword", placeholder="e.g. pharmacy online", key="gt_cat")
        with gt_c3:
            st.markdown("<br>", unsafe_allow_html=True)
            gt_run = st.button("Fetch Trends", use_container_width=True, key="gt_run")

        if gt_run and (gt_brand or gt_category):
            keywords_to_fetch = [k for k in [gt_brand, gt_category] if k.strip()]
            with st.spinner("Fetching Google Trends data (AU)..."):
                t_range_start = df["p_date"].min().date()
                t_range_end   = df["p_date"].max().date()
                df_trends, gt_err = fetch_google_trends(keywords_to_fetch, t_range_start, t_range_end)
            if gt_err:
                st.markdown(f'<div class="card card-a">Google Trends error: {gt_err}</div>', unsafe_allow_html=True)
            elif df_trends is not None:
                st.session_state.gt_data = df_trends
                st.session_state.gt_keywords = keywords_to_fetch

        if st.session_state.get("gt_data") is not None:
            df_gt = st.session_state.gt_data
            gt_kws = st.session_state.get("gt_keywords", [])

            # Choose which channel to overlay against
            overlay_col = next((c for c in conv_cols_sot if "TikTok" in c or "Direct" in c or "Organic" in c), conv_cols_sot[0] if conv_cols_sot else None)
            overlay_options = conv_cols_sot + sess_cols_sot
            gt_overlay_col = st.selectbox("Overlay channel metric", overlay_options, key="gt_overlay")

            fig_gt = go.Figure()
            # Primary channel on left axis
            if gt_overlay_col and gt_overlay_col in df.columns:
                s7 = df[gt_overlay_col].rolling(7, center=True, min_periods=1).mean()
                fig_gt.add_trace(go.Scatter(
                    x=df["p_date"], y=s7, mode="lines",
                    name=gt_overlay_col.replace("_"," ") + " (7d avg)",
                    line=dict(color=C["purple"], width=2.5), yaxis="y1"
                ))
            # Trends on right axis
            TREND_COLORS = ["#EA4335","#34A853","#FBBC05","#4285F4"]
            for i, kw in enumerate(gt_kws):
                if kw in df_gt.columns:
                    # Resample trends to daily (it comes weekly from pytrends)
                    trend_daily = df_gt[kw].reindex(pd.date_range(df_gt.index.min(), df_gt.index.max(), freq="D")).interpolate()
                    fig_gt.add_trace(go.Scatter(
                        x=trend_daily.index, y=trend_daily.values,
                        mode="lines", name=f"Search: {kw}",
                        line=dict(color=TREND_COLORS[i % len(TREND_COLORS)], width=2, dash="dot"),
                        yaxis="y2", opacity=0.8
                    ))
            add_treatment(fig_gt)
            _gt_layout = {k: v for k, v in CHART.items() if k not in ("yaxis","yaxis2")}
            fig_gt.update_layout(
                **_gt_layout,
                title=dict(text="Channel Metric vs Google Search Interest (AU)", font=dict(size=14)),
                yaxis=dict(title="Channel metric", gridcolor="#E5E7EB", showgrid=True),
                yaxis2=dict(title="Search interest (0–100)", overlaying="y", side="right",
                            showgrid=False, range=[0, 110],
                            tickfont=dict(color=C["grey"], size=10)),
                legend=dict(orientation="h", y=-0.2)
            )
            st.plotly_chart(fig_gt, use_container_width=True)

            # Correlation between trends and channel
            if gt_overlay_col and gt_overlay_col in df.columns:
                corr_rows = []
                for kw in gt_kws:
                    if kw in df_gt.columns:
                        trend_daily = df_gt[kw].reindex(pd.date_range(df_gt.index.min(), df_gt.index.max(), freq="D")).interpolate()
                        merged = pd.merge(
                            df[["p_date", gt_overlay_col]].rename(columns={"p_date":"date"}),
                            trend_daily.rename("trend").reset_index().rename(columns={"index":"date"}),
                            on="date", how="inner"
                        )
                        if len(merged) > 5:
                            r = merged[gt_overlay_col].corr(merged["trend"])
                            corr_rows.append((kw, r))
                if corr_rows:
                    html_corr = '<div class="mrow">'
                    for kw, r in corr_rows:
                        mc = "mvc" if abs(r) > 0.6 else "mva" if abs(r) > 0.3 else "mvr"
                        interp = "Strong" if abs(r) > 0.6 else "Moderate" if abs(r) > 0.3 else "Weak"
                        direction = "positive" if r > 0 else "negative"
                        html_corr += (f'<div class="mtile"><div class="ml">"{kw}" vs channel</div>'
                                      f'<div class="mv {mc}">{r:.2f} R</div>'
                                      f'<div style="font-size:11px;color:{C["grey"]}">{interp} {direction} correlation</div></div>')
                    st.markdown(html_corr + "</div>", unsafe_allow_html=True)
                    # Narrative interpretation
                    high_corr = [kw for kw, r in corr_rows if abs(r) > 0.5]
                    low_corr  = [kw for kw, r in corr_rows if abs(r) <= 0.3]
                    if high_corr:
                        st.markdown(f'<div class="card card-a">⚠️ <strong>High correlation with {", ".join(high_corr)}</strong> — the channel lift may partly reflect broader search demand rather than TikTok incrementality alone. Flag this when contextualising results.</div>', unsafe_allow_html=True)
                    elif low_corr:
                        st.markdown(f'<div class="card card-c">✓ <strong>Low correlation with search trends</strong> — the channel lift is unlikely to be explained by broader category demand. Stronger evidence for TikTok incrementality.</div>', unsafe_allow_html=True)

        # ── PRE-MODELLING SUMMARY REPORT ───────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div style="height:1px;background:{C["border"]};margin:0 0 20px"></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:14px;font-weight:700;color:{C["text"]};margin-bottom:4px">📋 Pre-Modelling Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:12px">AI reads all EDA signals and generates a structured brief — channel performance, stability assessment, covariate recommendations, and readiness rating.</div>', unsafe_allow_html=True)

        sum_col1, sum_col2 = st.columns([1, 5])
        with sum_col1:
            gen_summary = st.button("Generate Summary", key="gen_summary_btn", use_container_width=True)
        with sum_col2:
            if st.session_state.get("summary_report"):
                if st.button("Clear", key="clear_summary_btn"):
                    st.session_state.summary_report = None
                    st.rerun()

        if gen_summary:
            with st.spinner("Generating pre-modelling brief..."):
                summary_ctx = build_eda_summary(df, pre_df, post_df,
                    st.session_state.hypothesis, "sot",
                    st.session_state.sot_channel_roles,
                    st.session_state.target_col,
                    conv_cols_sot, sess_cols_sot)
                report, err = call_ark_summary(
                    st.session_state.ark_key,
                    st.session_state.ark_endpoint,
                    summary_ctx
                )
                if report:
                    st.session_state.summary_report = report
                else:
                    st.error(f"Summary error: {err}")

        if st.session_state.get("summary_report"):
            st.markdown(f'''<div style="background:{C["surface"]};border:1px solid {C["border"]};border-left:3px solid {C["purple"]};border-radius:12px;padding:20px 24px;margin-bottom:16px">''', unsafe_allow_html=True)
            st.markdown(st.session_state.summary_report)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── VISUAL STUDIO ──────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div style="height:1px;background:{C["border"]};margin:0 0 20px"></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:14px;font-weight:700;color:{C["text"]};margin-bottom:4px">💬 Visual Studio</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-bottom:14px">Ask for any visualisation in plain English. The AI reads your data and builds it — no code needed.</div>', unsafe_allow_html=True)

        # Context-aware suggested prompts
        std_prompts, adv_prompts = get_suggested_prompts(st.session_state.hypothesis, conv_cols_sot, sess_cols_sot, channels)
        if std_prompts or adv_prompts:
            st.markdown(f'<div style="font-size:10px;font-weight:600;color:{C["grey"]};letter-spacing:.08em;margin-bottom:6px">STANDARD</div>', unsafe_allow_html=True)
            sug_cols = st.columns(4)
            for idx_s, prompt_s in enumerate(std_prompts):
                with sug_cols[idx_s % 4]:
                    if st.button(f"💡 {prompt_s}", key=f"sug_std_{idx_s}", use_container_width=True):
                        st.session_state.vs_queued = prompt_s
                        st.rerun()
            st.markdown(f'<div style="font-size:10px;font-weight:600;color:{C["grey"]};letter-spacing:.08em;margin:8px 0 6px">ADVANCED VISUALS</div>', unsafe_allow_html=True)
            adv_cols = st.columns(3)
            for idx_a, prompt_a in enumerate(adv_prompts):
                with adv_cols[idx_a % 3]:
                    if st.button(f"📊 {prompt_a}", key=f"sug_adv_{idx_a}", use_container_width=True):
                        st.session_state.vs_queued = prompt_a
                        st.rerun()

        # Chat history
        if "visual_studio_history" not in st.session_state:
            st.session_state.visual_studio_history = []

        for hi, turn in enumerate(st.session_state.visual_studio_history):
            if turn["role"] == "user":
                with st.chat_message("user"):
                    st.write(turn["content"])
            else:
                with st.chat_message("assistant"):
                    saved_spec = turn.get("spec", {})
                    ct = saved_spec.get("chart_type", "text")
                    if ct != "text" and saved_spec:
                        fig_hist = render_chart_from_spec(
                            saved_spec, df, pre_df, post_df,
                            channels, ch_col, ch_lbl,
                            t_start, t_end, CHART, C
                        )
                        if fig_hist:
                            st.plotly_chart(fig_hist, use_container_width=True, key=f"vs_hist_{hi}")
                    if turn.get("insight"):
                        st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-top:6px;font-style:italic">💡 {turn["insight"]}</div>', unsafe_allow_html=True)
                    if ct == "text" and turn.get("message"):
                        st.write(turn["message"])

        # Chat input + queued prompt processing
        if not (st.session_state.ark_key and st.session_state.ark_endpoint):
            st.markdown(f'<div style="font-size:12px;color:{C["amber"]};padding:10px 14px;background:{C["amber"]}11;border:1px solid {C["amber"]}44;border-radius:8px;margin-top:10px">Add your Ark API key and endpoint ID in the sidebar to enable Visual Studio.</div>', unsafe_allow_html=True)
        else:
            user_input_vs = st.chat_input("Ask for a visualisation — e.g. 'Show 7-day rolling TikTok conversions vs Direct'", key="vs_chat_input")
            queued = st.session_state.pop("vs_queued", None)
            active = queued or user_input_vs

            if active:
                st.session_state.visual_studio_history.append({"role": "user", "content": active})
                with st.spinner("Building your chart..."):
                    ctx = build_visual_context(df, pre_df, post_df, st.session_state.hypothesis,
                                               conv_cols_sot, sess_cols_sot, rev_cols_sot, channels)
                    spec_new, err_vs = call_ark_visual_studio(
                        st.session_state.ark_key, st.session_state.ark_endpoint,
                        active, ctx, st.session_state.visual_studio_history
                    )
                if err_vs:
                    st.error(f"API error: {err_vs}")
                elif spec_new:
                    ct_new = spec_new.get("chart_type", "text")
                    ins_new = spec_new.get("insight", "")
                    fig_new = None
                    if ct_new != "text":
                        fig_new = render_chart_from_spec(
                            spec_new, df, pre_df, post_df,
                            channels, ch_col, ch_lbl,
                            t_start, t_end, CHART, C
                        )
                    st.session_state.visual_studio_history.append({
                        "role": "assistant",
                        "spec": spec_new,
                        "insight": ins_new,
                        "message": spec_new.get("message", ""),
                        "raw_response": json.dumps(spec_new),
                    })
                    with st.chat_message("assistant"):
                        if fig_new:
                            st.plotly_chart(fig_new, use_container_width=True, key="vs_new")
                        if ins_new:
                            st.markdown(f'<div style="font-size:12px;color:{C["grey"]};margin-top:6px;font-style:italic">💡 {ins_new}</div>', unsafe_allow_html=True)
                        if ct_new == "text":
                            st.write(spec_new.get("message", "I couldn't build that chart from the available data — try rephrasing or ask for a different metric."))
                    st.rerun()

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

