"""
main.py — Streamlit Cryptocurrency Regime-Conditional Prediction Dashboard
Deploy: push to GitHub → connect to Streamlit Cloud → set main file = main.py
Run locally: streamlit run main.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from scipy.stats import binomtest

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Regime Prediction",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .mono { font-family: 'JetBrains Mono', monospace; }

    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 32px;
        font-weight: 700;
        line-height: 1;
    }
    .metric-sub {
        font-size: 12px;
        color: #6b7280;
        margin-top: 4px;
    }
    .bullish  { color: #10b981; }
    .bearish  { color: #ef4444; }
    .neutral  { color: #60a5fa; }
    .warning  { color: #f59e0b; }

    .regime-high {
        background: #1f0a0a;
        border: 1px solid #ef444440;
        border-left: 3px solid #ef4444;
        border-radius: 8px;
        padding: 14px 18px;
    }
    .regime-low {
        background: #0a1f15;
        border: 1px solid #10b98140;
        border-left: 3px solid #10b981;
        border-radius: 8px;
        padding: 14px 18px;
    }
    .evidence-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #1f2937;
        font-size: 13px;
    }
    .evidence-key   { color: #9ca3af; }
    .evidence-value { font-family: 'JetBrains Mono', monospace;
                      font-weight: 600; color: #f9fafb; }
    .section-title {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #4b5563;
        margin: 24px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #1f2937;
    }
    .demo-banner {
        background: #1c1407;
        border: 1px solid #f59e0b40;
        border-left: 3px solid #f59e0b;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13px;
        color: #fcd34d;
        margin-bottom: 20px;
    }
    .stSlider > div > div { background: #1f2937; }
    div[data-testid="stSidebar"] { background: #0d1117; }
    .main .block-container { padding-top: 2rem; max-width: 1400px; }
</style>
""", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    models, meta = {}, {}
    model_files = {
        "lgbm"      : "saved_models/lgbm_baseline.pkl",
        "lgbm_high" : "saved_models/lgbm_high_vol.pkl",
        "lgbm_low"  : "saved_models/lgbm_low_vol.pkl",
        "xgb"       : "saved_models/xgb_baseline.pkl",
        "rf"        : "saved_models/rf_baseline.pkl",
        "hmm"       : "saved_models/hmm_model.pkl",
    }
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)

    meta_path = "saved_models/pipeline_metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {
            "conformal_quantile" : 0.052057,
            "best_thresh_lgbm"   : 0.0,
            "high_vol_state"     : 0,
            "markov_ofi_high_vol": 0.0558,
            "markov_ofi_low_vol" : -0.0147,
            "variance_ratio"     : 12.0,
            "metrics": {
                "xgb_rmse": 0.0320, "lgbm_rmse": 0.0289,
                "rf_rmse": 0.0239,  "rw_rmse": 0.0232,
                "xgb_da": 0.500,    "lgbm_da": 0.4969,
                "rf_da": 0.5219,    "high_vol_da": 0.6071,
                "low_vol_da": 0.4902, "regime_cond_da": 0.4953,
            },
            "feature_cols": [
                'lag1','lag2','lag5','volatility_10','volatility_20',
                'momentum_5','momentum_10','ofi','ofi_ratio',
                'taker_buy_ratio','trade_intensity','volume_change',
                'price_range','close_open_diff','vwap',
                'hour','day_of_week','high_vol_regime'
            ]
        }
    return models, meta

models, meta = load_models()
DEMO_MODE    = "lgbm" not in models
FEATURE_COLS = meta["feature_cols"]

# ── Sidebar inputs ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ₿ Feature Inputs")
    st.caption("Set current market microstructure values")

    st.markdown('<div class="section-title">Order Flow</div>',
                unsafe_allow_html=True)
    ofi_ratio       = st.slider("OFI Ratio",        -1.0,  1.0,  0.05, 0.01,
                        help="Order flow imbalance — positive = more buying")
    taker_buy_ratio = st.slider("Taker Buy Ratio",   0.0,  1.0,  0.52, 0.01,
                        help="Proportion of volume from market buyers")
    trade_intensity = st.slider("Trade Intensity",   0.0,  3.0,  1.0,  0.05,
                        help="Trade count vs 20-period rolling mean")
    volume_change   = st.slider("Volume Change",    -1.0,  2.0,  0.0,  0.05)

    st.markdown('<div class="section-title">Returns & Momentum</div>',
                unsafe_allow_html=True)
    lag1            = st.slider("Lag 1 Return",  -0.15, 0.15, 0.0,  0.005)
    lag2            = st.slider("Lag 2 Return",  -0.15, 0.15, 0.0,  0.005)
    lag5            = st.slider("Lag 5 Return",  -0.15, 0.15, 0.0,  0.005)
    momentum_5      = st.slider("Momentum 5",    -0.30, 0.30, 0.0,  0.01)
    momentum_10     = st.slider("Momentum 10",   -0.40, 0.40, 0.0,  0.01)

    st.markdown('<div class="section-title">Price Structure</div>',
                unsafe_allow_html=True)
    price_range     = st.slider("Price Range",    0.0,  0.15, 0.02, 0.005)
    close_open_diff = st.slider("Close-Open",    -0.1,  0.1,  0.0,  0.005)
    vwap            = st.slider("VWAP (USD)",    10000.0, 150000.0, 60000.0, 500.0)

    st.markdown('<div class="section-title">Volatility & Regime</div>',
                unsafe_allow_html=True)
    volatility_10   = st.slider("Volatility 10",  0.001, 0.15, 0.025, 0.001)
    volatility_20   = st.slider("Volatility 20",  0.001, 0.15, 0.022, 0.001)
    high_vol_regime = st.selectbox(
        "Volatility Regime",
        options=[0, 1],
        format_func=lambda x: (
            "🔴  High Volatility — Stress" if x == 1
            else "🟢  Low Volatility — Calm"
        )
    )

    st.markdown('<div class="section-title">Time</div>',
                unsafe_allow_html=True)
    hour        = st.slider("Hour of Day",  0, 23, 12)
    day_of_week = st.slider("Day of Week",  0,  6,  2,
                    help="0 = Monday   6 = Sunday")

# ── Build feature vector ───────────────────────────────────────────────────
feature_values = {
    'lag1': lag1, 'lag2': lag2, 'lag5': lag5,
    'volatility_10': volatility_10, 'volatility_20': volatility_20,
    'momentum_5': momentum_5, 'momentum_10': momentum_10,
    'ofi': ofi_ratio, 'ofi_ratio': ofi_ratio,
    'taker_buy_ratio': taker_buy_ratio,
    'trade_intensity': trade_intensity,
    'volume_change': volume_change,
    'price_range': price_range,
    'close_open_diff': close_open_diff,
    'vwap': vwap,
    'hour': hour,
    'day_of_week': day_of_week,
    'high_vol_regime': high_vol_regime,
}
X_input = np.array([[feature_values[f] for f in FEATURE_COLS]])

# ── Prediction ─────────────────────────────────────────────────────────────
def predict(X, regime):
    if DEMO_MODE:
        coef = (meta["markov_ofi_high_vol"] if regime == 1
                else meta["markov_ofi_low_vol"])
        return float(coef * ofi_ratio + 0.05 * momentum_5 + 0.02 * lag1)
    m = models.get("lgbm_high" if regime == 1 else "lgbm_low",
                   models.get("lgbm"))
    return float(m.predict(X)[0])

pred       = predict(X_input, high_vol_regime)
q          = meta["conformal_quantile"]
lower      = pred - q
upper      = pred + q
thresh     = meta.get("best_thresh_lgbm", 0.0)
is_bullish = pred > thresh
direction  = "BULLISH" if is_bullish else "BEARISH"
dir_color  = "bullish" if is_bullish else "bearish"
ofi_coef   = (meta["markov_ofi_high_vol"] if high_vol_regime == 1
              else meta["markov_ofi_low_vol"])
exp_da     = (meta["metrics"]["high_vol_da"] if high_vol_regime == 1
              else meta["metrics"]["low_vol_da"])

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:8px;'>
    <span style='font-size:11px; font-weight:600; letter-spacing:0.1em;
                 text-transform:uppercase; color:#4b5563;'>
        MSc Data Science Research Project
    </span>
</div>
<h1 style='font-size:28px; font-weight:700; margin:0 0 4px 0;
           letter-spacing:-0.02em;'>
    BTC/USDT Regime-Conditional Prediction
</h1>
<p style='color:#6b7280; font-size:14px; margin:0 0 24px 0;'>
    LightGBM + Hidden Markov Model regime detection ·
    Binance 2017–2026 · Conformal prediction intervals
</p>
""", unsafe_allow_html=True)

if DEMO_MODE:
    st.markdown("""
    <div class="demo-banner">
        ⚠ DEMO MODE — Place your saved_models/ folder in the same
        directory as main.py to load trained models.
    </div>""", unsafe_allow_html=True)

# ── Row 1 — four key metrics ───────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Direction</div>
        <div class="metric-value {dir_color}">{direction}</div>
        <div class="metric-sub">threshold = {thresh:+.4f}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    ret_color = "bullish" if pred > 0 else "bearish"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Return</div>
        <div class="metric-value {ret_color}">{pred:+.4f}</div>
        <div class="metric-sub">log return units</div>
    </div>""", unsafe_allow_html=True)

with c3:
    r_label = "HIGH VOL" if high_vol_regime == 1 else "LOW VOL"
    r_color = "bearish" if high_vol_regime == 1 else "bullish"
    r_desc  = "Stress regime" if high_vol_regime == 1 else "Calm regime"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Market Regime</div>
        <div class="metric-value {r_color}">{r_label}</div>
        <div class="metric-sub">{r_desc} · OFI coef {ofi_coef:+.4f}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    da_color = "bullish" if exp_da > 0.52 else "warning"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Historical DA (this regime)</div>
        <div class="metric-value {da_color}">{exp_da:.1%}</div>
        <div class="metric-sub">direction accuracy on test set</div>
    </div>""", unsafe_allow_html=True)

# ── Row 2 — interval chart + regime panel ─────────────────────────────────
col_chart, col_regime = st.columns([3, 2])

with col_chart:
    st.markdown('<div class="section-title">Conformal Prediction Interval</div>',
                unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(9, 2.8))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # Interval fill
    ax.fill_betweenx([0], [lower], [upper], alpha=0.15,
                     color='#60a5fa', label='90% interval')
    ax.fill_betweenx([0], [max(lower, 0) if is_bullish else lower],
                     [upper if is_bullish else min(upper, 0)],
                     alpha=0.25,
                     color='#10b981' if is_bullish else '#ef4444')

    # Lines
    ax.axvline(pred,  color='#10b981' if is_bullish else '#ef4444',
               linewidth=2.5, label=f'Prediction {pred:+.4f}', zorder=5)
    ax.axvline(0,     color='#ffffff', linewidth=0.8,
               linestyle='--', alpha=0.4, label='Zero')
    ax.axvline(lower, color='#ef4444', linewidth=0.8,
               linestyle=':', alpha=0.7, label=f'Lower {lower:+.4f}')
    ax.axvline(upper, color='#10b981', linewidth=0.8,
               linestyle=':', alpha=0.7, label=f'Upper {upper:+.4f}')

    ax.set_yticks([])
    ax.tick_params(colors='#6b7280', labelsize=9)
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#1f2937')
    ax.set_xlabel('Log Return', color='#6b7280', fontsize=10)
    ax.set_xlim(lower - q * 0.5, upper + q * 0.5)

    legend = ax.legend(loc='upper right', fontsize=8,
                       facecolor='#111827', labelcolor='#d1d5db',
                       framealpha=0.9, edgecolor='#1f2937')
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.caption(
        f"Quantile: {q:.4f}  ·  "
        f"Width: {2*q:.4f}  ·  "
        f"Empirical coverage: 88.09% (target 90%)  ·  "
        f"Method: split-conformal (Vovk et al., 2005)"
    )

with col_regime:
    st.markdown('<div class="section-title">Regime Evidence</div>',
                unsafe_allow_html=True)

    regime_class = "regime-high" if high_vol_regime == 1 else "regime-low"
    regime_title = "🔴 HIGH VOLATILITY REGIME" if high_vol_regime == 1 \
                   else "🟢 LOW VOLATILITY REGIME"
    st.markdown(f"""
    <div class="{regime_class}" style="margin-bottom:14px;">
        <div style="font-size:11px; font-weight:700;
                    letter-spacing:0.08em; margin-bottom:4px;">
            {regime_title}
        </div>
        <div style="font-size:13px; color:#9ca3af;">
            {"OFI signals amplified · High predictability window"
             if high_vol_regime == 1
             else "Near-efficient market · OFI signal weak"}
        </div>
    </div>""", unsafe_allow_html=True)

    evidence = [
        ("OFI coef — high vol",   "+0.0558"),
        ("OFI coef — low vol",    "−0.0147"),
        ("Variance ratio",        "12.0×"),
        ("OFI corr — high vol",   "0.325"),
        ("OFI corr — low vol",    "0.210"),
        ("DA — high vol (n=28)",  "60.71%"),
        ("DA — low vol (n=610)",  "49.02%"),
        ("Markov log-likelihood", "4209.6"),
    ]
    for k, v in evidence:
        st.markdown(f"""
        <div class="evidence-row">
            <span class="evidence-key">{k}</span>
            <span class="evidence-value">{v}</span>
        </div>""", unsafe_allow_html=True)

# ── Row 3 — model comparison + RQ summary ─────────────────────────────────
col_perf, col_rq = st.columns([3, 2])

with col_perf:
    st.markdown('<div class="section-title">Model Performance</div>',
                unsafe_allow_html=True)

    m   = meta["metrics"]
    perf = pd.DataFrame({
        'Model': [
            'Random Walk', 'XGBoost', 'LightGBM',
            'Random Forest', 'LightGBM\nRegime-Cond',
            'High-Vol\nOnly (n=28)'
        ],
        'DA': [0.500, m['xgb_da'], m['lgbm_da'],
               m['rf_da'], m['regime_cond_da'], m['high_vol_da']],
        'type': ['base','model','model','model','improved','best']
    })

    color_map = {
        'base'    : '#374151',
        'model'   : '#1d4ed8',
        'improved': '#d97706',
        'best'    : '#059669'
    }
    bar_colors = [color_map[t] for t in perf['type']]

    fig2, ax2 = plt.subplots(figsize=(9, 3.5))
    fig2.patch.set_facecolor('#0d1117')
    ax2.set_facecolor('#0d1117')

    bars = ax2.bar(perf['Model'], perf['DA'],
                   color=bar_colors, alpha=0.9,
                   edgecolor='#0d1117', linewidth=0.5)
    ax2.axhline(0.50, color='#ffffff', linewidth=1.0,
                linestyle='--', alpha=0.5, label='Random (50%)')

    for bar, val in zip(bars, perf['DA']):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.002,
                 f'{val:.1%}',
                 ha='center', va='bottom',
                 color='#d1d5db', fontsize=8.5,
                 fontfamily='monospace')

    ax2.set_ylim(0.44, 0.67)
    ax2.set_ylabel('Direction Accuracy', color='#6b7280', fontsize=10)
    ax2.tick_params(colors='#6b7280', labelsize=8)
    ax2.spines[['top','right']].set_visible(False)
    ax2.spines[['bottom','left']].set_color('#1f2937')
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.legend(facecolor='#111827', labelcolor='#d1d5db',
               edgecolor='#1f2937', fontsize=9)
    st.pyplot(fig2, use_container_width=True)
    plt.close()

with col_rq:
    st.markdown('<div class="section-title">Research Questions</div>',
                unsafe_allow_html=True)

    rqs = [
        ("RQ1", "Market Efficiency",
         "Short-term inefficiency confirmed. Daily returns approach weak-form efficiency. GARCH α+β > 0.85. Supports AMH.",
         "#10b981"),
        ("RQ2", "OFI Predictability",
         "OFI top predictor across all models. Sign reversal confirmed econometrically: +0.0558 stress vs −0.0147 calm.",
         "#60a5fa"),
        ("RQ3", "Signal Correlation",
         "OFI correlation 0.210 → 0.325 in stress. Momentum weakens. Opposite regime sensitivity confirmed.",
         "#a78bfa"),
    ]

    for tag, title, body, color in rqs:
        st.markdown(f"""
        <div style="border-left:3px solid {color};
                    padding:10px 14px; margin-bottom:10px;
                    background:#111827; border-radius:0 8px 8px 0;">
            <div style="display:flex; gap:8px; align-items:center;
                        margin-bottom:4px;">
                <span style="font-family:'JetBrains Mono',monospace;
                             font-size:10px; font-weight:700;
                             color:{color}; letter-spacing:0.05em;">
                    {tag}
                </span>
                <span style="font-size:12px; font-weight:600;
                             color:#f9fafb;">{title}</span>
            </div>
            <div style="font-size:12px; color:#9ca3af;
                        line-height:1.5;">{body}</div>
        </div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="display:flex; justify-content:space-between;
            font-size:11px; color:#4b5563; padding:4px 0;">
    <span>MSc Data Science · Binance BTCUSDT 2017–2026 ·
          linxy/CryptoCoin (Hugging Face)</span>
    <span class="mono">{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</span>
</div>""", unsafe_allow_html=True)
