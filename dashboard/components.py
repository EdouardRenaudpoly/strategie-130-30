import streamlit as st

COLORS = {
    "strategy":  "#00D4AA",
    "benchmark": "#6B7280",
    "long":      "#10B981",
    "short":     "#EF4444",
    "neutral":   "#6366F1",
    "accent":    "#F59E0B",
}

LAYOUT_COMMON = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)

def explain(text: str):
    st.markdown(f'<div class="explain-box">{text}</div>', unsafe_allow_html=True)

GLOBAL_CSS = """
<style>
    .metric-card {
        background: #1E293B; border-radius: 12px; padding: 20px;
        border-left: 4px solid #00D4AA;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #00D4AA; }
    .metric-label { font-size: 0.85rem; color: #94A3B8; margin-top: 4px; }
    .explain-box {
        background: #0F172A; border-left: 3px solid #6366F1;
        padding: 12px 16px; border-radius: 6px; margin-bottom: 16px;
        font-size: 0.9rem; color: #CBD5E1;
    }
</style>
"""
