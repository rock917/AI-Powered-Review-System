# app/components/cards.py
# FeedbackIQ — Reusable UI Card Components

import streamlit as st


def kpi_card(label: str, value: str, delta: str = "",
             accent_color: str = "#2563EB") -> str:
    """Returns HTML for a premium KPI card."""
    return f"""
    <div class="kpi-card">
        <div class="kpi-accent" style="background:{accent_color};"></div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-delta">{delta}</div>
    </div>
    """


def render_kpi_row(metrics: list):
    """
    Render a row of KPI cards.
    metrics = list of (label, value, delta, color) tuples
    """
    cols = st.columns(len(metrics))
    for col, (label, value, delta, color) in zip(cols, metrics):
        with col:
            st.markdown(
                kpi_card(label, value, delta, color),
                unsafe_allow_html=True
            )


def result_card(sentiment: str, confidence: float,
                prob_positive: float, prob_negative: float):
    """Render the sentiment analysis result card."""

    is_positive  = sentiment == 'Positive'
    emoji        = "😊" if is_positive else "😞"
    label_color  = "#16A34A" if is_positive else "#DC2626"
    bg_color     = "#F0FDF4" if is_positive else "#FEF2F2"
    border_color = "#86EFAC" if is_positive else "#FCA5A5"
    pos_width    = int(prob_positive * 100)
    neg_width    = int(prob_negative * 100)
    conf_pct     = f"{confidence*100:.1f}%"
    pos_pct      = f"{prob_positive*100:.1f}%"
    neg_pct      = f"{prob_negative*100:.1f}%"

    html = (
        f'<div class="result-card" style="background:{bg_color};'
        f'border-color:{border_color};">'
        f'<div class="sentiment-emoji">{emoji}</div>'
        f'<div style="font-size:1.6rem;font-weight:800;color:{label_color};'
        f'letter-spacing:-0.5px;">{sentiment.upper()}</div>'
        f'<div class="confidence-value">{conf_pct}</div>'
        f'<div class="confidence-label">Confidence</div>'
        f'<div class="prob-container" style="margin-top:1.5rem;">'

        f'<div class="prob-row">'
        f'<div class="prob-label">Positive</div>'
        f'<div class="prob-bar-bg">'
        f'<div class="prob-bar-fill-pos" style="width:{pos_width}%;"></div>'
        f'</div>'
        f'<div class="prob-pct">{pos_pct}</div>'
        f'</div>'

        f'<div class="prob-row">'
        f'<div class="prob-label">Negative</div>'
        f'<div class="prob-bar-bg">'
        f'<div class="prob-bar-fill-neg" style="width:{neg_width}%;"></div>'
        f'</div>'
        f'<div class="prob-pct">{neg_pct}</div>'
        f'</div>'

        f'</div>'
        f'<div style="margin-top:1.2rem;font-size:0.75rem;color:#6B7280;">'
        f'Model: Fine-tuned DistilBERT &nbsp;|&nbsp; Max length: 256 tokens'
        f'</div>'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


def insight_card(title: str, text: str):
    """Render a data insight card."""
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">💡 {title}</div>
        <div class="insight-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)


def empty_state(icon: str, message: str):
    """Render an empty state placeholder."""
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-state-icon">{icon}</div>
        <div class="empty-state-text">{message}</div>
    </div>
    """, unsafe_allow_html=True)


def pipeline_display():
    """Render the model pipeline visualization."""
    steps = [
        ("📝", "Review Text",          "Raw customer review input"),
        ("🔤", "Tokenizer",            "DistilBERT WordPiece tokenizer → input_ids + attention_mask"),
        ("🤖", "DistilBERT",           "6-layer transformer — 66M parameters"),
        ("🎯", "Classification Head",  "Linear layer: 768 → 2 logits"),
        ("📊", "Softmax",              "Converts logits to probabilities"),
        ("✅", "Prediction",           "Positive / Negative + confidence score"),
    ]

    for i, (icon, title, desc) in enumerate(steps):
        st.markdown(f"""
        <div class="pipeline-step">
            <span style="font-size:1.2rem;">{icon}</span>
            <div>
                <strong>{title}</strong>
                <div style="font-size:0.78rem;color:#6B7280;margin-top:1px;">
                    {desc}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if i < len(steps) - 1:
            st.markdown(
                '<div class="pipeline-arrow">↓</div>',
                unsafe_allow_html=True
            )