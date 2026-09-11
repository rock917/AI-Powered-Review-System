# app/pages/overview.py
# FeedbackIQ — Overview / Command Center Page

import streamlit as st
from datetime import datetime
from components.cards import render_kpi_row, insight_card
from components.charts import (
    sentiment_donut, rating_bar,
    sentiment_trend, confidence_histogram,
    sentiment_by_rating
)
from utils.data_loader import load_demo_with_predictions


def render():
    # ── Header ───────────────────────────────────────────────
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Customer Feedback Intelligence</div>
        <div class="page-subtitle">
            Analyze customer feedback using fine-tuned DistilBERT.
            Real predictions on Amazon Fine Food Reviews.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────
    with st.spinner("Loading demo intelligence..."):
        df = load_demo_with_predictions()

    if df is None:
        st.error("Demo dataset not found. Check data/processed/test.csv exists.")
        return

    # ── Compute KPIs ─────────────────────────────────────────
    total    = len(df)
    n_pos    = (df['predicted_sentiment'] == 'Positive').sum()
    n_neg    = (df['predicted_sentiment'] == 'Negative').sum()
    pct_pos  = n_pos / total * 100
    pct_neg  = n_neg / total * 100
    avg_conf = df['confidence'].mean() * 100
    avg_star = df['Score'].mean()

    # ── KPI Cards ─────────────────────────────────────────────
    st.markdown("<div style='margin-bottom:1rem;'></div>",
                unsafe_allow_html=True)

    render_kpi_row([
        ("TOTAL REVIEWS",    f"{total:,}",
         "Demo dataset (500 reviews)", "#2563EB"),
        ("POSITIVE SENTIMENT", f"{pct_pos:.1f}%",
         f"{n_pos:,} reviews predicted positive", "#16A34A"),
        ("NEGATIVE SENTIMENT", f"{pct_neg:.1f}%",
         f"{n_neg:,} reviews predicted negative", "#DC2626"),
        ("AVG STAR RATING",  f"{avg_star:.2f} ⭐",
         "Customer-assigned rating", "#D97706"),
        ("MODEL ACCURACY",   "95.01%",
         "On 11,000 unseen test reviews", "#7C3AED"),
    ])

    st.markdown("<div style='margin:1.5rem 0;'></div>",
                unsafe_allow_html=True)

    # ── Row 1: Donut + Rating ─────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Sentiment Distribution**")
        st.plotly_chart(
            sentiment_donut(n_pos, n_neg),
            use_container_width=True
        )

    with col2:
        st.markdown("**Rating Distribution**")
        rating_counts = df['Score'].value_counts().sort_index()
        st.plotly_chart(
            rating_bar(rating_counts),
            use_container_width=True
        )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Row 2: Trend + Confidence ─────────────────────────────
    col3, col4 = st.columns([1, 1])

    with col3:
        st.markdown("**Sentiment Trend Over Time**")
        if 'year' in df.columns and df['year'].notna().sum() > 0:
            st.plotly_chart(
                sentiment_trend(df),
                use_container_width=True
            )
        else:
            st.info("Date information not available in this dataset.")

    with col4:
        st.markdown("**Prediction Confidence**")
        st.plotly_chart(
            confidence_histogram(df),
            use_container_width=True
        )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Sentiment by Rating ───────────────────────────────────
    st.markdown("**Sentiment vs Star Rating**")
    st.plotly_chart(
        sentiment_by_rating(df),
        use_container_width=True
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── AI Insights (calculated from actual data) ─────────────
    st.markdown("### 💡 Data Insights")
    st.caption("Calculated from demo dataset predictions — not hardcoded.")

    # Insight 1: Dominant sentiment
    dominant   = "Positive" if pct_pos > pct_neg else "Negative"
    dominant_pct = max(pct_pos, pct_neg)
    insight_card(
        "Sentiment Concentration",
        f"{dominant_pct:.1f}% of analyzed reviews are predicted {dominant}. "
        f"This reflects a dataset skewed toward satisfied customers "
        f"({avg_star:.2f} average star rating)."
    )

    # Insight 2: High confidence
    high_conf = (df['confidence'] >= 0.90).sum()
    high_conf_pct = high_conf / total * 100
    insight_card(
        "Model Certainty",
        f"The model predicted with ≥90% confidence on "
        f"{high_conf_pct:.1f}% of reviews ({high_conf:,} out of {total:,}). "
        f"Low-confidence predictions indicate genuinely ambiguous reviews."
    )

    # Insight 3: Rating-sentiment alignment
    five_star = df[df['Score'] == 5]
    if len(five_star) > 0:
        five_pos = (five_star['predicted_sentiment'] == 'Positive').sum()
        five_pos_pct = five_pos / len(five_star) * 100
        insight_card(
            "Rating Alignment",
            f"{five_pos_pct:.1f}% of 5-star reviews are predicted Positive — "
            f"confirming the model correctly associates high ratings "
            f"with positive language."
        )

    # ── CTA ───────────────────────────────────────────────────
    st.markdown("<div style='margin:2rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    col_a, col_b, col_c = st.columns([1, 1, 1])

    with col_a:
        if st.button("✨ Analyze a Review",
                     use_container_width=True, type="primary"):
            st.session_state['current_page'] = 'Review Analyzer'
            st.rerun()

    with col_b:
        if st.button("📂 Upload Your Reviews",
                     use_container_width=True):
            st.session_state['current_page'] = 'Batch Intelligence'
            st.rerun()

    with col_c:
        if st.button("🤖 View Model Performance",
                     use_container_width=True):
            st.session_state['current_page'] = 'Model Lab'
            st.rerun()

    st.markdown(
        f"<div style='text-align:right;font-size:0.75rem;color:#9CA3AF;"
        f"margin-top:1rem;'>Last updated: {datetime.now().strftime('%d %b %Y, %H:%M')}"
        f"</div>",
        unsafe_allow_html=True
    )