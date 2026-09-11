# app/pages/customer_insights.py
# FeedbackIQ — Customer Insights / Business Intelligence Page

import streamlit as st
from components.charts import (
    sentiment_donut, rating_bar,
    sentiment_trend, sentiment_by_rating,
    confidence_histogram
)
from components.cards import empty_state
from utils.data_loader import load_demo_with_predictions


def render():
    st.markdown("""
    <div class="page-header">
        <div class="page-title">💡 Customer Insights</div>
        <div class="page-subtitle">
            Turn feedback into product intelligence.
            Interactive analytics from the demo dataset.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading insights..."):
        df = load_demo_with_predictions()

    if df is None:
        st.error("Demo data unavailable.")
        return

    # ── Filters ───────────────────────────────────────────────
    st.markdown("**Filters**")
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        sentiment_filter = st.selectbox(
            "Sentiment", ["All", "Positive", "Negative"]
        )
    with fc2:
        ratings = sorted(df['Score'].dropna().unique().astype(int).tolist())
        rating_filter = st.multiselect(
            "Star Rating", ratings, default=ratings
        )
    with fc3:
        years = sorted(df['year'].dropna().unique().astype(int).tolist())
        year_filter = st.multiselect(
            "Year", years, default=years
        )

    # Apply filters
    filtered = df.copy()
    if sentiment_filter != "All":
        filtered = filtered[
            filtered['predicted_sentiment'] == sentiment_filter
        ]
    if rating_filter:
        filtered = filtered[filtered['Score'].isin(rating_filter)]
    if year_filter:
        filtered = filtered[filtered['year'].isin(year_filter)]

    if len(filtered) == 0:
        empty_state("🔍", "No reviews match the selected filters.")
        return

    st.caption(f"Showing {len(filtered):,} reviews after filters")
    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────
    st.markdown("### 📊 Sentiment Overview")
    col1, col2 = st.columns(2)

    n_pos = (filtered['predicted_sentiment'] == 'Positive').sum()
    n_neg = (filtered['predicted_sentiment'] == 'Negative').sum()

    with col1:
        st.markdown("**Sentiment Distribution**")
        st.plotly_chart(
            sentiment_donut(n_pos, n_neg),
            use_container_width=True
        )
    with col2:
        st.markdown("**Rating Distribution**")
        rating_counts = filtered['Score'].value_counts().sort_index()
        st.plotly_chart(
            rating_bar(rating_counts),
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("### 📈 Trends & Patterns")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Sentiment Over Time**")
        if filtered['year'].notna().sum() > 0:
            st.plotly_chart(
                sentiment_trend(filtered),
                use_container_width=True
            )
        else:
            st.info("Date data not available.")

    with col4:
        st.markdown("**Sentiment vs Rating**")
        st.plotly_chart(
            sentiment_by_rating(filtered),
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("### 🔍 Review Explorer")

    tab1, tab2, tab3 = st.tabs([
        "⭐ Most Positive", "⚠️ Most Negative", "🔥 Most Helpful"
    ])

    with tab1:
        top_pos = filtered[
            filtered['predicted_sentiment'] == 'Positive'
        ].nlargest(5, 'confidence')

        for _, row in top_pos.iterrows():
            with st.container():
                st.markdown(f"""
                <div style="background:#F0FDF4;border:1px solid #86EFAC;
                            border-radius:8px;padding:1rem;margin-bottom:0.8rem;">
                    <div style="font-size:0.85rem;color:#374151;">
                        {str(row['cleaned_text'])[:300]}...
                    </div>
                    <div style="margin-top:0.5rem;font-size:0.75rem;color:#6B7280;">
                        ⭐ {int(row['Score'])} stars &nbsp;|&nbsp;
                        Confidence: {row['confidence']*100:.1f}% &nbsp;|&nbsp;
                        {int(row.get('year', 0))}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        top_neg = filtered[
            filtered['predicted_sentiment'] == 'Negative'
        ].nlargest(5, 'confidence')

        for _, row in top_neg.iterrows():
            st.markdown(f"""
            <div style="background:#FEF2F2;border:1px solid #FCA5A5;
                        border-radius:8px;padding:1rem;margin-bottom:0.8rem;">
                <div style="font-size:0.85rem;color:#374151;">
                    {str(row['cleaned_text'])[:300]}...
                </div>
                <div style="margin-top:0.5rem;font-size:0.75rem;color:#6B7280;">
                    ⭐ {int(row['Score'])} stars &nbsp;|&nbsp;
                    Confidence: {row['confidence']*100:.1f}% &nbsp;|&nbsp;
                    {int(row.get('year', 0))}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        if 'HelpfulnessNumerator' in filtered.columns:
            most_helpful = filtered.nlargest(5, 'HelpfulnessNumerator')
            for _, row in most_helpful.iterrows():
                sentiment_color = "#F0FDF4" if row[
                    'predicted_sentiment'] == 'Positive' else "#FEF2F2"
                st.markdown(f"""
                <div style="background:{sentiment_color};border:1px solid #E5E7EB;
                            border-radius:8px;padding:1rem;margin-bottom:0.8rem;">
                    <div style="font-size:0.85rem;color:#374151;">
                        {str(row['cleaned_text'])[:300]}...
                    </div>
                    <div style="margin-top:0.5rem;font-size:0.75rem;color:#6B7280;">
                        👍 {int(row['HelpfulnessNumerator'])} helpful votes &nbsp;|&nbsp;
                        Sentiment: {row['predicted_sentiment']} &nbsp;|&nbsp;
                        ⭐ {int(row['Score'])} stars
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(
                "Helpfulness data not available in this dataset. "
                "Upload a CSV with 'HelpfulnessNumerator' column to enable this view."
            )