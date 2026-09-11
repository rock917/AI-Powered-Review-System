# app/pages/review_analyzer.py
# FeedbackIQ — Single Review Analysis Page

import streamlit as st
from src.inference import predict_single
from components.cards import result_card, pipeline_display, empty_state
from utils.validators import validate_review_text

EXAMPLES = {
    "⭐ Excellent Product": (
        "This coffee is absolutely outstanding. The flavor is rich, smooth, "
        "and perfectly balanced. I have been ordering this for three years and "
        "it never disappoints. The packaging is excellent and it always arrives "
        "fresh. Highly recommend to anyone who takes their coffee seriously."
    ),
    "💔 Terrible Experience": (
        "Complete waste of money. The product arrived damaged and smelled "
        "terrible. I contacted customer service twice and got no response. "
        "It stopped working after just two days. The quality is shockingly "
        "poor for the price. Do not buy this under any circumstances."
    ),
    "😐 Mixed Experience": (
        "The taste is actually quite good and I enjoyed it at first. "
        "However, the packaging was damaged when it arrived and some "
        "of the product had spilled. Customer service was helpful but "
        "the overall experience left me uncertain about ordering again."
    ),
}


def render():
    # ── Header ───────────────────────────────────────────────
    st.markdown("""
    <div class="page-header">
        <div class="page-title">✨ Review Analyzer</div>
        <div class="page-subtitle">
            Understand any customer review using fine-tuned DistilBERT.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Example buttons ───────────────────────────────────────
    st.markdown("**Try an example:**")
    cols = st.columns(len(EXAMPLES))
    for col, (label, text) in zip(cols, EXAMPLES.items()):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state['analyzer_text'] = text

    st.markdown("<div style='margin:0.5rem 0;'></div>",
                unsafe_allow_html=True)

    # ── Text input ────────────────────────────────────────────
    review_text = st.text_area(
        "Customer Review",
        value=st.session_state.get('analyzer_text', ''),
        height=160,
        placeholder="Paste a customer review here...",
        label_visibility="collapsed",
        max_chars=5000,
        key="review_input"
    )

    # Live character/word count
    char_count = len(review_text)
    word_count = len(review_text.split()) if review_text.strip() else 0
    is_long    = word_count > 180

    count_color = "#D97706" if is_long else "#6B7280"
    st.markdown(
        f"<div style='font-size:0.78rem;color:{count_color};"
        f"text-align:right;margin-top:-0.5rem;margin-bottom:0.8rem;'>"
        f"{char_count} characters · {word_count} words"
        f"{'  ⚠️ Long review — will be truncated to 256 tokens' if is_long else ''}"
        f"</div>",
        unsafe_allow_html=True
    )

    # ── Analyze button ────────────────────────────────────────
    analyze_clicked = st.button(
        "✨ Analyze Review",
        type="primary",
        use_container_width=True,
        disabled=word_count == 0
    )

    # ── Inference ─────────────────────────────────────────────
    if analyze_clicked:
        validation = validate_review_text(review_text)
        if not validation['valid']:
            st.warning(f"⚠️ {validation['error']}")
            return

        with st.spinner("Running DistilBERT inference..."):
            result = predict_single(review_text)

        if result['error']:
            st.error(f"❌ {result['error']}")
            return

        st.session_state['last_result'] = result
        st.session_state['last_review'] = review_text

    # ── Show result ───────────────────────────────────────────
    if 'last_result' in st.session_state and st.session_state['last_result']:
        result = st.session_state['last_result']

        st.markdown("<div style='margin:1.5rem 0;'></div>",
                    unsafe_allow_html=True)
        st.markdown("### Analysis Result")

        col1, col2 = st.columns([1, 1])

        with col1:
            result_card(
                sentiment=result['sentiment'],
                confidence=result['confidence'],
                prob_positive=result['prob_positive'],
                prob_negative=result['prob_negative']
            )

        with col2:
            # Review stats
            review  = st.session_state.get('last_review', '')
            w_count = result.get('word_count', len(review.split()))

            st.markdown("**Review Details**")
            st.markdown(f"""
            <div class="upload-info">
                <div class="upload-info-row">
                    <span class="upload-info-label">Word count</span>
                    <span class="upload-info-value">{w_count}</span>
                </div>
                <div class="upload-info-row">
                    <span class="upload-info-label">Tokens used</span>
                    <span class="upload-info-value">
                        {'256 (truncated)' if result.get('truncated') else f'~{min(w_count*1.3, 256):.0f}'}
                    </span>
                </div>
                <div class="upload-info-row">
                    <span class="upload-info-label">Predicted class</span>
                    <span class="upload-info-value">{result['sentiment']}</span>
                </div>
                <div class="upload-info-row">
                    <span class="upload-info-label">Confidence</span>
                    <span class="upload-info-value">{result['confidence']*100:.2f}%</span>
                </div>
                <div class="upload-info-row">
                    <span class="upload-info-label">P(Positive)</span>
                    <span class="upload-info-value">{result['prob_positive']*100:.2f}%</span>
                </div>
                <div class="upload-info-row">
                    <span class="upload-info-label">P(Negative)</span>
                    <span class="upload-info-value">{result['prob_negative']*100:.2f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if result.get('truncated'):
                st.warning(
                    "⚠️ Review exceeded 256 tokens. "
                    "Analysis based on first ~180 words."
                )

        # ── Model transparency ────────────────────────────────
        with st.expander("🔍 How did the model decide?"):
            pipeline_display()
            st.markdown("<div style='margin-top:1rem;'></div>",
                        unsafe_allow_html=True)
            st.markdown("**Raw model output:**")
            col_a, col_b = st.columns(2)
            col_a.metric("P(Positive)",
                         f"{result['prob_positive']:.6f}")
            col_b.metric("P(Negative)",
                         f"{result['prob_negative']:.6f}")
            st.caption(
                "Logits are converted to probabilities via softmax. "
                "The class with higher probability is the prediction. "
                "Note: attention weights are not shown — doing so without "
                "proper attribution methods would be misleading."
            )

        st.caption(
            "⚠️ Predictions are probabilistic. "
            f"Model test accuracy: 95.01% on 11,000 unseen reviews."
        )
    else:
        st.markdown("<div style='margin-top:2rem;'></div>",
                    unsafe_allow_html=True)
        empty_state("✨", "Enter a review above and click Analyze to see results.")