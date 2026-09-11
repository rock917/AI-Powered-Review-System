# app/pages/about.py
# FeedbackIQ — About Page

import streamlit as st


def render():
    st.markdown("""
    <div class="page-header">
        <div class="page-title">ℹ️ About FeedbackIQ</div>
        <div class="page-subtitle">
            AI-Powered Customer Feedback Intelligence System
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#F9FAFB;border:1px solid #E5E7EB;
                border-radius:12px;padding:2rem;margin-bottom:2rem;">
        <div style="font-size:2rem;font-weight:800;color:#111827;
                    letter-spacing:-1px;">
            💬 FeedbackIQ
        </div>
        <div style="font-size:1rem;color:#6B7280;margin-top:0.5rem;">
            Turn customer reviews into actionable intelligence.
        </div>
        <div style="margin-top:1.2rem;font-size:0.92rem;color:#374151;
                    line-height:1.7;">
            FeedbackIQ is an NLP-powered platform that analyzes customer
            feedback using a fine-tuned DistilBERT transformer model.
            Built as a placement portfolio project to demonstrate
            end-to-end ML engineering — from raw data to deployed product.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tech Stack ────────────────────────────────────────────
    st.markdown("### 🛠️ Technology Stack")
    tech_items = [
        ("Python 3.11",             "Core language"),
        ("PyTorch 2.5.1",           "Deep learning framework"),
        ("Hugging Face Transformers","DistilBERT model + tokenizer"),
        ("DistilBERT",              "66M parameter transformer"),
        ("Scikit-learn",            "TF-IDF baseline model"),
        ("Pandas / NumPy",          "Data processing"),
        ("Plotly",                  "Interactive visualizations"),
        ("Streamlit",               "Web application"),
    ]

    cols = st.columns(4)
    for i, (tech, desc) in enumerate(tech_items):
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background:#F9FAFB;border:1px solid #E5E7EB;
                        border-radius:8px;padding:0.8rem;
                        margin-bottom:0.8rem;text-align:center;">
                <div style="font-size:0.85rem;font-weight:700;
                            color:#111827;">{tech}</div>
                <div style="font-size:0.72rem;color:#6B7280;
                            margin-top:3px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Dataset ───────────────────────────────────────────────
    st.markdown("### 📦 Dataset")
    st.markdown("""
    <div class="upload-info">
        <div class="upload-info-row">
            <span class="upload-info-label">Name</span>
            <span class="upload-info-value">Amazon Fine Food Reviews</span>
        </div>
        <div class="upload-info-row">
            <span class="upload-info-label">Source</span>
            <span class="upload-info-value">McAuley et al. — Stanford / Kaggle</span>
        </div>
        <div class="upload-info-row">
            <span class="upload-info-label">Total reviews</span>
            <span class="upload-info-value">568,454</span>
        </div>
        <div class="upload-info-row">
            <span class="upload-info-label">Date range</span>
            <span class="upload-info-value">1999 – 2012</span>
        </div>
        <div class="upload-info-row">
            <span class="upload-info-label">Training subset</span>
            <span class="upload-info-value">
                110,000 (55K positive + 55K negative, balanced)
            </span>
        </div>
        <div class="upload-info-row">
            <span class="upload-info-label">Labeling strategy</span>
            <span class="upload-info-value">
                4–5 stars → Positive · 1–2 stars → Negative · 3 stars → Dropped
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Pipeline ──────────────────────────────────────────────
    st.markdown("### 🔄 ML Pipeline")

    steps = [
        ("📦", "Data",          "568K Amazon reviews · CSV format"),
        ("🧹", "Preprocessing", "Clean · label · balance · split 80/10/10"),
        ("📊", "Baseline",      "TF-IDF + Logistic Regression · 92.12% accuracy"),
        ("🤖", "DistilBERT",    "Fine-tune 66M param transformer · 3 epochs"),
        ("📈", "Evaluation",    "95.01% accuracy on 11K unseen reviews"),
        ("⚡", "Inference",     "src/inference.py · GPU-accelerated"),
        ("💡", "Intelligence",  "FeedbackIQ dashboard · batch analysis"),
    ]

    for i, (icon, title, desc) in enumerate(steps):
        st.markdown(f"""
        <div class="pipeline-step">
            <span style="font-size:1.3rem;">{icon}</span>
            <div>
                <strong>{title}</strong>
                <div style="font-size:0.78rem;color:#6B7280;">
                    {desc}
                </div>
            </div>
        </div>
        {'<div class="pipeline-arrow">↓</div>' if i < len(steps)-1 else ''}
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        "FeedbackIQ — Built as a placement portfolio project. "
        "All predictions are probabilistic. "
        "Not affiliated with Amazon."
    )