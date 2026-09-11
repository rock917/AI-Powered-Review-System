# app/_pages/model_lab.py
# FeedbackIQ — Model Lab / Performance Page

import streamlit as st
import pandas as pd
from components.charts import (
    model_comparison_bar,
    training_curves,
    accuracy_curves,
)
from components.cards import render_kpi_row
from utils.data_loader import (
    load_baseline_metrics,
    load_distilbert_metrics,
    load_training_history,
    get_model_info,
)


def render():

    st.markdown("""
    <div class="page-header">
        <div class="page-title">🤖 Model Lab</div>
        <div class="page-subtitle">
            ML engineering behind FeedbackIQ.
            Real evaluation results from 11,000 unseen test reviews.
        </div>
    """, unsafe_allow_html=True)

    baseline = load_baseline_metrics()
    distilbert = load_distilbert_metrics()
    history = load_training_history()
    model_info = get_model_info()

    # ── KPIs ──────────────────────────────────────────────────
    improvement = (distilbert["accuracy"] - baseline["accuracy"]) * 100

    render_kpi_row([
        (
            "TEST ACCURACY",
            f"{distilbert['accuracy'] * 100:.2f}%",
            f"+{improvement:.2f}% vs baseline",
            "#2563EB",
        ),
        (
            "PRECISION",
            f"{distilbert['precision'] * 100:.2f}%",
            "Positive predictive value",
            "#16A34A",
        ),
        (
            "RECALL",
            f"{distilbert['recall'] * 100:.2f}%",
            "Sensitivity / true positive rate",
            "#D97706",
        ),
        (
            "F1-SCORE",
            f"{distilbert['f1'] * 100:.2f}%",
            "Harmonic mean of precision & recall",
            "#7C3AED",
        ),
    ])

    st.markdown(
        "<div style='margin:1.5rem 0;'></div>",
        unsafe_allow_html=True,
    )

    # ── Model Comparison ──────────────────────────────────────
    st.markdown("### 📊 Baseline vs DistilBERT")
    st.caption(
        "All metrics measured on the same 11,000 unseen test reviews."
    )

    st.plotly_chart(
        model_comparison_bar(baseline, distilbert),
        use_container_width=True,
    )

    metrics_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
        ],
        "Baseline": [
            baseline["accuracy"],
            baseline["precision"],
            baseline["recall"],
            baseline["f1"],
        ],
        "DistilBERT": [
            distilbert["accuracy"],
            distilbert["precision"],
            distilbert["recall"],
            distilbert["f1"],
        ],
    })

    metrics_df["Improvement"] = (
        metrics_df["DistilBERT"] - metrics_df["Baseline"]
    )

    st.dataframe(
        metrics_df.style.format({
            "Baseline": "{:.4f}",
            "DistilBERT": "{:.4f}",
            "Improvement": "+{:.4f}",
        }).background_gradient(
            subset=["DistilBERT"],
            cmap="Blues",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # ── Training Curves ───────────────────────────────────────
    st.markdown("### 📈 Training History")
    st.caption(
        "Epoch 2 selected as best model — "
        "val_loss increased in epoch 3 indicating early overfitting."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Loss Curves**")
        st.plotly_chart(
            training_curves(history),
            use_container_width=True,
        )

    with col2:
        st.markdown("**Accuracy Curves**")
        st.plotly_chart(
            accuracy_curves(history),
            use_container_width=True,
        )

    st.markdown("---")

    # ── Model Info ────────────────────────────────────────────
    st.markdown("### ⚙️ Model Configuration")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Architecture**")

        config_rows = [
            ("Model", model_info["model_name"]),
            ("Task", model_info["task"]),
            ("Classes", " · ".join(model_info["classes"])),
            ("Parameters", "66 million"),
            ("Max Seq Length", str(model_info["max_length"])),
        ]

        for label, value in config_rows:
            st.markdown(
                f'<div class="upload-info-row" style="padding:5px 0;'
                f'border-bottom:1px solid #F3F4F6;">'
                f'<span class="upload-info-label">{label}</span>'
                f'<span class="upload-info-value">{value}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col_b:
        st.markdown("**Training Setup**")

        train_rows = [
            ("Training samples", f"{model_info['training_samples']:,}"),
            ("Test samples", f"{model_info['test_samples']:,}"),
            ("Batch size", str(model_info["batch_size"])),
            ("Learning rate", model_info["learning_rate"]),
            (
                "Epochs",
                f"{model_info['epochs']} "
                f"(best: {model_info['best_epoch']})",
            ),
            ("Hardware", model_info["hardware"]),
            ("Training time", model_info["training_time"]),
        ]

        for label, value in train_rows:
            st.markdown(
                f'<div class="upload-info-row" style="padding:5px 0;'
                f'border-bottom:1px solid #F3F4F6;">'
                f'<span class="upload-info-label">{label}</span>'
                f'<span class="upload-info-value">{value}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Known Limitations ─────────────────────────────────────
    st.markdown("### ⚠️ Known Limitations")

    limitations = [
        (
            "Labeling Noise",
            "Star rating occasionally contradicts review text. "
            "Some 5-star reviews contain negative language.",
        ),
        (
            "Truncation",
            "Reviews exceeding 256 tokens are truncated. "
            "Sentiment appearing late in long reviews may be missed.",
        ),
        (
            "Domain Specificity",
            "Trained on food reviews — may generalize less "
            "to electronics, fashion, or other product categories.",
        ),
        (
            "Sarcasm",
            "Sarcastic reviews using positive words negatively "
            "remain a challenge for any sentiment model.",
        ),
    ]

    for title, text in limitations:
        st.markdown(
            f'<div class="insight-card">'
            f'<div class="insight-title">{title}</div>'
            f'<div class="insight-text">{text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Inference Cache Stats ─────────────────────────────────
    st.markdown("### ⚡ Inference Cache")

    st.caption(
        "Previously analyzed reviews are returned instantly "
        "from SQLite cache without re-running the model."
    )

    try:
        from utils.cache import get_cache_stats, clear_cache

        stats = get_cache_stats()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Cached Reviews",
            f"{stats['total']:,}",
        )
        c2.metric(
            "Positive Cached",
            f"{stats['positive']:,}",
        )
        c3.metric(
            "Negative Cached",
            f"{stats['negative']:,}",
        )
        c4.metric(
            "Cache DB Size",
            f"{stats['db_size']:.1f} KB",
        )

        db_path = stats["db_path"]

        st.markdown(
            '<div class="upload-info" style="margin-top:1rem;">'
            '<div class="upload-info-row">'
            '<span class="upload-info-label">Database</span>'
            '<span class="upload-info-value">SQLite</span>'
            '</div>'
            '<div class="upload-info-row">'
            '<span class="upload-info-label">Lookup method</span>'
            '<span class="upload-info-value">SHA256 hash of review text</span>'
            '</div>'
            '<div class="upload-info-row">'
            '<span class="upload-info-label">Cache hit time</span>'
            '<span class="upload-info-value">&lt; 1ms per review</span>'
            '</div>'
            '<div class="upload-info-row">'
            '<span class="upload-info-label">DB location</span>'
            f'<span class="upload-info-value">{db_path}</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='margin-top:1rem;'></div>",
            unsafe_allow_html=True,
        )

        with st.expander("🔍 How does the cache work?"):
            st.markdown("""
            **Flow for every batch upload:**

            1. **Review text is received** from the uploaded dataset.
            2. **SHA256 hash is generated** from the review text.
            3. The hash is checked against the **SQLite cache**.
            4. If a match exists, the previously calculated sentiment
               is returned immediately.
            5. If no match exists, the review is sent to the
               **DistilBERT model**.
            6. The prediction is stored in the cache for future requests.

            This avoids running the model again for reviews that have
            already been analyzed.
            """)

        if st.button("🗑️ Clear Inference Cache"):
            clear_cache()
            st.success("Inference cache cleared successfully.")
            st.rerun()

    except Exception as e:
        st.info(
            "Inference cache statistics are currently unavailable. "
            f"Details: {e}"
        )