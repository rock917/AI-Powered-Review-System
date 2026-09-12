# app/_pages/batch_intelligence.py
# FeedbackIQ — Batch CSV Analysis Page

import streamlit as st
import pandas as pd
import torch
import io
from src.inference import predict_dataframe
from components.charts import sentiment_donut, confidence_histogram
from components.cards import render_kpi_row, empty_state
from utils.validators import validate_csv

# Environment detection
IS_GPU     = torch.cuda.is_available()
HARD_LIMIT = 10000 if IS_GPU else 500


def estimate_time(n: int) -> str:
    """Estimate inference time based on environment."""
    secs = n * (0.05 if IS_GPU else 0.4)
    if secs < 60:
        return f"~{int(secs)} seconds"
    elif secs < 3600:
        return f"~{int(secs/60)} minutes"
    else:
        return f"~{secs/3600:.1f} hours"


def render():

    # ── Header ───────────────────────────────────────────────
    st.markdown("""
    <div class="page-header">
        <div class="page-title">📂 Batch Intelligence</div>
        <div class="page-subtitle">
            Analyze thousands of customer reviews at once.
            Upload a CSV and get sentiment predictions for every row.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Instructions ─────────────────────────────────────────
    with st.expander("📋 How to prepare your CSV"):
        st.markdown(f"""
        **Required:** At least one column containing review text.

        **Accepted column names:**
        `reviewText` · `review_text` · `cleaned_text` · `text` ·
        `Text` · `review` · `Review` · `feedback` · `Review Text`

        **Optional columns (preserved in output):**
        `rating` · `Score` · `product_id` · `date`

        **Limits:**
        Maximum **{HARD_LIMIT:,} reviews** on this environment
        ({'GPU — local' if IS_GPU else 'CPU — cloud deployment'}).
        CSV format · UTF-8 encoding recommended.
        """)

    # ── Upload ────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Drop your CSV file here",
        type=['csv'],
        label_visibility="collapsed"
    )

    if uploaded is None:
        empty_state(
            "📂",
            "Upload a CSV file to begin batch sentiment analysis."
        )
        return

    # ── Load CSV ──────────────────────────────────────────────
    try:
        df_raw = pd.read_csv(
            uploaded,
            encoding='utf-8',
            on_bad_lines='skip',
            engine='python',
            quoting=0,
            dtype=str
        )
    except UnicodeDecodeError:
        try:
            uploaded.seek(0)
            df_raw = pd.read_csv(
                uploaded,
                encoding='latin-1',
                on_bad_lines='skip',
                engine='python',
                dtype=str
            )
        except Exception as e:
            st.error(
                f"❌ Could not read file. "
                f"Please save your CSV as UTF-8 and try again.\n\n"
                f"Technical detail: {e}"
            )
            return
    except Exception as e:
        st.error(
            f"❌ Could not read file. "
            f"The CSV may be malformed or use an unsupported format.\n\n"
            f"Technical detail: {e}"
        )
        return

    # ── Validate ──────────────────────────────────────────────
    validation = validate_csv(df_raw)

    if not validation['valid']:
        st.error(f"❌ {validation['error']}")
        st.markdown("**Your file preview:**")
        st.dataframe(df_raw.head(5), use_container_width=True)
        return

    text_col = validation['text_column']

    # ── File info card ────────────────────────────────────────
    name_val = uploaded.name
    rows_val = f"{len(df_raw):,}"
    cols_val = str(len(df_raw.columns))
    st.markdown(
        '<div class="upload-info">'
        '<div class="upload-info-row">'
        '<span class="upload-info-label">File</span>'
        f'<span class="upload-info-value">{name_val}</span>'
        '</div>'
        '<div class="upload-info-row">'
        '<span class="upload-info-label">Rows detected</span>'
        f'<span class="upload-info-value">{rows_val}</span>'
        '</div>'
        '<div class="upload-info-row">'
        '<span class="upload-info-label">Columns</span>'
        f'<span class="upload-info-value">{cols_val}</span>'
        '</div>'
        '<div class="upload-info-row">'
        '<span class="upload-info-label">Review column</span>'
        f'<span class="upload-info-value">✅ {text_col}</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Preview ───────────────────────────────────────────────
    st.markdown("**Data Preview**")
    st.dataframe(
        df_raw[[text_col]].head(5),
        use_container_width=True,
        hide_index=True
    )

    # ── Row count slider ──────────────────────────────────────
    total_available = min(len(df_raw), HARD_LIMIT)

    if len(df_raw) > 50:
        st.markdown("**How many reviews do you want to analyze?**")

        selected_rows = st.slider(
            label="Number of reviews",
            min_value=50,
            max_value=total_available,
            value=min(200, total_available),
            step=50,
            help=f"Maximum {total_available:,} reviews on this environment"
        )

        col_e1, col_e2, col_e3 = st.columns(3)
        col_e1.metric("Selected",    f"{selected_rows:,} reviews")
        col_e2.metric("Est. Time",   estimate_time(selected_rows))
        col_e3.metric("Environment", "GPU ⚡" if IS_GPU else "CPU ☁️")

        st.caption(
            f"📌 **Maximum limit: {total_available:,} reviews** "
            f"on this environment "
            f"({'GPU — local' if IS_GPU else 'CPU — cloud deployment'}). "
            f"Analysis will stop automatically at this limit. "
            f"Beyond {total_available:,} rows the app will not process "
            f"further to avoid timeouts."
        )

        if not IS_GPU and selected_rows > 200:
            st.warning(
                f"⚠️ Analyzing {selected_rows:,} reviews on CPU will take "
                f"{estimate_time(selected_rows)}. "
                f"Consider starting with 200 reviews to test first."
            )

    else:
        selected_rows = total_available

    # ── Analyze button ────────────────────────────────────────
    if st.button("✨ Analyze Reviews", type="primary",
                 use_container_width=True):

        df_work = df_raw.head(selected_rows).copy()
        df_work[text_col] = df_work[text_col].fillna('').astype(str)
        df_work = df_work[df_work[text_col].str.strip() != '']

        progress      = st.progress(0, text="Starting analysis...")
        status        = st.empty()
        progress_text = st.empty()

        def update_progress(current, total, message):
            pct = int(current / total * 100) if total > 0 else 0
            progress.progress(pct, text=message)
            progress_text.caption(message)

        from utils.cached_inference import cached_predict_dataframe

        df_results, cache_stats = cached_predict_dataframe(
            df_work,
            text_column=text_col,
            progress_callback=update_progress
        )

        progress.progress(100, text="Analysis complete!")

        hit_rate = cache_stats['hit_rate']
        if cache_stats['cache_hits'] > 0:
            status.success(
                f"✅ Analyzed {len(df_results):,} reviews — "
                f"⚡ {cache_stats['cache_hits']:,} from cache "
                f"({hit_rate:.1f}% hit rate) · "
                f"🤖 {cache_stats['cache_misses']:,} new predictions"
            )
        else:
            status.success(
                f"✅ Analyzed {len(df_results):,} reviews · "
                f"All new — results cached for next upload"
            )

        st.session_state['batch_results']  = df_results
        st.session_state['batch_text_col'] = text_col

    # ── Show results ──────────────────────────────────────────
    if 'batch_results' not in st.session_state:
        return

    df_results = st.session_state['batch_results']
    text_col   = st.session_state['batch_text_col']

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### Results")

    n_total  = len(df_results)
    n_pos    = (df_results['predicted_sentiment'] == 'Positive').sum()
    n_neg    = (df_results['predicted_sentiment'] == 'Negative').sum()
    avg_conf = df_results['confidence'].mean()

    render_kpi_row([
        ("ANALYZED",       f"{n_total:,}",
         "Total reviews", "#2563EB"),
        ("POSITIVE",       f"{n_pos:,}",
         f"{n_pos/n_total*100:.1f}%", "#16A34A"),
        ("NEGATIVE",       f"{n_neg:,}",
         f"{n_neg/n_total*100:.1f}%", "#DC2626"),
        ("AVG CONFIDENCE", f"{avg_conf*100:.1f}%",
         "Model certainty", "#7C3AED"),
    ])

    st.markdown("<div style='margin:1.5rem 0;'></div>",
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sentiment Split**")
        st.plotly_chart(
            sentiment_donut(n_pos, n_neg),
            use_container_width=True
        )
    with col2:
        st.markdown("**Confidence Distribution**")
        st.plotly_chart(
            confidence_histogram(df_results),
            use_container_width=True
        )

    # ── Filters + Table ───────────────────────────────────────
    st.markdown("**Explore Results**")
    f1, f2, f3 = st.columns(3)

    with f1:
        sentiment_filter = st.selectbox(
            "Sentiment", ["All", "Positive", "Negative"]
        )
    with f2:
        conf_min = st.slider(
            "Min Confidence", 0.0, 1.0, 0.0, 0.05,
            key="conf_filter"
        )
    with f3:
        search = st.text_input(
            "Search reviews", placeholder="Search..."
        )

    filtered = df_results.copy()
    if sentiment_filter != "All":
        filtered = filtered[
            filtered['predicted_sentiment'] == sentiment_filter
        ]
    filtered = filtered[filtered['confidence'] >= conf_min]
    if search.strip():
        filtered = filtered[
            filtered[text_col].str.contains(
                search, case=False, na=False
            )
        ]

    display_cols = [
        text_col, 'predicted_sentiment',
        'confidence', 'prob_positive', 'prob_negative'
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[display_cols].style.format({
            'confidence'   : '{:.3f}',
            'prob_positive': '{:.3f}',
            'prob_negative': '{:.3f}'
        }),
        use_container_width=True,
        height=380,
        hide_index=True
    )
    st.caption(f"Showing {len(filtered):,} of {n_total:,} reviews")

    # ── Download ──────────────────────────────────────────────
    csv_buffer = io.StringIO()
    df_results.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download Results CSV",
        data=csv_buffer.getvalue(),
        file_name="feedbackiq_results.csv",
        mime="text/csv",
        use_container_width=True
    )