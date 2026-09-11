# app/components/sidebar.py
# FeedbackIQ — Sidebar Navigation

import streamlit as st
from styles.theme import load_css

PAGES = [
    ("🏠", "Overview",           "Command Center"),
    ("✨", "Review Analyzer",    "Single Review Analysis"),
    ("📂", "Batch Intelligence", "Bulk CSV Analysis"),
    ("💡", "Customer Insights",  "Business Intelligence"),
    ("🤖", "Model Lab",          "Performance Metrics"),
    ("ℹ️",  "About",             "Project Information"),
]

def render_sidebar():
    """
    Renders the FeedbackIQ sidebar with navigation and model status.
    Returns the name of the currently selected page.
    """
    with st.sidebar:

        # ── Brand ────────────────────────────────────────────
        st.markdown("""
        <div class="brand-container">
            <div class="brand-name">💬 FeedbackIQ</div>
            <div class="brand-tagline">AI-Powered Customer Feedback Intelligence</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Model status ──────────────────────────────────────
        model_loaded = st.session_state.get('model_loaded', False)
        if model_loaded:
            st.markdown("""
            <div class="status-online">
                <span class="status-dot"></span>
                AI Model Online
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="color:#EF4444;font-size:0.75rem;margin-top:0.5rem;">
                ⚠️ Model Loading...
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Navigation ────────────────────────────────────────
        if 'current_page' not in st.session_state:
            st.session_state['current_page'] = 'Overview'

        for icon, name, description in PAGES:
            is_active = st.session_state['current_page'] == name
            button_style = "primary" if is_active else "secondary"

            if st.button(
                f"{icon}  {name}",
                key=f"nav_{name}",
                use_container_width=True,
                type=button_style
            ):
                st.session_state['current_page'] = name
                st.rerun()

        # ── Footer ────────────────────────────────────────────
        st.markdown("<br>" * 3, unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.72rem;color:#6B7280;padding-top:1rem;
                    border-top:1px solid #1F2937;">
            Model: distilbert-base-uncased<br>
            Dataset: Amazon Fine Food Reviews<br>
            Built for placement portfolio
        </div>
        """, unsafe_allow_html=True)

    return st.session_state['current_page']