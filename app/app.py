# app/app.py
# FeedbackIQ — Main Entry Point
# Handles routing only. All page logic lives in pages/

import streamlit as st
import sys
import os

# ── Path setup ───────────────────────────────────────────────
APP_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, APP_DIR)

# ── Page config — MUST be first Streamlit call ───────────────
st.set_page_config(
    page_title="FeedbackIQ — Customer Feedback Intelligence",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load CSS ─────────────────────────────────────────────────
from styles.theme import load_css
st.markdown(load_css(), unsafe_allow_html=True)

# ── Load model once at startup ───────────────────────────────
@st.cache_resource(show_spinner=False)
def initialize_model():
    from src.inference import load_model
    load_model()
    return True

with st.spinner("Initializing FeedbackIQ..."):
    try:
        initialize_model()
        st.session_state['model_loaded'] = True
    except Exception as e:
        st.session_state['model_loaded'] = False
        st.error(f"❌ Model failed to load: {e}")
        st.stop()

# ── Sidebar + routing ────────────────────────────────────────
from components.sidebar import render_sidebar
page = render_sidebar()

if page == "Overview":
    from _pages.overview import render
    render()

elif page == "Review Analyzer":
    from _pages.review_analyzer import render
    render()

elif page == "Batch Intelligence":
    from _pages.batch_intelligence import render
    render()

elif page == "Customer Insights":
    from _pages.customer_insights import render
    render()

elif page == "Model Lab":
    from _pages.model_lab import render
    render()

elif page == "About":
    from _pages.about import render
    render()