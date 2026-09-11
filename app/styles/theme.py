# app/styles/theme.py
# FeedbackIQ — Global Design System
# All CSS, colors, and typography in one place.

# ── Color Palette ────────────────────────────────────────────
COLORS = {
    'primary'       : '#2563EB',   # blue — primary actions
    'primary_light' : '#EFF6FF',   # light blue — backgrounds
    'positive'      : '#16A34A',   # green — positive sentiment
    'positive_light': '#F0FDF4',   # light green — backgrounds
    'negative'      : '#DC2626',   # red — negative sentiment
    'negative_light': '#FEF2F2',   # light red — backgrounds
    'neutral'       : '#6B7280',   # gray — neutral elements
    'surface'       : '#F9FAFB',   # off-white — card backgrounds
    'border'        : '#E5E7EB',   # light gray — borders
    'text_primary'  : '#111827',   # near black — headings
    'text_secondary': '#6B7280',   # gray — subtitles
    'warning'       : '#D97706',   # amber — warnings
    'background'    : '#FFFFFF',   # white — page background
}

# ── Plotly Chart Theme ───────────────────────────────────────
PLOTLY_TEMPLATE = 'plotly_white'

CHART_COLORS = {
    'positive'  : '#16A34A',
    'negative'  : '#DC2626',
    'primary'   : '#2563EB',
    'secondary' : '#7C3AED',
    'neutral'   : '#6B7280',
    'scale'     : ['#DC2626', '#F97316', '#EAB308', '#22C55E', '#16A34A']
}

# ── Global CSS ───────────────────────────────────────────────
def load_css():
    return """
<style>

/* ── Reset & Base ──────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                 'Segoe UI', sans-serif;
}

/* ── Hide Streamlit Default Elements ───────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Sidebar ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: 1px solid #1F2937;
}
[data-testid="stSidebar"] * {
    color: #F9FAFB !important;
}

/* ── Brand Header ───────────────────────────────────────── */
.brand-container {
    padding: 1.5rem 0 1rem 0;
    border-bottom: 1px solid #1F2937;
    margin-bottom: 1.5rem;
}
.brand-name {
    font-size: 1.5rem;
    font-weight: 800;
    color: #FFFFFF !important;
    letter-spacing: -0.5px;
}
.brand-tagline {
    font-size: 0.72rem;
    color: #9CA3AF !important;
    margin-top: 0.2rem;
    letter-spacing: 0.3px;
}

/* ── Model Status Indicator ─────────────────────────────── */
.status-online {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #064E3B;
    color: #6EE7B7 !important;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 20px;
    margin-top: 0.8rem;
    letter-spacing: 0.3px;
}
.status-dot {
    width: 7px;
    height: 7px;
    background: #10B981;
    border-radius: 50%;
    display: inline-block;
}

/* ── Navigation ─────────────────────────────────────────── */
.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.6rem 0.8rem;
    border-radius: 8px;
    margin-bottom: 2px;
    cursor: pointer;
    font-size: 0.88rem;
    font-weight: 500;
    color: #D1D5DB !important;
    transition: background 0.15s;
    text-decoration: none;
}
.nav-item:hover {
    background: #1F2937;
}
.nav-item.active {
    background: #1E3A5F;
    color: #93C5FD !important;
    font-weight: 600;
}

/* ── Page Header ────────────────────────────────────────── */
.page-header {
    margin-bottom: 2rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid #E5E7EB;
}
.page-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #111827;
    letter-spacing: -0.5px;
    margin: 0;
}
.page-subtitle {
    font-size: 0.92rem;
    color: #6B7280;
    margin-top: 0.3rem;
}

/* ── KPI Cards ──────────────────────────────────────────── */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #6B7280;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #111827;
    letter-spacing: -0.5px;
    line-height: 1;
}
.kpi-delta {
    font-size: 0.78rem;
    color: #6B7280;
    margin-top: 0.4rem;
}
.kpi-accent {
    position: absolute;
    top: 0; right: 0;
    width: 4px;
    height: 100%;
    border-radius: 0 12px 12px 0;
}

/* ── Sentiment Result Card ──────────────────────────────── */
.result-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.sentiment-emoji {
    font-size: 3rem;
    margin-bottom: 0.5rem;
}
.sentiment-label-positive {
    font-size: 1.6rem;
    font-weight: 800;
    color: #16A34A;
    letter-spacing: -0.5px;
}
.sentiment-label-negative {
    font-size: 1.6rem;
    font-weight: 800;
    color: #DC2626;
    letter-spacing: -0.5px;
}
.confidence-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #111827;
    letter-spacing: -1px;
    margin: 0.5rem 0;
}
.confidence-label {
    font-size: 0.8rem;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Probability Bar ────────────────────────────────────── */
.prob-container {
    margin: 1.2rem 0;
}
.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}
.prob-label {
    font-size: 0.82rem;
    font-weight: 500;
    color: #374151;
    width: 70px;
    text-align: right;
}
.prob-bar-bg {
    flex: 1;
    background: #F3F4F6;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}
.prob-bar-fill-pos {
    height: 100%;
    background: #16A34A;
    border-radius: 4px;
}
.prob-bar-fill-neg {
    height: 100%;
    background: #DC2626;
    border-radius: 4px;
}
.prob-pct {
    font-size: 0.82rem;
    font-weight: 600;
    color: #374151;
    width: 45px;
}

/* ── Section Divider ────────────────────────────────────── */
.section-divider {
    border: none;
    border-top: 1px solid #E5E7EB;
    margin: 2rem 0;
}

/* ── Info Badge ─────────────────────────────────────────── */
.info-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #EFF6FF;
    color: #1D4ED8;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
}

/* ── Upload Zone ────────────────────────────────────────── */
.upload-info {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
}
.upload-info-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    padding: 3px 0;
}
.upload-info-label {
    color: #6B7280;
    font-weight: 500;
}
.upload-info-value {
    color: #111827;
    font-weight: 600;
}

/* ── Insight Card ───────────────────────────────────────── */
.insight-card {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 4px solid #D97706;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.insight-title {
    font-size: 0.8rem;
    font-weight: 700;
    color: #92400E;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.insight-text {
    font-size: 0.9rem;
    color: #78350F;
    margin-top: 0.3rem;
}

/* ── Model Pipeline ─────────────────────────────────────── */
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.7rem 1rem;
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 0.88rem;
    color: #374151;
    font-weight: 500;
}
.pipeline-arrow {
    text-align: center;
    color: #9CA3AF;
    font-size: 1rem;
    margin: 2px 0;
}

/* ── Tech Badge ─────────────────────────────────────────── */
.tech-badge {
    display: inline-block;
    background: #F3F4F6;
    color: #374151;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    margin: 3px;
    border: 1px solid #E5E7EB;
}

/* ── Empty State ────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #9CA3AF;
}
.empty-state-icon {
    font-size: 2.5rem;
    margin-bottom: 0.8rem;
}
.empty-state-text {
    font-size: 0.95rem;
    color: #6B7280;
}

/* ── Metric Comparison ──────────────────────────────────── */
.metric-better {
    color: #16A34A;
    font-weight: 700;
}
.metric-worse {
    color: #DC2626;
}

</style>
"""