import streamlit as st

from utils.config import (
    APP_TAGLINE,
    APP_TITLE,
    APP_VERSION,
    AUTHOR_NAME,
    DETECTION_MODES,
    GITHUB_URL,
    LINKEDIN_URL,
    MODEL_OPTIONS,
    ORG_NAME,
)
from utils.ui import inject_global_styles, render_metric_cards, render_page_footer

st.set_page_config(
    page_title=f"{APP_TITLE} — Home",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_styles()

st.markdown(
    f"""
    <div class="hero">
        <div class="badge">Version {APP_VERSION}</div>
        <h1>{APP_TITLE}</h1>
        <p>{APP_TAGLINE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

render_metric_cards(
    [
        ("3", "Voice Detection Modules"),
        ("4", "ML Models Available"),
        ("40", "MFCC Features / Sample"),
        ("1", "Defect Prediction Engine"),
    ]
)

st.markdown("## Explore Detection Modules")
st.caption("Choose a specialized page from the sidebar or the cards below.")

cols = st.columns(3)
page_links = [
    ("pages/1_Urdu_Deepfake_Detection.py", DETECTION_MODES["urdu"]),
    ("pages/2_General_Voice_Detection.py", DETECTION_MODES["general"]),
    ("pages/3_AI_Voice_Clone_Detection.py", DETECTION_MODES["clone"]),
]

for col, (_, mode) in zip(cols, page_links):
    with col:
        st.markdown(
            f"""
            <div class="section-card" style="border-top: 4px solid {mode['accent']};">
                <h3 style="margin-top:0;">{mode['icon']} {mode['title']}</h3>
                <p style="color:#64748b;">{mode['description']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("## Additional Tools")

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
        <div class="section-card">
            <h3 style="margin-top:0;">🐛 Software Defect Prediction</h3>
            <p style="color:#64748b;">
                Analyze 40 software metrics to predict potential defects using
                SVM, Logistic Regression, Perceptron, or DNN.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="section-card">
            <h3 style="margin-top:0;">ℹ️ About & Documentation</h3>
            <p style="color:#64748b;">
                Learn about the project architecture, models, deployment options,
                and developer information.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown("## Supported Models")
st.write("All detection pages support the following classifiers:")
st.markdown(
    " · ".join([f"**{m}**" for m in MODEL_OPTIONS])
)

with st.expander("Quick Start Guide"):
    st.markdown(
        """
        1. Open any **Voice Detection** page from the sidebar  
        2. Select your preferred **ML model** in the sidebar  
        3. Upload a **WAV / MP3 / FLAC / OGG** audio file  
        4. Review prediction, confidence, waveform, and spectrogram  
        5. For defect prediction, enter 40 comma-separated metrics or upload CSV
        """
    )

st.sidebar.markdown("### 🔗 Connect")
st.sidebar.markdown(f"[GitHub Profile]({GITHUB_URL})")
st.sidebar.markdown(f"[LinkedIn Profile]({LINKEDIN_URL})")
st.sidebar.markdown("---")
st.sidebar.caption(f"© {AUTHOR_NAME} · {ORG_NAME}")

render_page_footer()
