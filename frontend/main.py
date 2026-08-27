#code for dashboard in this folder
import streamlit as st
from state import uploads as uploads_state, page as page_state
from views.helpers import load_css, spacer
from views.components import performance_footer
from views.pages import main_page, insights_page, about_page

st.set_page_config(
    page_title="iSeeU",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()
uploads_state.init()
page_state.init()

PAGES = {
    "main":     ("main",     main_page),
    "insights": ("insights", insights_page),
    "about":    ("about",    about_page),
}

with st.sidebar:
    st.markdown('<div class="sidebar-logo">iSeeU</div>', unsafe_allow_html=True)
    for key, (label, _) in PAGES.items():
        if st.button(label, use_container_width=True, key=f"nav_{key}"):
            page_state.switch_to(key)
    spacer("sm")
    st.caption("bread&jam")

title_col, repo_col = st.columns([3, 1])

with st.container(key="main_content"):
    title_col, repo_col = st.columns([3, 1])
    with title_col:
        st.markdown(
            '''<div class="prod-title">iSeeU</div>
            <div class="prod-subtitle">AIGC Detector</div>''',
            unsafe_allow_html=True
        )

    with repo_col:
        spacer("sm")
        st.link_button(
            "click here to view our repo",
            "https://github.com/hu-jx/breadandjam",
            use_container_width=True
        )

    spacer("md")
    PAGES[page_state.current()][1].render()

performance_footer.render()