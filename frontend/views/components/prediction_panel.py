import streamlit as st
from services.detector import make_heatmap
from state import uploads as uploads_state
from views.helpers import render as render_template

def render():
    with st.container(key="predict_square"):
        st.markdown("#### Prediction")
        current = uploads_state.current()
        if (current is None):
            render_template("empty_message", message="upload an image to see predictions!!")
            return
        _render_images(current)
        _render_verdict(current)
        _render_nav()

def _render_images(current):
    img_c, heat_c = st.columns(2)
    with img_c:
        st.image(current.bytes, caption="image preview", use_container_width=True)
    with heat_c:
        try:
            st.image(make_heatmap(current.bytes), caption="ELA heatmap", use_container_width=True)
        except Exception:
            st.info("heatmap unavailable sorry gng :(")

def _render_verdict(current):
    pct = int(current.prob_ai * 100)
    st.markdown(f"**{pct}% likely AI-generated**")
    st.progress(current.prob_ai)
    st.caption(f"confidence score: {current.confidence * 100:.1f}%")

def _render_nav():
    prev_c, next_c, _ = st.columns([1, 1, 3])
    with prev_c:
        if st.button("prev", use_container_width=True):
            uploads_state.prev_image()
            st.rerun()
    with next_c:
        if (st.button("next", use_container_width=True)):
            uploads_state.next_image()
            st.rerun()