import streamlit as st
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
    st.image(current.bytes, caption="image preview", use_container_width=True)

def _render_verdict(current):
    pct = int(current.prob_ai * 100)
    st.markdown(f"**{pct}% likely AI-generated**")
    st.progress(current.prob_ai)
    st.caption(f"confidence score: {current.confidence * 100:.1f}%")

def _render_nav():
    prev_c, next_c, counter_c = st.columns([1, 1, 3])
    with prev_c:
        if st.button("prev", use_container_width=True):
            uploads_state.prev_image()
            st.rerun()
    with next_c:
        if st.button("next", use_container_width=True):
            uploads_state.next_image()
            st.rerun()

    with counter_c:
        current_num, total = uploads_state.current_position()
        st.markdown(
            f"<div class='image-counter'>{current_num}/{total}</div>",
            unsafe_allow_html=True,
        )