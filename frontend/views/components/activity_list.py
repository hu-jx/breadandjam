import streamlit as st
from state import uploads as uploads_state
from views.helpers import render as render_template

def render(limit: int = 10):
    st.markdown("#### Activity")
    recent = uploads_state.recent(limit=limit)
    if (not recent):
        render_template("empty_card", message="no activity yet upload smt plz")
        return

    for i, upload in enumerate(recent):
        _render_row(i, upload)

def _render_row(index, upload):
    c1, c2, c3, c4, c5 = st.columns([1, 3, 2, 2, 1])
    with c1:
        st.image(upload.bytes, width=52)

    with c2:
        st.markdown(
            f"<div class='activity-name'>{upload.name[:28]}</div>"
            f"<div class='activity-time'>{upload.timestamp}</div>",
            unsafe_allow_html=True,
        )
    with c3:
        badge = "badge-ai" if upload.verdict == "AI" else "badge-human"
        st.markdown(f"<span class='{badge}'>{upload.verdict}</span>", unsafe_allow_html=True)

    with c4:
        st.markdown(f"conf · **{upload.confidence * 100:.0f}%**")

    with c5:
        st.button("open", key=f"open_{index}", use_container_width=True)