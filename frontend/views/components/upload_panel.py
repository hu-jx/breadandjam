import streamlit as st
from controllers.upload_controller import handle_new_files
from state import uploads as uploads_state

def render():
    with st.container(key="upload_square"):
        st.markdown("#### Upload")
        files = st.file_uploader(
            "drop image(s) here pleasee",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if (files):
            handle_new_files(files)

        st.caption(f"{len(uploads_state.all_uploads())} image(s) in session")