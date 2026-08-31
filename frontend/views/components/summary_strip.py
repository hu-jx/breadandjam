import streamlit as st
from services.export import uploads_to_json
from state import uploads as uploads_state
from views.helpers import render as render_template

def render():
    st.markdown("#### Summary")
    stats = uploads_state.stats()
    all_uploads = uploads_state.all_uploads()
    with st.container(key="summary_bar"):
        c1, c2, c3 = st.columns(3)
        with c1:
            render_template("bar_stat", label="total images", value=stats["total"], 
                accent_class="")
                
        with c2:
            render_template("bar_stat", label="ai flagged", value=stats["flagged"], 
                accent_class="accent")

        with c3:
            if (all_uploads):
                st.download_button(
                    "download json",
                    data=uploads_to_json(all_uploads),
                    file_name="iSeeU_results.json",
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                st.button("download json", disabled=True, use_container_width=True)