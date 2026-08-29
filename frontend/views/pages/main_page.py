import streamlit as st
from views.components import (upload_panel, prediction_panel, summary_strip, activity_list,)
from views.helpers import spacer

def render():
    upload_col, predict_col = st.columns(2, gap="large")
    with upload_col:
        upload_panel.render()
    with predict_col:
        prediction_panel.render()
    spacer("md")
    summary_strip.render()
    spacer("md")
    activity_list.render()