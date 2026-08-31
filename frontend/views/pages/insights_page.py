import streamlit as st
from views.helpers import render as render_template

_CLEAN_CM = {"title": "clean test set", "tn": 134, "fp": 66, "fn": 43, "tp": 157}
_ROBUST_CM = {"title": "robust test set", "tn": 141, "fp": 59, "fn": 43, "tp": 157}

def render():
    st.markdown("#### Confusion Matrix")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        render_template("confusion_matrix", **_CLEAN_CM)
    with col2:
        render_template("confusion_matrix", **_ROBUST_CM)