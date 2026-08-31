from pathlib import Path
from string import Template
from textwrap import dedent
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
_TEMPLATES = _ROOT / "templates"

def load_css(filename: str = "styles.css"):
    css = (_ROOT / filename).read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def render(template_name: str, **fields):
    path = _TEMPLATES / f"{template_name}.html"
    raw = path.read_text()
    html = Template(raw).safe_substitute(**fields)
    st.html(html)

def spacer(size: str = "md"):
    st.markdown(f"<div class='spacer-{size}'></div>", unsafe_allow_html=True)