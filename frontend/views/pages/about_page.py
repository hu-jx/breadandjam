from textwrap import dedent
import streamlit as st

def render():
    html = dedent("""
    <div class='card'>
    <p><b>problem framing.</b> 
        whaaaaaaat 
    </p>
    <p><b>impact</b> 
        whaaaaaaaaaaat gys we're so impactful now give us 50k
    </p>
    </div>
    """).strip()
    st.markdown(html, unsafe_allow_html=True)