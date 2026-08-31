from textwrap import dedent
import streamlit as st

def render():
    html = dedent("""
    <div class='card'>
    <p><h3><b>problem statement</b></h3> <br>
        As image generation capabilities of LLMs improve, 
        it has become harder to differentiate an AI-generated image from a real one, 
        which poses serious implications, especially in the spread of disinformation on 
        social media platforms. Furthermore, existing AI image detectors often struggle 
        when the images have been slightly edited or compressed, which is often the case 
        with images shared on social media.  
    </p>
    <p><h3><b>aim</b> </h3> <br>
        We aim to build a robust AI image detector, ISeeU, that maintains 
        good performance and accuracy even under image transformations. 
        This can help reduce fraud that occur due to AI images circulating, 
        such as people falling for scams due to real-looking visuals that support 
        the scams' claims.
    </p>
    </div>
    """).strip()
    st.markdown(html, unsafe_allow_html=True)