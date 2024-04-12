import streamlit as st
from streamlit_option_menu import option_menu
from app_utils import switch_page
import streamlit as st
from PIL import Image
import os

os.environ["OPENAI_API_KEY"] = "#####"


# im = Image.open(r"C:\Users\a\Downloads\GPTInterviewer-main (3)\GPTInterviewer-main\icon.png")
st.set_page_config(page_title = "SamvaadAI", layout = "centered")

lan = st.selectbox("#### Language", ["English"])

if lan == "English":
    home_title = "SamvaadAI"
    home_introduction = "Welcome to SamvaadAI, empowering your interview preparation with generative AI."

    st.markdown(
        "<style>#MainMenu{visibility:hidden;}</style>",
        unsafe_allow_html=True
    )
    # st.image(im, width=100)
    st.markdown("""\n""")
    #st.markdown("#### Greetings")
    st.markdown("Introducing SamvaadAI! Your dedicated AI-powered mock interviewer, SamvaadAI, is here to help you practice your interview skills."
                "Upload your resume or input job descriptions to receive customized interview questions from SamvaadAI.")
    st.markdown("""\n""")
    
    st.markdown("""\n""")
    st.markdown("#### Select a Mode")
    selected = option_menu(
            menu_title= None,
            options=["Topic", "Resume"],
            # icons = ["cast", "cloud-upload", "cast"],
            default_index=0,
            orientation="horizontal",
        )
    if selected == 'Topic':
        st.info("""
            SamvaadAI will assess your technical skills as they relate to the job description. """)
        if st.button("Start Interview!"):
            switch_page("Topic Mode")
    if selected == 'Resume':
        st.info("""
        SamvaadAI will review your resume and discuss your past experiences."""
        )
        if st.button("Start Interview!"):
            switch_page("Resume Mode")
    st.markdown("""\n""")
    #st.write(
    #        f'<iframe src="https://17nxkr0j95z3vy.embednotionpage.com/AI-Interviewer-Wiki-8d962051e57a48ccb304e920afa0c6a8" style="width:100%; height:100%; min-height:500px; border:0; padding:0;"/>',
    #        unsafe_allow_html=True,
    #    )

