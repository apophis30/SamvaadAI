import streamlit as st
from streamlit_lottie import st_lottie
from typing import Literal
from dataclasses import dataclass
import json
import soundfile as sf
import numpy as np
import os
import base64
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import NLTKTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import nltk
import importlib
from prompts.prompts import templates
# Audio
from speech_recognition.openai_whisper import save_wav_file, transcribe
from audio_recorder_streamlit import audio_recorder
from aws.synthesize_speech import synthesize_speech
from IPython.display import Audio



#st.markdown("""solutions to potential errors:""")

jd = st.text_area("Enter Job Description / Interview Topic ")
auto_play = st.checkbox("Enable Voice for Questions")


@dataclass
class Message:
    """class for keeping track of interview history."""
    origin: Literal["human", "ai"]
    message: str

def save_vector(text):
    """embeddings"""

    nltk.download('punkt')
    text_splitter = NLTKTextSplitter()
    texts = text_splitter.split_text(text)
     # Create emebeddings
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    return docsearch

def initialize_session_state_jd():
    """ initialize session states """
    if 'jd_docsearch' not in st.session_state:
        st.session_state.jd_docserch = save_vector(jd)
    if 'jd_retriever' not in st.session_state:
        st.session_state.jd_retriever = st.session_state.jd_docserch.as_retriever(search_type="similarity")
    if 'jd_chain_type_kwargs' not in st.session_state:
        Interview_Prompt = PromptTemplate(input_variables=["context", "question"],
                                          template=templates.jd_template)
        st.session_state.jd_chain_type_kwargs = {"prompt": Interview_Prompt}
    if 'jd_memory' not in st.session_state:
        st.session_state.jd_memory = ConversationBufferMemory()
    # interview history
    if "jd_history" not in st.session_state:
        st.session_state.jd_history = []
        st.session_state.jd_history.append(Message("ai",
                                                   "Hello and welcome to your interview. I will be conducting the interview today. Let's discuss some questions related to the job description you provided."
                                                   "Start with your interview.  "))
    # token count
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "jd_guideline" not in st.session_state:
        llm = ChatOpenAI(
        model_name = "gpt-3.5-turbo",
        temperature = 0.8,)
        st.session_state.jd_guideline = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type_kwargs=st.session_state.jd_chain_type_kwargs, chain_type='stuff',
            retriever=st.session_state.jd_retriever, memory = st.session_state.jd_memory).run("Create an interview guideline and prepare only one questions for each topic. Make sure the questions tests the technical knowledge")
    # llm chain and memory
    if "jd_screen" not in st.session_state:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.8, )
        PROMPT = PromptTemplate(
            input_variables=["history", "input"],
            template="""Please assume the role of an interviewer named GPTInterviewer and adhere strictly to the guidelines provided for this interaction. The candidate is unaware of these guidelines. Your task is to engage with me by posing questions based on our ongoing conversation. Ensure each question is unique, without repetitions or explanations. Approach the questioning as if in a real-life interview scenario, asking only one question at a time. Follow up with additional questions if necessary. Respond only in your role as an interviewer, and do not write out the entire conversation at once. If an error occurs, please point it out.


                            Current Conversation:
                            {history}

                            Candidate: {input}
                            AI: """)

        st.session_state.jd_screen = ConversationChain(prompt=PROMPT, llm=llm,
                                                           memory=st.session_state.jd_memory)
    if 'jd_feedback' not in st.session_state:
        llm = ChatOpenAI(
            model_name="gpt-4-turbo-2024-04-09",
            temperature=0.8, )
        st.session_state.jd_feedback = ConversationChain(
            prompt=PromptTemplate(input_variables=["history", "input"], template=templates.feedback_template),
            llm=llm,
            memory=st.session_state.jd_memory,
        )

def answer_call_back():
    with get_openai_callback() as cb:
        # user input
        human_answer = st.session_state.answer
        # transcribe audio
        if voice:
            save_wav_file(r"C:\Users\Arpan Kumar\Downloads\GPTInterviewer-main (3)\GPTInterviewer-main\GPTInterviewer-main\temp\audio.wav", human_answer)
            try:
                # Load existing transcriptions from file
                try:
                    with open(r"C:\Users\Arpan Kumar\Downloads\GPTInterviewer-main (3)\GPTInterviewer-main\GPTInterviewer-main\temp\transcriptions.json", 'r') as f:
                        transcriptions = json.load(f)
                        print("transcriptions",transcriptions)
                except FileNotFoundError:
                    transcriptions = []  # Initialize as empty list if file does not exist

                print("1")
                # Load existing audio data
                if os.path.exists(r"C:\Users\Arpan Kumar\Downloads\GPTInterviewer-main (3)\GPTInterviewer-main\GPTInterviewer-main\temp\fullaudio.wav"):
                    existing_data, _ = sf.read(r"C:\Users\Arpan Kumar\Downloads\GPTInterviewer-main (3)\GPTInterviewer-main\GPTInterviewer-main\temp\fullaudio.wav")
                else:
                    existing_data = np.empty((0,2))  # Empty 2D array
                print("2")
                # Append new audio data
                new_data, _ = sf.read(r"C:\Users\Arpan Kumar\Downloads\GPTInterviewer-main (3)\GPTInterviewer-main\GPTInterviewer-main\temp\audio.wav")
                print("2.1")
                # If the audio is mono, add an extra dimension
                if len(existing_data.shape) == 1:
                    existing_data = np.expand_dims(existing_data, axis=1)
                if len(new_data.shape) == 1:
                    new_data = np.expand_dims(new_data, axis=1)

                full_data = np.concatenate((existing_data, new_data))
                print("3")
                # Save full audio data
                sf.write(r"C:\Users\Arpan Kumar\Downloads\GPTInterviewer-main (3)\GPTInterviewer-main\GPTInterviewer-main\temp\fullaudio.wav", full_data, 44100)
                print("4")
                input = transcribe(r"C:\Users\Arpan Kumar\Downloads\GPTInterviewer-main (3)\GPTInterviewer-main\GPTInterviewer-main\temp\audio.wav")
                print("5")
                transcriptions.append(input)
                print("6")
                with open(r"C:\Users\Arpan Kumar\Downloads\GPTInterviewer-main (3)\GPTInterviewer-main\GPTInterviewer-main\temp\transcriptions.json", 'w') as f:
                    json.dump(transcriptions, f)
                print("7")
                # save human_answer to history
            except Exception as e:
                print("exception faced",e)
                st.session_state.jd_history.append(Message("ai", "Sorry, I didn't get that."))
                return "Please try again."
        else:
            input = human_answer

        st.session_state.jd_history.append(
            Message("human", input)
        )
        # OpenAI answer and save to history
        llm_answer = st.session_state.jd_screen.run(input)
        audio_file_path = synthesize_speech(llm_answer)
        # create audio widget with autoplay
        audio_widget = Audio(audio_file_path, autoplay=True)
        # save audio data to history
        st.session_state.jd_history.append(
            Message("ai", llm_answer)
        )
        st.session_state.token_count += cb.total_tokens

        return audio_widget

if jd:
    # initialize session states
    initialize_session_state_jd()
    #st.write(st.session_state.jd_guideline)
    credit_card_placeholder = st.empty()
    col1, col2, col3 = st.columns(3)
    with col1:
        feedback = st.button("Get Interview Feedback")
    with col2:
        guideline = st.button("Show interview guideline!")
    with col3:
        analysis = st.button("Get Speech Analytics")
    chat_placeholder = st.container()
    answer_placeholder = st.container()
    audio = None
    # if submit email adress, get interview feedback imediately
    if guideline:
        st.write(st.session_state.jd_guideline)
    if feedback:
        evaluation = st.session_state.jd_feedback.run("please give evalution regarding the interview")
        st.markdown(evaluation, unsafe_allow_html=True)
        st.download_button(label="Download Interview Feedback", data=evaluation, file_name="interview_feedback.txt")
        st.stop()
    if analysis:
        st.write("Analysis")
        module_spec = importlib.util.spec_from_file_location("report", r"C:\Users\Arpan Kumar\Downloads\GPTInterviewer-main (3)\GPTInterviewer-main\GPTInterviewer-main\report.py")
        report_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(report_module)
        # report_module = importlib.import_module("report")
        report_module.generate_report()
    else:
        with answer_placeholder:
            voice: bool = st.checkbox("I would like to speak with SamvaadAI")
            if voice:
                answer = audio_recorder(pause_threshold = 2.5, sample_rate = 44100)
                #st.warning("An UnboundLocalError will occur if the microphone fails to record.")
            else:
                answer = st.chat_input("Your answer")
            if answer:
                st.session_state['answer'] = answer
                audio = answer_call_back()
        with chat_placeholder:
            for answer in st.session_state.jd_history:
                if answer.origin == 'ai':
                    if auto_play and audio:
                        with st.chat_message("assistant"):
                            st.write(answer.message)
                            st.write(audio)
                    else:
                        with st.chat_message("assistant"):
                            st.write(answer.message)
                else:
                    with st.chat_message("user"):
                        st.write(answer.message)

        credit_card_placeholder.caption(f"""
        Progress: {int(len(st.session_state.jd_history) / 30 * 100)}% completed.""")
else:
    st.info("Submit a job description to start the interview.")
