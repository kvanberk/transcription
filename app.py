import os
import pandas as pd
import streamlit as st

from tempfile import NamedTemporaryFile

from nlp import EntityRecognition
from summarizer import LexSummarizer
from transcribe import Whisper

def remove_temp(dir="./temp"):
    items = os.listdir(dir)
    for item in items:
       os.remove(f"{dir}/{item}")

def create_session(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

# generate session states for parameters
create_session("transcript", None)
create_session("text", "")
create_session("audio", False)

# create app title
st.title("Whisper OpenAI Transcription")

# create audio upload widget
audio_upload = st.file_uploader("Upload Audio File")

# create transcribe button widget
transcribe_button = st.button("Transcribe")

# generate sidebar
with st.sidebar:
    st.session_state["option1"] = st.selectbox(
        label="Model Type",
        options=(
            "tiny",
            "base",
            "small",
            "medium",
            "large"
        ),
        index=1,
    )
    st.session_state["option2"] = st.selectbox(
        label="Language",
        options=([
            "English"
        ]),
        index=0
    )
    st.session_state["option3"] = st.checkbox(
        label="Summary" 
    )
    st.session_state["option4"] = st.checkbox(
        label="Insights" 
    )
    
if audio_upload is not None:
    st.session_state["audio"] = True
else:
    st.session_state["audio"] = False

if st.session_state["audio"]:
    if transcribe_button:
        with NamedTemporaryFile(dir="./temp", delete=False, suffix=".mp3") as temp:
            temp.write(audio_upload.getvalue())
            temp.seek(0)
            st.session_state["transcript"] = Whisper(st.session_state["option1"], st.session_state["option2"]).transcript(temp.name)
            temp.close()
            remove_temp()

    if st.session_state["transcript"] is not None:
        st.session_state["text"] = ""
        for segement in st.session_state["transcript"]["segments"]:
            st.session_state["text"] += f'[{segement["start"]:.2f} - {segement["end"]:.2f}]  \n {segement["text"]}  \n\n'
  
        with st.expander(label="Transcript", expanded=True):
            st.download_button("Download Transcript", st.session_state["text"], file_name=os.path.splitext(audio_upload.name)[0]+".txt")
            st.write(st.session_state["text"])

        if st.session_state["option3"]:
            transcript_summary = LexSummarizer(st.session_state["transcript"]["text"], 6)
            
            with st.expander(label="Summary", expanded=False):
                st.write(transcript_summary.summary)

        if st.session_state["option4"]:
            ner = EntityRecognition(st.session_state["transcript"]["text"])
            
            dataframe = pd.DataFrame(ner.entity, columns=["Entity", "Identified"])

            with st.expander(label="Insights", expanded=False):
                st.table(dataframe)            
     
else:
    pass