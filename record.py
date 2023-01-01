import streamlit as st
import transcribe

st.title("Whisper OpenAI Transcription")
st.file_uploader("Upload Audio File")

if "run" not in st.session_state:
    st.session_state["run"] = False

def button_state():
    if st.session_state["run"]:
        st.session_state["run"] = False
    else:
        st.session_state["run"] = True

def button_label():
    if st.session_state["run"]:
        return "Stop Recording"
    return "Start Recording"


st.button(label=button_label(), on_click=button_state())

with st.expander("Model Options"):

    exp1_col1, exp1_col2 = st.columns(2)

    with exp1_col1:
        option1 = st.selectbox(
            label="Model Type",
            options=(
                "tiny",
                "base",
                "small",
                "medium",
                "large"
            ),
            index=1

        )
    with exp1_col2:
        option2 = st.selectbox(
            label="Language",
            options=("English")
        )

def transcribe(model, language):
    model = transcribe.load_model(model)

