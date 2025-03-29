import streamlit as st
from openai import OpenAI
from moviepy.video.io.VideoFileClip import VideoFileClip
import google.generativeai as genai
import tempfile
import os
import time

# --- Session state init ---
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "gemini_response" not in st.session_state:
    st.session_state.gemini_response = None

# --- Audio extraction ---
def extract_audio(video_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_temp:
        video = VideoFileClip(video_file.name)
        video.audio.write_audiofile(audio_temp.name, codec='libmp3lame')
        return audio_temp.name

# --- Whisper via OpenAI API ---
def transcribe_with_openai(audio_path, api_key):
    client = OpenAI(api_key=api_key)
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    return transcript

# --- Transcript upload handling ---
def load_transcript_text(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    return uploaded_file.read().decode("utf-8")

# --- Gemini Prompt ---
def send_to_gemini(api_key, transcript_text):
    genai.configure(api_key=api_key)
    system_prompt = """
You are a marketing specialist who just produced and hosted a webinar showcasing the Golf Genius product, Golf Shop.

Your task is to review the webinar transcript and extract the most valuable content. Specifically, look for:

* Direct testimonials about the product or feature
* Positive quotes highlighting benefits
* Stories that demonstrate real-world improvements

Whenever possible, include the speaker's full name, not just their first name, when referencing quotes or insights.

Focus on content that can be repurposed for marketing materials, social media, emails, or case studies.
"""
    model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
    response = model.generate_content(
        [system_prompt, transcript_text],
        generation_config={"temperature": 0.7}
    )
    return response.text

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="NuggetMiner", layout="wide")
st.title("🜚 NuggetMiner: Customer Testimonial Extractor")

st.sidebar.header("🛈 API Keys")
openai_api_key = st.sidebar.text_input(
    "🔑 OpenAI Whisper API Key", 
    type="password", 
    help="Used for transcribing audio via OpenAI's Whisper API"
)
gemini_api_key = st.sidebar.text_input(
    "🔑 Google Gemini API Key", 
    type="password", 
    help="Used for extracting insights with Gemini 2.5"
)

input_mode = st.sidebar.radio("Choose Input Type", ["Video File", "Transcript File"])

uploaded_file = st.file_uploader(
    "Upload your video or transcript",
    type=["mp4", "txt", "vtt", "srt"],
    accept_multiple_files=False
)

if uploaded_file:
    if (input_mode == "Video File" and (not openai_api_key or not gemini_api_key)) or \
       (input_mode == "Transcript File" and not gemini_api_key):
        st.warning("Please enter the required API key(s).")
    else:
        st.info("Processing input...")

        if st.session_state.transcript is None:
            if input_mode == "Video File" and uploaded_file.type.startswith("video"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                    temp_video.write(uploaded_file.read())
                    temp_video.flush()

                    with st.status("🜚 Processing Video...", expanded=True) as status:
                        st.write("🜚 Step 1: Extracting audio from video...")
                        audio_path = extract_audio(temp_video)

                        st.write("🜚 Step 2: Sending audio to OpenAI Whisper API...")
                        progress_bar = st.progress(0)

                        for i in range(60):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)

                        transcript = transcribe_with_openai(audio_path, openai_api_key)

                        for i in range(60, 100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)

                        st.session_state.transcript = transcript
                        st.toast("Transcription complete ✅")
                        os.remove(audio_path)

            elif input_mode == "Transcript File":
                st.session_state.transcript = load_transcript_text(uploaded_file)
                st.toast("Transcript loaded ✅")
            else:
                st.error("Invalid file type.")
                st.stop()

if st.session_state.transcript:
    transcript = st.session_state.transcript
    st.subheader("🜚 Transcript Preview")
    st.text_area("Transcript", transcript, height=300)

    if st.button("🜚 Mine for Nuggets"):
        st.session_state.gemini_response = None
        with st.spinner("We are mining for marketing gold..."):
            gemini_response = send_to_gemini(gemini_api_key, transcript)
            st.session_state.gemini_response = gemini_response
            st.toast("Gemini has spoken 💎")

if st.session_state.gemini_response:
    st.subheader("🜚 Nuggets Found")
    st.text_area("Marketing Nuggets", st.session_state.gemini_response, height=400)

    st.download_button(
        label="🜚 Download Nuggets",
        data=st.session_state.gemini_response,
        file_name="nugget_output.txt",
        mime="text/plain"
    )

# Option to Submit Another
if st.session_state.transcript or st.session_state.gemini_response:
    if st.button("🔁 Submit Another"):
        st.session_state.transcript = None
        st.session_state.gemini_response = None
        st.rerun()

