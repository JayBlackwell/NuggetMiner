import streamlit as st
import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
import google.generativeai as genai
import tempfile
import os
import time

# --- Initialize session state ---
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "gemini_response" not in st.session_state:
    st.session_state.gemini_response = None

# --- Utilities ---
def extract_audio(video_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_temp:
        video = VideoFileClip(video_file.name)
        video.audio.write_audiofile(audio_temp.name, codec='pcm_s16le')
        return audio_temp.name

def format_timestamp(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{mins:02}:{secs:02}.{millis:03}"

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, verbose=False)

    lines = []
    for segment in result["segments"]:
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()
        lines.append(f"[{start} --> {end}] {text}")
    
    return "\n".join(lines)

def load_transcript_text(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    lines = uploaded_file.read().decode("utf-8").splitlines()
    if ext in [".txt", ".vtt", ".srt"]:
        return "\n".join(
            line.strip() for line in lines if line.strip() and not line.strip().isdigit()
        )
    return ""

def send_to_gemini(api_key, transcript_text):
    genai.configure(api_key=api_key)
    system_prompt = """
You are a marketing specialist who just produced and hosted a webinar showcasing the Golf Genius product, Golf Shop.

Your task is to review the webinar transcript and extract the most valuable content. Specifically, look for:

* Direct testimonials about the product or feature
* Positive quotes highlighting benefits
* Stories that demonstrate real-world improvements

Whenever possible, include the speaker's **full name**, not just their first name, when referencing quotes or insights.

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

st.sidebar.header("🜚 Gemini Config")
api_key = st.sidebar.text_input("Google Gemini API Key", type="password")
input_mode = st.sidebar.radio("Choose Input Type", ["Video File", "Transcript File"])

uploaded_file = st.file_uploader(
    "Upload your video or transcript",
    type=["mp4", "txt", "vtt", "srt"],
    accept_multiple_files=False
)

if uploaded_file and api_key:
    st.info("Processing input...")

    # Load transcript (if not already done)
    if st.session_state.transcript is None:
        if input_mode == "Video File" and uploaded_file.type.startswith("video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(uploaded_file.read())
                temp_video.flush()

                with st.status("🔍 Processing Video...", expanded=True) as status:
                    st.write("📁 Step 1: Extracting audio from video...")
                    audio_path = extract_audio(temp_video)

                    st.write("📝 Step 2: Transcribing with Whisper model...")
                    progress_bar = st.progress(0)

                    for i in range(60):
                        time.sleep(0.03)
                        progress_bar.progress(i + 1)

                    transcript = transcribe_audio(audio_path)

                    for i in range(60, 100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)

                    st.write("✅ Transcription complete.")
                    status.update(label="✅ Whisper Processing Complete!", state="complete")
                    st.toast("Transcription ready to review ✅")

                    st.session_state.transcript = transcript
                    os.remove(audio_path)

        elif input_mode == "Transcript File":
            st.session_state.transcript = load_transcript_text(uploaded_file)
            st.toast("Transcript loaded ✅")

        else:
            st.error("Invalid file type.")
            st.stop()

    # Display transcript
    transcript = st.session_state.transcript
    st.subheader("🜚 Transcript Preview")
    st.text_area("Transcript", transcript, height=300)

    # Generate with Gemini
    if st.button("🜚 Mine transcript for Nuggets"):
        st.session_state.gemini_response = None  # Reset previous output
        with st.spinner("Gemini is mining for marketing gold..."):
            gemini_response = send_to_gemini(api_key, transcript)
            st.session_state.gemini_response = gemini_response
            st.toast("Gemini has spoken 💎")

    if st.session_state.gemini_response:
        st.subheader("🜚 Gold Strikes")
        st.text_area("Marketing Nuggets", st.session_state.gemini_response, height=400)

        st.download_button(
            label="🜚 Download Nuggets",
            data=st.session_state.gemini_response,
            file_name="nugget_output.txt",
            mime="text/plain"
        )

else:
    st.warning("Please upload a file and enter your API key.")

