import streamlit as st
import openai
from moviepy.editor import VideoFileClip, AudioFileClip
import google.generativeai as genai
import tempfile
import os
import time

# --- Session state init ---
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "gemini_response" not in st.session_state:
    st.session_state.gemini_response = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

# --- Audio extraction and chunking using moviepy ---
def extract_audio_chunks(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio_duration = audio.duration  # in seconds

    chunk_paths = []
    chunk_length = 300  # seconds (5 minutes)
    start = 0

    while start < audio_duration:
        end = min(start + chunk_length, audio_duration)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            audio.subclip(start, end).write_audiofile(
                temp_audio.name,
                codec="libmp3lame",
                bitrate="64k",
                ffmpeg_params=["-ac", "1", "-ar", "16000"]
            )
            chunk_paths.append((temp_audio.name, start))
        start += chunk_length

    return chunk_paths

# --- Whisper via OpenAI API with chunking ---
def transcribe_with_openai_chunks(chunk_paths, api_key):
    openai.api_key = api_key
    all_segments = []

    for idx, (chunk_path, offset) in enumerate(chunk_paths):
        try:
            with open(chunk_path, "rb") as audio_file:
                response = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )
                for segment in response.segments:
                    segment["start"] += offset
                    segment["end"] += offset
                    all_segments.append(segment)
        except Exception as e:
            st.error(f"❌ OpenAI Whisper API failed on chunk {idx+1}.")
            st.exception(e)
            return None
        finally:
            os.remove(chunk_path)

    lines = []
    for segment in all_segments:
        start = time.strftime('%H:%M:%S', time.gmtime(segment['start']))
        end = time.strftime('%H:%M:%S', time.gmtime(segment['end']))
        lines.append(f"[{start} --> {end}] {segment['text'].strip()}")
    return "\n".join(lines)

# --- Transcript upload handling ---
def load_transcript_text(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    lines = uploaded_file.read().decode("utf-8").splitlines()
    if ext in [".txt", ".vtt", ".srt"]:
        return "\n".join(
            line.strip() for line in lines if line.strip() and not line.strip().isdigit()
        )
    return ""

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

st.sidebar.header("🔐 API Keys")
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

if st.button("🔁 Submit Another"):
    for key in ["transcript", "gemini_response", "uploaded_filename"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

uploaded_file = st.file_uploader(
    "Upload your video or transcript",
    type=["mp4", "txt", "vtt", "srt"],
    accept_multiple_files=False
)

if uploaded_file:
    st.session_state.uploaded_filename = uploaded_file.name

if uploaded_file and gemini_api_key and (input_mode == "Transcript File" or openai_api_key):
    st.info("Processing input...")

    if st.session_state.transcript is None:
        if input_mode == "Video File" and uploaded_file.type.startswith("video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(uploaded_file.read())
                temp_video.flush()

                with st.status("🜚 Processing Video...", expanded=True) as status:
                    st.write("🜚 Step 1: Extracting audio chunks from video...")
                    chunk_paths = extract_audio_chunks(temp_video.name)

                    st.write("🜚 Step 2: Sending chunks to OpenAI Whisper API...")
                    progress_bar = st.progress(0)
                    transcript = transcribe_with_openai_chunks(chunk_paths, openai_api_key)
                    for i in range(80, 100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)

                    if transcript:
                        st.session_state.transcript = transcript
                        st.toast("Transcription complete ✅")

        elif input_mode == "Transcript File":
            st.session_state.transcript = load_transcript_text(uploaded_file)
            st.toast("Transcript loaded ✅")

        else:
            st.error("Invalid file type.")
            st.stop()

    transcript = st.session_state.transcript
    if transcript:
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
else:
    st.warning("Please upload a file and enter the required API keys.")
