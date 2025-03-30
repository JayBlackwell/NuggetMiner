import streamlit as st
from openai import OpenAI
from moviepy.video.io.VideoFileClip import VideoFileClip
import google.generativeai as genai
import tempfile
import os
import time
import ffmpeg
from pydub import AudioSegment

# --- Session state init ---
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "gemini_response" not in st.session_state:
    st.session_state.gemini_response = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

# --- Audio extraction with compression ---
def extract_audio(video_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_temp:
        input_path = video_file.name
        output_path = audio_temp.name

        # Compress audio to mono, 16kHz, 64k bitrate
        ffmpeg.input(input_path).output(
            output_path,
            acodec='libmp3lame',
            audio_bitrate='64k',
            ac=1,
            ar=16000
        ).run(quiet=True, overwrite_output=True)

        return output_path

# --- Whisper via OpenAI API with chunking ---
def transcribe_with_openai(audio_path, api_key):
    max_size_mb = 25
    client = OpenAI(api_key=api_key)
    audio = AudioSegment.from_file(audio_path)
    chunk_length_ms = 5 * 60 * 1000  # 5 minutes
    chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

    all_segments = []
    current_offset = 0.0

    for idx, chunk in enumerate(chunks):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_chunk:
            chunk.export(temp_chunk.name, format="mp3")
            chunk_size_mb = os.path.getsize(temp_chunk.name) / (1024 * 1024)
            if chunk_size_mb > max_size_mb:
                st.error(f"❌ Chunk {idx+1} is {chunk_size_mb:.2f}MB and exceeds OpenAI's 25MB limit.")
                return None

            try:
                with open(temp_chunk.name, "rb") as audio_file:
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json"
                    )
                    for segment in response.segments:
                        segment["start"] += current_offset
                        segment["end"] += current_offset
                        all_segments.append(segment)
                current_offset += chunk.duration_seconds
            except Exception as e:
                st.error(f"❌ OpenAI Whisper API failed on chunk {idx+1}.")
                st.exception(e)
                return None
            finally:
                os.remove(temp_chunk.name)

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
                    st.write("🜚 Step 1: Extracting audio from video...")
                    audio_path = extract_audio(temp_video)

                    st.write("🜚 Step 2: Sending audio to OpenAI Whisper API...")
                    size_mb = os.path.getsize(audio_path) / 1024 / 1024
                    st.write(f"🔎 Compressed audio size: {size_mb:.2f}MB")
                    progress_bar = st.progress(0)

                    for i in range(80):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)

                    transcript = transcribe_with_openai(audio_path, openai_api_key)

                    for i in range(80, 100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)

                    if transcript:
                        st.session_state.transcript = transcript
                        st.toast("Transcription complete ✅")
                    os.remove(audio_path)

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

