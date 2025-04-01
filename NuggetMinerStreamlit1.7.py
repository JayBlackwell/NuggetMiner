import streamlit as st
from openai import OpenAI
import httpx # For custom client
import pkg_resources # For version checking (optional, can remove if httpx issue resolved)
from moviepy.editor import VideoFileClip
import google.generativeai as genai
import tempfile
import os
import time
import re # <-- Import regular expressions for VTT/SRT parsing

# --- Session state init ---
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "gemini_response" not in st.session_state:
    st.session_state.gemini_response = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

# --- Audio extraction and chunking using moviepy ---
def extract_audio_chunks(video_path):
    """Extracts audio from video, saves as chunks, returns paths."""
    chunk_paths = []
    video = None
    audio = None
    try:
        st.write("Opening video file...")
        video = VideoFileClip(video_path)
        audio = video.audio
        if audio is None:
            st.error("Could not extract audio from the video.")
            if video: video.close()
            return []
        audio_duration = audio.duration
        st.write(f"Audio duration: {audio_duration:.2f} seconds")

        chunk_length = 300
        start = 0
        idx = 1

        while start < audio_duration:
            end = min(start + chunk_length, audio_duration)
            st.write(f"Processing chunk {idx}: {start:.2f}s to {end:.2f}s")
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_chunk{idx}.mp3", prefix="audio_") as temp_audio:
                temp_audio_path = temp_audio.name
                try:
                    audio.subclip(start, end).write_audiofile(
                        temp_audio_path, codec="libmp3lame", bitrate="64k",
                        ffmpeg_params=["-ac", "1", "-ar", "16000"], logger=None
                    )
                    chunk_paths.append((temp_audio_path, start))
                    st.write(f"Saved chunk {idx} to {os.path.basename(temp_audio_path)}")
                except Exception as e:
                    st.error(f"Error writing audio chunk {idx}: {e}")
                    if os.path.exists(temp_audio_path): os.remove(temp_audio_path)
                    for path, _ in chunk_paths:
                        if os.path.exists(path): os.remove(path)
                    if audio: audio.close()
                    if video: video.close()
                    return []

            start += chunk_length
            idx += 1
        st.write("Audio extraction complete.")
    except Exception as e:
        st.error(f"Error during video processing or audio extraction: {e}")
        st.exception(e)
        for path, _ in chunk_paths:
            if os.path.exists(path):
                try: os.remove(path)
                except OSError: pass
        return []
    finally:
        if audio: audio.close()
        if video: video.close()
    return chunk_paths


# --- via OpenAI API with chunking (Includes httpx Debugging) ---
def transcribe_with_openai_chunks(chunk_paths, api_key):
    """Transcribes audio chunks using OpenAI Whisper API, explicitly disabling proxies."""
    custom_http_client = None
    client = None
    try:
        # --- Debugging httpx Initialization (Optional - can be removed if stable) ---
        # st.write("--- Starting httpx Client Setup ---")
        # ... (Optional version checks using pkg_resources) ...
        # Initialize httpx client directly
        # st.write("Initializing httpx.Client(proxies=None, timeout=60.0)...")
        custom_http_client = httpx.Client(proxies=None, timeout=60.0)
        # st.write("Custom httpx client created.")
        # st.write("--- End httpx Client Setup ---")
    except Exception as e:
        st.error(f"❌ Failed during httpx client creation.")
        st.exception(e)
        # st.write("--- End httpx Client Setup ---")
        # Cleanup chunks if client creation fails
        for chunk_path, _ in chunk_paths:
            if os.path.exists(chunk_path):
                try: os.remove(chunk_path)
                except OSError: pass
        return None

    # --- Proceed with OpenAI client and transcription ---
    try:
        try:
            client = OpenAI(api_key=api_key, http_client=custom_http_client)
            # st.write("OpenAI client initialized.")
        except Exception as e_openai:
            st.error("❌ Failed to initialize OpenAI client.")
            st.exception(e_openai)
            # Cleanup chunks
            for chunk_path, _ in chunk_paths:
                 if os.path.exists(chunk_path):
                    try: os.remove(chunk_path)
                    except OSError: pass
            return None # Exit, finally will close httpx client

        # --- Main Transcription Loop ---
        all_formatted_lines = []
        total_chunks = len(chunk_paths)
        if total_chunks == 0:
            st.warning("No audio chunks found for transcription.")
            return ""

        progress_bar = st.progress(0, text="Initializing transcription...")
        processed_chunks = 0
        for idx, (chunk_path, offset) in enumerate(chunk_paths):
            # ... (rest of the transcription loop as in previous version) ...
             current_chunk_number = idx + 1
             progress_text = f"Transcribing chunk {current_chunk_number}/{total_chunks}..."
             progress_bar.progress(processed_chunks / total_chunks, text=progress_text)
             # st.write(f"Sending chunk {current_chunk_number} ({os.path.basename(chunk_path)}) to Whisper...") # Less verbose

             try:
                 if not os.path.exists(chunk_path):
                     st.error(f"Chunk file {chunk_path} not found. Skipping.")
                     continue

                 with open(chunk_path, "rb") as audio_file:
                     response = client.audio.transcriptions.create(
                         model="whisper-1", file=audio_file,
                         response_format="verbose_json", timestamp_granularities=["segment"]
                     )

                 segments = getattr(response, 'segments', [])
                 if not segments:
                      st.warning(f"No segments returned for chunk {current_chunk_number}.")

                 for segment in segments:
                     segment_start_abs = segment.get('start', 0) + offset
                     segment_end_abs = segment.get('end', 0) + offset
                     segment_text = segment.get('text', '').strip()
                     start_str = time.strftime('%H:%M:%S', time.gmtime(segment_start_abs))
                     end_str = time.strftime('%H:%M:%S', time.gmtime(segment_end_abs))
                     # Ensure text exists before adding line
                     if segment_text:
                        all_formatted_lines.append(f"[{start_str} --> {end_str}] {segment_text}")

                 # st.write(f"Chunk {current_chunk_number} transcribed successfully.") # Less verbose
                 processed_chunks += 1
                 # Update progress bar here to ensure it reflects actual progress even if segment processing is quick
                 progress_bar.progress(processed_chunks / total_chunks, text=progress_text)

             except Exception as e:
                 st.error(f"❌ OpenAI Whisper API failed on chunk {current_chunk_number}.")
                 st.exception(e)
                 progress_bar.empty()
                 for i in range(idx, len(chunk_paths)): # Cleanup remaining
                     path_to_remove = chunk_paths[i][0]
                     if os.path.exists(path_to_remove):
                         try: os.remove(path_to_remove)
                         except OSError: pass
                 return None # Exit loop
             finally:
                 # Ensure *this* chunk file is always removed
                 if os.path.exists(chunk_path):
                     try: os.remove(chunk_path)
                     except OSError as rm_error:
                          st.warning(f"Could not remove temp chunk file {chunk_path}: {rm_error}")

        progress_bar.progress(1.0, text="Transcription complete!")
        time.sleep(1.5)
        progress_bar.empty()
        return "\n".join(all_formatted_lines)

    finally:
        # Ensure custom httpx client is always closed
        if custom_http_client:
            try:
                custom_http_client.close()
                # st.write("Custom httpx client closed.") # Less verbose
            except Exception as e_close:
                st.warning(f"Error closing httpx client: {e_close}")


# --- Transcript upload handling (Revised for VTT/SRT) ---
def load_transcript_text(uploaded_file):
    """Reads transcript from VTT/SRT or TXT files."""
    try:
        file_name = uploaded_file.name
        file_ext = os.path.splitext(file_name)[1].lower()
        content_bytes = uploaded_file.read()
        encoding = 'utf-8-sig' if content_bytes.startswith(b'\xef\xbb\xbf') else 'utf-8'
        content = content_bytes.decode(encoding)
        lines = content.splitlines()
        output_lines = []

        if file_ext in ['.vtt', '.srt']:
            # st.write(f"Parsing {file_ext} file format...") # Less verbose
            i = 0
            while i < len(lines):
                # ... (VTT/SRT parsing logic as in previous version) ...
                 line = lines[i].strip()
                 if not line or line.upper() == "WEBVTT" or line.isdigit() or line.upper().startswith("NOTE"):
                     i += 1
                     continue
                 match = re.search(r'(\d{2}:\d{2}:\d{2})\.?\d*\s+-->\s+(\d{2}:\d{2}:\d{2})\.?\d*', line)
                 if match:
                     start_time, end_time = match.group(1), match.group(2)
                     text_parts = []
                     i += 1
                     while i < len(lines) and lines[i].strip() and '-->' not in lines[i] and not lines[i].strip().isdigit():
                         text_parts.append(lines[i].strip())
                         i += 1
                     full_text = " ".join(text_parts).strip() # Added strip here
                     if full_text: # Check if text exists after join/strip
                          output_lines.append(f"[{start_time} --> {end_time}] {full_text}")
                     continue
                 i += 1
            # st.write(f"Finished parsing {file_ext}. Found {len(output_lines)} segments.") # Less verbose
        elif file_ext == '.txt':
            # st.write("Processing plain .txt file...") # Less verbose
            output_lines = [line.strip() for line in lines if line.strip()]
        else:
            st.warning(f"Unsupported file extension for transcript loading: {file_ext}")
            return ""
        return "\n".join(output_lines)
    except Exception as e:
        st.error(f"Error reading or processing transcript file '{uploaded_file.name}': {e}")
        st.exception(e)
        return None


# --- Gemini Prompt ---
def send_to_gemini(api_key, transcript_text):
    """Sends transcript to Gemini API and returns the response text."""
    try:
        genai.configure(api_key=api_key)
        # --- MODIFIED SYSTEM PROMPT ---
        system_prompt = """
You are a marketing specialist reviewing a webinar transcript (Golf Genius Golf Shop product). Your task is to extract valuable marketing content:

*   **Direct Testimonials:** Quotes from named speakers praising the product/features.
*   **Benefit Highlights:** Positive quotes showing specific advantages or results.
*   **Success Stories:** Examples of real-world improvements or problem-solving.
*   **Positive Feature Mentions:** Key features discussed positively or in detail.

**Formatting Requirements:**
*   Include speaker's full name if possible (from transcript context). Otherwise, use generic attribution (e.g., "A speaker...").
*   **Prefix each nugget with the approximate start timestamp** from the transcript line it came from (e.g., `[Around 00:15:10]`).
*   Use bullet points for each extracted nugget.
*   Focus *only* on repurposable marketing content. Ignore introductions, filler, off-topic chat.

**Exclusions:**
*   Do NOT include any quotes, testimonials, or insights that are attributed to Bart Rickard or Brian Morrison.

Example Output:
*   [Around 00:15:10] Jane Smith: "The Golf Shop product dramatically reduced our inventory reconciliation time."
*   [Around 00:28:45] A speaker highlighted how the special order tracking saved them hours each week.
"""
        # --- END OF MODIFIED SYSTEM PROMPT ---

        model = genai.GenerativeModel(
             model_name="gemini-1.5-flash", # Changed model name back for stability/cost? Revert if needed.
             system_instruction=system_prompt
        )
        response = model.generate_content(
            transcript_text, generation_config={"temperature": 0.7}
        )
        # ... (Gemini response handling as in previous version) ...
        if response and hasattr(response, 'text'):
             return response.text
        elif response and hasattr(response, 'prompt_feedback'):
              safety_ratings_str = "N/A"
              try:
                  if response.candidates and response.candidates[0].safety_ratings:
                      safety_ratings_str = str(response.candidates[0].safety_ratings)
              except Exception: pass
              st.warning(f"Gemini content generation potentially blocked: {response.prompt_feedback}. Ratings: {safety_ratings_str}")
              return f"Error: Content generation blocked by safety settings. Feedback: {response.prompt_feedback}. Ratings: {safety_ratings_str}"
        else:
              st.error("Received an unexpected response structure from Gemini.")
              return "Error: Could not process Gemini response."
    except Exception as e:
        st.error("❌ Google Gemini API failed.")
        st.exception(e)
        return None

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="NuggetMiner", layout="wide")
st.title("🜚 NuggetMiner: Customer Testimonial Extractor")
st.caption("Upload a webinar video (MP4) or transcript (TXT, VTT, SRT) to extract marketing gold!")

# Sidebar for controls
with st.sidebar:
    st.header("🔐 API Keys")
    openai_api_key = st.text_input(
        "🔑 OpenAI API Key", type="password", help="Required for transcribing video files."
    )
    gemini_api_key = st.text_input(
        "🔑 Google Gemini API Key", type="password", help="Required for analyzing the transcript."
    )

    st.header("⚙️ Input Settings")
    input_mode = st.radio(
        "Choose Input Type:", ["Video File", "Transcript File"],
        key="input_mode_radio", # Key to identify this widget's state
        help="Select input type."
    )

    st.divider()

    # --- Start Over / Clear Button (FIXED) ---
    if st.button("🔁 Start Over / Clear"):
        keys_to_clear = ["transcript", "gemini_response", "uploaded_filename"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        # Rerun the script - This implicitly clears the file uploader state visually
        st.rerun()

    st.divider()
    st.info("NuggetMiner v1.7 (Exclude Speakers)") # Updated version

# Main area for file upload and results
st.header("📤 Upload Your File")
uploaded_file = st.file_uploader(
    f"Upload your {st.session_state.get('input_mode_radio', 'Video File').lower()}", # Use persisted state for label
    type=["mp4", "txt", "vtt", "srt"],
    accept_multiple_files=False,
    key="file_uploader" # Assign key
)

# --- Processing Logic ---
if uploaded_file is not None:
    # Check if filename state needs updating (indicates a new upload action)
    if st.session_state.get("uploaded_filename") != uploaded_file.name:
        keys_to_reset = ["transcript", "gemini_response"]
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.uploaded_filename = uploaded_file.name
        st.info(f"Processing new file: '{st.session_state.uploaded_filename}'.")

    # Determine prerequisites based on persisted input mode state
    gemini_ready = bool(gemini_api_key)
    whisper_ready = bool(openai_api_key)
    current_input_mode = st.session_state.get("input_mode_radio", "Video File")

    can_process_video = current_input_mode == "Video File" and whisper_ready and gemini_ready
    can_process_transcript = current_input_mode == "Transcript File" and gemini_ready

    # Proceed if prerequisites met
    if (can_process_video or can_process_transcript):
        # --- Step 1: Get Transcript (if not already processed) ---
        if st.session_state.get("transcript") is None:
            st.write("---") # Separator only when processing starts
            temp_video_path = None
            try:
                if current_input_mode == "Video File":
                    st.info("Processing video file...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                         temp_video.write(uploaded_file.getvalue())
                         temp_video_path = temp_video.name
                         # st.write(f"Temporary video file created: {temp_video_path}") # Less verbose

                    with st.status("🜚 Processing Video...", expanded=True) as status:
                         status.update(label="➡️ Step 1: Extracting audio chunks...", state="running")
                         chunk_paths = extract_audio_chunks(temp_video_path)
                         if not chunk_paths:
                             st.error("Audio extraction failed.")
                             status.update(label="Audio Extraction Failed ❌", state="error")
                             st.stop()

                         status.update(label=f"➡️ Step 2: Transcribing {len(chunk_paths)} audio chunk(s)...", state="running")
                         transcript_text = transcribe_with_openai_chunks(chunk_paths, openai_api_key)
                         if transcript_text is not None:
                             st.session_state.transcript = transcript_text
                             status.update(label="✅ Video Processed & Transcribed!", state="complete")
                             st.toast("Transcription complete ✅")
                         else:
                             status.update(label="Transcription Failed ❌", state="error")
                             st.stop()

                elif current_input_mode == "Transcript File":
                     st.info(f"Loading transcript file: {uploaded_file.name}")
                     with st.spinner("Reading transcript..."):
                         transcript_text = load_transcript_text(uploaded_file)
                         if transcript_text is not None:
                              st.session_state.transcript = transcript_text
                              st.toast("Transcript loaded ✅")
                         else:
                              # Error message shown in function
                              st.stop()
            finally:
                # Cleanup temp video file
                if temp_video_path and os.path.exists(temp_video_path):
                    try:
                        os.remove(temp_video_path)
                        # st.write(f"Cleaned up temporary video file: {temp_video_path}") # Less verbose
                    except OSError as e:
                        st.warning(f"Could not remove temp video file {temp_video_path}: {e}")

        # --- Step 2: Display Transcript and Process with Gemini ---
        transcript = st.session_state.get("transcript")
        if transcript is not None and transcript.strip():
            st.subheader("🜚 Transcript Preview")
            st.text_area("Transcript Text", transcript, height=300, key="transcript_display")
            st.write("---")
            if gemini_ready:
                 # Add unique key to avoid button state issues across reruns if needed,
                 # but "mine_button" might be sufficient if only shown once per transcript load.
                 if st.button("🜚 Mine for Nuggets", key="mine_button"):
                     st.session_state.gemini_response = None # Reset before call
                     with st.spinner("🜚 Asking Gemini to find the marketing gold..."):
                         gemini_response_text = send_to_gemini(gemini_api_key, transcript)
                         st.session_state.gemini_response = gemini_response_text
                         if gemini_response_text and not gemini_response_text.startswith("Error:"):
                             st.toast("Gemini analysis complete! ✨")
                         else:
                              st.toast("Gemini analysis finished (check output).", icon="⚠️")
            else:
                  st.warning("☝️ Enter your Google Gemini API key in the sidebar to analyze this transcript.")

            # Display results if they exist
            if st.session_state.get("gemini_response"):
                 is_error_response = st.session_state.gemini_response.startswith("Error:")
                 st.subheader("✨ Nuggets Found!" if not is_error_response else "⚠️ Gemini Response")
                 st.text_area("Marketing Nuggets" if not is_error_response else "Gemini Output",
                              st.session_state.gemini_response, height=400, key="nuggets_display")
                 st.download_button(
                     label="💾 Download Output", data=st.session_state.gemini_response.encode('utf-8'),
                     file_name=f"nuggetminer_output_{os.path.splitext(st.session_state.uploaded_filename)[0]}.txt",
                     mime="text/plain"
                 )
        elif st.session_state.get("transcript") is not None: # Transcript processed but empty
             st.warning("The generated or loaded transcript appears to be empty.")
        # If transcript is None, it means processing failed or hasn't started, so no message needed here.

    else:
        # Prerequisites not met
        st.write("---")
        st.warning(f"☝️ Please ensure prerequisites are met for the selected '{current_input_mode}' mode:")
        if current_input_mode == "Video File":
             if not whisper_ready: st.error("   - OpenAI API key is missing.")
             if not gemini_ready: st.error("   - Google Gemini API key is missing.")
        elif current_input_mode == "Transcript File":
              if not gemini_ready: st.error("   - Google Gemini API key is missing.")

elif not uploaded_file:
    # Initial state - no file uploaded yet
    st.info("👈 Upload a file and provide API keys in the sidebar to get started.")

# --- End of Script ---
