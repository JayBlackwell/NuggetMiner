import streamlit as st
from openai import OpenAI
import httpx # <-- Import httpx
from moviepy.editor import VideoFileClip # Removed AudioFileClip if not used directly
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
    """Extracts audio from video, saves as chunks, returns paths."""
    chunk_paths = []
    try:
        st.write("Opening video file...")
        video = VideoFileClip(video_path)
        audio = video.audio
        if audio is None:
            st.error("Could not extract audio from the video.")
            if video: video.close() # Ensure video object is closed
            return []
        audio_duration = audio.duration  # in seconds
        st.write(f"Audio duration: {audio_duration:.2f} seconds")

        chunk_length = 300  # seconds (5 minutes) - adjust if needed
        start = 0
        idx = 1

        while start < audio_duration:
            end = min(start + chunk_length, audio_duration)
            st.write(f"Processing chunk {idx}: {start:.2f}s to {end:.2f}s")
            # Use a unique name to avoid potential conflicts in high concurrency
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_chunk{idx}.mp3", prefix="audio_") as temp_audio:
                temp_audio_path = temp_audio.name # Get path before potential close/error
                try:
                    # Write audio chunk
                    audio.subclip(start, end).write_audiofile(
                        temp_audio_path,
                        codec="libmp3lame",
                        bitrate="64k",
                        ffmpeg_params=["-ac", "1", "-ar", "16000"], # Mono, 16kHz sample rate
                        logger=None # Suppress excessive moviepy logging if desired
                    )
                    chunk_paths.append((temp_audio_path, start))
                    st.write(f"Saved chunk {idx} to {os.path.basename(temp_audio_path)}")
                except Exception as e:
                    st.error(f"Error writing audio chunk {idx}: {e}")
                    # Attempt to clean up the failed chunk file
                    if os.path.exists(temp_audio_path):
                         os.remove(temp_audio_path)
                    # Clean up previously successful chunks
                    for path, _ in chunk_paths:
                        if os.path.exists(path): os.remove(path)
                    # Close moviepy objects before returning
                    if audio: audio.close()
                    if video: video.close()
                    return [] # Return empty list indicating failure

            start += chunk_length
            idx += 1

        # Close video and audio objects to release resources
        if audio: audio.close()
        if video: video.close()
        st.write("Audio extraction complete.")

    except Exception as e:
        st.error(f"Error during video processing or audio extraction: {e}")
        # Clean up any chunks that might have been created before the error
        for path, _ in chunk_paths:
            if os.path.exists(path):
                try: os.remove(path)
                except OSError: pass # Ignore errors during cleanup
        # Ensure moviepy objects are closed on general error
        if 'audio' in locals() and audio: audio.close()
        if 'video' in locals() and video: video.close()
        return [] # Return empty list

    return chunk_paths


# --- Whisper via OpenAI API with chunking (Revised with Custom httpx Client) ---
def transcribe_with_openai_chunks(chunk_paths, api_key):
    """Transcribes audio chunks using OpenAI Whisper API, explicitly disabling proxies."""

    # Initialize custom_http_client to None
    custom_http_client = None
    client = None

    try:
        # Create a custom httpx client that ignores environment proxies
        try:
            custom_http_client = httpx.Client(proxies=None, timeout=60.0) # Explicitly disable proxies, set timeout
            st.write("Custom httpx client created (proxies disabled).")
        except Exception as e:
            st.error("❌ Failed to create custom httpx client.")
            st.exception(e)
            # Clean up chunks before returning
            for chunk_path, _ in chunk_paths:
                try:
                    if os.path.exists(chunk_path): os.remove(chunk_path)
                except OSError: pass
            return None

        # Instantiate the OpenAI client using the custom httpx client
        try:
            client = OpenAI(
                api_key=api_key,
                http_client=custom_http_client # <-- Pass the custom client here
            )
            st.write("OpenAI client initialized with custom httpx client.")
        except Exception as e:
            st.error("❌ Failed to initialize OpenAI client even with custom httpx client.")
            st.exception(e)
            # Clean up chunks before returning
            for chunk_path, _ in chunk_paths:
                try:
                    if os.path.exists(chunk_path): os.remove(chunk_path)
                except OSError: pass
            return None # Return None as OpenAI client init failed

        # --- Main Transcription Loop ---
        all_formatted_lines = []
        total_chunks = len(chunk_paths)
        if total_chunks == 0:
            st.warning("No audio chunks provided for transcription.")
            return "" # Return empty string if no chunks

        progress_bar = st.progress(0, text="Initializing transcription...")
        processed_chunks = 0

        for idx, (chunk_path, offset) in enumerate(chunk_paths):
            current_chunk_number = idx + 1
            progress_text = f"Transcribing chunk {current_chunk_number}/{total_chunks}..."
            # Update progress at the start of processing each chunk
            progress_bar.progress(processed_chunks / total_chunks, text=progress_text)
            st.write(f"Sending chunk {current_chunk_number} ({os.path.basename(chunk_path)}) to Whisper...")

            try:
                if not os.path.exists(chunk_path):
                    st.error(f"Chunk file {chunk_path} not found. Skipping.")
                    # Consider if skipping should count towards progress or not.
                    # Here, we'll let the progress update naturally on the next successful chunk.
                    continue

                with open(chunk_path, "rb") as audio_file:
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json",
                        timestamp_granularities=["segment"]
                    )

                segments = getattr(response, 'segments', [])
                if not segments:
                     st.warning(f"No segments returned for chunk {current_chunk_number}.")

                for segment in segments:
                    segment_start = segment.get('start', 0) + offset
                    segment_end = segment.get('end', 0) + offset
                    segment_text = segment.get('text', '').strip()
                    start_str = time.strftime('%H:%M:%S', time.gmtime(segment_start))
                    end_str = time.strftime('%H:%M:%S', time.gmtime(segment_end))
                    all_formatted_lines.append(f"[{start_str} --> {end_str}] {segment_text}")
                st.write(f"Chunk {current_chunk_number} transcribed successfully.")

                # Increment processed count ONLY after successful processing
                processed_chunks += 1
                # Update progress bar again *after* successful processing
                progress_bar.progress(processed_chunks / total_chunks, text=progress_text)

            except Exception as e:
                st.error(f"❌ OpenAI Whisper API failed on chunk {current_chunk_number}.")
                st.exception(e)
                progress_bar.empty() # Clear progress bar on error
                # Clean up remaining unprocessed chunks
                for i in range(idx, len(chunk_paths)):
                    path_to_remove = chunk_paths[i][0]
                    try:
                        if os.path.exists(path_to_remove): os.remove(path_to_remove)
                    except OSError: pass
                return None # Stop processing
            finally:
                # Ensure *this* chunk file is always removed after processing attempt
                try:
                    if os.path.exists(chunk_path): os.remove(chunk_path)
                except OSError as rm_error:
                     st.warning(f"Could not remove temp chunk file {chunk_path}: {rm_error}")

        # Final progress update after loop completes successfully
        progress_bar.progress(1.0, text="Transcription complete!")
        time.sleep(1.5) # Keep message visible
        progress_bar.empty() # Clear the progress bar

        return "\n".join(all_formatted_lines)

    finally:
        # IMPORTANT: Close the custom httpx client if it was created
        if custom_http_client:
            custom_http_client.close()
            st.write("Custom httpx client closed.")


# --- Transcript upload handling ---
def load_transcript_text(uploaded_file):
    """Reads transcript from uploaded file, cleaning basic formatting."""
    try:
        content_bytes = uploaded_file.read()
        encoding = 'utf-8-sig' if content_bytes.startswith(b'\xef\xbb\xbf') else 'utf-8'
        content = content_bytes.decode(encoding)
        lines = content.splitlines()
        cleaned_lines = [
            line.strip() for line in lines
            if line.strip() and not line.strip().isdigit() and '-->' not in line
        ]
        return "\n".join(cleaned_lines)
    except Exception as e:
        st.error(f"Error reading or processing transcript file: {e}")
        return None


# --- Gemini Prompt ---
def send_to_gemini(api_key, transcript_text):
    """Sends transcript to Gemini API and returns the response text."""
    try:
        genai.configure(api_key=api_key)
        system_prompt = """
You are a marketing specialist who just produced and hosted a webinar showcasing the Golf Genius product, Golf Shop.

Your task is to review the webinar transcript and extract the most valuable content. Specifically, look for:

* Direct testimonials about the product or feature from named speakers.
* Positive quotes highlighting specific benefits or results.
* Stories that demonstrate real-world improvements or problem-solving using the product.
* Key feature mentions that received positive reactions or detailed explanations.

Whenever possible, include the speaker's full name (if mentioned near the quote in the transcript) when referencing quotes or insights. If the name isn't clear, attribute it generically (e.g., "A speaker mentioned...").

Focus *only* on content that can be directly repurposed for marketing materials, social media posts, email snippets, or case study elements. Avoid generic introductions, filler words, or off-topic discussions. Format the output clearly, perhaps using bullet points for each extracted nugget.
"""
        model = genai.GenerativeModel(
             model_name="gemini-1.5-flash", # Or try "gemini-1.5-pro" / other available models
             system_instruction=system_prompt
             )
        response = model.generate_content(
            transcript_text,
            generation_config={"temperature": 0.7}
        )

        if response and hasattr(response, 'text'):
            return response.text
        elif response and hasattr(response, 'prompt_feedback'):
             st.warning(f"Gemini content generation might be blocked: {response.prompt_feedback}")
             # Try to access candidate information if available
             safety_ratings = None
             if response.candidates and response.candidates[0].safety_ratings:
                 safety_ratings = response.candidates[0].safety_ratings
             return f"Error: Content generation blocked by safety settings. Feedback: {response.prompt_feedback}. Ratings: {safety_ratings}"
        else:
             st.error("Received an unexpected response structure from Gemini.")
             # Log the actual response if possible (be careful with sensitive data)
             # st.write("Gemini Response:", response)
             return "Error: Could not process Gemini response."

    except Exception as e:
        st.error("❌ Google Gemini API failed.")
        st.exception(e)
        return None

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="NuggetMiner", layout="wide")
st.title("💎 NuggetMiner: Customer Testimonial Extractor")
st.caption("Upload a webinar video (MP4) or transcript (TXT, VTT, SRT) to extract marketing gold!")

# Place API keys and input selection in the sidebar
with st.sidebar:
    st.header("🔐 API Keys")
    openai_api_key = st.text_input(
        "🔑 OpenAI Whisper API Key",
        type="password",
        help="Required for transcribing video files via OpenAI Whisper."
    )
    gemini_api_key = st.text_input(
        "🔑 Google Gemini API Key",
        type="password",
        help="Required for analyzing the transcript with Google Gemini."
    )

    st.header("⚙️ Input Settings")
    input_mode = st.radio(
        "Choose Input Type:",
        ["Video File", "Transcript File"],
        key="input_mode_radio",
        help="Select whether you are uploading a video to transcribe or an existing transcript file."
    )

    st.divider()

    if st.button("🔁 Start Over / Clear"):
        keys_to_clear = ["transcript", "gemini_response", "uploaded_filename", "input_mode_radio"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


# Main area for file upload and results
st.header("📤 Upload Your File")
uploaded_file = st.file_uploader(
    f"Upload your {input_mode.lower()}", # Dynamic label
    type=["mp4", "txt", "vtt", "srt"],
    accept_multiple_files=False,
    key="file_uploader"
)

# --- Processing Logic ---
if uploaded_file is not None:
    # Store filename only if it hasn't been processed yet or if it's a new file
    # This check helps prevent reprocessing if the user interacts with UI elements
    # causing a rerun without a new file upload.
    if st.session_state.get("uploaded_filename") != uploaded_file.name:
        # Clear previous results when a new file is uploaded
        keys_to_clear = ["transcript", "gemini_response"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.uploaded_filename = uploaded_file.name
        st.info(f"New file uploaded: '{st.session_state.uploaded_filename}'. Previous results cleared.")

    # Check if required keys are present based on input mode
    gemini_ready = bool(gemini_api_key)
    whisper_ready = bool(openai_api_key)
    can_process_video = input_mode == "Video File" and whisper_ready and gemini_ready
    can_process_transcript = input_mode == "Transcript File" and gemini_ready

    if (can_process_video or can_process_transcript):

        # --- Step 1: Get Transcript (only if not already in session state) ---
        if st.session_state.get("transcript") is None:
            st.write("---") # Separator
            temp_video_path = None # Initialize path variable

            try: # Wrap the whole transcript generation process
                if input_mode == "Video File" and uploaded_file.type.startswith("video"):
                    st.info("Processing video file...")
                    # Create temp file for video upload
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                        temp_video.write(uploaded_file.getvalue())
                        temp_video_path = temp_video.name
                        st.write(f"Temporary video file created at: {temp_video_path}")

                    with st.status("🜚 Processing Video...", expanded=True) as status:
                        status.update(label="➡️ Step 1: Extracting audio chunks...", state="running")
                        chunk_paths = extract_audio_chunks(temp_video_path)

                        if not chunk_paths:
                            st.error("Audio extraction failed. Cannot proceed.")
                            status.update(label="Audio Extraction Failed ❌", state="error")
                            # Stop only if extraction fails. Cleanup handled below in finally.
                            st.stop()

                        status.update(label=f"➡️ Step 2: Transcribing {len(chunk_paths)} audio chunk(s)...", state="running")
                        # Transcription with integrated progress happens here
                        transcript_text = transcribe_with_openai_chunks(chunk_paths, openai_api_key)
                        # Note: Chunk files are deleted inside transcribe_with_openai_chunks

                        if transcript_text is not None:
                            st.session_state.transcript = transcript_text
                            status.update(label="✅ Video Processed & Transcribed!", state="complete")
                            st.toast("Transcription complete ✅")
                        else:
                            st.error("Transcription failed.")
                            status.update(label="Transcription Failed ❌", state="error")
                            # Stop if transcription fails. Cleanup handled below.
                            st.stop()

                elif input_mode == "Transcript File":
                     st.info("Loading transcript file...")
                     with st.spinner("Reading transcript..."):
                        transcript_text = load_transcript_text(uploaded_file)
                        if transcript_text is not None:
                             st.session_state.transcript = transcript_text
                             st.toast("Transcript loaded ✅")
                        else:
                             st.error("Failed to load transcript from file.")
                             st.stop()
                # No 'else' needed here as file type check handles other cases

            finally:
                 # Ensure temporary video file is always cleaned up if it was created
                 if temp_video_path and os.path.exists(temp_video_path):
                    try:
                        os.remove(temp_video_path)
                        st.write(f"Cleaned up temporary video file: {temp_video_path}")
                    except OSError as e:
                        st.warning(f"Could not remove temporary video file {temp_video_path}: {e}")


        # --- Step 2: Display Transcript and Process with Gemini ---
        transcript = st.session_state.get("transcript") # Use .get for safety

        if transcript is not None and transcript.strip(): # Check if transcript exists and is not just whitespace
            st.subheader("📜 Transcript Preview")
            st.text_area("Transcript Text", transcript, height=300, key="transcript_display")

            st.write("---") # Separator

            # Show Mine button only if Gemini key is provided
            if gemini_ready:
                if st.button("💎 Mine for Nuggets", key="mine_button"):
                    st.session_state.gemini_response = None # Reset previous response
                    with st.spinner("🧠 Asking Gemini to find the marketing gold..."):
                        gemini_response_text = send_to_gemini(gemini_api_key, transcript)
                        # Store response regardless of success/error message for display
                        st.session_state.gemini_response = gemini_response_text
                        if gemini_response_text and not gemini_response_text.startswith("Error:"):
                            st.toast("Gemini analysis complete! ✨")
                        else:
                             st.toast("Gemini analysis finished (potential issues found).", icon="⚠️")
            elif transcript: # Transcript exists but no Gemini key
                 st.warning("☝️ Enter your Google Gemini API key in the sidebar to analyze this transcript.")


            # Display Gemini results if available in session state
            if st.session_state.get("gemini_response"):
                st.subheader("✨ Nuggets Found!" if not st.session_state.gemini_response.startswith("Error:") else "⚠️ Gemini Response")
                st.text_area("Marketing Nuggets" if not st.session_state.gemini_response.startswith("Error:") else "Gemini Output",
                             st.session_state.gemini_response,
                             height=400,
                             key="nuggets_display")

                # Allow download even if it's an error message (for debugging)
                st.download_button(
                    label="💾 Download Output",
                    data=st.session_state.gemini_response.encode('utf-8'), # Encode to bytes
                    file_name=f"nuggetminer_output_{os.path.splitext(st.session_state.uploaded_filename)[0]}.txt",
                    mime="text/plain"
                )

        elif st.session_state.get("transcript") is not None: # Transcript exists but is empty/whitespace
             st.warning("The generated or loaded transcript appears to be empty or contains only whitespace.")

    else:
        # Warnings if keys are missing for the selected mode
        st.write("---") # Separator
        if input_mode == "Video File":
            if not whisper_ready:
                st.warning("☝️ Please enter your OpenAI Whisper API key in the sidebar to process video files.")
            if not gemini_ready:
                st.warning("☝️ Please enter your Google Gemini API key in the sidebar to analyze the transcript.")
        elif input_mode == "Transcript File":
             if not gemini_ready:
                st.warning("☝️ Please enter your Google Gemini API key in the sidebar to analyze transcript files.")

elif not uploaded_file:
    # Initial state message
    st.info("👈 Upload a file and provide API keys in the sidebar to get started.")

# Add a footer or version in the sidebar
st.sidebar.divider()
st.sidebar.info("NuggetMiner v1.2")
