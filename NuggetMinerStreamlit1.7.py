import streamlit as st
# import openai # Keep this if you use openai elsewhere, otherwise remove
from openai import OpenAI # Add this import
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
    """Extracts audio from video, saves as chunks, returns paths."""
    chunk_paths = []
    try:
        st.write("Opening video file...")
        video = VideoFileClip(video_path)
        audio = video.audio
        if audio is None:
            st.error("Could not extract audio from the video.")
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
                try:
                    # Write audio chunk
                    audio.subclip(start, end).write_audiofile(
                        temp_audio.name,
                        codec="libmp3lame",
                        bitrate="64k",
                        ffmpeg_params=["-ac", "1", "-ar", "16000"], # Mono, 16kHz sample rate
                        logger=None # Suppress excessive moviepy logging if desired
                    )
                    chunk_paths.append((temp_audio.name, start))
                    st.write(f"Saved chunk {idx} to {temp_audio.name}")
                except Exception as e:
                    st.error(f"Error writing audio chunk {idx}: {e}")
                    # Attempt to clean up the failed chunk file
                    if os.path.exists(temp_audio.name):
                         os.remove(temp_audio.name)
                    # Decide if you want to stop or continue with other chunks
                    # For now, let's stop if one chunk fails
                    # Clean up previously successful chunks
                    for path, _ in chunk_paths:
                        if os.path.exists(path): os.remove(path)
                    return [] # Return empty list indicating failure

            start += chunk_length
            idx += 1

        # Close video and audio objects to release resources
        audio.close()
        video.close()
        st.write("Audio extraction complete.")

    except Exception as e:
        st.error(f"Error during video processing or audio extraction: {e}")
        # Clean up any chunks that might have been created before the error
        for path, _ in chunk_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass # Ignore if already deleted
        return [] # Return empty list

    return chunk_paths


# --- Whisper via OpenAI API with chunking (Revised) ---
def transcribe_with_openai_chunks(chunk_paths, api_key):
    """Transcribes audio chunks using OpenAI Whisper API."""
    # Instantiate the client (recommended for openai >= 1.0.0)
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        st.error("❌ Failed to initialize OpenAI client.")
        st.exception(e)
        # Clean up any created chunks before returning
        for chunk_path, _ in chunk_paths:
            try:
                if os.path.exists(chunk_path): os.remove(chunk_path)
            except OSError:
                pass # Ignore errors during cleanup
        return None

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
        progress_bar.progress(processed_chunks / total_chunks, text=progress_text)
        st.write(f"Sending chunk {current_chunk_number} ({os.path.basename(chunk_path)}) to Whisper...")

        try:
            if not os.path.exists(chunk_path):
                st.error(f"Chunk file {chunk_path} not found. Skipping.")
                continue # Skip to the next chunk

            with open(chunk_path, "rb") as audio_file:
                # Use the client instance to make the API call
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"] # Request segment-level timestamps
                )

                # Process segments from the response
                # Ensure 'segments' key exists in the response dictionary/object
                segments = getattr(response, 'segments', []) # Safely get segments
                if not segments:
                     st.warning(f"No segments returned for chunk {current_chunk_number}.")

                for segment in segments:
                    # Adjust segment timings based on the chunk's offset
                    segment_start = segment.get('start', 0) + offset
                    segment_end = segment.get('end', 0) + offset
                    segment_text = segment.get('text', '').strip()

                    # Format timestamp (HH:MM:SS)
                    start_str = time.strftime('%H:%M:%S', time.gmtime(segment_start))
                    end_str = time.strftime('%H:%M:%S', time.gmtime(segment_end))

                    # Append formatted line directly
                    all_formatted_lines.append(f"[{start_str} --> {end_str}] {segment_text}")
                st.write(f"Chunk {current_chunk_number} transcribed successfully.")

            # Update progress after successful processing
            processed_chunks += 1
            progress_bar.progress(processed_chunks / total_chunks, text=progress_text)

        except Exception as e:
            st.error(f"❌ OpenAI Whisper API failed on chunk {current_chunk_number}.")
            st.exception(e)
            # Clean up remaining chunks on failure
            progress_bar.empty() # Clear the progress bar on error
            for i in range(idx, len(chunk_paths)):
                path_to_remove = chunk_paths[i][0]
                try:
                    if os.path.exists(path_to_remove): os.remove(path_to_remove)
                except OSError:
                    pass
            return None # Stop processing further chunks
        finally:
            # Ensure chunk file is always removed after processing (success or specific error)
            try:
                if os.path.exists(chunk_path): os.remove(chunk_path)
            except OSError as rm_error:
                 st.warning(f"Could not remove temp chunk file {chunk_path}: {rm_error}")

    # Final progress update and cleanup
    progress_bar.progress(1.0, text="Transcription complete!")
    time.sleep(1.5) # Keep the 100% message visible briefly
    progress_bar.empty() # Clear the progress bar

    # Join the already formatted lines
    return "\n".join(all_formatted_lines)


# --- Transcript upload handling ---
def load_transcript_text(uploaded_file):
    """Reads transcript from uploaded file, cleaning basic formatting."""
    try:
        # Read as bytes first, then decode
        content_bytes = uploaded_file.read()
        # Detect encoding (simple check for utf-8 with BOM)
        encoding = 'utf-8-sig' if content_bytes.startswith(b'\xef\xbb\xbf') else 'utf-8'
        content = content_bytes.decode(encoding)
        lines = content.splitlines()

        # Simple cleaning: remove empty lines and lines that are just numbers (like SRT indices)
        # More robust VTT/SRT parsing could be added if needed
        cleaned_lines = [
            line.strip() for line in lines
            if line.strip() and not line.strip().isdigit()
            # Add more specific VTT/SRT pattern removal if necessary
            and '-->' not in line # Basic check to remove timestamp lines
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
        # Consider using a newer or more appropriate model if available
        # e.g., gemini-1.5-flash or gemini-1.5-pro
        # Check Google AI documentation for current model names
        model = genai.GenerativeModel(
             model_name="gemini-1.5-flash", # Or try "gemini-1.5-pro"
             system_instruction=system_prompt # Use system_instruction for newer models
             )
        response = model.generate_content(
            transcript_text, # Pass transcript directly as user content
            generation_config={"temperature": 0.7}
        )
        # Add basic error handling for the response object
        if response and hasattr(response, 'text'):
            return response.text
        elif response and hasattr(response, 'prompt_feedback'):
             st.warning(f"Gemini content generation might be blocked: {response.prompt_feedback}")
             return "Error: Content generation blocked by safety settings."
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

    # Separator
    st.divider()

    # Reset button
    if st.button("🔁 Start Over / Clear"):
        # Clear relevant session state keys
        keys_to_clear = ["transcript", "gemini_response", "uploaded_filename", "input_mode_radio"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        # Clear the file uploader state is harder, rerun is usually sufficient
        st.rerun()


# Main area for file upload and results
st.header("📤 Upload Your File")
uploaded_file = st.file_uploader(
    f"Upload your {input_mode.lower()}", # Dynamic label
    type=["mp4", "txt", "vtt", "srt"],
    accept_multiple_files=False,
    key="file_uploader" # Give it a key for potential state management
)

# --- Processing Logic ---
if uploaded_file is not None:
    st.session_state.uploaded_filename = uploaded_file.name
    st.info(f"File '{st.session_state.uploaded_filename}' uploaded.")

    # Check if required keys are present based on input mode
    gemini_ready = bool(gemini_api_key)
    whisper_ready = bool(openai_api_key)
    can_process_video = input_mode == "Video File" and whisper_ready and gemini_ready
    can_process_transcript = input_mode == "Transcript File" and gemini_ready

    # Proceed only if keys and file are ready for the selected mode
    if (can_process_video or can_process_transcript):

        # --- Step 1: Get Transcript (either by transcribing or loading) ---
        if st.session_state.transcript is None:
            st.write("---") # Separator
            if input_mode == "Video File" and uploaded_file.type.startswith("video"):
                st.info("Processing video file...")
                # Use a temporary file for the video
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                    temp_video.write(uploaded_file.getvalue()) # Use getvalue() for uploaded file
                    temp_video_path = temp_video.name

                with st.status("🜚 Processing Video...", expanded=True) as status:
                    st.write("➡️ Step 1: Extracting audio chunks...")
                    # Ensure temp_video_path is passed correctly
                    chunk_paths = extract_audio_chunks(temp_video_path)

                    if not chunk_paths:
                        st.error("Audio extraction failed. Cannot proceed.")
                        status.update(label="Audio Extraction Failed ❌", state="error")
                        # Clean up the temp video file if extraction fails
                        if os.path.exists(temp_video_path): os.remove(temp_video_path)
                        st.stop() # Stop execution here

                    st.write(f"➡️ Step 2: Transcribing {len(chunk_paths)} audio chunk(s)...")
                    # Transcription with integrated progress happens here
                    transcript_text = transcribe_with_openai_chunks(chunk_paths, openai_api_key)

                    # Clean up the temporary video file after use
                    if os.path.exists(temp_video_path):
                         os.remove(temp_video_path)
                         st.write("Cleaned up temporary video file.")

                    if transcript_text is not None:
                        st.session_state.transcript = transcript_text
                        status.update(label="✅ Video Processed & Transcribed!", state="complete")
                        st.toast("Transcription complete ✅")
                    else:
                        st.error("Transcription failed.")
                        status.update(label="Transcription Failed ❌", state="error")
                        st.stop() # Stop if transcription failed

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

            else:
                # This case should ideally not be reached due to file type constraints
                st.error(f"Invalid file type ({uploaded_file.type}) for selected mode '{input_mode}'.")
                st.stop()

        # --- Step 2: Display Transcript and Process with Gemini ---
        transcript = st.session_state.get("transcript") # Use .get for safety
        if transcript:
            st.subheader("📜 Transcript Preview")
            st.text_area("Transcript Text", transcript, height=300, key="transcript_display")

            st.write("---") # Separator

            if st.button("💎 Mine for Nuggets", key="mine_button"):
                st.session_state.gemini_response = None # Reset previous response
                with st.spinner("🧠 Asking Gemini to find the marketing gold..."):
                    gemini_response_text = send_to_gemini(gemini_api_key, transcript)
                    if gemini_response_text:
                        st.session_state.gemini_response = gemini_response_text
                        st.toast("Gemini analysis complete! ✨")
                    else:
                        st.error("Failed to get response from Gemini.")
                        # Keep spinner from showing success state implicitly

            # Display Gemini results if available in session state
            if st.session_state.get("gemini_response"):
                st.subheader("✨ Nuggets Found!")
                st.text_area("Marketing Nuggets", st.session_state.gemini_response, height=400, key="nuggets_display")

                # Add download button for the nuggets
                st.download_button(
                    label="💾 Download Nuggets",
                    data=st.session_state.gemini_response.encode('utf-8'), # Encode to bytes
                    file_name=f"marketing_nuggets_{st.session_state.uploaded_filename}.txt",
                    mime="text/plain"
                )
        elif st.session_state.transcript is not None: # Transcript exists but is empty
             st.warning("The generated or loaded transcript appears to be empty.")

    else:
        # Warnings if keys are missing for the selected mode
        if input_mode == "Video File":
            if not whisper_ready:
                st.warning("☝️ Please enter your OpenAI Whisper API key in the sidebar to process video files.")
            if not gemini_ready:
                st.warning("☝️ Please enter your Google Gemini API key in the sidebar to analyze the transcript.")
        elif input_mode == "Transcript File":
             if not gemini_ready:
                st.warning("☝️ Please enter your Google Gemini API key in the sidebar to analyze transcript files.")

elif not uploaded_file:
    st.info("👈 Upload a file and provide API keys in the sidebar to get started.")

st.sidebar.info("App Version 1.1")
