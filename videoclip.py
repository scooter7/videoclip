import os
import whisper
from moviepy.editor import VideoFileClip, concatenate_videoclips
import requests
import json
import ast
import streamlit as st

# Step 1: Transcribe the Video
def transcribe_video(video_path, model_name="tiny"):  # Using "tiny" model for faster processing
    model = whisper.load_model(model_name)
    audio_path = "temp_audio.wav"
    
    # Add logging to track the progress of ffmpeg
    st.write("Extracting audio from video using ffmpeg...")
    result = os.system(f"ffmpeg -i {video_path} -ar 16000 -ac 1 -b:a 64k -f mp3 {audio_path}")
    
    # Check if ffmpeg command failed
    if result != 0:
        st.error("Failed to extract audio using ffmpeg.")
        return None
    
    # Check if audio file is created
    if not os.path.exists(audio_path):
        st.error("Audio extraction failed, no audio file created.")
        return None
    
    st.write("Audio extracted. Now transcribing...")

    # Whisper transcription process
    result = model.transcribe(audio_path)
    
    transcription = []
    for segment in result['segments']:
        transcription.append({
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text'].strip()
        })
    
    st.write("Transcription completed.")
    return transcription

import re
import json

def get_relevant_segments(transcript, user_query):
    groq_key = st.secrets["groq_key"]  # Get the API key from Streamlit secrets
    
    # Create the prompt
    prompt = f"""You are an expert video editor who can read video transcripts and perform video editing. Given a transcript with segments, your task is to identify all the conversations related to a user query. Follow these guidelines when choosing conversations. A group of continuous segments in the transcript is a conversation.

    Guidelines:
    1. The conversation should be relevant to the user query. The conversation should include more than one segment to provide context and continuity.
    2. Include all the before and after segments needed in a conversation to make it complete.
    3. The conversation should not cut off in the middle of a sentence or idea.
    4. Choose multiple conversations from the transcript that are relevant to the user query.
    5. Match the start and end time of the conversations using the segment timestamps from the transcript.
    6. The conversations should be a direct part of the video and should not be out of context.

    Output format: {{ "conversations": [{{"start": "s1", "end": "e1"}}, {{"start": "s2", "end": "e2"}}] }}

    Transcript:
    {transcript}

    User query:
    {user_query}"""
    
    # API request
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {groq_key}"  # Use the key from Streamlit secrets
    }

    data = {
        "messages": [
            {
                "role": "system",
                "content": prompt
            }
        ],
        "model": "llama-3.1-70b-versatile",
        "temperature": 1,
        "max_tokens": 1024,
        "top_p": 1,
        "stream": False,
        "stop": None
    }

    response = requests.post(url, headers=headers, json=data)
    raw_response = response.text
    st.write("Raw content before cleaning:", raw_response)

    # Use a focused regex to extract the JSON part starting from "conversations"
    try:
        json_match = re.search(r'\"conversations\":\s*\[.*\]', raw_response, re.DOTALL)
        if json_match:
            json_str = '{' + json_match.group() + '}'
            st.write("Extracted JSON string:", json_str)

            # Parse the cleaned JSON string
            conversations = json.loads(json_str)["conversations"]
            return conversations
        else:
            st.error("No valid JSON content found in the API response.")
            return None
    except Exception as e:
        st.error(f"Error while decoding JSON content: {e}")
        return None

# Step 3: Edit the video based on relevant segments
from moviepy.editor import VideoFileClip, concatenate_videoclips

def edit_video(original_video_path, segments, output_video_path, fade_duration=0.5, default_fps=24):
    try:
        video = VideoFileClip(original_video_path)
        clips = []
        for seg in segments:
            start = float(seg['start'])  # Convert start time to float
            end = float(seg['end'])  # Convert end time to float
            if start < 0 or end > video.duration:
                st.write(f"Invalid segment start ({start}) or end ({end}) time. Skipping this segment.")
                continue
            clip = video.subclip(start, end).fadein(fade_duration).fadeout(fade_duration)
            clips.append(clip)
        
        if clips:
            final_clip = concatenate_videoclips(clips, method="compose")
            
            # Set FPS to avoid NoneType error
            if video.fps is None:
                fps = default_fps
            else:
                fps = video.fps
            
            final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac", fps=fps)
        else:
            st.write("No valid segments to include in the edited video.")
    
    except Exception as e:
        st.error(f"An error occurred during video editing: {e}")

# Streamlit App Interface
def main():
    st.title("Video Transcription and Editing App")
    
    # Upload video file
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    
    if video_file:
        # Check the file size before proceeding
        if video_file.size > 50 * 1024 * 1024:  # Example: 50MB limit
            st.warning("The video file is large, and transcription may take longer than expected.")
        
        # Save uploaded file to a temporary location
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(video_file.read())
        
        # User Query
        user_query = st.text_input("Enter your query (e.g., 'Find clips about GPT-4 Turbo')")
        
        if st.button("Transcribe and Edit Video"):
            # Transcribe the video
            st.write("Transcribing video...")
            transcription = transcribe_video(video_path, model_name="tiny")  # Using "tiny" model for faster processing
            
            if transcription is not None:
                # Get relevant segments based on user query
                st.write("Finding relevant segments...")
                relevant_segments = get_relevant_segments(transcription, user_query)
                
                if relevant_segments:
                    # Edit video based on relevant segments
                    output_video_path = "edited_video.mp4"
                    st.write("Editing video...")
                    edit_video(video_path, relevant_segments, output_video_path)
                    
                    # Provide download link for the edited video
                    st.video(output_video_path)
                    with open(output_video_path, "rb") as file:
                        st.download_button(label="Download Edited Video", data=file, file_name="edited_output.mp4")
                else:
                    st.error("No relevant segments found based on the query.")
            else:
                st.error("Transcription failed. Please try again.")
            
if __name__ == "__main__":
    main()
