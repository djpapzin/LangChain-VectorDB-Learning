import os
from dotenv import load_dotenv
import streamlit as st
import yt_dlp

# Load environment variables (API keys in this case)
load_dotenv()

# Set the title of the app
st.title("Youtube Video Summarizer")

# Retrieve and set API keys from environment variables
openai_api_key = os.environ['OPENAI_API_KEY']
activeloop_token = os.environ['ACTIVELOOP_TOKEN']

# Function to download MP4 videos from YouTube
def download_mp4_from_youtube(url):
    # Define options to list available video formats
    ydl_opts = {
        'listformats': True,
    }
    # Extract video information without downloading
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=False)

    # Parse and display available video quality options
    available_formats = set()
    for format in result['formats']:
        if format['ext'] == 'mp4' and 'height' in format:
            height = format['height']
            available_formats.add(height)
    available_formats_str = [str(height) + 'p' for height in sorted(available_formats)]

    # Add a placeholder to the beginning of the list
    placeholder = "Select an option..."
    options = [placeholder] + available_formats_str

    # Display the radio buttons with the available formats
    selected_quality = st.radio("Select video quality:", options)

    # If the user hasn't made a selection, set selected_quality to None
    if selected_quality == placeholder:
        selected_quality = None
    
    # Download the video based on the selected quality
    if selected_quality is not None:
        ydl_opts = {
            'format': f'bestvideo[height={selected_quality}][ext=mp4]+bestaudio[ext=m4a]/best[height={selected_quality}][ext=mp4]',
            'outtmpl': '%(title)s.mp4',
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=True)
            video_title = result['title']
    else:
        video_title = None

    return video_title, selected_quality

# Get YouTube video URL from user
url = st.text_input("Enter the YouTube video URL: ")
selected_quality = None

if st.button("Start"):
    if url:
        video_title, selected_quality = download_mp4_from_youtube(url)

        if video_title is not None:
            st.success(f"Video '{video_title}' downloaded successfully!")
        else:
            st.error("Failed to download the video.")

        st.write(f"Selected video quality: {selected_quality}")
