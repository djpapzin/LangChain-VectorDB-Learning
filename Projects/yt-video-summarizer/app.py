import streamlit as st
import yt_dlp

def download_mp4_from_youtube(url, selected_quality):
    # Download the video based on the selected quality
    ydl_opts = {
        'format': f'bestvideo[height={selected_quality}][ext=mp4]+bestaudio[ext=m4a]/best[height={selected_quality}][ext=mp4]',
        'outtmpl': '%(title)s.mp4',
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=True)
            video_title = result['title']
    except yt_dlp.DownloadError as e:
        st.error(f"Error: {str(e)}")
        st.stop()

    return video_title, selected_quality

# Streamlit app code
st.title("YouTube Video Summarizer")

url = st.text_input("Enter YouTube URL")
selected_quality = st.selectbox("Select Video Quality", ["720", "480", "360"])

if st.button("Start"):
    video_title, selected_quality = download_mp4_from_youtube(url, selected_quality)
    st.success(f"Video '{video_title}' downloaded successfully in {selected_quality}p quality.")
