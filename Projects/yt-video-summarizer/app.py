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
