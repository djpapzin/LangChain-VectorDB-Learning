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

    # Get user input for desired video quality
    selected_quality = st.radio("Select video quality:", available_formats_str, index=None)

    # Set a default value for selected_quality if it is None
    if selected_quality is None:
        selected_quality = available_formats_str[0]

    # Download the video based on the selected quality
    ydl_opts = {
        'format': f'bestvideo[height={selected_quality}][ext=mp4]+bestaudio[ext=m4a]/best[height={selected_quality}][ext=mp4]',
        'outtmpl': '%(title)s.mp4',
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)
        video_title = result['title']

    return video_title, selected_quality
