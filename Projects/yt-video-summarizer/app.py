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

    # Display the radio buttons with the placeholder as the default selection
    selected_quality = st.radio("Select video quality:", options)

    # If the user hasn't made a selection, set selected_quality to the placeholder
    if selected_quality == placeholder:
        selected_quality = None
    
    # Check if the selected quality is available
    if selected_quality is not None and selected_quality not in available_formats_str:
        st.error("Selected video quality is not available. Please choose a different option.")
        return None, None

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
