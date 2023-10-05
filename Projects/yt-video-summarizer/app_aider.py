import os
from dotenv import load_dotenv
import streamlit as st
import yt_dlp
import whisper
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import textwrap

# Load environment variables (API keys in this case)
load_dotenv()

# Set the title of the app
st.title("Youtube Video Summarizer")

# Retrieve and set API keys from environment variables
openai_api_key = os.environ['OPENAI_API_KEY']
activeloop_token = os.environ['ACTIVELOOP_TOKEN']

# Function to download MP4 videos from YouTube
def download_mp4_from_youtube(url, selected_quality):
    ydl_opts = {
        'format': f'bestvideo[height={selected_quality[:-1]}][ext=mp4]+bestaudio[ext=m4a]/best[height={selected_quality[:-1]}][ext=mp4]',
        'outtmpl': '%(title)s.mp4',
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)
        video_title = result['title']
    return video_title

# Get YouTube video URL from user
url = st.session_state.get('url', '')
selected_quality = st.session_state.get('selected_quality', None)

url = st.text_input("Enter the YouTube video URL:", value=url, key="video_url")
st.session_state['url'] = url

# Check if video has already been downloaded
if 'video_downloaded' not in st.session_state:
    st.session_state['video_downloaded'] = False

if not st.session_state['video_downloaded']:
    available_formats = ["720p", "480p", "360p"]  # Define the available video formats here
    options = ["Select preferred video format"] + available_formats
    selected_quality = st.selectbox("Select video quality:", options, index=0, key="video_quality")
    st.session_state['selected_quality'] = selected_quality

    if st.button("Start", key="start_button"):
        st.session_state['start_button_pressed'] = True
        if selected_quality != "Select preferred video format":
            video_title = download_mp4_from_youtube(url, selected_quality)
            st.session_state['video_downloaded'] = True
            st.session_state['video_title'] = video_title
else:
    video_title = st.session_state['video_title']

if st.session_state.get('start_button_pressed', False):
    if url and selected_quality:
        if video_title and selected_quality:
            # Show progress updates to the user
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Load the Whisper model for transcription
            model = whisper.load_model("tiny")
            result = model.transcribe(f"{video_title}.mp4")

            # Extract transcription from the result
            transcription = result['text']

            # Save the transcription to a file
            with open(f'{video_title}_transcription.txt', 'w') as file:
                file.write(transcription)

            # Initialize the LangChain model for summarization
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

            # Split the transcription text into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
            )
            with open(f'{video_title}_transcription.txt') as f:
                text = f.read()
            texts = text_splitter.split_text(text)
            docs = [Document(page_content=t) for t in texts[:4]]

            # Use the 'refine' summarization chain with the custom prompt
            chain = load_summarize_chain(llm, chain_type="refine")
            output_summary = chain.run(docs)
            wrapped_text = textwrap.fill(output_summary, width=100)

            # Update progress bar and status text
            progress_bar.progress(50)
            status_text.text("Transcription complete")

            # Expand icon to display the rest of the transcription
            with st.expander("Transcription", expanded=False):
                st.write(transcription)

            # Button to download transcript
            if st.button("Download Transcript", key="download_transcript"):
                st.download_button(
                    label="Download Transcript",
                    data=f'{video_title}_transcription.txt',
                    file_name=f'{video_title}_transcription.txt',
                    on_click=None
                )

            # Display the refined summary
            st.subheader("Summary:")
            st.write(wrapped_text)

            # Button to download summary
            if st.button("Download Summary", key="download_summary"):
                st.download_button(
                    label="Download Summary",
                    data=wrapped_text,
                    file_name=f'{video_title}_summary.txt',
                    on_click=None
                )

            # Update progress bar and status text
            progress_bar.progress(100)
            status_text.text("Summary complete")

        else:
            st.write("Please select a video quality.")
    else:
        st.write("Please enter a YouTube video URL.")
