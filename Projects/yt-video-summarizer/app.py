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
    selected_quality = st.radio("Select video quality:", available_formats_str, index=-1)

    # Define options for video download based on user's choice
    ydl_opts = {
        'format': f'bestvideo[height={selected_quality}][ext=mp4]+bestaudio[ext=m4a]/best[height={selected_quality}][ext=mp4]',
        'outtmpl': '%(title)s.mp4',
        'quiet': True,
    }

    # Download the video based on the selected quality
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)
        video_title = result['title']

    return video_title

# Get YouTube video URL from user
url = st.text_input("Enter the YouTube video URL: ")
selected_quality = None

if st.button("Start"):
    if url:
        selected_quality = download_mp4_from_youtube(url, selected_quality)

        # Load the Whisper model for transcription
        model = whisper.load_model("tiny")
        result = model.transcribe(f"{selected_quality}.mp4")

        # Extract transcription from the result
        transcription = result['text']

        # Save the transcription to a file
        with open(f'{selected_quality}_transcription.txt', 'w') as file:
            file.write(transcription)

        # Initialize the LangChain model for summarization
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        # Split the transcription text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
        )
        with open(f'{selected_quality}_transcription.txt') as f:
            text = f.read()
        texts = text_splitter.split_text(text)
        docs = [Document(page_content=t) for t in texts[:4]]

        # Use the 'refine' summarization chain with the custom prompt
        chain = load_summarize_chain(llm, chain_type="refine")
        output_summary = chain.run(docs)
        wrapped_text = textwrap.fill(output_summary, width=100)

        # Display the refined summary
        st.subheader("Summary:")
        st.write(wrapped_text)

        # Button to download transcript
        if st.button("Download Transcript"):
            st.download_button(
                label="Download Transcript",
                data=f'{selected_quality}_transcription.txt',
                file_name=f'{selected_quality}_transcription.txt',
            )

        # Button to download summary
        if st.button("Download Summary"):
            st.download_button(
                label="Download Summary",
                data=wrapped_text,
                file_name=f'{selected_quality}_summary.txt',
            )

        # Expand icon to display the rest of the transcription
        with st.expander("Transcription", expanded=True):
            st.write(transcription)
