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
def download_mp4_from_youtube(url):
    # Extract video information without downloading
    with yt_dlp.YoutubeDL() as ydl:
        result = ydl.extract_info(url, download=False)

    # Parse and display available video quality options
    available_formats = []
    for format in result['formats']:
        if format['ext'] == 'mp4' and 'height' in format:
            height = format['height']
            available_formats.append(str(height) + 'p')
    available_formats = sorted(list(set(available_formats)))

    # Add "Select preferred video format" to the beginning of the list
    options = ["Select preferred video format"] + available_formats

    # Display the radio buttons
    selected_quality = st.selectbox("Select video quality:", options, index=0, key="video_quality")

    # Download the video based on the selected quality
    if selected_quality != "Select preferred video format":
        ydl_opts = {
            'format': f'bestvideo[height={selected_quality[:-1]}][ext=mp4]+bestaudio[ext=m4a]/best[height={selected_quality[:-1]}][ext=mp4]',
            'outtmpl': '%(title)s.mp4',
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=True)
            video_title = result['title']
        return video_title, selected_quality
    else:
        return None, None

# Get YouTube video URL from user
url = st.session_state.get('url', '')
selected_quality = None

url = st.text_input("Enter the YouTube video URL:", value=url, key="video_url")
st.session_state['url'] = url

if st.button("Start", key="start_button"):
    st.session_state['start_button_pressed'] = True
else:
    st.session_state['start_button_pressed'] = False

if st.session_state.get('start_button_pressed', False):
    if url:
        video_title, selected_quality = download_mp4_from_youtube(url)

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

            # Display the refined summary
            st.subheader("Summary:")
            st.write(wrapped_text)

            # Button to download transcript
            if st.button("Download Transcript"):
                st.download_button(
                    label="Download Transcript",
                    data=f'{video_title}_transcription.txt',
                    file_name=f'{video_title}_transcription.txt',
                )

            # Button to download summary
            if st.button("Download Summary"):
                st.download_button(
                    label="Download Summary",
                    data=wrapped_text,
                    file_name=f'{video_title}_summary.txt',
                )

            # Expand icon to display the rest of the transcription
            with st.expander("Transcription", expanded=True):
                st.write(transcription)
        else:
            st.write("Please select a video quality.")
    else:
        st.write("Please enter a YouTube video URL.")
