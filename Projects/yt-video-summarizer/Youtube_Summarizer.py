# Import necessary libraries and modules
import os
from dotenv import load_dotenv
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
import streamlit as st

# Function to download MP4 videos from YouTube
def download_mp4_from_youtube(url, selected_quality):
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

# Function to summarize a YouTube video
def summarize_youtube_video(url, selected_quality):
    # Download the video from YouTube and get the title
    video_title = download_mp4_from_youtube(url, selected_quality)

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

    # Define the prompt for the 'refine' summarization chain
    prompt_template = """Write a concise bullet point summary of the following:
{text}
CONSCISE SUMMARY IN BULLET POINTS:"""
    BULLET_POINT_PROMPT = PromptTemplate(template=prompt_template, 
                        input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="refine", refine_prompt=BULLET_POINT_PROMPT)

    # Use the 'refine' summarization chain to generate summaries for each chunk of text
    summaries = []
    for doc in docs:
        summary_doc, _ = chain(doc)
        summaries.append(summary_doc.page_content)

    # Join the summaries into one string and wrap it to fit in a column width of 80 characters 
    summary = "\n".join(summaries)
    summary_wrapped = "\n".join(textwrap.wrap(summary, width=80))

    return transcription, summary_wrapped, video_title

# Function to run the YouTube video summarizer app
def run_youtube_summarizer():
    # Load environment variables (API keys in this case)
    load_dotenv()

    # Retrieve and set API keys from environment variables
    openai_api_key = os.environ['OPENAI_API_KEY']
    activeloop_token = os.environ['ACTIVELOOP_TOKEN']

    # Set the title of the streamlit app
    st.title("YouTube Video Summarizer")

    # Get YouTube video URL from user using a text input widget
    url = st.text_input("Enter the YouTube video URL: ")

    # Check if the URL is valid and not empty
    if url and yt_dlp.validate_url(url):
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
        selected_quality = st.radio("Select video quality:", available_formats_str)

        # Button to start the process of transcribing and summarizing
        if st.button("Start"):
            # Summarize the YouTube video
            transcription, summary, video_title = summarize_youtube_video(url, selected_quality)

            # Display the video using a player component
            st.video(f"{video_title}.mp4")

            # Display the transcription and the summary using expander components
            with st.expander("Transcription"):
                st.write(transcription)

            with st.expander("Summary"):
                st.write(summary)

            # Button to download the transcription and the summary as text files
            st.download_button(label="Download transcription", data=transcription, file_name=f"{video_title}_transcription.txt", mime="text/plain")
            st.download_button(label="Download summary", data=summary, file_name=f"{video_title}_summary.txt", mime="text/plain")
