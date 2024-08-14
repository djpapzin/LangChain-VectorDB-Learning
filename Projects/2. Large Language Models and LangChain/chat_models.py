from langchain_google_genai import GoogleGenerativeAI # Import GoogleGenerativeAI from langchain_google_genai 
from langchain.schema import ( # Import HumanMessage and SystemMessage from langchain.schema
  HumanMessage,
  SystemMessage
)
from dotenv import load_dotenv # Import load_dotenv
import os

load_dotenv() # Load environment variables from .env file

llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY")) # Create GoogleGenerativeAI instance

messages = [ # Create a list of messages
	SystemMessage(content="You are a helpful assistant that translates English to French."), # Create a SystemMessage with the content "You are a helpful assistant that translates English to French."
	HumanMessage(content="Translate the following sentence: I love programming.") # Create a HumanMessage with the content "Translate the following sentence: I love programming."
]

llm(messages) # Call the llm object with the list of messages