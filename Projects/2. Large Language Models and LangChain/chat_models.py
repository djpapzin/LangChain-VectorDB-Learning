# Import ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI 
# Import load_dotenv
from dotenv import load_dotenv 
import os

# Load environment variables from .env file
load_dotenv() 

# Create ChatGoogleGenerativeAI instance
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GEMINI_API_KEY")) 

# Use invoke for single messages
response = llm.invoke("Translate the following sentence: I love programming.") 
# Print the generated response text only
print(response.content) 