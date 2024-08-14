from langchain.chat_models import ChatGemini # Import ChatGemini instead of ChatOpenAI
from langchain.schema import ( # Import HumanMessage and SystemMessage from langchain.schema
  HumanMessage,
  SystemMessage
)
import getpass
import os

if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Provide your GEMINI API Key")

chat = ChatGemini(model_name="gemini-1.5-pro-exp-0801", temperature=0) # Create a ChatGemini object with the specified model name and temperature

messages = [ # Create a list of messages
	SystemMessage(content="You are a helpful assistant that translates English to French."), # Create a SystemMessage with the content "You are a helpful assistant that translates English to French."
	HumanMessage(content="Translate the following sentence: I love programming.") # Create a HumanMessage with the content "Translate the following sentence: I love programming."
]

chat(messages) # Call the chat object with the list of messages