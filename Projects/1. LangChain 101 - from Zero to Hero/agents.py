from langchain.llms import OpenAI # create an instance of the OpenAI language model
from langchain.agents import AgentType 
from langchain.agents import load_tools # load a list of tools that an AI agent can use
from langchain.agents import initialize_agent # initializes an AI agent that can use a given set of tools and a language model to interact with users
from langchain.agents import Tool # define a tool that an AI agent can use.
from langchain.utilities import GoogleSearchAPIWrapper #wrapper for the Google Search API, allowing it to be used as a tool by an AI agent
import os
from dotenv import load_dotenv

# Load environment viriables
load_dotenv()

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
GOOGLE_CSE_ID = os.environ["GOOGLE_CSE_ID"]

# Initialize the LLM and set the temperature to 0 for the precise answer. 
llm = OpenAI(model="text-davinci-003", temperature=0)

# Define the Google search wrapper
# remember to set the environment variables
# “GOOGLE_API_KEY” and “GOOGLE_CSE_ID” to be able to use
# Google Search via API.
search = GoogleSearchAPIWrapper()

# The Tool object represents a specific capability or function the system can use. In this case, it's a tool for performing Google searches.
tools = [
    Tool(
        name = "google-search",
        func=search.run,
        description="useful for when you need to search google to answer questions about current events"
    )
]

# create an agent that uses our Google Search tool
agent = initialize_agent(tools, 
                         llm, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                         verbose=True,
                         max_iterations=6)

# check out the response
response = agent("What's the latest news about the Mars rover?")
print(response['output'])