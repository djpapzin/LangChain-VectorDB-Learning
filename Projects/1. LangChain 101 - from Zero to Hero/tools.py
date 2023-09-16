from langchain.llms import OpenAI
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
import os
from dotenv import load_dotenv

# Load environment viriables
load_dotenv()

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
GOOGLE_CSE_ID = os.environ["GOOGLE_CSE_ID"]

# instantiate a  LLMChain specifically for text summarization.
llm = OpenAI(model="text-davinci-003", temperature=0)

prompt = PromptTemplate(
    input_variables=["query"],
    template="Write a summary of the following text: {query}"
)

summarize_chain = LLMChain(llm=llm, prompt=prompt)

# create the tools that our agent will use.
# remember to set the environment variables
# “GOOGLE_API_KEY” and “GOOGLE_CSE_ID” to be able to use
# Google Search via API.
search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for finding information about recent events"
    ),
    Tool(
       name='Summarizer',
       func=summarize_chain.run,
       description='useful for summarizing texts'
    )
]

# create our agent that leverages two tools.
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  
)

#  run the agent with a question about summarizing the latest news about the Mars rover.
response = agent("What's the latest news about the Mars rover? Then please summarize the results.")
print(response['output'])