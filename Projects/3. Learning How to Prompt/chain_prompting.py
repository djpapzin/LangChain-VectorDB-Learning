from langchain import PromptTemplate, LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Initialize LLM
api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
print(
    llm.invoke(
        "What are some of the pros and cons of Python as a programming language?"
    )
)

# Prompt 1
template_question = """What is the name of the famous scientist who developed the theory of general relativity?
Answer: """
prompt_question = PromptTemplate(
    template=template_question, 
    input_variables=[])

# Prompt 2
template_fact = """Provide a brief description of {scientist}'s theory of general relativity.
Answer: """
prompt_fact = PromptTemplate(
    input_variables=["scientist"], 
    template=template_fact)

# Create the LLMChain for the first prompt
chain_question = LLMChain(llm=llm, prompt=prompt_question)

# Run the LLMChain for the first prompt with an empty dictionary
response_question = chain_question.run({})

# Extract the scientist's name from the response
scientist = response_question.strip()

# Create the LLMChain for the second prompt
chain_fact = LLMChain(llm=llm, prompt=prompt_fact)

# Input data for the second prompt
input_data = {"scientist": scientist}

# Run the LLMChain for the second prompt
response_fact = chain_fact.run(input_data)

print("Scientist:", scientist.text)
print("Fact:", response_fact.text)