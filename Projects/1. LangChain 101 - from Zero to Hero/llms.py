from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Call the LLM
llm = OpenAI(model="text-davinci-003", temperature=0.9)

# The Prompt
prompt = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."

# pass the prompt to the LLM
print(llm(prompt))