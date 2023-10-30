# Import modules
from langchain.llms import GPT4All
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = GPT4All(model="E:/Backup/Documents/Softwares/GPT4All/Models/ggml-wizardLM-7B.q4_2.bin", callback_manager=callback_manager, verbose=True)
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What happens when it rains somewhere?"
llm_chain.run(question)