from langchain import PromptTemplate, OpenAI, LLMChain

prompt_template = "What is a word to replace the following: {word}?"

# Set the "OPENAI_API_KEY" environment variable before running following line.
llm = OpenAI(model_name="text-davinci-003", temperature=0)

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

# __call__ method
llm_chain("artificial")

# .apply() method
input_list = [
    {"word": "artificial"},
    {"word": "intelligence"},
    {"word": "robot"}
]

llm_chain.apply(input_list)

# .generate() method
llm_chain.generate(input_list)

# .predict() method
prompt_template = "Looking at the context of '{context}'. What is an appropriate word to replace the following: {word}?"

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(template=prompt_template, input_variables=["word", "context"]))

llm_chain.predict(word="fan", context="object")

# or .run() method
llm_chain.run(word="fan", context="object")

