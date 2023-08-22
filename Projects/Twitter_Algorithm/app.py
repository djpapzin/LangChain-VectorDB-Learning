import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Load the .env file
load_dotenv()

# Access the keys from the .env file
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ACTIVELOOP_TOKEN = os.getenv('ACTIVELOOP_TOKEN')

# Define the OpenAI embeddings.
embeddings = OpenAIEmbeddings()

# load all files inside the repository.
root_dir = './the-algorithm-new'
docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        try: 
            loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
            docs.extend(loader.load_and_split())
        except Exception as e: 
            pass

# Divide the loaded files into chunks:
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

# Perform the indexing process and upload embeddings to Active-Loop
username = "djpapzin" # replace with your username from app.activeloop.ai
db = DeepLake(dataset_path=f"hub://{username}/twitter-algorithm", embedding_function=embeddings)
db.add_documents(texts)


# Define the retriever
retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 100
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 10

# load the dataset, 
db = DeepLake(dataset_path="hub://djpapzin/twitter-algorithm", read_only=True, embedding_function=embeddings)

# Connect to GPT-4 for question answering
model = ChatOpenAI(model='gpt-3.5-turbo') # switch to 'gpt-4'


# Establish the retriever
qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)

# Create the Conversational Chain
# Define all the juicy questions you want to be answered:
questions = [
    "What does favCountParams do?",
    "is it Likes + Bookmarks, or not clear from the code?",
    "What are the major negative modifiers that lower your linear ranking parameters?",   
    "How do you get assigned to SimClusters?",
    "What is needed to migrate from one SimClusters to another SimClusters?",
    "How much do I get boosted within my cluster?",   
    "How does Heavy ranker work. what are it’s main inputs?",
    "How can one influence Heavy ranker?",
    "why threads and long tweets do so well on the platform?",
    "Are thread and long tweet creators building a following that reacts to only threads?",
    "Do you need to follow different strategies to get most followers vs to get most likes and bookmarks per tweet?",
    "Content meta data and how it impacts virality (e.g. ALT in images).",
    "What are some unexpected fingerprints for spam factors?",
    "Is there any difference between company verified checkmarks and blue verified individual checkmarks?",
] 
chat_history = []

for question in questions:  
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result['answer']))
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")
