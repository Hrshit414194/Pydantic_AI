import os
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader

# Load API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Step 1: Define structured schema
class RAGAnswer(BaseModel):
    answer: str
    sources: List[str]

# Step 2: Setup LLM
model = OpenAIModel("gpt-4o-mini", provider=OpenAIProvider(api_key=api_key))
agent = Agent(model, output_type=RAGAnswer)

# Step 3: Scrape webpage
url = "https://en.wikipedia.org/wiki/Intermittent_fasting"
print(f"Scraping {url} ...")
loader = WebBaseLoader(url)
documents = loader.load()

# Step 4: Chunk scraped text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Step 5: Build FAISS vector store
embeddings = OpenAIEmbeddings(api_key=api_key)
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("Ask me anything (type 'exit' to quit).")

# Step 6: Interactive Q&A
while True:
    query = input("Your question: ")
    if query.lower() in ["exit", "quit"]:
        print("Thank You!")
        break

    # Retrieve docs
    retrieved_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Build prompt
    prompt = f"""
    Question: {query}
    Here are relevant documents:
    {context}
    Answer clearly in 'answer' and provide source URLs in 'sources'.
    """

    # Run Agent
    result = agent.run_sync(prompt)

    print("\n Answer:", result.output.answer)
    print("Sources:", result.output.sources)