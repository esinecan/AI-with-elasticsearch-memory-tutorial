import logging
from fastapi import FastAPI
from urllib.request import urlopen
import json

# ...existing logging configuration...
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("elasticsearch").setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.DEBUG)

app = FastAPI()

# Elasticsearch credentials (using cloud id from reference)
CLOUD_ID = "RAGTest:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGUzZmNkZTI3ODFiMjRkZmViYmNmYjhkMTY3NWU2NzQ5JDVlY2NjN2Q1OWY1ZTRjMDlhYTAyZDk1ZWU5MTcwNDVi"
API_KEY = "YOUR KEY"
INDEX_NAME = "rag_index"

# New Langchain-related imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_elasticsearch import ElasticsearchStore
from langchain_elasticsearch import SparseVectorStrategy
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.llms import Ollama

# Initialize Langchain ElasticsearchStore with ELSER v2 strategy
es_vector_store = ElasticsearchStore(
    es_cloud_id=CLOUD_ID,
    es_api_key=API_KEY,
    index_name=INDEX_NAME,
    strategy=SparseVectorStrategy(model_id="my-elser-model"),
)
""" es_vector_store = ElasticsearchStore(
    es_cloud_id=CLOUD_ID,
    es_api_key=API_KEY,
    index_name=INDEX_NAME,
    strategy=SparseVectorStrategy(
        model_id=".elser_model_2_linux-x86_64",
        service="elasticsearch",
        service_settings={
            "num_threads": 1,
            "adaptive_allocations": {
                "enabled": True,
                "min_number_of_allocations": 1,
                "max_number_of_allocations": 10
            }
        }
    )
) """

# Initialize LLM using Langchain's Ollama wrapper
llm = Ollama(model="your-boss:latest")

# Remove unused llama_index imports and settings
# ...existing code removed...

# Refactored load_data() to use Langchain text splitter
def load_data():
    url = "https://raw.githubusercontent.com/elastic/elasticsearch-labs/main/datasets/workplace-documents.json"
    response = urlopen(url)
    workplace_docs = json.loads(response.read())
    content = []
    metadata = []
    for doc in workplace_docs:
        content.append(doc["content"])
        metadata.append({
            "name": doc["name"],
            "summary": doc["summary"],
            "rolePermissions": doc["rolePermissions"],
        })
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=256
    )
    docs = text_splitter.create_documents(content, metadatas=metadata)
    logging.info("Loaded %d documents for ingestion.", len(docs))
    return docs

# Startup ingestion: load data and add documents to Elasticsearch
@app.on_event("startup")
def ingest_data():
    try:
        logging.info("Starting ingestion...")
        docs = load_data()
        es_vector_store.add_documents(documents=docs)
        logging.info("Documents added to vector store.")
    except Exception as e:
        logging.error("Error during ingestion: %s", e)
        raise

# / endpoint remains unchanged
@app.get("/")
def home():
    return {"message": "RAG with Elastic ELSER and Llama3.2 using Langchain is running!"}

# Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# /query endpoint using Langchain chain
@app.get("/query")
def query_llm(question: str):
    try:
        logging.info("Received query: %s", question)
        retriever = es_vector_store.as_retriever()
        template = """Answer the question based only on the following context:

{context}

Question: {question}"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        response = chain.invoke(question)
        return {"response": response}
    except Exception as e:
        logging.error("Error handling query: %s", e)
        return {"error": str(e)}

# Interactive Console Mode using Langchain chain
if __name__ == '__main__':
    try:
        logging.info("Starting ingestion (manual run) ...")
        ingest_data()
        logging.info("Ingestion completed. System is coming alive!")
    except Exception as e:
        logging.error("Failed to complete ingestion: %s", e)
        exit(1)
    print("Interactive Console Mode: Enter your questions below (type 'exit' to quit):")
    while True:
        try:
            question = input("Question: ").strip()
            if question.lower() in ['exit', 'quit']:
                logging.info("Exiting interactive console.")
                break
            logging.info("Processing question from console.")
            retriever = es_vector_store.as_retriever()
            template = """Answer the question based only on the following context:

{context}

Question: {question}"""
            prompt = ChatPromptTemplate.from_template(template)
            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            response = chain.invoke(question)
            print("Response:", response)
        except Exception as e:
            logging.error("Error during interactive query: %s", e)
            print("Error:", e)
