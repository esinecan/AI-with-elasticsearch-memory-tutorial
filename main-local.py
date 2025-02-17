import logging
from fastapi import FastAPI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core import VectorStoreIndex, QueryBundle, Document, Settings
from llama_index.llms.ollama import Ollama
from urllib.request import urlopen
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# Enable debug logging for elasticsearch and urllib3 to show request/response payloads
logging.getLogger("elasticsearch").setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.DEBUG)

app = FastAPI()

# Updated Elasticsearch credentials for local installation
ES_ENDPOINT = "http://localhost:9200"
API_KEY = "YOUR KEY"
INDEX_NAME = "rag_index"

# Update vector_store instantiation to use es_endpoint instead of es_cloud_id
vector_store = ElasticsearchStore(
    es_endpoint=ES_ENDPOINT,
    es_api_key=API_KEY,
    index_name=INDEX_NAME,
    text_field="content",
    vector_field="content_vector"
)

# LLM setup
llm = Ollama(model="your-boss:latest")

# Embedding model
embed_model = OllamaEmbedding(model_name="llama3.2:latest")
# Global Embedding Registration
Settings.embed_model = embed_model

# Load dataset from the tutorial
def load_data():
    url = "https://raw.githubusercontent.com/elastic/elasticsearch-labs/main/datasets/workplace-documents.json"
    response = urlopen(url)
    workplace_docs = json.loads(response.read())

    documents = [
        Document(
            text=doc["content"],
            metadata={
                "name": doc["name"],
                "summary": doc["summary"],
                "rolePermissions": doc["rolePermissions"]
            }
        )
        for doc in workplace_docs
    ]
    logging.info("Loaded %d documents for ingestion.", len(documents))
    return documents

# Ingestion pipeline with updated transformations
@app.on_event("startup")
def ingest_data():
    try:
        logging.info("Starting ingestion pipeline...")
        docs = load_data()

        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=100),
                embed_model
            ],
            vector_store=vector_store
        )
        logging.info("Running ingestion pipeline with payload:\n%s", json.dumps({"num_documents": len(docs)}, indent=2)[:50])
        pipeline.run(documents=docs)

        index = VectorStoreIndex.from_vector_store(vector_store)
        app.state.query_engine = index.as_query_engine(llm=llm, similarity_top_k=10)
        logging.info("Ingestion pipeline completed successfully.")
    except Exception as e:
        logging.error("Error during ingestion pipeline: %s", e)
        raise

@app.get("/")
def home():
    return {"message": "LlamaIndex RAG with Elastic is running!"}

@app.get("/query")
def query_llm(question: str):
    try:
        logging.info("Received query: %s", question)
        query_embedding = Settings.embed_model.get_query_embedding(query=question)
        bundle = QueryBundle(query_str=question, embedding=query_embedding)
        rq_payload = json.dumps(bundle.__dict__, indent=2)[:50]
        logging.info("Elastic Request Payload (QueryBundle): %s", rq_payload)
        response = app.state.query_engine.query(bundle)
        # Use string representation for response payload
        resp_payload = str(response)[:50]
        logging.info("Query processed successfully. Elastic Response Payload: %s", resp_payload)
        return {"answer": str(response)}
    except Exception as e:
        logging.error("Error processing query: %s", e)
        return {"error": str(e)}

# Interactive Console Mode
if __name__ == '__main__':
    try:
        # Run ingestion manually if not using ASGI server
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
            query_embedding = Settings.embed_model.get_query_embedding(query=question)
            bundle = QueryBundle(query_str=question, embedding=query_embedding)
            rq_payload_console = json.dumps(bundle.__dict__, indent=2)[:50]
            logging.info("Console Elastic Request Payload (QueryBundle): %s", rq_payload_console)
            response = app.state.query_engine.query(bundle)
            answer = str(response)
            # Use string representation for response payload
            resp_payload_console = str(response)[:50]
            logging.info("Answer generated successfully. Elastic Response Payload: %s", resp_payload_console)
            print("Answer:", answer)
        except Exception as e:
            logging.error("Error during interactive query: %s", e)
            print("Error:", e)
