# ElasticRAG (USABLE)

ElasticRAG is a project designed to provide a flexible and scalable solution for integrating Elasticsearch with LlamaIndex for Retrieval-Augmented Generation (RAG) applications.

## Features

- Integration with Elasticsearch for scalable vector storage
- Support for LlamaIndex and Langchain for advanced query processing
- FastAPI-based API for easy deployment and interaction

## Installation

To install the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/elasticrag.git
    ```
2. Navigate to the project directory:
    ```bash
    cd elasticrag
    ```
3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use the project, follow these steps:

1. Start the development server:
    ```bash
    uvicorn main:app --reload
    ```
2. Open your browser and navigate to `http://localhost:8000`.

## Endpoints

- `GET /` - Home endpoint to check if the server is running.
- `GET /query` - Endpoint to query the LLM with a question. Example:
    ```bash
    curl -X GET "http://localhost:8000/query?question=Your+question+here"
    ```

## Contributing

We welcome contributions! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature-branch
    ```
3. Make your changes and commit them:
    ```bash
    git commit -m "Description of your changes"
    ```
4. Push to the branch:
    ```bash
    git push origin feature-branch
    ```
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
