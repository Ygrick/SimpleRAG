import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Датасеты с Hugging Face
DATASET = "neural-bridge/rag-dataset-1200"
SPLIT_DATASET = "test"

# Модели
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
LLM_MODEL = "meta-llama/llama-3.1-8b-instruct:free"

# API-конфигурация к LLM
CLIENT = OpenAI(
    base_url="https://openrouter.ai/api/v1", 
    api_key=os.getenv("TOKEN_OPENAI")
)

# Аппаратные параметры
DEVICE = "cuda"

DOC_RETRIEVAL_PROMPT = (
    "You are an assistant specialized in document retrieval. "
    "Your task is to identify the most relevant document IDs and chunk IDs from the provided documents. "
    "Return only a JSON object with the following format: "
    '{"relevant_documents": [{"document_id": <doc_id>, "chunk_id": <chunk_id>}, ...]}. '
    "Do not include any explanations or additional text."
)

ANSWER_GENERATION_PROMPT = (
    "You are an assistant that answers user questions based strictly on the provided documents. "
    "Use only the content from the relevant documents and chunks listed below: "
    "{retrieved_data} "
    "Now, generate a well-structured answer to the user's question."
    "Do not make up information. If the answer is unclear from the documents, say 'Insufficient information'."
)

