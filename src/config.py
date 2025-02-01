import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Кэш ответов
ANSWER_CACHE_FILE = "./cache/answer_cache.json"

# Датасет с Hugging Face
DATASET: str = "neural-bridge/rag-dataset-1200"
SPLIT_DATASET: str = "test"

# Модель-LLM для генерации ответа
LLM_MODEL: str = "meta-llama/llama-3.1-8b-instruct"

# Модель эмбедингов для индексирования документов
EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-large"

# Модель-кросс-энкодера для переоценки документов
CROSS_ENCODER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# API-конфигурация к LLM
# Можно заменить на API к вашей LLM (к примеру, развернутую с помощью vLLM)
CLIENT = OpenAI(
    base_url="https://openrouter.ai/api/v1", 
    api_key=os.getenv("TOKEN_OPENAI")
)

# Создадим два промпта для уменьшения вероятности нерелевантного ответа
    # Промпт для LLM, который просит определить только релевантные документы
DOC_RETRIEVAL_PROMPT: str = (
    "You are an AI assistant specialized in document retrieval. "
    "Your task is to extract only the most relevant document IDs and chunk IDs from the provided documents. "
    "Strictly follow these rules: "
    "1. Return only a JSON object in this exact format: "
    '{"relevant_documents": [{"document_id": <doc_id>, "chunk_id": <chunk_id>}, ...]}. '
    "2. Do not modify, summarize, or explain the documents. "
    "3. Do not include any additional text, explanations, reasoning, or commentary. "
    "4. Do not return the document content, only IDs. "
    "5. If no relevant documents exist, return an empty JSON: {\"relevant_documents\": []}. "
    "6. Any deviation from these rules is strictly prohibited."
)
    # Промпт для LLM, который просит составить ответ только на релевантных документах
ANSWER_GENERATION_PROMPT: str = (
    "You are an assistant that answers user questions based strictly on the provided documents. "
    "Use only the content from the relevant documents and chunks listed below: "
    "{retrieved_data} "
    "Now, generate a well-structured answer to the user's question."
    "Do not make up information. If the answer is unclear from the documents, say 'Insufficient information'."
)

