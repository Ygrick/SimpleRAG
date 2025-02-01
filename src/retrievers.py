import gc
import logging
from typing import List

import torch
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from .config import EMBEDDING_MODEL_NAME


def create_retriever(documents: List[Document]) -> EnsembleRetriever:
    """
    Создаёт EnsembleRetriever с использованием GPU для индексирования и CPU для обработки запросов.
    
    Этапы:
      1. Инициализируется модель эмбеддингов на GPU для вычисления эмбеддингов документов.
      2. Создаётся FAISS‑индекс с использованием этой модели.
      3. Модель для индексирования удаляется, и вызывается torch.cuda.empty_cache() для освобождения GPU‑памяти.
      4. Для вычисления эмбеддингов запроса и поиска создаётся модель на CPU, и векторное хранилище перенастраивается.
      5. Создаются FAISS‑retriever и BM25‑retriever, которые объединяются в EnsembleRetriever.
    
    Args:
        documents (List[Document]): Список документов для индексирования.
    
    Returns:
        EnsembleRetriever: Комбинированный ретривер для поиска.
    """
    logging.info("Инициализация модели эмбеддингов для индексирования на GPU...")
    # Модель для индексирования на GPU (для ускорения процесса)
    indexing_embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"}
    )
    
    # Создаём FAISS‑индекс с использованием модели на GPU
    vector_store = FAISS.from_documents(documents, indexing_embedding_model)
    
    # Удаляем объект модели для индексирования и освобождаем GPU‑память
    del indexing_embedding_model
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("GPU-память очищена после индексирования.")
    
    # Создаём модель эмбеддингов для запросов на CPU (для оптимизации использования видеопамяти)
    query_embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"}
    )
    # Переопределяем функцию вычисления эмбеддингов запроса
    vector_store.embedding_function = query_embedding_model.embed_query
    
    # Создаём FAISS ретривер с обновлённой функцией эмбеддингов
    faiss_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 2}
    )
    
    # Создаём BM25 ретривер
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 2
    
    # Объединяем ретриверы в EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]
    )
    
    logging.info("EnsembleRetriever успешно создан.")
    return ensemble_retriever
