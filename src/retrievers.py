import logging
from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from .config import EMBEDDING_MODEL, DEVICE


def create_retriever(documents: List[Document]) -> EnsembleRetriever:
    """
    Создаёт EnsembleRetriever на основе FAISS и BM25.

    Args:
        documents (List[Document]): Список документов для индексации.

    Returns:
        EnsembleRetriever: Комбинированный ретривер для поиска.
    """
    logging.info("Создаём векторное представление документов...")

    # Загружаем модель эмбедингов
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE}
    )

    # FAISS retriever
    vector_store = FAISS.from_documents(documents, embedding_model)
    faiss_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 2}  # Количество возвращаемых документов
    )

    # BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 2  # Количество возвращаемых документов

    # Объединяем их в EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]  # Больше веса на FAISS
    )

    logging.info("EnsembleRetriever успешно создан.")
    return ensemble_retriever
