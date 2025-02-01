import logging

import streamlit as st
from datasets import load_dataset

from src.caching import load_answer_cache, save_answer_cache
from src.chunking import chunk_documents
from src.config import DATASET, SPLIT_DATASET
from src.rag_pipeline import get_answer, get_docs
from src.retrievers import create_ensemble_retriever, create_reranked_retriever

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Заголовок приложения
st.title("RAG-Агент: Поиск и Генерация Ответов")

# Инициализация данных и моделей (один раз при запуске)
@st.cache_resource
def load_retriever():
    st.info("Загружаем и индексируем документы...")

    # Загружаем текстовые документы из Hugging Face
    rag_dataset = load_dataset(DATASET, split=SPLIT_DATASET)
    documents = rag_dataset["context"]
    
    # Чанкуем и создаём ретривер
    chunked_docs = chunk_documents(documents)
    ensemble_retriever = create_ensemble_retriever(chunked_docs)
    reranked_retriever = create_reranked_retriever(ensemble_retriever, top_n=3)
    return reranked_retriever

retriever = load_retriever()

# Ввод вопроса пользователем
query = st.text_input("Введите ваш вопрос:")

if query:
    # Обращение к RAG
    st.info("Обрабатываем запрос...")

    # Загружаем кэш ответов
    cache = load_answer_cache()
    if query in cache:
        logging.info("Ответ найден в кэше, возвращаем кэшированный ответ.")
        answer = cache[query]
    
    else:
        logging.info("Ответ не найден в кэше, генерируем ответ с нуля.")
        # Поиск релевантных документов
        relevant_json_docs = get_docs(query, retriever)
        
        # Генерация ответа
        answer = get_answer(query, relevant_json_docs)
        
        # Если ошибки не произошло, то сохраняем ответ в кэш
        if answer != "Произошла ошибка.":
            # Обновляем кэш ответов
            cache[query] = answer
            save_answer_cache(cache)

    # Вывод ответа
    st.success("Ответ:")
    st.write(answer)
