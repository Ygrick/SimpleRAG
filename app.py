import logging

import streamlit as st
from datasets import load_dataset

from src.chunking import chunk_documents
from src.config import DATASET, SPLIT_DATASET
from src.rag_pipeline import get_answer
from src.retrievers import create_retriever


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
    return create_retriever(chunked_docs)

retriever = load_retriever()

# Ввод вопроса пользователем
query = st.text_input("Введите ваш вопрос:")

if query:
    # Обращение к RAG
    st.info("Обрабатываем запрос...")
    answer = get_answer(query, retriever)

    # Вывод ответа
    st.success("Ответ:")
    st.write(answer)
