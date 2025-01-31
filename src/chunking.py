import logging
import re
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    """
    Очистка текста перед разбиением на чанки.

    Args:
        text (str): Исходный текст.

    Returns:
        str: Очищенный текст.
    """
    text = re.sub(r'(\r\n|\r|\n){2,}', r'\n', text)  # Удаляем лишние пустые строки
    text = re.sub(r'[ \t]+', ' ', text)  # Заменяем табуляции на пробелы
    return text.strip()


def chunk_documents(documents: List[str], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    """
    Разбивает документы на чанки.

    Args:
        documents (List[str]): Список документов.
        chunk_size (int): Размер чанка в символах.
        chunk_overlap (int): Перекрытие чанков.

    Returns:
        List[Document]: Разбитые на чанки документы.
    """
    logging.info(f"Разбиваем {len(documents)} документов на чанки (размер: {chunk_size}, перекрытие: {chunk_overlap})")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_documents = []

    # Предобработка всех текстовых файлов\документов\текстов
    for i, doc in enumerate(documents):
        clean_doc = clean_text(doc)
        # Создадим langchain-документ (совместимый формат для RecursiveCharacterTextSplitter)
        langchain_doc = Document(page_content=clean_doc)
        # Разобьём длинный текст на чанки
        chunks = text_splitter.split_documents([langchain_doc])

        # Добавим номер документа (текста) и номер чанка как доп.информацию
        # И добавим сам чанк в список всех чанков
        for j, chunk in enumerate(chunks):
            chunk.metadata["document_id"] = i + 1
            chunk.metadata["chunk_id"] = j + 1
            chunked_documents.append(chunk)

    logging.info(f"Создано {len(chunked_documents)} чанков")
    return chunked_documents
