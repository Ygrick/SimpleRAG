import json
import logging

from langchain.retrievers import EnsembleRetriever

from .config import (ANSWER_GENERATION_PROMPT, CLIENT, DOC_RETRIEVAL_PROMPT,
                     LLM_MODEL)


def get_docs(query: str, retriever: EnsembleRetriever) -> str:
    """
    Поиск релевантных документов.

    Args:
        query (str): Вопрос пользователя.
        retriever (EnsembleRetriever): Комбинированный ретривер (FAISS + BM25).

    Returns:
        str: Релевантные документы в формате JSON.
    """
    
    # Поиск релевантных документов
    relevant_docs = retriever.invoke(query)
    
    # Преобразуем найденные документы в нужный формат
    relevant_docs_data = [
        {
            "document_id": doc.metadata.get("document_id", -1),
            "chunk_id": doc.metadata.get("chunk_id", -1),
            "content": doc.page_content
        }
        for doc in relevant_docs
    ]
    logging.info(f"Найдено {len(relevant_docs_data)} релевантных документов.")

    json_docs = json.dumps(relevant_docs_data, ensure_ascii=False)
    return json_docs


def get_llm_response(system_prompt: str, docs: str, query: str, temperature: float) -> str:
    """
    Отправляет запрос в LLM, используя заданный системный промпт.

    Args:
        system_prompt (str): Промпт, определяющий задачу для модели.
        docs (str): Документы в формате JSON, содержащие релевантную информацию.
        query (str): Вопрос пользователя.
        temperature (float): Температура ответа (насколько модель креативна)
    
    Returns:
        str: Ответ LLM.
    """
    # Формируем контекст для LLM
    chat_history = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'documents', 'content': docs},
        {'role': 'user', 'content': query}
    ]
    
    # Отправляем запрос в LLM
    response = CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=chat_history,
        temperature=temperature,
        max_tokens=2048
    ).choices[0].message.content
    
    return response


def get_answer(query: str, json_docs: str) -> str:
    """
    Основной RAG-конвейер: поиск документов + генерация ответа.

    Args:
        query (str): Вопрос пользователя.
        json_docs (str): Релевантные документы в формате JSON.

    Returns:
        str: Сгенерированный ответ RAG-системы.
    """
    logging.info(f"Запрос пользователя: {query}")
 
    try:
        # Первый запрос к LLM: получение списка ID релевантных документов
        logging.info("Запрос к LLM: получение идентификаторов документов...")
        id_docs_response = get_llm_response(
            system_prompt=DOC_RETRIEVAL_PROMPT, 
            docs=json_docs, 
            query=query,
            temperature=0.0, # Жёсткий режим для строгого ответа
        )
        logging.info(f"Полученные идентификаторы документов: {id_docs_response}")

        # Второй запрос к LLM: генерация финального ответа
        logging.info("Запрос к LLM: генерация финального ответа...")
        response = get_llm_response(
            system_prompt=ANSWER_GENERATION_PROMPT.format(retrieved_data=id_docs_response), 
            docs=json_docs, 
            query=query,
            temperature=0.3, # Разрешаем быть слегка креативными, для получения более живого ответа
        )
        logging.info(f"Ответ успешно получен: {response}")
    
    except Exception as e:
        logging.error(f"Произошла ошибка: {e}. Проверьте ваш token и подключение к LLM.")
        response = "Произошла ошибка."
    
    return response
