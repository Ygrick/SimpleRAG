import json
import logging
import os

from .config import ANSWER_CACHE_FILE


def load_answer_cache() -> dict:
    """
    Загружает кэш ответов из файла.

    Returns:
        dict: Словарь кэша, где ключ — запрос, значение — ответ.
    """
    if os.path.exists(ANSWER_CACHE_FILE):
        try:
            with open(ANSWER_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Ошибка загрузки кэша ответов: {e}")
    return {}


def save_answer_cache(cache: dict) -> None:
    """
    Сохраняет кэш ответов в файл.

    Args:
        cache (dict): Словарь кэша.
    """
    try:
        os.makedirs(os.path.dirname(ANSWER_CACHE_FILE), exist_ok=True)
        with open(ANSWER_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.error(f"Ошибка сохранения кэша ответов: {e}")