# 🔍 SimpleRAG: Retrieval-Augmented Generation Agent  

SimpleRAG — это лёгкий Retrieval-Augmented Generation (RAG) агент, который использует поиск по базе знаний для генерации точных и обоснованных ответов.  

## 🚀 Основные возможности  
- **Поиск релевантных документов** с помощью векторной базы данных FAISS и BM25 (EnsembleRetriever).  
- **Генерация ответов** с использованием малой LLM (meta-llama/llama-3.1-8b-instruct).  
- **Двуэтапный процесс**:  
  1. Определение **релевантных документов и чанков**.  
  2. Генерация ответа **только на основе найденных данных**.  
- **Простой UI** на Streamlit для удобного взаимодействия.  


## ⚙️ Установка и запуск  

### 1. Получение токена

- Перейти на сайт https://openrouter.ai/
- Зарегестрироваться и получить токен вида `"sk-or-v1-1d9..."`
- Создать `.env` файл с токеном по примеру `.env.example`

### 2. Установка зависимостей  
```bash
python -m venv .venv
source .venv/bin/activate  # Для Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Запуск Streamlit-приложения  
```bash
streamlit run app.py
```

## 🛠 Используемые технологии  

| Компонент      | Описание |
|---------------|----------|
| **LangChain** | Чанкинг, ретриверы, работа с документами |
| **FAISS**     | Векторный поиск (поиск похожих документов) |
| **BM25**      | Классический поиск по тексту |
| **PyTorch**   | Работа с моделью эмбедингов |
| **OpenAI API** | Взаимодействие с LLM |
| **Streamlit** | UI для взаимодействия с пользователем |

---

## Примеры запросов  


**Вопрос:**  
'What change has been observed in the banking industry in Hong Kong since the 2008 financial crisis?'

**Ответ:**  
Specifically, the following changes have been observed:

1. Banks are less generous about compensation.
2. Banks have reined in their spending and growth plans.
3. Banks are not encouraging bankers to innovate.
4. It's "not as fun as before" (Simon Loong).

These changes have led to a shift in the career choices of some bankers, with some starting their own businesses.



## TODO  

**1. Поддержка файловых форматов (DOCX, PDF)**  
- Добавить возможность загружать и обрабатывать документы в форматах **.docx** и **.pdf**, используя **LangChain Document loaders**

**2. Загрузка пользовательских файлов**  
 - Добавить в Streamlit-приложение возможность **загружать файлы** (PDF/DOCX/TXT).  
 - При загрузке файлы должны автоматически **обрабатываться** и **индексироваться** для поиска.  

**3. Проверка на галлюцинации LLM**  
- Добавить механизм проверки ответа на соответствие фактам из найденных документов - cравнивать **ключевые факты** из ответа с извлечённой информацией.   

**4. Интеграция с поисковыми системами**  
- Добавить возможность **искать информацию в интернете**, если данных в базе знаний недостаточно.  
 Возможные методы:  
   - **API поисковиков** (например, Google Search API, Bing API).  
   -  **LangChain Web Search Retriever**.

**5. Предоставление релевантных документов**  
- Добавить вывод документов, на основе которых составлен ответ. 

## 📩 Контакты  
Если у вас есть вопросы или предложения: 👨‍💻 **https://t.me/Ygrickkk**  

