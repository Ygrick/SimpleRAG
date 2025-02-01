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

### 2. Без Docker-контейнера
#### 2.1 Установка зависимостей  
```bash
python -m venv .venv
source .venv/bin/activate  # Для Windows: .venv\Scripts\activate
pip install -r requirements.txt 
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 # или другую необходимую cuda-версию для torch
```

#### 2.2 Запуск Streamlit-приложения  
```bash
streamlit run app.py
```
* Первый запуск может быть долгим, ~5-10 минут (зависит от конфигурации вашего ПК и скорости интернета)

### 3. Используя Docker-контейнер
#### 3.1 Сборка образа 
```bash
docker build -t simple-rag .
```

#### 3.2 Запуск контейнера
```bash
docker run --gpus all -p 8501:8501 simple-rag
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
```
What change has been observed in the banking industry in Hong Kong since the 2008 financial crisis?
```
**Ответ:**  
```
Specifically, the following changes have been observed:

1. Banks are less generous about compensation.
2. Banks have reined in their spending and growth plans.
3. Banks are not encouraging bankers to innovate.
4. It's "not as fun as before" (Simon Loong).

These changes have led to a shift in the career choices of some bankers, with some starting their own businesses.
```


## TODO  

1. Добавить возможность загружать и обрабатывать документы в форматах **.docx** и **.pdf**, используя **LangChain Document loaders** (Сложность: **easy** 🔵🔵);

2. Провести различные эксперименты с промптами, LLM-моделями, весами для ретриверов, галлюцинациями, эмбединг-моделями (Сложность: **medium** 🟡🟡🟡);

3. Проверка на галлюцинации LLM - cравнивать **ключевые факты** из ответа с извлечённой информацией (Сложность: **very easy** 🟢);  

4. Добавить возможность искать информацию в интернете через `Langchain Community GoogleSearchAPIWrapper` (Сложность: **easy** 🔵🔵); 

5. Добавить вывод документов, на основе которых составлен ответ (Сложность: **very easy** 🟢);

6. Добавить более строгую очистку текста (Сложность: **very easy** 🟢).

7. ✅ ~~Добавить `CrossEncoder` (к примеру, `cross-encoder/ms-marco-MiniLM-L-6-v2`) для дополнительного переранжирования документов (Сложность: **easy** 🔵🔵);~~

8. ✅ ~~Добавить кэширование ответов для оптимизации системы (Сложность: **very easy** 🟢);~~

9. ✅ ~~Оптимизировать использования видеопамяти за счёт перевода `Embedding-модели` на `CPU` после индексации документов (Сложность: **very easy** 🟢).~~

## 📩 Контакты  
Если у вас есть вопросы или предложения: 👨‍💻 **https://t.me/Ygrickkk**  

