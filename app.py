import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings  # Для multilingual-e5-large
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile
import requests
import torch

# Настройка страницы
st.set_page_config(
    page_title="Customer Support RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Customer Support RAG Chatbot (100% локальный)")
st.markdown("---")

# Инициализация сессии
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_history = ChatMessageHistory()
    st.session_state.vectorstore_initialized = False

# Проверка Ollama
def check_ollama():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

@st.cache_resource
def init_ollama_chat():
    try:
        llm = ChatOllama(
            model="gpt-oss:20b",
            temperature=0.7,
            num_predict=2048,
            top_k=40,
            top_p=0.9,
            num_ctx=8192,
            repeat_penalty=1.1,
            format="json",  # Для структурированных ответов
        )
        return llm
    except Exception as e:
        st.error(f"Ошибка инициализации ChatOllama: {str(e)}")
        return None

# Инициализация E5-large embeddings (мультиязычные)
@st.cache_resource
def init_e5_embeddings():
    try:
        # Проверяем наличие CUDA для ускорения
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            st.sidebar.success(f"✅ Используется GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.sidebar.info("ℹ️ Используется CPU (рекомендуется GPU для E5-large)")
        
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={
                'device': device,
                'torch_dtype': torch.float16 if device == "cuda" else torch.float32
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32  # Оптимизация для больших документов
            }
        )
        return embeddings
    except Exception as e:
        st.error(f"Ошибка инициализации E5 embeddings: {str(e)}")
        return None

# Функция для форматирования текста с префиксом для E5
def prepare_text_for_e5(text, is_query=False):
    """E5 требует специальные префиксы для запросов и документов"""
    if is_query:
        return f"query: {text}"
    else:
        return f"passage: {text}"

# Проверяем Ollama
if not check_ollama():
    st.error("""
    ⚠️ **Ollama не запущена!**
    
    Пожалуйста, выполните:
    1. Запустите Ollama: `ollama serve` (в отдельном терминале)
    2. Скачайте модель: `ollama pull gpt-oss:20b`
    3. Обновите эту страницу
    """)
    st.stop()

# Инициализируем модели
llm = init_ollama_chat()
embeddings = init_e5_embeddings()

if llm is None or embeddings is None:
    st.stop()

# Боковая панель
with st.sidebar:
    st.header("📁 Загрузка документа")
    
    # Информация о моделях
    st.subheader("🤖 Модели")
    st.info("""
    **Chat model:** `gpt-oss:20b` (20B параметров)
    **Embeddings:** `intfloat/multilingual-e5-large` (мультиязычная)
    
    🌍 Поддержка языков: 100+ языков
    📊 Размер модели: ~2.2 GB
    """)
    
    # Дополнительные настройки
    st.subheader("⚙️ Настройки поиска")
    k_results = st.slider("Количество фрагментов для поиска", min_value=2, max_value=10, value=5)
    chunk_size = st.slider("Размер фрагмента (токенов)", min_value=500, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Перекрытие фрагментов", min_value=0, max_value=500, value=200, step=50)
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Выберите PDF файл с информацией",
        type="pdf"
    )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Очистить историю", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = ChatMessageHistory()
            st.rerun()
    
    with col2:
        if st.button("🔄 Сбросить БД", use_container_width=True):
            import shutil
            if os.path.exists("./chroma_db_e5"):
                shutil.rmtree("./chroma_db_e5")
            st.session_state.vectorstore_initialized = False
            st.rerun()
    
    st.markdown("---")
    
    # Показать доступные модели
    if st.button("📋 Показать модели Ollama", use_container_width=True):
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                st.write("**Доступные модели:**")
                for model in models:
                    st.write(f"- {model['name']} ({model['size'] / 1e9:.1f} GB)")
        except:
            st.error("Не удалось получить список моделей")

# Обработка загруженного файла
if uploaded_file is not None and not st.session_state.vectorstore_initialized:
    with st.status("🔄 Обработка документа...", expanded=True) as status:
        try:
            # Сохраняем временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            st.write("📄 Загрузка PDF...")
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            st.write(f"✅ Загружено {len(documents)} страниц")
            
            # Разбиваем на chunks с учетом настроек
            st.write("✂️ Разбивка на фрагменты...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                length_function=len,
            )
            splits = text_splitter.split_documents(documents)
            st.write(f"✅ Создано {len(splits)} фрагментов")
            
            # Добавляем префиксы для E5 к документам
            st.write("🔧 Подготовка для E5 embeddings...")
            for doc in splits:
                doc.page_content = prepare_text_for_e5(doc.page_content, is_query=False)
            
            # СОЗДАЕМ ВЕКТОРНОЕ ХРАНИЛИЩЕ С E5-large
            st.write("💾 Создание векторной базы данных с multilingual-e5-large...")
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory="./chroma_db_e5",
                collection_metadata={"hnsw:space": "cosine"}  # E5 использует cosine similarity
            )
            
            # Функция для обработки запросов с префиксом
            class E5Retriever:
                def __init__(self, vectorstore, k=5):
                    self.vectorstore = vectorstore
                    self.k = k
                
                def get_relevant_documents(self, query):
                    # Добавляем префикс для запроса
                    query_with_prefix = prepare_text_for_e5(query, is_query=True)
                    return self.vectorstore.similarity_search(query_with_prefix, k=self.k)
            
            retriever = E5Retriever(vectorstore, k=k_results)
            
            # Prompt для переформулирования вопроса с учетом истории
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", """Given a chat history and the latest user question 
                which might reference context in the chat history, formulate a standalone question 
                which can be understood without the chat history. Do NOT answer the question, 
                just reformulate it if needed and otherwise return it as is.
                
                Important: The question might be in any language. Preserve the original language."""),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt
            )
            
            # Prompt для ответа на вопросы (мультиязычный)
            system_prompt = """You are a helpful customer support assistant. You can communicate in multiple languages.
            Use the following pieces of retrieved context to answer the user's question.
            If you don't know the answer based on the context, say that you don't know.
            Be concise, friendly, and professional. Show empathy when users express frustration.
            
            Important: Answer in the SAME LANGUAGE as the user's question.
            
            Context: {context}"""
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            st.session_state.conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                lambda session_id: st.session_state.chat_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
            
            st.session_state.vectorstore_initialized = True
            os.unlink(tmp_file_path)
            
            status.update(label="✅ Документ успешно обработан с multilingual-e5-large!", state="complete")
            
        except Exception as e:
            st.error(f"❌ Ошибка: {str(e)}")
            st.exception(e)

# Отображение чата
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Приветственное сообщение
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 **Привет! Я мультиязычный бот службы поддержки!**
        
        **🔧 Текущая конфигурация:**
        - 💬 **Chat model:** `gpt-oss:20b` (20B параметров)
        - 🔍 **Embeddings model:** `intfloat/multilingual-e5-large` (2.2GB)
        - 🌍 **Поддержка:** 100+ языков
        - 📊 **Поиск:** настраиваемое количество фрагментов
        - 💾 **Хранилище:** ChromaDB с cosine similarity
        
        **📝 Что я умею:**
        - Отвечать на вопросы на любом языке
        - Понимать контекст на разных языках
        - Работать с большими документами
        - Помнить историю разговора
        
        **⬅️ Загрузите PDF** в боковой панели, чтобы начать!
        """)

# Поле ввода
if st.session_state.vectorstore_initialized:
    user_input = st.chat_input("💬 Введите ваш вопрос на любом языке...")
    
    if user_input:
        # Добавляем сообщение пользователя
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Получаем ответ
        with st.chat_message("assistant"):
            with st.spinner("🤔 Думаю..."):
                try:
                    response = st.session_state.conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": "default"}}
                    )
                    
                    bot_response = response['answer']
                    st.markdown(bot_response)
                    
                    # Показываем использованные документы (опционально)
                    show_context = st.sidebar.checkbox("📚 Показать контекст", False)
                    if show_context and 'context' in response:
                        with st.expander("📚 Использованные фрагменты"):
                            for i, doc in enumerate(response['context']):
                                # Убираем префикс для отображения
                                content = doc.page_content.replace("passage: ", "")
                                st.markdown(f"**Фрагмент {i+1}:**")
                                st.info(content)
                                st.markdown("---")
                    
                except Exception as e:
                    bot_response = f"❌ Ошибка: {str(e)}"
                    st.error(bot_response)
                    st.exception(e)
        
        # Добавляем ответ в историю
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

else:
    # Отключаем ввод, пока не загружен PDF
    st.chat_input("💬 Сначала загрузите PDF в боковой панели...", disabled=True)
    
    # Показываем подсказку
    if not uploaded_file:
        col1, col2, col3 = st.columns(3)
        with col2:
            st.info("👈 Загрузите PDF файл в боковой панели, чтобы начать общение")
