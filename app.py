import streamlit as st
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile
import requests

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

# Инициализация Ollama Chat
@st.cache_resource
def init_ollama_chat():
    try:
        llm = ChatOllama(
            model="llama3",  # Модель для ответов
            temperature=0.7,
            num_predict=2048,
            top_k=10,
            top_p=0.95,
            num_ctx=4096
        )
        return llm
    except Exception as e:
        st.error(f"Ошибка инициализации ChatOllama: {str(e)}")
        return None

# Инициализация Ollama Embeddings
@st.cache_resource
def init_ollama_embeddings():
    try:
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",  # Специализированная модель для эмбеддингов
        )
        return embeddings
    except Exception as e:
        st.error(f"Ошибка инициализации OllamaEmbeddings: {str(e)}")
        return None

# Проверяем Ollama
if not check_ollama():
    st.error("""
    ⚠️ **Ollama не запущена!**
    
    Пожалуйста, выполните:
    1. Запустите Ollama: `ollama serve` (в отдельном терминале)
    2. Скачайте модели: 
       - `ollama pull llama3` (для ответов)
       - `ollama pull nomic-embed-text` (для эмбеддингов)
    3. Обновите эту страницу
    """)
    st.stop()

# Инициализируем модели
llm = init_ollama_chat()
embeddings = init_ollama_embeddings()

if llm is None or embeddings is None:
    st.stop()

# Боковая панель
with st.sidebar:
    st.header("📁 Загрузка документа")
    
    # Информация о моделях
    st.subheader("🤖 Модели Ollama")
    st.info("""
    **Chat model:** llama3 (для ответов)
    **Embeddings model:** nomic-embed-text (для поиска)
    
    ✅ Оптимальная комбинация!
    """)
    
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
            if os.path.exists("./chroma_db_ollama"):
                shutil.rmtree("./chroma_db_ollama")
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
            
            # Разбиваем на chunks
            st.write("✂️ Разбивка на фрагменты...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            splits = text_splitter.split_documents(documents)
            st.write(f"✅ Создано {len(splits)} фрагментов")
            
            # СОЗДАЕМ ВЕКТОРНОЕ ХРАНИЛИЩЕ С NOMIC-EMBED-TEXT
            st.write("💾 Создание векторной базы данных с nomic-embed-text...")
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory="./chroma_db_ollama"
            )
            
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}  # Возвращаем 4 наиболее релевантных фрагмента
            )
            
            # Prompt для переформулирования вопроса с учетом истории
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", """Given a chat history and the latest user question 
                which might reference context in the chat history, formulate a standalone question 
                which can be understood without the chat history. Do NOT answer the question, 
                just reformulate it if needed and otherwise return it as is."""),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt
            )
            
            # Prompt для ответа на вопросы
            system_prompt = """You are a helpful customer support assistant. 
            Use the following pieces of retrieved context to answer the user's question.
            If you don't know the answer based on the context, say that you don't know.
            Be concise, friendly, and professional. Show empathy when users express frustration.
            
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
                lambda: st.session_state.chat_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
            
            st.session_state.vectorstore_initialized = True
            os.unlink(tmp_file_path)
            
            status.update(label="✅ Документ успешно обработан с nomic-embed-text!", state="complete")
            
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
        👋 **Привет! Я полностью локальный бот службы поддержки на Ollama!**
        
        **🔧 Текущая конфигурация:**
        - 💬 **Chat model:** `llama3` (для ответов)
        - 🔍 **Embeddings model:** `nomic-embed-text` (для поиска)
        - 📊 **Поиск:** 4 наиболее релевантных фрагмента
        - 💾 **Хранилище:** ChromaDB
        
        **📝 Что я умею:**
        - Отвечать на вопросы на основе загруженного PDF
        - Помнить историю разговора
        - Понимать контекст вопросов
        
        **⬅️ Загрузите PDF** в боковой панели, чтобы начать!
        """)

# Поле ввода
if st.session_state.vectorstore_initialized:
    user_input = st.chat_input("💬 Введите ваш вопрос...")
    
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
                        {"input": user_input}
                    )
                    
                    bot_response = response['answer']
                    st.markdown(bot_response)
                    
                    # Показываем использованные документы (опционально)
                    show_context = st.sidebar.checkbox("📚 Показать контекст", False)
                    if show_context and 'context' in response:
                        with st.expander("📚 Использованные фрагменты"):
                            for i, doc in enumerate(response['context']):
                                st.markdown(f"**Фрагмент {i+1}:**")
                                st.info(doc.page_content)
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