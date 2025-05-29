import os
import logging
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.callbacks import BaseCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings
from src.ocr import extract_text
from langchain_community.document_loaders import PyPDFLoader

os.makedirs(os.path.join("logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "app.log")),
        logging.StreamHandler()
    ]
)

grok_api_key = os.getenv("GROQ_API_KEY")
if not grok_api_key:
    st.error("❌ Missing GROQ API Token!")
    st.stop()

def manage_chat_history(func):
    def wrapper(self, *args, **kwargs):
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = "chatbot"
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you with your medical report?"}]
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])
        return func(self, *args, **kwargs)
    return wrapper

def display_msg(msg, author):
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

def configure_llm():
    llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.3, groq_api_key=grok_api_key)
    st.sidebar.success("✅ Active Model: LLaMA 4")
    return llm

def configure_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_file(file):
    folder = "tmp"
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getvalue())
    try:
        if file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
            text = extract_text(file_path)
        else:  # PDF
            loader = PyPDFLoader(file_path)
            text = "\n".join([page.page_content for page in loader.load()])
        return text
    finally:
        os.unlink(file_path)

def setup_retrieval_system(uploaded_files):
    docs = []
    for file in uploaded_files:
        text = process_file(file)
        if text:
            docs.append(Document(page_content=text))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embedding_model = configure_embedding_model()
    vector_db = FAISS.from_documents(documents=splits, embedding=embedding_model)
    return vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

def print_qa(question, answer):
    log_str = f"\nUsecase: MedicalChatbot\nQuestion: {question}\nAnswer: {answer}\n" + "-" * 50
    logging.info(log_str)

class MedicalChatbot:
    def __init__(self, uploaded_files=None):
        self.llm = configure_llm()
        self.uploaded_files = uploaded_files if uploaded_files else []

    @manage_chat_history
    def main(self):
        st.title("💬 Medical Diagnosis Chatbot 🌡️")
        st.markdown("<p style='color:#e0f7fa; font-size: 18px;'>Ask questions about your medical report or general health! 📋</p>", unsafe_allow_html=True)
        if not self.uploaded_files:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<p style='color:#e0f7fa'>Please upload a medical report using the sidebar to start chatting! 📤</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown('<div style="max-height: 70vh; overflow-y: auto; padding: 10px;">', unsafe_allow_html=True)
                if st.session_state.get("messages"):
                    for msg in st.session_state["messages"]:
                        st.chat_message(msg["role"]).write(msg["content"])
                st.markdown('</div>', unsafe_allow_html=True)

            with st.container():
                user_query = st.chat_input(placeholder="🔎 Ask something about your medical report!")
                if user_query:
                    retriever = setup_retrieval_system(self.uploaded_files)
                    display_msg(user_query, "user")
                    with st.chat_message("assistant"):
                        stream_container = st.empty()
                        stream_handler = StreamHandler(stream_container)
                        retrieved_docs = retriever.get_relevant_documents(user_query)
                        context = "\n".join([doc.page_content for doc in retrieved_docs])
                        prompt = f"Based on this context: {context}\n\nUser question: {user_query}\nAnswer:"
                        response = ""
                        for token in self.llm.stream(prompt):
                            token_text = next(iter(token.content.values())) if isinstance(token.content, dict) else token.content
                            response += token_text
                            stream_handler.on_llm_new_token(token_text)
                        stream_container.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        print_qa(user_query, response)

if __name__ == "__main__":
    obj = MedicalChatbot()
    obj.main()