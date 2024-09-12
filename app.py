import streamlit as st
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatMessagePromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")  # Fixed typo
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Setup streamlit
st.title("Conversational RAG with PDF uploads and Chat History")
st.write("Upload Pdf's and chat with their content")

api_key = st.text_input("Enter your Groq API Key:", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model="Gemma2-9b-It")
    session_id = st.text_input("Session ID", value="default_session")

    # Manage chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False)

    # Process Uploaded PDF
    if uploaded_file:
        documents = []

        temppdf = f"./temp.pdf"

        with open(temppdf, "wb") as file:
            file.write(uploaded_file.getvalue())  # Directly write the bytes
            file_name = uploaded_file.name

        # Load the PDF and split documents
        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)

        # Split and create embedding for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)

        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            "Given a Chat History and the latest user question, "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do not answer the question, "
            "just reformulate it if needed, and otherwise return it as is."
        )

        # Fixing how we format messages
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Answer Question
        system_prompt = (
            "You are an assistant for question answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder('chat_history'),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Session history management
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # User input and response generation
        user_input = st.text_input("Your Question")

        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {'input': user_input},
                config={
                    "configurable": {"session_id": session_id}
                }
            )

            # Displaying the results
            st.write(st.session_state.store)  # Show stored session state
            st.success(f"Assistant: {response['answer']}")  # Display answer
            st.write("Chat History:", session_history.messages)  # Show chat history

else:
    st.warning("Hey, please enter your Groq API key")
