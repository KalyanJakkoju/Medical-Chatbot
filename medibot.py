import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

def load_llm(repo_id, token):
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=token,
        temperature=0.5,
        max_new_tokens=512
    )
    return ChatHuggingFace(llm=llm)

# Streamlit App UI
st.title("🩺 Medical Chatbot ")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Show chat history
for user_msg, bot_msg in st.session_state.chat_history:
    st.chat_message("user").markdown(user_msg)
    st.chat_message("assistant").markdown(bot_msg)

prompt = st.chat_input("Ask your medical query:")

if prompt:
    st.chat_message("user").markdown(prompt)

    HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"
    HF_TOKEN = os.environ.get("HF_TOKEN")

    vectorstore = get_vectorstore()
    llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)

    prompt_template = """You are a professional medical assistant. Use the provided information to answer the user's question, but do NOT ever mention that you are using "provided information" or "context" in your response.
    Answer the user's question directly and professionally. Do NOT use phrases like "Based on the information provided," "According to the context," or "The context mentions."
    Your response should look like it comes from your own professional medical knowledge.
    If you don't know the answer, just say that you don't know — don't try to make up an answer.
    Synthesize the information clearly and concisely, similar to ChatGPT.

    Context: {context}
    Question: {question}

    Helpful Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )

    response = qa_chain.invoke({
        "question": prompt,
        "chat_history": st.session_state.chat_history
    })

    answer = response["answer"]
    st.chat_message("assistant").markdown(answer)

    # Save to history
    st.session_state.chat_history.append((prompt, answer))
