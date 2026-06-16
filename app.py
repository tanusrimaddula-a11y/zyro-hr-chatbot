import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Zyro Dynamics HR Help Desk")
st.write("Ask questions about Zyro Dynamics HR policies.")

# Get API key from Streamlit Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

@st.cache_resource
def build_rag():

    # Load all PDFs from repository root
    loader = PyPDFDirectoryLoader(".")
    documents = loader.load()

    if len(documents) == 0:
        st.error("No PDF files found in repository.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    return retriever, llm


retriever, llm = build_rag()

question = st.chat_input("Ask an HR question...")

if question:

    with st.chat_message("user"):
        st.write(question)

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are Zyro Dynamics HR Assistant.

Use ONLY the context provided.

If the answer is not present in the context, reply exactly:

I can only answer questions based on Zyro Dynamics HR policy documents.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    with st.chat_message("assistant"):
        st.write(answer)

        with st.expander("📄 Sources"):
            for i, doc in enumerate(docs, start=1):
                source = doc.metadata.get("source", "Unknown")
                st.write(f"{i}. {source}")
