import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------- CONFIG ----------------

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

load_dotenv()

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

.stChatMessage {
    border-radius: 15px;
    padding: 12px;
}

.title {
    text-align:center;
    font-size:3rem;
    font-weight:bold;
    color:white;
}

.subtitle{
    text-align:center;
    color:#94a3b8;
    margin-bottom:30px;
}

.glass {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(15px);
    border-radius:20px;
    padding:20px;
    border:1px solid rgba(255,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown(
    """
    <div class='title'>🤖 RAG Knowledge Assistant</div>
    <div class='subtitle'>
        Ask questions from your vector database powered by Mistral AI
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD MODELS ----------------

@st.cache_resource
def load_components():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10
        }
    )

    llm = ChatMistralAI(
        model="mistral-small-2506"
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant that answers questions based on the provided context."
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}"
        )
    ])

    return retriever, llm, prompt


retriever, llm, prompt = load_components()

# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown("## ⚙️ Settings")

    st.markdown("""
    ### Features
    ✅ Mistral AI

    ✅ ChromaDB

    ✅ MMR Retrieval

    ✅ Semantic Search

    ✅ RAG Pipeline
    """)

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- CHAT HISTORY ----------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- USER INPUT ----------------

query = st.chat_input("Ask anything from your knowledge base...")

if query:

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner("Searching knowledge base..."):

            docs = retriever.invoke(query)

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            final_prompt = prompt.invoke({
                "context": context,
                "question": query
            })

            result = llm.invoke(final_prompt)

            answer = result.content

            st.markdown(answer)

            with st.expander("📚 Retrieved Context"):

                for i, doc in enumerate(docs, 1):
                    st.markdown(f"### Chunk {i}")
                    st.write(doc.page_content[:1000])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )