import tempfile
import os
import streamlit as st
from src.agent import ResearchAgent
from src.ingestion import DataIngestion
from src.vectorstore import VectorStoreManager
from src.retrieval import AdvancedRetriever
from src.scraper import WebScraper

st.set_page_config(
    page_title="Autonomous Research Assistant",
    layout="wide",
    page_icon="🕵️",
)


# ---------------------------------------------------------------------------
# Cached singletons
# ---------------------------------------------------------------------------

@st.cache_resource
def init_agent() -> ResearchAgent:
    return ResearchAgent(llm_model="llama3")


@st.cache_resource
def init_ingestion() -> DataIngestion:
    return DataIngestion()


@st.cache_resource
def init_vectorstore() -> VectorStoreManager:
    return VectorStoreManager()


agent = init_agent()
ingestion = init_ingestion()
vectorstore = init_vectorstore()
# Retriever used independently for source attribution display
retriever = AdvancedRetriever(vectorstore, similarity_threshold=0.3, k=4)


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("🕵️ Autonomous Research Assistant")
st.markdown(
    "This system combines **RAG** with an **Autonomous Agent** powered by a local "
    "Ollama model. It searches your knowledge base first, then scrapes the web if needed."
)


# ---------------------------------------------------------------------------
# Sidebar — control panel
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Control Panel")

    # --- 1. Ingest sample docs ---
    st.subheader("1. Ingest Sample Documents")
    if st.button("📂 Ingest from data/sample_docs"):
        with st.spinner("Ingesting data/sample_docs …"):
            chunks = ingestion.ingest_data_folder("data/sample_docs")
            if chunks:
                vectorstore.add_documents(chunks)
                st.success(f"Added {len(chunks)} chunks to the knowledge base.")
            else:
                st.warning("No documents found in data/sample_docs.")

    st.divider()

    # --- 2. Upload files ---
    st.subheader("2. Upload Files")
    uploaded_files = st.file_uploader(
        "Upload TXT or PDF files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        help="Files are chunked and added to the local vector database.",
    )

    if st.button("📥 Ingest Uploaded Files") and uploaded_files:
        all_chunks = []
        with st.spinner("Processing uploads …"):
            for uploaded_file in uploaded_files:
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    if suffix.lower() == ".pdf":
                        docs = ingestion.load_pdf(tmp_path)
                    else:
                        docs = ingestion.load_single_text_file(tmp_path)

                    # Restore original filename as source metadata
                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name

                    chunks = ingestion.process_and_chunk(docs)
                    all_chunks.extend(chunks)
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                finally:
                    os.unlink(tmp_path)

        if all_chunks:
            vectorstore.add_documents(all_chunks)
            st.success(f"Added {len(all_chunks)} chunks from {len(uploaded_files)} file(s).")
        else:
            st.warning("No content could be extracted from the uploaded files.")

    st.divider()

    # --- 3. Scrape a URL ---
    st.subheader("3. Scrape a Website")
    url_input = st.text_input(
        "Enter a URL to scrape",
        placeholder="https://example.com/article",
    )
    if st.button("🌐 Scrape & Ingest URL"):
        if url_input.startswith("http"):
            with st.spinner(f"Scraping {url_input} …"):
                scraper = WebScraper()
                doc = scraper.scrape_url(url_input)
                if doc:
                    chunks = ingestion.process_and_chunk([doc])
                    vectorstore.add_documents(chunks)
                    st.success(
                        f"Scraped **{doc.metadata.get('title', url_input)}** "
                        f"and added {len(chunks)} chunks."
                    )
                else:
                    st.error("Failed to scrape the URL. Check the address and try again.")
        else:
            st.warning("Please enter a valid URL starting with http:// or https://")

    st.divider()

    # --- 4. Settings ---
    st.subheader("4. Settings")
    st.info(
        "Running locally via **Ollama**. "
        "Ensure `ollama serve` is active and `llama3` is pulled:\n\n"
        "```\nollama pull llama3\n```"
    )


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.header("💬 Chat with the Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Re-render source attribution stored alongside the message
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Sources used", expanded=False):
                for src in message["sources"]:
                    st.markdown(
                        f"- **{src['source']}** &nbsp; "
                        f"<span style='color:grey;font-size:0.85em;'>score: {src['score']:.2f}</span>",
                        unsafe_allow_html=True,
                    )

# New message
if prompt := st.chat_input("Ask a research question …"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking … (the agent may query the DB or scrape the web)"):
            response = agent.run_query(prompt)

        st.markdown(response)

        # --- Source attribution ---
        # Run retriever independently so we can show which chunks influenced
        # the answer, with their source filenames and similarity scores.
        source_docs = retriever.get_relevant_documents(prompt)
        sources = []
        seen = set()
        for doc in source_docs:
            src_name = doc.metadata.get("source", "Unknown")
            score = doc.metadata.get("similarity_score", 0.0)
            key = (src_name, round(score, 2))
            if key not in seen:
                seen.add(key)
                sources.append({"source": src_name, "score": score})

        if sources:
            with st.expander("📚 Sources used", expanded=False):
                for src in sources:
                    st.markdown(
                        f"- **{src['source']}** &nbsp; "
                        f"<span style='color:grey;font-size:0.85em;'>score: {src['score']:.2f}</span>",
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "sources": sources}
    )