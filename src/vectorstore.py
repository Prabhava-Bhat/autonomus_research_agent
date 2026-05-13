import os
import hashlib
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStoreManager:
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.persist_directory = persist_directory
        # HuggingFace embeddings run fully locally — no API key needed.
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        os.makedirs(self.persist_directory, exist_ok=True)

        self.vector_store = Chroma(
            collection_name="research_agent_docs",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    # ------------------------------------------------------------------
    # ID helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_id(page_content: str) -> str:
        """Derive a stable ID from chunk content using MD5.

        ChromaDB treats IDs as primary keys: if a chunk with the same ID is
        added again the existing record is *updated* rather than duplicated.
        This prevents the silent inflation that occurred when users clicked
        "Ingest" multiple times with random uuid4() IDs.
        """
        return hashlib.md5(page_content.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add_documents(self, chunks: list) -> list[str]:
        """Add document chunks to the vector store, skipping exact duplicates."""
        if not chunks:
            print("No chunks to add.")
            return []

        # Stable, content-derived IDs — re-ingesting the same file is a no-op.
        ids = [self._chunk_id(chunk.page_content) for chunk in chunks]

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            self.vector_store.add_documents(documents=batch_chunks, ids=batch_ids)
            print(f"Added batch {i // batch_size + 1} ({len(batch_chunks)} chunks)")

        print("Finished adding documents to ChromaDB.")
        return ids

    def get_retriever(self, k: int = 4):
        """Return a LangChain retriever interface for the vector store."""
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )


if __name__ == "__main__":
    manager = VectorStoreManager()
    print("Vector store initialised successfully.")