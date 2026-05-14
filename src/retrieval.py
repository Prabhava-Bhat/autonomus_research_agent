from typing import List
from langchain_core.documents import Document

# Note: BaseRetriever was imported but never used as a base class in the
# original. Removed to keep the module clean.


class AdvancedRetriever:
    def __init__(self, vectorstore_manager, similarity_threshold: float = 0.3, k: int = 4):
        """Custom retriever that filters results by a similarity score threshold.

        Important — score direction:
        ---------------------------------------------------------------
        ChromaDB's LangChain wrapper (`similarity_search_with_relevance_scores`)
        should return *similarity* scores in [0, 1] where 1 = most similar.
        However, depending on the collection's distance metric configuration it
        can return raw *distance* scores instead (lower = more similar).

        Run the diagnostic below during development to confirm which direction
        your store uses before choosing a threshold:

            results = manager.vector_store.similarity_search_with_relevance_scores(
                "a test sentence", k=3
            )
            for doc, score in results:
                print(score, doc.page_content[:60])

        If the best match shows the LOWEST score, you are getting distances and
        should either reconfigure ChromaDB or invert the filter:
            `if score <= self.similarity_threshold`
        ---------------------------------------------------------------
        """
        self.vectorstore_manager = vectorstore_manager
        self.similarity_threshold = similarity_threshold
        self.k = k

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents and filter by similarity score."""
        results = self.vectorstore_manager.vector_store.similarity_search_with_relevance_scores(
            query, k=self.k
        )

        filtered_docs: List[Document] = []
        for doc, score in results:
            # Log raw scores so you can catch an inverted distance metric early.
            print(f"[Retriever] score={score:.4f} | source={doc.metadata.get('source', '?')}")

            if score >= self.similarity_threshold:
                doc.metadata["similarity_score"] = score
                filtered_docs.append(doc)

        return filtered_docs

    def get_context_string(self, query: str) -> str:
        """Return a formatted string of retrieved documents for LLM context."""
        docs = self.get_relevant_documents(query)

        if not docs:
            return "No relevant information found in the local knowledge base."

        context = ""
        for doc in docs:
            source = doc.metadata.get("source", "Unknown Source")
            score = doc.metadata.get("similarity_score", 0.0)
            context += f"\n[Source: {source} | Score: {score:.2f}]\n{doc.page_content}\n"

        return context