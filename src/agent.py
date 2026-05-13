import os
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool
from .retrieval import AdvancedRetriever
from .scraper import WebScraper
from .vectorstore import VectorStoreManager
from .ingestion import DataIngestion

# LangChain 1.x agents use a system prompt to define behavior.
SYSTEM_PROMPT = """You are a professional autonomous research assistant. 
Your goal is to provide accurate answers by using your available tools.

CRITICAL INSTRUCTIONS:
1. If a question requires specific knowledge, ALWAYS use the 'query_local_knowledge_base' tool first.
2. If the local database doesn't have enough info, use 'scrape_website' if a URL is provided or relevant.
3. Once you have the information from a tool, summarize it and provide a final answer to the user.
4. Do NOT output raw JSON tool calls as your final answer. Use the tools, then answer in plain text.
"""


class ResearchAgent:
    def __init__(self, llm_model: str = "llama3"):
        """Initialise the autonomous research agent using LangChain 1.x and Ollama."""
        try:
            # ChatOllama supports tool calling natively in modern versions.
            self.llm = ChatOllama(model=llm_model, temperature=0)
        except Exception as e:
            print(f"Failed to initialise LLM: {e}. Please ensure Ollama is running.")
            self.llm = None

        self.vectorstore_manager = VectorStoreManager()
        self.retriever = AdvancedRetriever(
            self.vectorstore_manager, similarity_threshold=0.3, k=4
        )
        self.scraper = WebScraper()
        self.ingestion = DataIngestion()

        self.agent = self._setup_agent()

    def _setup_agent(self):
        if not self.llm:
            return None

        # Define tools using the modern @tool decorator
        @tool
        def query_local_knowledge_base(query: str) -> str:
            """Search the local vector database for information on a topic. 
            Always try this tool FIRST before scraping the web.
            """
            return self.retriever.get_context_string(query)

        @tool
        def scrape_website(url: str) -> str:
            """Scrape a specific URL to obtain real-time or external information.
            Input must be a valid URL starting with http:// or https://.
            """
            doc = self.scraper.scrape_url(url)
            if doc:
                chunks = self.ingestion.process_and_chunk([doc])
                self.vectorstore_manager.add_documents(chunks)
                return (
                    f"Successfully scraped {url} and added {len(chunks)} chunks to the "
                    "knowledge base. You can now query the database for information about it."
                )
            return f"Failed to scrape {url}."

        tools = [query_local_knowledge_base, scrape_website]

        # create_agent in 1.x returns a CompiledGraph (which replaces AgentExecutor)
        return create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
        )

    def run_query(self, query: str) -> str:
        if not self.agent:
            return "Agent is not initialised. Please ensure Ollama is running locally."

        try:
            # Modern agents take a messages list.
            result = self.agent.invoke({"messages": [("user", query)]})
            # The final response is the content of the last message in the output state.
            return result["messages"][-1].content
        except Exception as e:
            return f"An error occurred during agent execution: {e}"