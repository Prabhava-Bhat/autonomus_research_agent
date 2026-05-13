import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from .retrieval import AdvancedRetriever
from .scraper import WebScraper
from .vectorstore import VectorStoreManager
from .ingestion import DataIngestion

SYSTEM_PROMPT = """You are a professional autonomous research assistant.
Answer the user's question by reasoning step-by-step and using the available tools.

Available tools:
- query_local_knowledge_base(query): Search the local vector database. Use this FIRST.
- scrape_website(url): Scrape a URL for real-time information.

Use the following format EXACTLY and stop after Final Answer:

Thought: <your reasoning>
Action: <tool name>
Action Input: <tool input>
Observation: <tool result>
Thought: <your reasoning>
Final Answer: <your final answer to the user>

Rules:
- Always start with querying the local knowledge base.
- Only use scrape_website if given an explicit URL by the user.
- Do NOT repeat an action you already took.
- Always end with "Final Answer:" on its own line.
"""


class ResearchAgent:
    def __init__(self, llm_model: str = "llama3"):
        """Initialise the autonomous research agent with a manual ReAct loop.
        Works with any Ollama model, including those without native tool calling."""
        try:
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

    def _run_tool(self, action: str, action_input: str) -> str:
        action = action.strip()
        action_input = action_input.strip()
        if action == "query_local_knowledge_base":
            return self.retriever.get_context_string(action_input)
        if action == "scrape_website":
            doc = self.scraper.scrape_url(action_input)
            if doc:
                chunks = self.ingestion.process_and_chunk([doc])
                self.vectorstore_manager.add_documents(chunks)
                return (
                    f"Successfully scraped {action_input} and added {len(chunks)} chunks. "
                    "You can now query the knowledge base for information about it."
                )
            return f"Failed to scrape {action_input}."
        return f"Unknown tool: {action}"

    def run_query(self, query: str) -> str:
        if not self.llm:
            return "Agent is not initialised. Please ensure Ollama is running locally."

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]

        for _ in range(5):  # max iterations
            response = self.llm.invoke(messages)
            text = response.content

            # Check for Final Answer
            fa_match = re.search(r"Final Answer\s*:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
            if fa_match:
                return fa_match.group(1).strip()

            # Check for Action / Action Input
            action_match = re.search(r"Action\s*:\s*(.+)", text, re.IGNORECASE)
            input_match = re.search(r"Action Input\s*:\s*(.+)", text, re.IGNORECASE)

            if action_match and input_match:
                action = action_match.group(1).strip()
                action_input = input_match.group(1).strip()
                observation = self._run_tool(action, action_input)

                # Append assistant turn + observation and loop
                messages.append(AIMessage(content=text))
                messages.append(HumanMessage(content=f"Observation: {observation}"))
            else:
                # Model gave a plain answer without tool use
                return text.strip()

        return "I was unable to find a conclusive answer within the allowed steps."