import os
import glob
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DataIngestion:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    def load_single_text_file(self, file_path: str):
        """Load a single text file and return a list of Documents."""
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()

    def load_directory(self, directory_path: str, glob_pattern: str = "**/*.txt"):
        """Load all matching files in a directory."""
        loader = DirectoryLoader(
            directory_path,
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        return loader.load()

    def load_pdf(self, file_path: str):
        """Load a single PDF file and return a list of Documents."""
        loader = PyMuPDFLoader(file_path)
        return loader.load()

    def process_and_chunk(self, documents: list) -> list:
        """Split a list of Document objects into chunks."""
        return self.text_splitter.split_documents(documents)

    def ingest_data_folder(self, folder_path: str) -> list:
        """Ingest all supported files (.txt, .pdf) in a folder and chunk them."""
        documents = []

        # Load text files
        try:
            txt_docs = self.load_directory(folder_path, glob_pattern="**/*.txt")
            documents.extend(txt_docs)
        except Exception as e:
            print(f"Error loading text files: {e}")

        # Load PDF files manually (DirectoryLoader can be unreliable with mixed types)
        pdf_files = glob.glob(os.path.join(folder_path, "**/*.pdf"), recursive=True)
        for pdf_file in pdf_files:
            try:
                pdf_docs = self.load_pdf(pdf_file)
                documents.extend(pdf_docs)
            except Exception as e:
                print(f"Error loading PDF {pdf_file}: {e}")

        if not documents:
            print("No documents found.")
            return []

        chunks = self.process_and_chunk(documents)
        print(f"Produced {len(chunks)} chunks from {len(documents)} document(s).")
        return chunks


if __name__ == "__main__":
    ingestion = DataIngestion()
    chunks = ingestion.ingest_data_folder("data/sample_docs")
    print(f"Generated {len(chunks)} chunks.")