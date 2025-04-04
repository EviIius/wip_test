import os
import json
import re
import pickle
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from PyPDF2 import PdfReader
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import pandas as pd  # for CSV reading

# Logging Configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# LLaMA 2 Model Configuration
LLAMA_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info("Loading tokenizer and models...")
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
# Fix: Set pad_token as eos_token
tokenizer.pad_token = tokenizer.eos_token

# Use AutoModelForCausalLM for both embeddings and text generation
embedding_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL_NAME,
    device_map="auto", 
    load_in_8bit=True, 
    torch_dtype=torch.float16,
)
completion_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL_NAME,
    device_map="auto",
    load_in_8bit=True,
    torch_dtype=torch.float16,
)
logger.info("LLaMA 2 model loaded successfully.")


def get_embedding(text: str) -> List[float]:
    """Get normalized embeddings using LLaMA 2 model."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = embedding_model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Last hidden state
        embedding = hidden_states.mean(dim=1).squeeze()
        embedding = embedding / embedding.norm(p=2)
        embedding = embedding.cpu().numpy().tolist()
    return embedding


def get_embeddings_batch(texts: List[str], batch_size: int = 50) -> List[List[float]]:
    """Get embeddings for multiple texts."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = [get_embedding(text) for text in batch]
        embeddings.extend(batch_embeddings)
    return embeddings


def get_completion(
    messages: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    timeout: int = 30
) -> str:
    """Get completion with proper attention mask and GPU termination."""
    # Explicitly set a max_length for tokenization if you want truncation:
    input_data = tokenizer(
        messages,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,  # or whatever fits your GPU memory
    )
    input_ids = input_data.input_ids.to(device)
    attention_mask = input_data.attention_mask.to(device)

    with torch.no_grad():
        try:
            with torch.amp.autocast("cuda"):
                output = completion_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2,  # helps avoid repeated phrases
                    num_return_sequences=1,
                )
            generated_text = tokenizer.decode(
                output[0], skip_special_tokens=True
            )
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            del input_ids, output, attention_mask  # free memory
            return generated_text

        except torch.cuda.TimeoutError:
            logger.warning("Generation timed out.")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return "No response due to timeout."


@dataclass
class Document:
    content: str
    metadata: Dict[str, Any] = None


class DocumentProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"\.{2,}", "", text)
        text = re.sub(r"\s*\u2002\s*", " ", text)
        text = re.sub(r"\s+", " ", text)

        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if not line or all(c in ".-" for c in line):
                continue
            if line.startswith("=== Page"):
                lines.append(line)
                continue
            if re.match(r"^\d+[-–]\d+$", line):
                continue
            lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def load_pdf(file_path: str) -> str:
        text = ""
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                text += f"\n=== Page {page_num} ===\n{page_text}"
        print("Cleaning extracted text...")
        return DocumentProcessor.clean_text(text)

    @staticmethod
    def load_txt(file_path: str) -> str:
        """Simply read text from a .txt file."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        print("Cleaning extracted text (TXT)...")
        return DocumentProcessor.clean_text(text)

    @staticmethod
    def load_csv(file_path: str) -> str:
        """Load CSV and convert to a readable text. 
           This is a simple approach converting CSV contents into a text blob."""
        df = pd.read_csv(file_path, dtype=str)  # read everything as string
        # Convert entire dataframe to a string
        text = df.to_string(index=False)
        print("Cleaning extracted text (CSV)...")
        return DocumentProcessor.clean_text(text)

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        separators = ["\n=== Page", "\n\n", "\n", ". ", "? ", "! "]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )
        
        chunks = splitter.split_text(text)
        processed_chunks = []
        
        for chunk in chunks:
            chunk = DocumentProcessor.clean_text(chunk)
            if chunk.strip():
                processed_chunks.append(Document(content=chunk))
        
        print(f"Final number of chunks: {len(processed_chunks)}")
        return processed_chunks


def load_file(file_path: str) -> List[Document]:
    """
    Load a file (PDF, TXT, or CSV), clean the text, and chunk it into Document objects.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        text = DocumentProcessor.load_pdf(file_path)
    elif ext == ".txt":
        text = DocumentProcessor.load_txt(file_path)
    elif ext == ".csv":
        text = DocumentProcessor.load_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Chunk the text
    documents = DocumentProcessor.chunk_text(text)
    return documents


class VectorStore:
    def __init__(self, persist_directory: str = "rag_index"):
        self.index = None
        self.documents = []
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

    def _get_index_path(self) -> str:
        return os.path.join(self.persist_directory, "faiss.index")

    def _get_documents_path(self) -> str:
        return os.path.join(self.persist_directory, "documents.pkl")

    def load_local_index(self) -> bool:
        index_path = self._get_index_path()
        documents_path = self._get_documents_path()

        try:
            if os.path.exists(index_path) and os.path.exists(documents_path):
                self.index = faiss.read_index(index_path)
                with open(documents_path, "rb") as f:
                    self.documents = pickle.load(f)
                print(f"Loaded existing index with {len(self.documents)} documents")
                return True
        except Exception as e:
            print(f"Error loading index: {e}")
            self.index = None
            self.documents = []

        return False

    def save_local_index(self):
        if self.index is None or not self.documents:
            return
        try:
            faiss.write_index(self.index, self._get_index_path())
            with open(self._get_documents_path(), "wb") as f:
                pickle.dump(self.documents, f)
            print(f"Saved index with {len(self.documents)} documents")
        except Exception as e:
            print(f"Error saving index: {e}")

    def create_index(self, documents: List[Document], force_recreate: bool = False):
        if not force_recreate and self.load_local_index():
            return

        print("Creating new index...")
        self.documents = documents
        contents = [doc.content for doc in documents]
        embeddings = get_embeddings_batch(contents)

        embedding_dim = len(embeddings[0])
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(np.array(embeddings).astype("float32"))

        self.save_local_index()

    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index first.")

        query_embedding = get_embedding(query)
        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"), k
        )
        # Convert distances to similarity scores
        similarities = 1 / (1 + distances)

        return [
            (self.documents[i], similarities[0][idx])
            for idx, i in enumerate(indices[0])
        ]


class HybridSearcher:
    def __init__(self, persist_directory: str = "rag_index"):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.vector_store = VectorStore(persist_directory)

    def create_index(self, documents: List[Document]):
        self.documents = documents
        # Initialize TF-IDF
        self._initialize_tfidf()

        # Initialize vector store
        self.vector_store.create_index(documents)

    def _initialize_tfidf(self):
        """Initialize TF-IDF matrix"""
        contents = [doc.content for doc in self.documents]
        self.tfidf_matrix = self.vectorizer.fit_transform(contents)

        # Optionally save TF-IDF data
        tfidf_path = os.path.join(self.vector_store.persist_directory, "tfidf.pkl")
        with open(tfidf_path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "matrix": self.tfidf_matrix}, f)

    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """Hybrid search combining vector and keyword search."""
        # Vector search results
        vector_results = self.vector_store.search(query, k)

        # Keyword (TF-IDF) search
        query_vec = self.vectorizer.transform([query])
        keyword_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        keyword_indices = np.argsort(keyword_scores)[-k:][::-1]
        keyword_results = [
            (self.documents[i], keyword_scores[i]) for i in keyword_indices
        ]

        # Combine and deduplicate
        seen = set()
        combined_results = []
        for doc, score in vector_results + keyword_results:
            if doc.content not in seen:
                seen.add(doc.content)
                combined_results.append((doc, score))

        # Sort by score descending and return top k
        return sorted(combined_results, key=lambda x: x[1], reverse=True)[:k]


def load_documents(file_path):
    """
    Load a file (PDF, TXT, or CSV) into Documents, then create a new VectorStore index.
    """
    # Load and chunk document
    documents = load_file(file_path)
    
    # Create and return vector store
    vector_store = VectorStore()  # or HybridSearcher() if you want both vector + keyword
    vector_store.create_index(documents, force_recreate=True)
    print(f"Processed {len(documents)} chunks from file: {file_path}")
    
    return vector_store
