import os, json, re, pickle
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from PyPDF2 import PdfReader
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch

# Logging Configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# LLaMA 2 Model Configuration
LLAMA_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Tokenizer and Model
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

# Get Embedding Function (Updated)
def get_embedding(text: str) -> List[float]:
    """Get normalized embeddings using LLaMA 2 model"""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = embedding_model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Last hidden state
        embedding = hidden_states.mean(dim=1).squeeze()
        embedding = embedding / embedding.norm(p=2)
        embedding = embedding.cpu().numpy().tolist()
    return embedding

# Get Batch Embeddings
def get_embeddings_batch(texts: List[str], batch_size: int = 50) -> List[List[float]]:
    """Get embeddings for multiple texts."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = [get_embedding(text) for text in batch]
        embeddings.extend(batch_embeddings)
    return embeddings

# Document Class
@dataclass
class Document:
    content: str
    metadata: Dict[str, Any] = None

# Document Processor
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
        
        return processed_chunks

class VectorStore:
    def __init__(self, persist_directory: str = "rag_index"):
        self.index = None
        self.documents = []
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

    def create_index(self, documents: List[Document]):
        self.documents = documents
        contents = [doc.content for doc in documents]
        embeddings = get_embeddings_batch(contents)

        embedding_dim = len(embeddings[0])
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(np.array(embeddings).astype("float32"))

    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        query_embedding = get_embedding(query)
        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"), k
        )
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
        self._initialize_tfidf()
        self.vector_store.create_index(documents)

    def _initialize_tfidf(self):
        contents = [doc.content for doc in self.documents]
        self.tfidf_matrix = self.vectorizer.fit_transform(contents)

    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        vector_results = self.vector_store.search(query, k)
        query_vec = self.vectorizer.transform([query])
        keyword_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        keyword_indices = np.argsort(keyword_scores)[-k:][::-1]
        keyword_results = [
            (self.documents[i], keyword_scores[i]) for i in keyword_indices
        ]

        seen = set()
        combined_results = []
        for doc, score in vector_results + keyword_results:
            if doc.content not in seen:
                seen.add(doc.content)
                combined_results.append((doc, score))

        return sorted(combined_results, key=lambda x: x[1], reverse=True)[:k]