import os
import re
import pickle
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import faiss

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------------------------
# CONFIG: Modify to suit your environment/model
# ------------------------------------------------------------------------------
LLAMA_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
# Ensure pad_token is set
tokenizer.pad_token = tokenizer.eos_token

# Load Model for embeddings and completion (no 8-bit quantization, just float16 if GPU)
embedding_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
).eval()

completion_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
).eval()


@dataclass
class Document:
    """
    Simple wrapper for chunk text and any associated metadata.
    """
    content: str
    metadata: Dict[str, Any] = None


# ------------------------------------------------------------------------------
# DOCUMENT PROCESSOR
# ------------------------------------------------------------------------------
class DocumentProcessor:
    """
    Helper to load PDF, TXT, CSV files, then chunk them into Document objects.
    """

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
        from PyPDF2 import PdfReader

        text = ""
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                text += f"\n=== Page {page_num} ===\n{page_text}"
        return DocumentProcessor.clean_text(text)

    @staticmethod
    def load_txt(file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return DocumentProcessor.clean_text(text)

    @staticmethod
    def load_csv(file_path: str) -> str:
        df = pd.read_csv(file_path, dtype=str)  # read all columns as string
        return DocumentProcessor.clean_text(df.to_string(index=False))

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """
        Very basic chunking approach—splits text into blocks of 'chunk_size' 
        with 'chunk_overlap' overlap between consecutive chunks.
        """
        chunks = []
        start = 0
        end = chunk_size
        while start < len(text):
            chunk = text[start:end]
            if not chunk.strip():
                break
            doc = Document(content=chunk.strip())
            chunks.append(doc)

            # Overlap region
            start = end - chunk_overlap
            end = start + chunk_size
            if start < 0:
                start = 0

        return chunks

    @staticmethod
    def load_and_chunk_file(file_path: str) -> List[Document]:
        """
        Load a PDF/TXT/CSV, clean it, and chunk into Document objects.
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

        return DocumentProcessor.chunk_text(text)


# ------------------------------------------------------------------------------
# EMBEDDING & COMPLETION HELPERS
# ------------------------------------------------------------------------------
def get_embedding(text: str) -> List[float]:
    """
    Creates a single text embedding using the last hidden state 
    of LLaMA, normalized to unit length.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = embedding_model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # last hidden layer
        embedding = hidden_states.mean(dim=1).squeeze()
        embedding = embedding / embedding.norm(p=2)
    return embedding.cpu().numpy().tolist()


def get_embeddings_batch(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Get embeddings in batches, each text is embedded by calling get_embedding.
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = [get_embedding(text) for text in batch]
        embeddings.extend(batch_embeddings)
    return embeddings


def truncate_text(text, max_tokens=900):
    """
    Truncate text to a maximum number of tokens (simple split by whitespace).
    This is a rough approximation to avoid exceeding the model's token limit.
    """
    tokens = text.split()
    if len(tokens) > max_tokens:
        return " ".join(tokens[:max_tokens])
    return text


def get_completion(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """
    Generates text using the LLaMA model with basic sampling.
    """
    input_data = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,  # guard input length
    ).to(device)

    with torch.no_grad():
        output = completion_model.generate(
            **input_data,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def get_completion_fixed(prompt, max_new_tokens=512, temperature=0.7):
    """
    Use the provided get_completion function (from LatestHelp.py) to get an output,
    then remove any echo of the prompt. This helps prevent the model from repeating
    its input instructions.
    """
    # Call the original get_completion method.
    answer = get_completion(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    # Remove any echo: If the output includes the prompt (or a part of it),
    # strip it so that only the answer remains.
    if "Answer:" in answer:
        # Split on the last occurrence of "Answer:" and take the part after it.
        answer = answer.split("Answer:")[-1].strip()
    return answer


# ------------------------------------------------------------------------------
# VECTOR STORE (FAISS) - Original Logic
# ------------------------------------------------------------------------------
class VectorStore:
    """
    Stores embeddings of Documents in a FAISS index and allows for
    similarity search. This uses the original 'create_index' logic 
    that was working for you before.
    """
    def __init__(self, persist_directory: str = "rag_index"):
        self.index = None
        self.documents: List[Document] = []
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

    def _get_index_path(self) -> str:
        return os.path.join(self.persist_directory, "faiss.index")

    def _get_documents_path(self) -> str:
        return os.path.join(self.persist_directory, "documents.pkl")

    def load_local_index(self) -> bool:
        index_path = self._get_index_path()
        docs_path = self._get_documents_path()
        if os.path.exists(index_path) and os.path.exists(docs_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(docs_path, "rb") as f:
                    self.documents = pickle.load(f)
                print(f"Loaded existing index with {len(self.documents)} documents.")
                return True
            except Exception as e:
                print("Error loading index:", e)
        return False

    def save_local_index(self):
        if self.index is None or not self.documents:
            return
        try:
            faiss.write_index(self.index, self._get_index_path())
            with open(self._get_documents_path(), "wb") as f:
                pickle.dump(self.documents, f)
            print(f"Saved index with {len(self.documents)} documents.")
        except Exception as e:
            print("Error saving index:", e)

    def create_index(self, documents: List[Document], force_recreate: bool = False):
        if not force_recreate and self.load_local_index():
            return

        print("Creating new index...")
        self.documents = documents
        contents = [doc.content for doc in documents]

        # Original approach: get all embeddings at once (in batches, default=32)
        embeddings = get_embeddings_batch(contents)

        embedding_dim = len(embeddings[0])
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(np.array(embeddings).astype("float32"))

        self.save_local_index()

    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """
        Searches for similar documents. L2 distance is converted to similarity=1/(1+distance).
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index first.")

        query_embedding = get_embedding(query)
        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"), k
        )
        similarities = 1 / (1 + distances)

        results = []
        for i, idx in enumerate(indices[0]):
            doc = self.documents[idx]
            sim_score = similarities[0][i]
            results.append((doc, sim_score))
        return results


# ------------------------------------------------------------------------------
# HYBRID SEARCHER (Optional)
# ------------------------------------------------------------------------------
class HybridSearcher:
    """
    Example hybrid approach combining a TF-IDF search with a VectorStore search.
    """
    def __init__(self, persist_directory: str = "rag_index"):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.documents: List[Document] = []
        self.vector_store = VectorStore(persist_directory)

    def create_index(self, documents: List[Document]):
        self.documents = documents
        self._initialize_tfidf()
        self.vector_store.create_index(documents, force_recreate=True)

    def _initialize_tfidf(self):
        contents = [doc.content for doc in self.documents]
        self.tfidf_matrix = self.vectorizer.fit_transform(contents)

    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        # Vector-based results
        vector_results = self.vector_store.search(query, k)

        # TF-IDF results
        query_vec = self.vectorizer.transform([query])
        keyword_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        keyword_indices = np.argsort(keyword_scores)[-k:][::-1]
        keyword_results = [(self.documents[i], keyword_scores[i]) for i in keyword_indices]

        # Combine
        seen = set()
        combined_results = []
        for doc, score in (vector_results + keyword_results):
            if doc.content not in seen:
                seen.add(doc.content)
                combined_results.append((doc, score))

        # Sort by final score desc
        return sorted(combined_results, key=lambda x: x[1], reverse=True)[:k]