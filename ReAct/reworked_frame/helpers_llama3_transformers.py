import os, json, re
from typing import List, Dict, Any, Tuple, Union
from dataclasses import dataclass
from PyPDF2 import PdfReader
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModel
import torch

# Load the local LLaMA model and tokenizer
MODEL_NAME = "path/to/local/llama-model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
model.eval()


def get_embedding(text: str) -> List[float]:
    """Get embeddings using LLaMA 3"""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        # Get the last layer hidden states
        hidden_states = outputs.hidden_states[-1]
        # Get the embedding for the [CLS] token or mean pooling
        embedding = hidden_states.mean(dim=1).squeeze().tolist()
    return embedding


def get_embeddings_batch(texts: List[str], batch_size: int = 50) -> List[List[float]]:
    """Get embeddings for multiple texts using LLaMA 3 with progress bar"""
    embeddings = []
    
    # Calculate total number of batches for progress bar
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    with tqdm(total=len(texts), desc="Generating embeddings") as pbar:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            for text in batch:
                inputs = tokenizer(text, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Get the last hidden state for the [CLS] token
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
                    batch_embeddings.append(embedding)
            embeddings.extend(batch_embeddings)
            pbar.update(len(batch))
    return embeddings



# Load LLaMA 3 model for text generation
generation_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3b")
generation_pipeline = pipeline("text-generation", model=generation_model, tokenizer=tokenizer)

def get_completion(
    messages: str,
    model: str = "llama-3b"
) -> Union[str, Dict]:
    """Get completion using LLaMA 3"""
    response = generation_pipeline(messages, max_length=100, do_sample=True, temperature=0.7)
    content = response[0]['generated_text']
    return content



@dataclass
class Document:
    """Document class with content and metadata"""
    content: str
    metadata: Dict[str, Any] = None

class DocumentProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by removing unwanted patterns and formatting"""
        
        # Remove repeated dots and spaces
        text = re.sub(r'\.{2,}', '', text)  # Remove multiple dots
        text = re.sub(r'\s*\u2002\s*', ' ', text)  # Remove special unicode spaces
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        
        # Remove empty lines and clean up page markers
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            # Skip empty lines or lines with just dots/dashes
            if not line or all(c in '.-' for c in line):
                continue
            # Clean up page markers but keep the page number
            if line.startswith('=== Page'):
                lines.append(line)
                continue
            # Remove lines that are just page numbers
            if re.match(r'^\d+[-–]\d+$', line):
                continue
            lines.append(line)
        
        return '\n'.join(lines)

    @staticmethod
    def load_pdf(file_path: str) -> str:
        """Load PDF content with improved cleaning"""
        text = ""
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            total_pages = len(reader.pages)
            
            with tqdm(total=total_pages, desc="Loading PDF pages") as pbar:
                for page_num, page in enumerate(reader.pages, 1):
                    page_text = page.extract_text()
                    text += f"\n=== Page {page_num} ===\n{page_text}"
                    pbar.update(1)
        
        print("Cleaning extracted text...")
        return DocumentProcessor.clean_text(text)

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Document]:
        """Split text into chunks with improved handling"""
        # Create custom separator for meaningful breaks
        separators = ["\n=== Page", "\n\n", "\n", ". ", "? ", "! "]
        
        print("Splitting text into initial chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False
        )
        
        # Split into initial chunks
        chunks = splitter.split_text(text)
        
        # Post-process chunks to ensure they start with complete sentences
        processed_chunks = []
        with tqdm(chunks, desc="Processing chunks", unit="chunk") as pbar:
            for chunk in pbar:
                # Clean any remaining formatting issues
                chunk = DocumentProcessor.clean_text(chunk)
                
                # Ensure chunks don't start mid-sentence unless they start with a page marker
                if not chunk.startswith('=== Page'):
                    # Find the first sentence boundary
                    sentence_boundaries = ['. ', '? ', '! ']
                    first_boundary = float('inf')
                    for boundary in sentence_boundaries:
                        pos = chunk.find(boundary)
                        if pos != -1 and pos < first_boundary:
                            first_boundary = pos
                    
                    # If we found a boundary and it's not too far in
                    if first_boundary < 100 and first_boundary != float('inf'):
                        chunk = chunk[first_boundary + 2:]
                
                if chunk.strip():  # Only add non-empty chunks
                    processed_chunks.append(Document(content=chunk))
                pbar.set_postfix({"total_chunks": len(processed_chunks)})
        
        print(f"Final number of chunks: {len(processed_chunks)}")
        return processed_chunks

class VectorStore:
    def __init__(self, persist_directory: str = "rag_index"):
        self.embeddings = []
        self.documents = []
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
    
    def save_local_index(self):
        """Save index and documents to disk using pickle"""
        if not self.embeddings or not self.documents:
            return
        
        try:
            np.save(os.path.join(self.persist_directory, "embeddings.npy"), self.embeddings)
            with open(os.path.join(self.persist_directory, 'documents.pkl'), 'wb') as f:
                pickle.dump(self.documents, f)
            print(f"Saved index with {len(self.documents)} documents")
        except Exception as e:
            print(f"Error saving index: {e}")

    def load_local_index(self) -> bool:
        """Load index and documents from disk"""
        try:
            self.embeddings = np.load(os.path.join(self.persist_directory, "embeddings.npy")).tolist()
            with open(os.path.join(self.persist_directory, 'documents.pkl'), 'rb') as f:
                self.documents = pickle.load(f)
            print(f"Loaded existing index with {len(self.documents)} documents")
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            self.embeddings = []
            self.documents = []
            return False

    def create_index(self, documents: List[Document], force_recreate: bool = False):
        """
        Create or load index
        Args:
            documents: List of documents to index
            force_recreate: If True, recreate index even if it exists
        """
        # Try to load existing index unless force_recreate is True
        if not force_recreate and self.load_local_index():
            return
        
        # Create new index
        print("Creating new index...")
        self.documents = documents
        
        # Get embeddings
        contents = [doc.content for doc in documents]
        embeddings = get_embeddings_batch(contents)
        self.embeddings = np.array(embeddings)
        
        # Save to disk
        self.save_local_index()

    def search(
        self,
        query: str,
        k: int = 3
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents using cosine similarity"""
        if not self.embeddings:
            raise ValueError("Index not initialized. Call create_index first.")
        
        query_embedding = get_embedding(query)
        
        # Compute cosine similarity
        query_embedding = np.array(query_embedding)
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k results
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [(self.documents[i], similarities[i]) for i in top_k_indices]


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
        with open(tfidf_path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'matrix': self.tfidf_matrix
            }, f)

    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """Hybrid search combining vector and keyword search"""
        # Get vector search results
        vector_results = self.vector_store.search(query, k)
        
        # Get keyword search results
        query_vec = self.vectorizer.transform([query])
        keyword_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        keyword_indices = np.argsort(keyword_scores)[-k:][::-1]
        keyword_results = [(self.documents[i], keyword_scores[i])
                          for i in keyword_indices]
        
        # Combine and deduplicate
        seen = set()
        combined_results = []
        for doc, score in vector_results + keyword_results:
            if doc.content not in seen:
                seen.add(doc.content)
                combined_results.append((doc, score))
        
        return sorted(combined_results, key=lambda x: x[1], reverse=True)[:k]
