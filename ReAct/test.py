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

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load the local LLaMA model and tokenizer
MODEL_NAME = "path/to/local/llama-model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(DEVICE)
model.eval()

def get_embedding(text: str) -> List[float]:
    """Get embeddings using LLaMA 3 with GPU support"""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states[-1]
        embedding = hidden_states.mean(dim=1).squeeze().cpu().tolist()
    return embedding

def get_embeddings_batch(texts: List[str], batch_size: int = 50) -> List[List[float]]:
    """Get embeddings for multiple texts using LLaMA 3 with GPU support and batch processing"""
    embeddings = []

    # Calculate total number of batches for progress bar
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    with tqdm(total=len(texts), desc="Generating embeddings") as pbar:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
                hidden_states = outputs.hidden_states[-1]
                batch_embeddings = hidden_states.mean(dim=1).cpu().tolist()
                embeddings.extend(batch_embeddings)
            pbar.update(len(batch))
    
    return embeddings

# Load LLaMA 3 model for text generation
generation_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
generation_pipeline = pipeline("text-generation", model=generation_model, tokenizer=tokenizer, device=0 if DEVICE == "cuda" else -1)

def get_completion(messages: str) -> Union[str, Dict]:
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
        text = re.sub(r'\.{2,}', '', text)  # Remove multiple dots
        text = re.sub(r'\s*\u2002\s*', ' ', text)  # Remove special unicode spaces
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if not line or all(c in '.-' for c in line):
                continue
            if line.startswith('=== Page'):
                lines.append(line)
                continue
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
        separators = ["\n=== Page", "\n\n", "\n", ". ", "? ", "! "]
        
        print("Splitting text into initial chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False
        )
        
        chunks = splitter.split_text(text)
        
        processed_chunks = []
        with tqdm(chunks, desc="Processing chunks", unit="chunk") as pbar:
            for chunk in pbar:
                chunk = DocumentProcessor.clean_text(chunk)
                if not chunk.startswith('=== Page'):
                    sentence_boundaries = ['. ', '? ', '! ']
                    first_boundary = float('inf')
                    for boundary in sentence_boundaries:
                        pos = chunk.find(boundary)
                        if pos != -1 and pos < first_boundary:
                            first_boundary = pos
                    
                    if first_boundary < 100 and first_boundary != float('inf'):
                        chunk = chunk[first_boundary + 2:]
                
                if chunk.strip():
                    processed_chunks.append(Document(content=chunk))
                pbar.set_postfix({"total_chunks": len(processed_chunks)})
        
        print(f"Final number of chunks: {len(processed_chunks)}")
        return processed_chunks

