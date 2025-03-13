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
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"")
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

def get_completion(messages: str, max_tokens: int = 1000, temperature: float = 0.7, timeout: int = 30) -> str:
    """Get completion with proper attention mask and GPU termination."""
    input_data = tokenizer(messages, return_tensors="pt", padding=True, truncation=True)
    input_ids = input_data.input_ids.to(device)
    attention_mask = input_data.attention_mask.to(device)
    
    with torch.no_grad():
        try:
            with torch.amp.autocast('cuda'):
                output = completion_model.generate(
                    input_ids,
                    attention_mask=attention_mask,  # Explicitly set attention mask
                    max_length=input_ids.shape[1] + max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            del input_ids, output, attention_mask  # Clear variables from memory
            return generated_text
        
        except torch.cuda.TimeoutError:
            logger.warning("Generation timed out.")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return "No response due to timeout."

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
        print("Cleaning extracted text...")
        return DocumentProcessor.clean_text(text)

    @staticmethod
    def chunk_text(
        text: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[Document]:
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
