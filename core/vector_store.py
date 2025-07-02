import os
import torch
import pickle
import logging
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class CodeVectorStore:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers",
        model_name: str = "microsoft/codebert-base",
        device: str = "cuda"  # or "cpu"
    ):
        self.embedding_model = embedding_model
        self.model_name = model_name
        self.device = device
        self.embeddings = self._init_embeddings()
        self.vector_store = None
        self.documents = []

    def _init_embeddings(self):
        """Initialize embedding model"""
        if self.embedding_model == "sentence-transformers":
            return HuggingFaceEmbeddingWrapper(self.model_name, self.device)
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model}")

    def add_documents(self, documents: List[Document]):
        """Add documents to vector store"""
        try:
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                self.vector_store.add_documents(documents)

            self.documents.extend(documents)
            logger.info(f"Added {len(documents)} documents to vector store")

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        if self.vector_store is None:
            return []
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []

    def save(self, path: str):
        """Save vector store to disk"""
        try:
            if self.vector_store:
                self.vector_store.save_local(path)
                with open(f"{path}/documents.pkl", "wb") as f:
                    pickle.dump(self.documents, f)
                logger.info(f"Vector store saved to {path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")

    def load(self, path: str):
        """Load vector store from disk"""
        try:
            if os.path.exists(path):
                self.vector_store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
                with open(f"{path}/documents.pkl", "rb") as f:
                    self.documents = pickle.load(f)
                logger.info(f"Vector store loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")

    def has_embeddings(self) -> bool:
        """Check if vector store has any embeddings loaded"""
        return self.vector_store is not None and len(self.documents) > 0

    def _normalize_parsed_info(self, parsed_info: Any) -> Dict[str, Any]:
        """Normalize parsed info to consistent dictionary format"""
        if hasattr(parsed_info, '__dict__'):
            return vars(parsed_info)
        elif isinstance(parsed_info, dict):
            return {
                "language": parsed_info.get("language", "unknown"),
                "functions": parsed_info.get("functions", []),
                "classes": parsed_info.get("classes", []),
                "lines": parsed_info.get("lines", 0),
                "imports": parsed_info.get("imports", []),
                "variables": parsed_info.get("variables", []),
                "exports": parsed_info.get("exports", []),
                "size": parsed_info.get("size", 0)
            }
        else:
            return {
                "language": "unknown",
                "functions": [],
                "classes": [],
                "lines": 0,
                "imports": [],
                "variables": [],
                "exports": [],
                "size": 0
            }

    def load_cached_embeddings(self, code_files: Dict[str, str], parsed_code: Dict[str, Any]):
        """Load embeddings from cached data if vector store is empty"""
        try:
            if self.has_embeddings() and self.documents:
                logger.info("Vector store already has embeddings, skipping reload")
                return

            logger.info("Recreating vector store from cached code files")

            documents = []
            for file_path, content in code_files.items():
                metadata = {
                    "source": file_path,
                    "type": "code_file"
                }

                if parsed_code and file_path in parsed_code:
                    parsed_info = parsed_code[file_path]
                    parsed_dict = self._normalize_parsed_info(parsed_info)

                    metadata.update({
                        "language": parsed_dict["language"],
                        "functions": len(parsed_dict["functions"]),
                        "classes": len(parsed_dict["classes"]),
                        "lines": parsed_dict["lines"]
                    })

                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)

            if documents:
                self.add_documents(documents)
                logger.info(f"Successfully recreated vector store with {len(documents)} documents")
            else:
                logger.warning("No documents to recreate vector store")

        except Exception as e:
            logger.error(f"Error loading cached embeddings: {e}")
            raise

    def clear(self):
        """Clear the vector store and documents"""
        self.vector_store = None
        self.documents = []
        logger.info("Vector store cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            "has_embeddings": self.has_embeddings(),
            "document_count": len(self.documents),
            "embedding_model": self.model_name,
            "device": self.device
        }

class HuggingFaceEmbeddingWrapper:
    """Adapter class to use sentence-transformers with LangChain-style interface"""

    def __init__(self, model_name: str, device: str = None):
        # Dynamically choose device: 'cuda' if available, else 'cpu'
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Support direct call style"""
        return self.embed_documents(texts)
