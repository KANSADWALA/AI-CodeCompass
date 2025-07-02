from typing import List, Dict, Any
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class CodeEmbeddingProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def create_code_chunks(self, file_content: str, file_path: str, 
                          language: str) -> List[Document]:
        """Create chunks from code file content"""
        try:
            # Split by functions/classes for better semantic chunks
            if language == "python":
                chunks = self._split_python_code(file_content)
            elif language in ["javascript", "typescript"]:
                chunks = self._split_js_code(file_content)
            else:
                chunks = self.text_splitter.split_text(file_content)
            
            documents = []
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 50:  # Skip very small chunks
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "file_path": file_path,
                            "language": language,
                            "chunk_id": i,
                            "token_count": len(self.encoding.encode(chunk))
                        }
                    )
                    documents.append(doc)
            
            logger.info(f"Created {len(documents)} chunks from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error creating chunks from {file_path}: {e}")
            return []
    
    def _split_python_code(self, content: str) -> List[str]:
        """Split Python code by functions and classes"""
        import ast
        import astunparse
        
        try:
            tree = ast.parse(content)
            chunks = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    try:
                        chunk = astunparse.unparse(node)
                        chunks.append(chunk)
                    except:
                        continue
            
            # If no functions/classes found, use regular splitting
            if not chunks:
                chunks = self.text_splitter.split_text(content)
            
            return chunks
            
        except:
            # Fallback to regular splitting if AST parsing fails
            return self.text_splitter.split_text(content)
    
    def _split_js_code(self, content: str) -> List[str]:
        """Split JavaScript code (basic implementation)"""
        # Simple splitting by function declarations
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        brace_count = 0
        
        for line in lines:
            current_chunk.append(line)
            brace_count += line.count('{') - line.count('}')
            
            # If we're at the end of a function/object
            if brace_count == 0 and current_chunk and any(
                keyword in ' '.join(current_chunk) 
                for keyword in ['function', 'class', 'const', 'let', 'var']
            ):
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
        
        # Add remaining lines
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        # Fallback to regular splitting if no good chunks
        if not chunks or len(chunks) == 1:
            chunks = self.text_splitter.split_text(content)
        
        return chunks