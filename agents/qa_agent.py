from typing import List, Dict, Any
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

# Importing the custom vector store class
from core.vector_store import CodeVectorStore

import logging

logger = logging.getLogger(__name__)

class QAAgent:
    def __init__(self, model_name: str = "llama2"):
        # Updated to use Ollama instead of ChatOpenAI
        self.llm = Ollama(model=model_name, temperature=0.1)
    
    def answer_question(self, question: str, vector_store: CodeVectorStore) -> str:
        """Answer questions about the codebase using RAG"""
        try:
            # Retrieve relevant code chunks
            relevant_docs = vector_store.similarity_search(question, k=5)

            logger.info(f"🔍 Retrieved {len(relevant_docs)} docs for question: {question}")
            for i, d in enumerate(relevant_docs[:3]):
                logger.info(f"  • Doc {i+1}: {d.metadata.get('source', 'unknown')}, lang: {d.metadata.get('language')}")

            if not relevant_docs:
                return "I couldn't find relevant information in the codebase to answer your question."
            
            # Create context from retrieved documents
            context = self._create_context(relevant_docs)
            
            # Generate answer
            prompt = self._create_qa_prompt(question, context)
            
            # Updated for Ollama (single string input)
            full_prompt = f"""You are an expert code analyst. Answer questions about codebases based on the provided context. Be precise and reference specific code when relevant.

                                {prompt}"""
            
            response = self.llm.invoke(full_prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"Error answering question: {e}"
    
    def _create_context(self, documents: List) -> str:
        """Create context string from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(documents):
            metadata = doc.metadata
            # file_path = metadata.get("source", "unknown")
            # language = metadata.get("language", "unknown")
            # doc_type = metadata.get("type", "code_file")
            file_path = metadata.get("source", "unknown")
            language = metadata.get("language") or "unknown"
            doc_type = metadata.get("type") or "code_file"

            # file_path = metadata.get("file_path", "unknown")
            # language = metadata.get("language", "unknown")
            
            context_parts.append(f"""
            **Code Chunk {i+1}** (from {file_path}, {language}):
            ```{language}
            {doc.page_content}
            ```
            """)
        
        return "\n".join(context_parts)
    
    def _create_qa_prompt(self, question: str, context: str) -> str:
        """Create prompt for question answering"""
        return f"""
            Based on the following code context, please answer the user's question:

            **User Question:** {question}

            **Relevant Code Context:**
            {context}

            **Instructions:**
            - Answer the question based on the provided code context
            - If the context doesn't contain enough information, say so
            - Reference specific files, functions, or code patterns when relevant
            - Provide code examples from the context when helpful
            - Be concise but thorough

            **Answer:**
            """