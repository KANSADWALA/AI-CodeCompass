# 🤖 AI-CodeCompass

## 📋 Overview

AI-CodeCompass is an autonomous codebase assistant that analyzes source code to generate summaries, documentation, answer questions, and suggest refactorings. It supports GitHub or zipped local repos, offering smart insights through a Streamlit UI. Powered by LLMs, it features modular agents for each task and uses semantic search with FAISS and Sentence Transformers. The system is fast, extensible, and designed for developer productivity.

## 📦 Features

<ul>
<li><strong>🔍 Analyze GitHub or local zipped repositories: </strong>Upload a repo or enter a URL and get insights instantly.</li>
  
<li><strong>🧠 Smart Summary Generation: </strong>Produces high-level overviews of your codebase including file types, language usage, and structure.</li>
  
<li><strong>❓ Question Answering over Code (RAG): </strong>Ask free-form questions like “What does this function do?” or “Where is the main logic?” using semantic search and context-aware LLMs.</li>

<li><strong>📄 Automated Documentation: </strong>Generates README-style documentation and API references from code.</li>

<li><strong>🛠️ Refactoring Suggestions: </strong>Identifies long functions, large classes, missing docstrings, duplicate code, and more.</li>

<li><strong>📁 File Chunking + Embeddings: </strong>Efficiently splits large files and generates semantic embeddings using FAISS + Sentence Transformers.</li>

<li><strong>🖥️ Streamlit UI: </strong>Clean, interactive interface with dashboards, file insights, and download buttons for logs/metrics.</li>

<li><strong>🧩 Modular Agents Architecture: </strong>Each task is handled by a dedicated agent (SummaryAgent, QAAgent, ParserAgent, DocGeneratorAgent, RefactorAgent) — easy to extend.</li>
</ul>

## 🧠 Technology Stack

### 💻 Frontend (User Interface)

<ul>
  <li><strong>Streamlit: </strong>Rapid development of data apps with built-in UI components (buttons, charts, forms).</li>
  
  <li><strong>Altair: </strong>Declarative charting library for generating interactive visualizations.</li>
  
  <li><strong>Streamlit Option Menu & Extras: </strong>For elegant sidebar navigation and UI enhancements.</li>

</ul>

### 🧠 Backend (Server & Processing)

<ul>
  <li><strong>FastAPI: </strong>High-performance async web framework to expose backend API endpoints.</li>
  
  <li><strong>Uvicorn: </strong>ASGI server to run FastAPI app efficiently.</li>
    
  <li><strong>Pydantic: </strong>For request/response data validation and modeling.</li>
</ul>

### 🤖 Agents & LLM Integration

<ul>
  <li><strong>LangChain: </strong>Framework for chaining LLM calls, agents, and workflows.</li>
  
  <li><strong>Ollama(LLaMA 2): </strong>Local LLM backend for generating summaries, answers, documentation, and refactoring suggestions.</li>
  
  <li><strong>LangGraph: </strong>Used to orchestrate workflows between multiple agents with conditional logic.</li>
</ul>


### 🧾 Code Parsing & Understanding

<ul>
  <li><strong>AST (Python standard library): </strong>For extracting structure from Python files.</li>
  
  <li><strong>Tree-sitter: </strong>Syntax-aware parser used for multi-language support (JS, Java, C++, Python etc.).</li>
  
  <li><strong>Regex & Heuristics: </strong>For fallback parsing in unsupported or malformed files.</li>
</ul>


### 🧠 Embeddings & Vector Store

<ul>
  <li><strong>Sentence Transformers (CodeBERT): </strong>For generating embeddings from code snippets.</li>
  
  <li><strong>FAISS (Facebook AI Similarity Search): </strong>High-performance vector store for semantic search (used in Q&A).</li>

  <li><strong>Tiktoken: </strong>Efficient token counting for chunking logic in embeddings.</li>
</ul>


### 🔄 Code Fetching & Processing

<ul>
  <li><strong>PyGitHub: </strong>Fetches files and metadata directly from public/private GitHub repositories.</li>
  
  <li><strong>Zipfile + Tempfile: </strong>Handles local zipped repos and temporary storage.</li>
</ul>





