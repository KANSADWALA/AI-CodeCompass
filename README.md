# AI-CodeCompass

## 📋 Overview

AI-CodeCompass is an autonomous codebase assistant designed to analyze source code, generate summary,documentation, answer questions, and suggest refactorings — all with AI. It provides a fast, extensible API powered by LLMs and a Streamlit frontend for ease of use.

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
