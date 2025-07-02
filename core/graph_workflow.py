from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel
import logging
from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import asdict

# Local imports
from agents.parser_agent import ParseResult

logger = logging.getLogger(__name__)

class DevCopilotState(BaseModel):
    """State for the DevCopilot workflow"""
    repo_url: Optional[str] = None
    repo_path: Optional[str] = None
    user_query: Optional[str] = None
    task_type: str = "summary"  # summary, qa, docs, refactor
    
    # New smart filter fields
    max_files: int = 100
    include_tests: bool = True
    file_extensions: Optional[List[str]] = None
    max_file_size_kb: int = 5_000_000  # Updated: skip files > 5GB by default
    skip_large_files: bool = True  # Skip files larger than max_file_size_kb

    # Data flowing through the workflow
    code_files: Dict[str, str] = {}
    parsed_code: Dict[str, Any] = {}
    embeddings_created: bool = False
    skip_processing: bool = False

    # Results
    summary: Optional[str] = None
    documentation: Optional[str] = None
    qa_response: Optional[str] = None
    refactor_suggestions: Optional[str] = None
    
    # Messages and context
    messages: List[BaseMessage] = []
    context: Dict[str, Any] = {}
    
    class Config:
        arbitrary_types_allowed = True

class DevCopilotWorkflow:
    def __init__(self, agents_dict: Dict[str, Any]):
        self.agents = agents_dict
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(DevCopilotState)
        
        # Add nodes
        workflow.add_node("fetch_code", self._fetch_code_node)
        workflow.add_node("parse_code", self._parse_code_node)
        workflow.add_node("create_embeddings", self._create_embeddings_node)
        workflow.add_node("route_task", self._route_task_node)
        workflow.add_node("generate_summary", self._generate_summary_node)
        workflow.add_node("answer_question", self._answer_question_node)
        workflow.add_node("generate_docs", self._generate_docs_node)
        workflow.add_node("suggest_refactors", self._suggest_refactors_node)
        workflow.add_node("finalize_output", self._finalize_output_node)
        
        # Define the workflow
        workflow.set_entry_point("fetch_code")
        workflow.add_edge("fetch_code", "parse_code")
        workflow.add_edge("parse_code", "create_embeddings")
        workflow.add_edge("create_embeddings", "route_task")
        
        # Conditional routing based on task type
        workflow.add_conditional_edges(
            "route_task",
            self._route_condition,
            {
                "summary": "generate_summary",
                "qa": "answer_question", 
                "docs": "generate_docs",
                "refactor": "suggest_refactors"
            }
        )
        
        # All tasks converge to finalize
        workflow.add_edge("generate_summary", "finalize_output")
        workflow.add_edge("answer_question", "finalize_output")
        workflow.add_edge("generate_docs", "finalize_output")
        workflow.add_edge("suggest_refactors", "finalize_output")
        workflow.add_edge("finalize_output", END)
        
        return workflow.compile()
    
    def _fetch_code_node(self, state: DevCopilotState) -> DevCopilotState:
        """Fetch code and process in filtered, chunked batches using max_files per chunk"""
        try:
            # Skip fetching if already done
            if state.context.get("skip_fetch", False):
                return state
            
            if state.repo_url:
                raw_files = self.agents["code_fetcher"].fetch_from_url(state.repo_url)
                logger.info(f"DEBUG: Raw files count: {len(raw_files)}")  # ADD THIS
                logger.info(f"DEBUG: First 3 files: {list(raw_files.keys())[:3]}")  # ADD THIS
            elif state.repo_path:
                raw_files = self.agents["code_fetcher"].fetch_from_path(state.repo_path)
            else:
                raise ValueError("No repository URL or path provided")
            
            # ADD THIS LINE HERE:
            logger.info(f"Fetched {len(raw_files)} raw files: {list(raw_files.keys())[:5]}...")  # ADD THIS LINE HERE
            
            state.context["skipped_large_files"] = []

            def split_large_file(path, content, max_kb=5120):
                max_bytes = max_kb * 1024
                content_bytes = content.encode("utf-8")
                for i in range(0, len(content_bytes), max_bytes):
                    chunk = content_bytes[i:i + max_bytes].decode("utf-8", errors="ignore")
                    yield f"{path}#chunk{i // max_bytes + 1}", chunk

            
            def is_valid(path, content):
                if state.file_extensions and not any(path.endswith(ext) for ext in state.file_extensions):
                    return False
                if not state.include_tests and ("test" in path.lower() or path.lower().startswith("test_")):
                    return False
                size_kb = len(content.encode("utf-8")) // 1024
                if size_kb > 1_000_000:
                    state.messages.append(AIMessage(content=f"⚠️ Warning: {path} is >1GB ({size_kb:,} KB)"))
                if size_kb > state.max_file_size_kb:
                    if state.skip_large_files:
                        state.context["skipped_large_files"].append({"path": path, "size_kb": size_kb})
                        return False
                    else:
                        return "split"
                return True

            # Apply filtering and chunking
            filtered = []
            for path, content in raw_files.items():
                result = is_valid(path, content)
                logger.debug(f"Filtering: {path} → {result}")
                if result is True:
                    filtered.append((path, content))
                elif result == "split":
                    for split_path, split_chunk in split_large_file(path, content):
                        filtered.append((split_path, split_chunk))

            total = len(filtered)

            # auto-adjust max_files for large repos
            if total > 1000:
                old = state.max_files
                state.max_files = min(state.max_files, 50)
                state.messages.append(AIMessage(content=f"⚙️ Auto-adjusted max_files from {old} to {state.max_files} due to large repo"))

            chunk_size = state.max_files
            chunks = [filtered[i:i + chunk_size] for i in range(0, total, chunk_size)]

            state.messages.append(AIMessage(content=f"Processing {total} files in {len(chunks)} chunks of {chunk_size}"))
            logger.info(f"Filtered {total} files into {len(chunks)} chunks")
            logger.info(f"✅ Final usable files: {len(filtered)} / Raw files: {len(raw_files)}")

            chunk_metrics = []
            def process_chunk(index, chunk):
                start = time.time()
                batch = {k: v for k, v in chunk}
                duration = round(time.time() - start, 3)
                return index, batch, duration

            final_code_files = {}

            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(process_chunk, idx, chunk) for idx, chunk in enumerate(chunks)]
                for future in futures:
                    idx, result, duration = future.result()
                    final_code_files.update(result)
                    state.messages.append(AIMessage(content=f"✅ Chunk {idx + 1}/{len(chunks)} ({len(result)} files) in {duration}s"))
                    chunk_metrics.append({"chunk": idx+1, "files": len(result), "duration_s": duration})

            state.code_files = final_code_files
            state.context["chunk_metrics"] = chunk_metrics

        except Exception as e:
            logger.error(f"Error in fetch_code_node: {e}")
            state.messages.append(AIMessage(content=f"Error fetching code: {e}"))

        return state
    
    
    def _parse_code_node(self, state: DevCopilotState) -> DevCopilotState:
        """Parse the fetched code"""
        try:
            # Skip parsing if already done
            if state.context.get("skip_parse", False):
                return state
            
            parsed_code = self.agents["parser"].parse_files(state.code_files)
            state.parsed_code = parsed_code
            state.messages.append(AIMessage(content="Code parsed successfully"))
            
        except Exception as e:
            logger.error(f"Error in parse_code_node: {e}")
            state.messages.append(AIMessage(content=f"Error parsing code: {e}"))
        
        return state
    
    def _create_embeddings_node(self, state: DevCopilotState) -> DevCopilotState:
        """Create embeddings for the code"""
        try:
            # Skip embeddings if already done
            if state.context.get("skip_embeddings", False):
                return state
            
            # Create embeddings using the vector store
            documents = []
            for file_path, content in state.code_files.items():
                file_docs = self.agents["embedding_processor"].create_code_chunks(
                    content, file_path, self._detect_language(file_path)
                )
                documents.extend(file_docs)
            
            if documents:
                self.agents["vector_store"].add_documents(documents)
                state.embeddings_created = True
                state.messages.append(AIMessage(content=f"Created embeddings for {len(documents)} chunks"))
            
        except Exception as e:
            logger.error(f"Error in create_embeddings_node: {e}")
            state.messages.append(AIMessage(content=f"Error creating embeddings: {e}"))
        
        return state
    
    def _route_task_node(self, state: DevCopilotState) -> DevCopilotState:
        """Route to appropriate task based on user input"""
        # This is just a routing node, actual routing happens in conditional edges
        state.messages.append(AIMessage(content=f"Routing to {state.task_type} task"))
        return state
    
    def _route_condition(self, state: DevCopilotState) -> str:
        """Determine which task to execute"""
        return state.task_type
    
    def _normalize_parsed_code(self, parsed_code: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parsed code to consistent dictionary format"""
        normalized = {}
        for k, v in parsed_code.items():
            if isinstance(v, ParseResult):
                normalized[k] = asdict(v)
            elif hasattr(v, '__dict__'):  # Handle other object types
                normalized[k] = {
                    "language": getattr(v, 'language', 'unknown'),
                    "functions": getattr(v, 'functions', []),
                    "classes": getattr(v, 'classes', []),
                    "lines": getattr(v, 'lines', 0),
                    "size": getattr(v, 'size', 0),
                    "imports": getattr(v, 'imports', []),
                    "variables": getattr(v, 'variables', []),
                    "exports": getattr(v, 'exports', []),
                    "error": getattr(v, 'error', None),
                    "tree_sitter_used": getattr(v, 'tree_sitter_used', False)
                }
            elif isinstance(v, dict):
                # Ensure dict has all required keys
                normalized[k] = {
                    "language": v.get("language", "unknown"),
                    "functions": v.get("functions", []),
                    "classes": v.get("classes", []),
                    "lines": v.get("lines", 0),
                    "size": v.get("size", 0),
                    "imports": v.get("imports", []),
                    "variables": v.get("variables", []),
                    "exports": v.get("exports", []),
                    "error": v.get("error"),
                    "tree_sitter_used": v.get("tree_sitter_used", False)
                }
            else:
                # Fallback for unknown types
                normalized[k] = {
                    "language": "unknown",
                    "functions": [],
                    "classes": [],
                    "lines": 0,
                    "size": 0,
                    "imports": [],
                    "variables": [],
                    "exports": [],
                    "error": None,
                    "tree_sitter_used": False
                }
        return normalized
        
    def _generate_summary_node(self, state: DevCopilotState) -> DevCopilotState:
        """Generate code summary"""
        try:
            # DEBUG: Check the structure of parsed_code
            logger.info(f"DEBUG: parsed_code keys: {list(state.parsed_code.keys())[:3]}")
            for k, v in list(state.parsed_code.items())[:2]:
                logger.info(f"DEBUG: {k} -> type: {type(v)}, hasattr lines: {hasattr(v, 'lines')}")
                if isinstance(v, dict):
                    logger.info(f"DEBUG: dict keys: {list(v.keys())}")
                    
            # Simple pass-through - let the summary agent handle normalization
            summary = self.agents["summary"].generate_summary(
                state.code_files, state.parsed_code
            )
            state.summary = summary
            state.messages.append(AIMessage(content="Summary generated"))
            
        except Exception as e:
            logger.error(f"Error in generate_summary_node: {e}")
            state.messages.append(AIMessage(content=f"Error generating summary: {e}"))
        
        return state
    
    def _answer_question_node(self, state: DevCopilotState) -> DevCopilotState:
        """Answer user question"""
        try:
            if state.user_query and state.embeddings_created:
                response = self.agents["qa"].answer_question(
                    state.user_query, self.agents["vector_store"]
                )
                state.qa_response = response
                state.messages.append(AIMessage(content="Question answered"))
            else:
                state.messages.append(AIMessage(content="No question provided or embeddings not ready"))
                
        except Exception as e:
            logger.error(f"Error in answer_question_node: {e}")
            state.messages.append(AIMessage(content=f"Error answering question: {e}"))
        
        return state
    
    def _generate_docs_node(self, state: DevCopilotState) -> DevCopilotState:
        """Generate documentation"""
        try:
            docs = self.agents["doc_generator"].generate_documentation(
                state.code_files, state.parsed_code
            )
            state.documentation = docs
            state.messages.append(AIMessage(content="Documentation generated"))
            
        except Exception as e:
            logger.error(f"Error in generate_docs_node: {e}")
            state.messages.append(AIMessage(content=f"Error generating docs: {e}"))
        
        return state
    
    def _suggest_refactors_node(self, state: DevCopilotState) -> DevCopilotState:
        """Suggest refactoring improvements"""
        try:
            suggestions = self.agents["refactor"].analyze_code(
                state.code_files, state.parsed_code
            )
            state.refactor_suggestions = suggestions
            state.messages.append(AIMessage(content="Refactor suggestions generated"))
            
        except Exception as e:
            logger.error(f"Error in suggest_refactors_node: {e}")
            state.messages.append(AIMessage(content=f"Error generating refactor suggestions: {e}"))
        
        return state
    
    def _finalize_output_node(self, state: DevCopilotState) -> DevCopilotState:
        """Finalize and format the output"""
        state.messages.append(AIMessage(content="Workflow completed successfully"))
        return state
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        extension = file_path.split('.')[-1].lower()
        language_map = {
            'py': 'python',
            'js': 'javascript', 
            'ts': 'typescript',
            'jsx': 'javascript',
            'tsx': 'typescript',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'go': 'go',
            'rs': 'rust',
            'php': 'php',
            'rb': 'ruby'
        }
        return language_map.get(extension, 'text')
    
    def run(self, initial_state: DevCopilotState) -> DevCopilotState:
        """Run the complete workflow with caching support"""
        try:
            # If embeddings already created, skip fetch/parse/embed steps
            if initial_state.embeddings_created:
                logger.info(f"🔁 Reusing cached code and parsed data for task: {initial_state.task_type}")
                initial_state.messages.append(AIMessage(content="🔁 Reusing cached embeddings"))
                initial_state.messages.append(AIMessage(content=f"Routing to {initial_state.task_type} task"))
                
                # Ensure vector store is populated with cached embeddings
                if hasattr(self.agents, 'vector_store') and self.agents['vector_store']:
                    try:
                        vector_store = self.agents['vector_store']

                        # Check if vector store has embeddings
                        if initial_state.code_files:
                            parsed_dict = {
                                k: vector_store._normalize_parsed_info(v)
                                for k, v in initial_state.parsed_code.items()
                            }
                            if not vector_store.has_embeddings() or not getattr(vector_store, "documents", []):
                                logger.info("🔄 Re-populating vector store from cached data")
                                vector_store.load_cached_embeddings(initial_state.code_files, parsed_dict)
                            else:
                                logger.info(f"✅ Vector store already has {len(vector_store.documents)} documents")

                        else:
                            logger.info(f"✅ Vector store already has {len(vector_store.documents)} documents")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not restore vector store: {e}")
                
                # Create a modified state that skips early stages
                cached_state = initial_state.copy()
                cached_state.context["skip_fetch"] = True
                cached_state.context["skip_parse"] = True  
                cached_state.context["skip_embeddings"] = True

                # ✅ Normalize parsed_code dict before passing to LangGraph again
                cached_state.parsed_code = {
                    k: self.agents["vector_store"]._normalize_parsed_info(v)
                    for k, v in cached_state.parsed_code.items()
                }

                result = self.graph.invoke(cached_state)
                #result = self.graph.invoke(initial_state, start_at="route_task")
                
                # Normalize return type (same as full pipeline)
                if isinstance(result, dict):
                    logger.warning("⚠️ LangGraph returned dict, converting to DevCopilotState")
                    result = DevCopilotState(**result)
                elif not isinstance(result, DevCopilotState):
                    raise ValueError("Unexpected return type from LangGraph workflow")
                
                return result

            # Else run full pipeline
            final_state = self.graph.invoke(initial_state)

            # Normalize return type
            if isinstance(final_state, dict):
                logger.warning("⚠️ LangGraph returned dict, converting to DevCopilotState")
                final_state = DevCopilotState(**final_state)
            elif not isinstance(final_state, DevCopilotState):
                raise ValueError("Unexpected return type from LangGraph workflow")

            return final_state

        except Exception as e:
            logger.error(f"❌ Error running workflow: {e}")
            raise





# # Great follow-up.

# Let’s break it down — here's **what the filtering logic does** in your updated workflow:

# ---

# ### ✅ 1. `max_files: int`

# **Purpose:**
# ⏱️ **Limit the total number of files** that DevCopilot will process (parse, embed, analyze).
# This is useful to:

# * Prevent overload on huge repos (e.g. 10k+ files)
# * Improve response time and memory usage

# **Example:**
# If a repo has 350 files and `max_files=100`, only the **first 100** valid files will be used.

# ---

# ### ✅ 2. `file_extensions: Optional[List[str]]`

# **Purpose:**
# 📂 **Only include files with specific extensions**, such as `.py`, `.ts`, `.java`, etc.
# Avoids wasting compute on files like:

# * `.md`, `.txt`, `.lock`, `.csv`, `.yml`, etc.

# **Example:**
# If `file_extensions = [".py", ".js"]`, then `.cpp`, `.java`, `.html` will be excluded.

# ---

# ### ✅ 3. `include_tests: bool`

# **Purpose:**
# 🧪 **Exclude test files** if not relevant to the task.
# It filters out files named like:

# * `test_something.py`
# * `something_test.js`
# * Any file path containing `"test"`

# **Use Case:**
# In summaries, docs, or refactor tasks, test files might add noise or irrelevant complexity.

# ---

# ### ✅ Combined Result

# These filters are applied **before parsing or embedding**, which means:

# * ✅ Less RAM and CPU usage
# * ✅ Faster results
# * ✅ Cleaner summaries or answers
# * ✅ Scalable to large codebases

# ---

# **a.** Want to show the count of skipped vs. included files in the UI dashboard?
# **b.** Want more advanced filters like “exclude files larger than X KB” or “only include top-level modules”?
