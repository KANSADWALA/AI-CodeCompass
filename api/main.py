# Import necessary modules
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Union
import os
import tempfile
import shutil
import logging
import traceback
import time
from contextlib import asynccontextmanager

# Local imports
from agents.code_fetcher import CodeFetcherAgent
from agents.summary_agent import SummaryAgent
from agents.doc_generator import DocumentationGenerator
from agents.parser_agent import ParserAgent
from agents.qa_agent import QAAgent
from agents.refactor_agent import RefactorAgent
from core.graph_workflow import DevCopilotState
from core.vector_store import CodeVectorStore
from core.embeddings import CodeEmbeddingProcessor

# Load environment variables
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=True)
print("✅ ENV DEBUG - GITHUB_TOKEN:", os.getenv("GITHUB_TOKEN")[:10])



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global agents - will be initialized in lifespan
agents = {}
workflow = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    # Startup
    logger.info("Starting DevCopilot AI initialization...")
    
    global agents, workflow
    
    try:
        # Initialize agents one by one with error handling
        logger.info("Initializing parser agent...")
        agents["parser"] = ParserAgent()
        logger.info("Parser agent initialized successfully")
        
        # Initialize other agents with fallback handling
        try:
            logger.info("Initializing vector store...")
            agents["vector_store"] = CodeVectorStore()
            logger.info("Vector store initialized")
        except Exception as e:
            logger.warning(f"Vector store initialization failed: {e}")
            agents["vector_store"] = None
        
        try:
            logger.info("Initializing embedding processor...")
            agents["embedding_processor"] = CodeEmbeddingProcessor()
            logger.info("Embedding processor initialized")
        except Exception as e:
            logger.warning(f"Embedding processor initialization failed: {e}")
            agents["embedding_processor"] = None
        
        try:
            logger.info("Initializing code fetcher...")
            agents["code_fetcher"] = CodeFetcherAgent()
            logger.info("Code fetcher initialized")
        except Exception as e:
            logger.warning(f"Code fetcher initialization failed: {e}")
            agents["code_fetcher"] = None
        
        try:
            logger.info("Initializing summary agent...")
            agents["summary"] = SummaryAgent()
            logger.info("Summary agent initialized")
        except Exception as e:
            logger.warning(f"Summary agent initialization failed: {e}")
            agents["summary"] = None
        
        try:
            logger.info("Initializing QA agent...")
            agents["qa"] = QAAgent()
            logger.info("QA agent initialized")
        except Exception as e:
            logger.warning(f"QA agent initialization failed: {e}")
            agents["qa"] = None
        
        try:
            logger.info("Initializing documentation generator...")
            agents["doc_generator"] = DocumentationGenerator()
            logger.info("Documentation generator initialized")
        except Exception as e:
            logger.warning(f"Documentation generator initialization failed: {e}")
            agents["doc_generator"] = None
        
        try:
            logger.info("Initializing refactor agent...")
            agents["refactor"] = RefactorAgent()
            logger.info("Refactor agent initialized")
        except Exception as e:
            logger.warning(f"Refactor agent initialization failed: {e}")
            agents["refactor"] = None
        
        # Initialize workflow only if core agents are available
        if agents.get("parser"):
            try:
                logger.info("Initializing workflow...")
                from core.graph_workflow import DevCopilotWorkflow
                workflow = DevCopilotWorkflow(agents)
                logger.info("Workflow initialized")
            except Exception as e:
                logger.warning(f"Workflow initialization failed: {e}")
                workflow = None
        
        # Log successful initialization
        initialized_agents = [name for name, agent in agents.items() if agent is not None]
        logger.info(f"DevCopilot AI initialized successfully with agents: {initialized_agents}")
        
        if not initialized_agents:
            logger.error("No agents were successfully initialized!")
        
    except Exception as e:
        logger.error(f"Critical error during initialization: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Don't raise the exception - let the app start with limited functionality
    
    yield
    
    # Shutdown
    logger.info("Shutting down DevCopilot AI...")

app = FastAPI(
    title="DevCopilot AI API",
    version="2.0.0",
    description="Enhanced AI-powered code analysis and development assistant",
    lifespan=lifespan
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time"]
)

# Define task types as constants
class TaskType(str):
    SUMMARY = "summary"
    QA = "qa"
    DOCS = "docs"
    REFACTOR = "refactor"
    PARSE = "parse"

# Request model for analysis
class AnalyzeRequest(BaseModel):
    repo_url: Optional[str] = Field(None, description="Git repository URL to analyze")
    repo_path: Optional[str] = Field(None, description="Local path to repository")
    task_type: str = Field(TaskType.SUMMARY, description="Type of analysis to perform")
    user_query: Optional[str] = Field(None, description="Specific question or query")
    max_files: Optional[int] = Field(100, ge=1, le=1000, description="Maximum number of files to process")
    include_tests: bool = Field(True, description="Whether to include test files in analysis")
    file_extensions: Optional[List[str]] = Field(None, description="Specific file extensions to analyze")
    code_files: Optional[Dict[str, str]] = None
    parsed_code: Optional[Dict[str, Any]] = None
    embeddings_created: Optional[bool] = False

    @validator('task_type')
    def validate_task_type(cls, v):
        valid_types = [TaskType.SUMMARY, TaskType.QA, TaskType.DOCS, TaskType.REFACTOR, TaskType.PARSE]
        if v not in valid_types:
            raise ValueError(f'task_type must be one of: {valid_types}')
        return v

# Model for parsed code results
class ParseResult(BaseModel):
    language: str
    functions: List[Dict[str, Any]]
    classes: List[Dict[str, Any]]
    imports: List[Dict[str, Any]]
    variables: List[Dict[str, Any]]
    exports: List[Dict[str, Any]]
    lines: int
    size: int
    error: Optional[str] = None
    tree_sitter_used: bool = False

# Response model for analysis results
class AnalyzeResponse(BaseModel):
    success: bool
    task_type: str
    result: Union[str, Dict[str, Any]]
    metadata: Dict[str, Any] = {}
    statistics: Optional[Dict[str, Any]] = None
    processing_time: float
    files_processed: int
    code_files: Optional[Dict[str, str]] = None
    parsed_code: Optional[Dict[str, Any]] = None

# Request model for Q&A
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    context: Optional[str] = Field(None, description="Additional context for the question")

# Response model for Q&A
class QuestionResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    confidence: Optional[float] = None

# Response model for health check
class HealthResponse(BaseModel):
    status: str
    version: str
    agents_loaded: List[str]
    parsers_available: List[str]

# Custom exception handler
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Validation Error", "detail": str(exc)}
    )

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if not agents:
        raise HTTPException(status_code=503, detail="Agents not initialized")
    
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        agents_loaded=list(agents.keys()),
        parsers_available=list(agents["parser"].parsers.keys()) if "parser" in agents else []
    )

# Main analysis endpoint
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_repo(request: AnalyzeRequest):
    """Analyze a repository with enhanced error handling and validation"""
    start_time = time.time()
    
    try:
        if not workflow:
            raise HTTPException(status_code=503, detail="Workflow not initialized")
        
        # Create state object
        state = DevCopilotState(
            repo_url=request.repo_url,
            repo_path=request.repo_path,
            task_type=request.task_type,
            user_query=request.user_query
        )
        
        # Add request parameters to state
        state.max_files = request.max_files
        state.include_tests = request.include_tests
        state.file_extensions = request.file_extensions

        # ✅ ADD THIS: preload previous data
        if request.code_files:
            logger.info(f"🔁 Reusing previous analysis data - {len(request.code_files)} files, embeddings: {request.embeddings_created}")
            state.code_files = request.code_files
            state.parsed_code = request.parsed_code or {}
            state.embeddings_created = request.embeddings_created or False
            # FLAG to skip processing steps
            state.skip_processing = True
        
        # Run workflow
        logger.info(f"Starting {request.task_type} analysis")
        result = workflow.run(state)
        logger.info(f"DEBUG: Final result has {len(result.code_files)} code files") 

        # Defensive fallback if result is unexpectedly a dict
        if isinstance(result, dict):
            logger.warning("⚠️ workflow.run() returned a dict instead of DevCopilotState")
            result = DevCopilotState(**result)
                
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response based on task type
        if request.task_type == TaskType.PARSE:
            # For parse tasks, return structured parsing results
            response_result = {
                "parsed_files": result.parsed_data if hasattr(result, 'parsed_data') else {},
                "statistics": agents["parser"].get_file_statistics(result.parsed_data) if hasattr(result, 'parsed_data') else {}
            }
        else:
            # For other tasks, return the appropriate result
            if result.task_type == TaskType.SUMMARY:
                response_result = result.summary or "⚠️ Summary generation failed or returned no output."
            elif result.task_type == TaskType.QA:
                response_result = result.qa_response or "⚠️ No answer generated."
            elif result.task_type == TaskType.DOCS:
                response_result = result.documentation or "⚠️ Documentation generation failed."
            elif result.task_type == TaskType.REFACTOR:
                response_result = result.refactor_suggestions or "⚠️ No refactor suggestions generated."
            else:
                response_result = "⚠️ No result generated"

        return AnalyzeResponse(
            success=True,
            task_type=result.task_type,
            result=response_result,
            code_files=result.code_files, # Include code files if available
            parsed_code=result.parsed_code, # Include parsed code if available
           
            # Include metadata for debugging
            metadata={
                "messages": [msg.content for msg in result.messages] if hasattr(result, 'messages') else [],
                "repo_info": {
                    "url": request.repo_url,
                    "path": request.repo_path,
                    "max_files": request.max_files,
                    "include_tests": request.include_tests
                }
            },
            statistics=agents["parser"].get_file_statistics(result.parsed_code) if hasattr(result, 'parsed_code') else None,
            processing_time=processing_time,
            files_processed=len(result.code_files) if hasattr(result, 'code_files') else 0
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        processing_time = time.time() - start_time
        
        return AnalyzeResponse(
            success=False,
            task_type=request.task_type,
            result=f"Analysis failed: {str(e)}",
            metadata={"error": str(e)},
            processing_time=processing_time,
            files_processed=0
        )

# Q&A endpoint
@app.post("/question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about the analyzed codebase"""
    try:
        if not agents or "qa" not in agents:
            raise HTTPException(status_code=503, detail="Q&A agent not available")
        
        # Use the QA agent to answer the question
        qa_agent = agents["qa"]
        
        # This would need to be implemented based on your QA agent's interface
        answer = qa_agent.answer_question(request.question, context=request.context)
        
        return QuestionResponse(
            answer=answer.get("answer", "No answer available"),
            sources=answer.get("sources", []),
            confidence=answer.get("confidence")
        )
        
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")

# File parsing endpoint
# File parsing endpoint completion
@app.post("/parse-files")
async def parse_files(files: Dict[str, str]):
    """Parse individual files without full repository analysis"""
    try:
        if not agents or "parser" not in agents:
            raise HTTPException(status_code=503, detail="Parser agent not available")
        
        parser_agent = agents["parser"]
        parsed_results = parser_agent.parse_files(files)
        
        # Generate statistics
        statistics = parser_agent.get_file_statistics(parsed_results)
        
        return {
            "success": True,
            "parsed_files": {
                file_path: {
                    "language": result.language,
                    "functions": result.functions,
                    "classes": result.classes,
                    "imports": result.imports,
                    "variables": result.variables,
                    "exports": result.exports,
                    "lines": result.lines,
                    "size": result.size,
                    "error": result.error,
                    "tree_sitter_used": result.tree_sitter_used
                }
                for file_path, result in parsed_results.items()
            },
            "statistics": statistics
        }
        
    except Exception as e:
        logger.error(f"File parsing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse files: {str(e)}")

# Get repository structure endpoint
@app.get("/repo-structure/{repo_id}")
async def get_repo_structure(repo_id: str):
    """Get the structure of a previously analyzed repository"""
    try:
        # This would need to be implemented based on your storage system
        # For now, return a placeholder response
        return {
            "repo_id": repo_id,
            "structure": "Repository structure would be returned here",
            "message": "This endpoint needs to be implemented based on your storage system"
        }
    except Exception as e:
        logger.error(f"Failed to get repository structure: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get repository structure: {str(e)}")

# Get supported languages endpoint
@app.get("/supported-languages")
async def get_supported_languages():
    """Get list of supported programming languages"""
    try:
        if not agents or "parser" not in agents:
            raise HTTPException(status_code=503, detail="Parser agent not available")
        
        parser_agent = agents["parser"]
        
        return {
            "supported_languages": list(parser_agent.supported_extensions.values()),
            "file_extensions": parser_agent.supported_extensions,
            "tree_sitter_parsers": list(parser_agent.parsers.keys()) if hasattr(parser_agent, 'parsers') else [],
            "tree_sitter_available": hasattr(parser_agent, 'parsers') and bool(parser_agent.parsers)
        }
        
    except Exception as e:
        logger.error(f"Failed to get supported languages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get supported languages: {str(e)}")

# Background task status endpoint
@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a background task"""
    # This would need to be implemented with a task tracking system
    return {
        "task_id": task_id,
        "status": "This endpoint needs to be implemented with a task tracking system",
        "message": "Consider using Celery or similar for background task management"
    }

# Cleanup endpoint for development
@app.delete("/cleanup")
async def cleanup_resources():
    """Clean up temporary resources (development only)"""
    try:
        # Clean up temporary directories
        temp_dirs_cleaned = 0
        temp_dir = tempfile.gettempdir()
        
        for item in os.listdir(temp_dir):
            if item.startswith("devcopilot_"):
                item_path = os.path.join(temp_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    temp_dirs_cleaned += 1
        
        return {
            "success": True,
            "message": f"Cleaned up {temp_dirs_cleaned} temporary directories",
            "temp_dirs_cleaned": temp_dirs_cleaned
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

# Error handling middleware
@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add processing time header to responses"""
    import time
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "DevCopilot AI API",
        "version": "2.0.0",
        "description": "Enhanced AI-powered code analysis and development assistant",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "question": "/question",
            "parse-files": "/parse-files",
            "supported-languages": "/supported-languages",
            "docs": "/docs"
        },
        "status": "ready"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )