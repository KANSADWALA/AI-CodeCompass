# src/ui/streamlit_app.py
import streamlit as st
import requests
import tempfile
import zipfile
import os
import json
from datetime import datetime
import altair as alt
import pandas as pd
from streamlit_option_menu import option_menu
from streamlit_extras.add_vertical_space import add_vertical_space
import hashlib

# Load environment variables
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

API_URL = "http://localhost:8000"

# Ensure the backend is running before proceeding
def backend_is_ready() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False

st.set_page_config(page_title="AI CodeCompass", layout="wide")

# Sidebar
with st.sidebar:
    st.image("https://1lm.me/cc.png", width=180)
    st.title("AI CodeCompass")
    st.markdown("### Your Autonomous Codebase Assistant")
    selected = option_menu(
        menu_title="Main Menu",
        options=["Analyze Repo", "Dashboard", "About"],
        icons=["code-slash", "bar-chart", "info-circle"],
        default_index=0
    )
    st.markdown("---")
    st.markdown("your AI-powered codebase navigator")
    add_vertical_space(2)

# Initialize session state
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_repo_url" not in st.session_state:
    st.session_state.last_repo_url = None
if "results_cache" not in st.session_state:
    st.session_state.results_cache = {}


# Helper function
def compute_file_hash(file_obj):
    """Compute MD5 hash of an uploaded file to detect if same zip reused."""
    file_obj.seek(0)
    data = file_obj.read()
    file_obj.seek(0)
    return hashlib.md5(data).hexdigest()

def render_result(data):
    """Render the analysis result in a structured way."""
    st.markdown(f"### 🧠 {data.get('task_type', 'Task').capitalize()} Result")
    st.code(data["result"], language="markdown")

    context = data.get("metadata", {}).get("repo_info", {})
    skipped = data.get("metadata", {}).get("skipped_large_files", [])
    chunk_metrics = data.get("metadata", {}).get("chunk_metrics", [])
    statistics = data.get("statistics") or {}
    total_files = statistics.get("total_files", "?")
    used_files = data.get("files_processed", 0)

    total_files_display = f"{total_files:,}" if isinstance(total_files, int) else str(total_files)
    used_files_display = f"{used_files:,}"

    st.markdown(f"**📊 Repo had {total_files_display} files → {used_files_display} used (filtered/split)**")

    # Show skipped files
    if skipped:
        with st.expander("📄 Skipped Large Files"):
            for file in skipped:
                st.write(f"{file['path']} — {file['size_kb']:,} KB")
            df = pd.DataFrame(skipped)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Skipped Files (CSV)", csv, "skipped_files.csv")

    # Show chunk metrics
    if chunk_metrics:
        with st.expander("⏱️ Chunk Performance"):
            df = pd.DataFrame(chunk_metrics)
            st.altair_chart(
                alt.Chart(df).mark_bar().encode(
                    x=alt.X("chunk:O", title="Chunk #"),
                    y=alt.Y("duration_s:Q", title="Duration (s)"),
                    tooltip=["files", "duration_s"]
                ).properties(title=f"Chunk Duration", width=600),
                use_container_width=True
            )
            json_data = json.dumps(chunk_metrics, indent=2).encode("utf-8")
            st.download_button("📥 Download Chunk Metrics (JSON)", json_data, "chunk_metrics.json")

if selected == "Analyze Repo":
    st.title("🔍 Analyze Codebase")

    # 🔹 MOVED: Clear Button and Tabs - NOW BEFORE the form
    if st.session_state.results_cache:
        if st.button("🧹 Clear All Results"):
            st.session_state.results_cache.clear()
            st.success("Cleared all previous results")
            st.rerun()

        # 🔹 Icon tab headers per task type
        task_icons = {
            "summary": "📝",
            "qa": "❓",
            "docs": "📄",
            "refactor": "🛠️"
        }
        tabs = st.tabs([
            f"{task_icons.get(ttype, '📦')} {ttype.capitalize()}"
            for ttype in st.session_state.results_cache
        ])
        for idx, (ttype, result) in enumerate(st.session_state.results_cache.items()):
            with tabs[idx]:
                render_result(result)

    # Form comes AFTER the tabs
    with st.form("analyze_form"):
        col1, col2 = st.columns(2)
        with col1:
            repo_url = st.text_input("GitHub Repo URL")
        with col2:
            repo_zip = st.file_uploader("Or upload zipped repo", type=["zip"])

        task_type = st.selectbox("Select Task", ["summary", "qa", "docs", "refactor"])
        user_query = st.text_area("Ask a question (only for QA)", height=100)
        submit = st.form_submit_button("🚀 Run Analysis")

    if submit:
        last = st.session_state.get("last_result", {})
        last_repo_url = st.session_state.get("last_repo_url", None)
        last_repo_zip_hash = st.session_state.get("last_repo_zip_hash", None)

        reuse_previous_analysis = False

        if isinstance(last, dict) and last.get("code_files"):
            if repo_zip:
                zip_hash = compute_file_hash(repo_zip)
                if last_repo_zip_hash == zip_hash:
                    reuse_previous_analysis = True
            elif repo_url and last_repo_url == repo_url:
                reuse_previous_analysis = True
            # NEW: Also reuse if same repo but different task
            elif (repo_url and last_repo_url == repo_url) or (repo_zip and last_repo_zip_hash == compute_file_hash(repo_zip)):
                reuse_previous_analysis = True

        if reuse_previous_analysis:
            st.info(f"🔁 Reusing analysis for `{repo_url or 'uploaded zip'}` and running new `{task_type}`.")
        else:
            if repo_zip:
                st.session_state["last_repo_zip_hash"] = compute_file_hash(repo_zip)
                st.session_state["last_repo_url"] = None
            elif repo_url:
                st.session_state["last_repo_url"] = repo_url
                st.session_state["last_repo_zip_hash"] = None

        payload = {
            "repo_url": repo_url if repo_url else None,
            "task_type": task_type,
            "user_query": user_query if task_type == "qa" else None
        }
        st.session_state["task_type"] = task_type

        if reuse_previous_analysis:
            payload["code_files"] = last["code_files"]
            payload["parsed_code"] = last.get("parsed_code")
            payload["embeddings_created"] = True
        else:
            st.session_state["last_repo_url"] = repo_url

        if not backend_is_ready():
            st.error(f"🚫 AI CodeCompass backend not reachable at {API_URL}. Please start the API.")
            st.stop()

        # 🔥 Make request
        with st.spinner(f"Running {task_type} analysis..."):
            try:
                res = requests.post(f"{API_URL}/analyze", json=payload)

                # Check if the response is successful
                if res.status_code == 200:
                    data = res.json()
                    data["code_files"] = data.get("code_files", {})
                    data["parsed_code"] = data.get("parsed_code", {})  
                    data["task_type"] = task_type
                    st.session_state.results_cache[task_type] = data
                    st.session_state.last_result = data
                    
                    # 🔥 AUTO-SAVE LOGS (moved from render_result function)
                    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    logs_dir = os.path.join("logs")
                    os.makedirs(logs_dir, exist_ok=True)
                    
                    # Save skipped files if any
                    metadata = data.get("metadata", {})
                    skipped = metadata.get("skipped_large_files", [])
                    if skipped:
                        df = pd.DataFrame(skipped)
                        csv = df.to_csv(index=False).encode("utf-8")
                        with open(os.path.join(logs_dir, f"skipped_files_{now_str}.csv"), "wb") as f:
                            f.write(csv)
                        st.info(f"💾 Saved {len(skipped)} skipped files to logs/skipped_files_{now_str}.csv")
                    
                    # Save chunk metrics if any
                    chunk_metrics = metadata.get("chunk_metrics", [])
                    if chunk_metrics:
                        json_data = json.dumps(chunk_metrics, indent=2).encode("utf-8")
                        with open(os.path.join(logs_dir, f"chunk_metrics_{now_str}.json"), "wb") as f:
                            f.write(json_data)
                        st.info(f"📊 Saved chunk metrics to logs/chunk_metrics_{now_str}.json")
                    
                    # Save analysis summary
                    summary_data = {
                        "task_type": task_type,
                        "timestamp": datetime.now().isoformat(),
                        "repo_url": repo_url,
                        "files_processed": data.get("files_processed", 0),
                        "statistics": data.get("statistics", {}),
                        "metadata": metadata
                    }
                    summary_json = json.dumps(summary_data, indent=2).encode("utf-8")
                    with open(os.path.join(logs_dir, f"analysis_summary_{task_type}_{now_str}.json"), "wb") as f:
                        f.write(summary_json)
                    
                    st.success(f"✅ Analysis completed! Logs saved to logs/ directory.")
                else:
                    st.error(f"❌ Error: {res.text}")
            except Exception as e:
                st.error(f"❗ Exception: {e}")

elif selected == "Dashboard":
    st.title("📊 Project Dashboard")
    
    # Check if we have analysis data
    last_result = st.session_state.get("last_result")
    
    if not last_result or not isinstance(last_result, dict):
        st.info("🔍 Run an analysis first to see dashboard data!")
        st.markdown("### 🚀 Get Started")
        st.markdown("1. Go to **Analyze Repo** tab")
        st.markdown("2. Enter a GitHub URL or upload a zip file")
        st.markdown("3. Select your analysis type")
        st.markdown("4. Come back here to see detailed insights!")
    
    # Extract data from last analysis
    metadata = last_result.get("metadata", {})
    statistics = last_result.get("statistics", {})
    code_files = last_result.get("code_files", {})
    chunk_metrics = metadata.get("chunk_metrics", [])
    skipped_files = metadata.get("skipped_large_files", [])
    repo_info = metadata.get("repo_info", {})
    
    # Calculate metrics
    total_files = statistics.get("total_files", 0)
    files_processed = last_result.get("files_processed", 0)
    
    # Analyze code files for language distribution
    language_stats = {}
    total_lines = 0
    total_functions = 0
    total_classes = 0
    
    for file_path, content in code_files.items():
        # Determine language from file extension
        ext = os.path.splitext(file_path)[1].lower()
        lang_map = {
            '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
            '.java': 'Java', '.cpp': 'C++', '.c': 'C', '.cs': 'C#',
            '.php': 'PHP', '.rb': 'Ruby', '.go': 'Go', '.rs': 'Rust',
            '.html': 'HTML', '.css': 'CSS', '.scss': 'SCSS',
            '.json': 'JSON', '.xml': 'XML', '.yaml': 'YAML', '.yml': 'YAML',
            '.md': 'Markdown', '.txt': 'Text', '.sh': 'Shell',
            '.sql': 'SQL', '.r': 'R', '.m': 'MATLAB'
        }
        
        language = lang_map.get(ext, 'Other')
        
        if language not in language_stats:
            language_stats[language] = {'files': 0, 'lines': 0}
        
        language_stats[language]['files'] += 1
        
        # Count lines (rough estimate)
        if isinstance(content, str):
            lines = len(content.split('\n'))
            language_stats[language]['lines'] += lines
            total_lines += lines
            
            # Simple heuristics for functions/classes (Python example)
            if language == 'Python':
                total_functions += content.count('def ')
                total_classes += content.count('class ')
            elif language in ['JavaScript', 'TypeScript']:
                total_functions += content.count('function ') + content.count('=> ')
                total_classes += content.count('class ')
            elif language == 'Java':
                total_functions += content.count('public ') + content.count('private ') + content.count('protected ')
                total_classes += content.count('class ') + content.count('interface ')
    
    # Main metrics row
    st.markdown("### 📈 Repository Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_files = files_processed - total_files if total_files > 0 else None
        st.metric(
            "Files Processed", 
            f"{files_processed:,}", 
            delta=f"{delta_files:+,}" if delta_files else None
        )
    
    with col2:
        languages_count = len(language_stats)
        st.metric("Languages Detected", languages_count)
    
    with col3:
        st.metric("Total Lines", f"{total_lines:,}")
    
    with col4:
        st.metric("Code Entities", f"{total_functions + total_classes:,}")
    
    # Secondary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Functions", f"{total_functions:,}")
    
    with col2:
        st.metric("Classes", f"{total_classes:,}")
    
    with col3:
        skipped_count = len(skipped_files)
        st.metric("Skipped Files", skipped_count)
    
    with col4:
        avg_chunk_time = sum(c.get('duration_s', 0) for c in chunk_metrics) / len(chunk_metrics) if chunk_metrics else 0
        st.metric("Avg Chunk Time", f"{avg_chunk_time:.1f}s")
    
    st.markdown("---")
    
    # Language distribution charts
    if language_stats:
        st.markdown("### 🌐 Language Distribution")
        
        # Create DataFrame for visualizations
        lang_df = pd.DataFrame([
            {'Language': lang, 'Files': stats['files'], 'Lines': stats['lines']}
            for lang, stats in language_stats.items()
        ]).sort_values('Files', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Files by Language**")
            chart = alt.Chart(lang_df.head(10)).mark_bar(color='#1f77b4').encode(
                x=alt.X('Files:Q', title='Number of Files'),
                y=alt.Y('Language:N', sort='-x', title='Language'),
                tooltip=['Language', 'Files', 'Lines']
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        
        with col2:
            st.markdown("**Lines by Language**")
            chart = alt.Chart(lang_df.head(10)).mark_bar(color='#ff7f0e').encode(
                x=alt.X('Lines:Q', title='Lines of Code'),
                y=alt.Y('Language:N', sort='-x', title='Language'),
                tooltip=['Language', 'Files', 'Lines']
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
    
    # Processing performance
    if chunk_metrics:
        st.markdown("### ⚡ Processing Performance")
        
        chunk_df = pd.DataFrame(chunk_metrics)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Chunk Processing Time**")
            chart = alt.Chart(chunk_df).mark_line(point=True).encode(
                x=alt.X('chunk:O', title='Chunk Number'),
                y=alt.Y('duration_s:Q', title='Duration (seconds)'),
                tooltip=['chunk', 'files', 'duration_s']
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        
        with col2:
            st.markdown("**Files per Chunk**")
            chart = alt.Chart(chunk_df).mark_bar().encode(
                x=alt.X('chunk:O', title='Chunk Number'),
                y=alt.Y('files:Q', title='Files Processed'),
                color=alt.Color('duration_s:Q', scale=alt.Scale(scheme='viridis')),
                tooltip=['chunk', 'files', 'duration_s']
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
    
    # Repository insights
    st.markdown("### 🔍 Repository Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Analysis Summary**")
        task_type = last_result.get("task_type", "unknown")
        st.info(f"📋 Last Analysis: **{task_type.upper()}**")
        
        if repo_info:
            st.markdown("**Repository Info:**")
            for key, value in repo_info.items():
                if isinstance(value, (str, int, float)):
                    st.write(f"• {key.replace('_', ' ').title()}: {value}")
        
        # File size distribution
        if skipped_files:
            st.markdown("**Large Files Skipped:**")
            for file in skipped_files[:5]:  # Show top 5
                st.write(f"• {os.path.basename(file['path'])}: {file['size_kb']:,} KB")
            if len(skipped_files) > 5:
                st.write(f"• ... and {len(skipped_files) - 5} more")
    
    with col2:
        st.markdown("**Quick Stats**")
        
        # Calculate some interesting metrics
        if language_stats:
            most_used_lang = max(language_stats.items(), key=lambda x: x[1]['files'])
            st.write(f"🏆 Most used language: **{most_used_lang[0]}** ({most_used_lang[1]['files']} files)")
            
            largest_lang = max(language_stats.items(), key=lambda x: x[1]['lines'])
            st.write(f"📏 Largest codebase: **{largest_lang[0]}** ({largest_lang[1]['lines']:,} lines)")
        
        if total_files > 0:
            processing_ratio = (files_processed / total_files) * 100
            st.write(f"⚡ Processing efficiency: **{processing_ratio:.1f}%**")
        
        if chunk_metrics:
            total_time = sum(c.get('duration_s', 0) for c in chunk_metrics)
            st.write(f"⏱️ Total processing time: **{total_time:.1f}s**")
        
        # Code complexity estimate
        if total_lines > 0:
            complexity_score = (total_functions + total_classes) / total_lines * 1000
            complexity_level = "Low" if complexity_score < 5 else "Medium" if complexity_score < 15 else "High"
            st.write(f"🧮 Code complexity: **{complexity_level}** ({complexity_score:.1f})")
    
    # Export options
    st.markdown("---")
    st.markdown("### 📤 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if language_stats:
            lang_csv = pd.DataFrame([
                {'Language': lang, 'Files': stats['files'], 'Lines': stats['lines']}
                for lang, stats in language_stats.items()
            ]).to_csv(index=False)
            st.download_button(
                "📊 Download Language Stats",
                lang_csv,
                "language_stats.csv",
                "text/csv"
            )
    
    with col2:
        if chunk_metrics:
            chunk_csv = pd.DataFrame(chunk_metrics).to_csv(index=False)
            st.download_button(
                "⚡ Download Performance Data",
                chunk_csv,
                "performance_data.csv",
                "text/csv"
            )
    
    with col3:
        # Complete analysis summary
        summary_data = {
            "analysis_type": task_type,
            "timestamp": datetime.now().isoformat(),
            "total_files": total_files,
            "files_processed": files_processed,
            "languages_detected": len(language_stats),
            "total_lines": total_lines,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "language_breakdown": language_stats
        }
        summary_json = json.dumps(summary_data, indent=2)
        st.download_button(
            "📋 Download Full Summary",
            summary_json,
            "analysis_summary.json",
            "application/json"
        )

elif selected == "About":
    st.title("ℹ️ About  AI CodeCompass")
    st.markdown("""
    **AI CodeCompass** is your autonomous programming assistant that can:

    - Ingest any GitHub repo or zipped source code
    - Summarize the entire project architecture
    - Answer intelligent questions about the codebase
    - Generate full module documentation
    - Suggest refactor and optimization tips

    Built with ❤️ using LLaMA2, LangChain, LangGraph, FAISS, and Streamlit.
    """)