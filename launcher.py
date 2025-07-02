# devcopilot_launcher.py 
import subprocess
import time
import requests
import sys
import os
from dotenv import load_dotenv
load_dotenv()  # Make sure it's before any API call or use of os.getenv

def wait_for_backend(url, timeout=30):
    print("⏳ Waiting for FastAPI backend to be ready...")
    for _ in range(timeout):
        try:
            if requests.get(url).status_code == 200:
                print("✅ FastAPI backend is live!")
                return True
        except:
            pass
        time.sleep(1)
    print("❌ Backend not responding after timeout.")
    return False

def main():
    cwd = os.getcwd()
    uvicorn_cmd = ["uvicorn", "api.main:app", "--port", "8000", "--reload"]
    streamlit_cmd = ["streamlit", "run", "ui/streamlit_app.py"]

    print("🚀 Starting FastAPI backend...")
    fastapi_proc = subprocess.Popen(uvicorn_cmd, cwd=cwd)

    if not wait_for_backend("http://localhost:8000/health"):
        fastapi_proc.terminate()
        sys.exit(1)

    print("🖥️ Launching Streamlit UI...")
    try:
        subprocess.run(streamlit_cmd, cwd=cwd)
    except KeyboardInterrupt:
        print("🛑 Stopping...")
    finally:
        print("🧹 Cleaning up...")
        fastapi_proc.terminate()
        fastapi_proc.wait()
        print("✅ Shutdown complete.")

if __name__ == "__main__":
    main()
