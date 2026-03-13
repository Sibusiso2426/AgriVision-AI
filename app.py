import subprocess
import time
import os

# 1. Start the FastAPI backend as a background process
backend_process = subprocess.Popen(["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"])

# 2. Wait a few seconds for the backend to wake up
time.sleep(5)

# 3. Start the Streamlit frontend
# Point this to wherever your main streamlit file is
subprocess.run(["streamlit", "run", "app/streamlit_app.py", "--server.port", "7860", "--server.address", "0.0.0.0"])