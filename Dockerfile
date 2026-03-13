# Use Python 3.10
FROM python:3.10

# Set working directory
WORKDIR /code

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create a shell script to run both services
RUN echo "#!/bin/bash\n\
uvicorn api:app --host 0.0.0.0 --port 8000 &\n\
streamlit run app/streamlit_app.py --server.port 7860 --server.address 0.0.0.0\n\
" > start.sh

RUN chmod +x start.sh

# Hugging Face uses port 7860
CMD ["./start.sh"]