# ********************* Production Dockerfile ********************

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Ollama (curl) and potentially for psycopg if not using psycopg[binary]
# and for building some Python packages if wheels aren't available
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    # build-essential libpq-dev # Only if not using psycopg[binary] and need to compile
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh
# This installs ollama to /usr/local/bin/ollama and sets up a service,
# but we'll run `ollama serve` directly with our script.

# Copy requirements and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./app /app/app
COPY ./start.sh /app/start.sh
RUN chmod +x /app/start.sh
# Executable bit is set here within the Linux build env

# Expose the port Uvicorn will listen on
EXPOSE 80

# Set default environment variables for Ollama if not overridden
ENV OLLAMA_HOST="0.0.0.0:11434"
# OLLAMA_MODELS="/app/ollama_models" # Point to a path inside the container for models
                                 # This path needs to be a persistent volume on Render/Fly.io

# The start.sh script will handle starting Ollama and then Uvicorn
CMD ["/app/start.sh"]