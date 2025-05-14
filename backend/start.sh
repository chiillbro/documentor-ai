#!/bin/bash

# start Ollama serve in the background

echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!
echo "Ollama server started with PID: $OLLAMA_PID"


# Wait a few seconds for the Ollama to initialize
# (Can be made more robust by checking for Ollama health)
sleep 10


# Pull the model if it doesn't exist (Ollama will handle this)
# This reliles on the OLLAMA_MODEL_NAME env var being set for the container
# The OLLAMA_MODELS env var in docker-compose points to /root/.ollama/models
# For Render/Fly.io, we'll need a persistant disk for /root/.ollama/models

echo "Ensuring Ollama model ${OLLAMA_MODEL_NAME} exists..."
ollama pull ${OLLAMA_MODEL_NAME}

echo "Ollama model check complete."

# Start Uvicorn server for FastAPI app
echo "Starting FastAPI application with Uvicorn..."
exec uvicorn app.main:app --host 0.0.0.0 --port 80 # Render typically expects port 80 or 10000
# Or use the PORT env var if set: --port ${PORT:-80}

# Graceful shutdown (optional, but good)
# trap "echo 'Shutting down Ollama...'; kill $OLLAMA_PID; wait $OLLAMA_PID; echo 'Ollama shut down.'; exit 0" SIGINT SIGTERM