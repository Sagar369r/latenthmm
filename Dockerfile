FROM python:3.12-slim

# Set non-interactive to avoid apt-get prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update apt and install essential C++ build tools for ONNX and Treelite
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgomp1 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip securely
RUN pip install --upgrade pip

# Copy the stripped down ML requirements
COPY docker-requirements.txt .

# Install dependencies (We use --no-cache-dir to keep the image slim)
RUN pip install --no-cache-dir -r docker-requirements.txt

# Copy the entire workspace into the container
# Note: we exclude caches using .dockerignore
COPY . .

# Set PYTHONPATH to the root directory so the orchestrators can resolve imports
ENV PYTHONPATH="/app"

# Default command (can be overridden by docker-compose)
CMD ["bash"]
