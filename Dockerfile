# ================================================================================
# Dockerfile — mlzero ML Training Dashboard
# ================================================================================
#
# WHAT THIS DOES:
#   Builds a container that runs the Streamlit training dashboard.
#   Everything needed to run is bundled inside the image:
#     - Python 3.11
#     - All pip dependencies (numpy, scikit-learn, streamlit, etc.)
#     - The mlzero package (installed in editable mode)
#     - The app/ and scripts/ code
#
# ANALOGY:
#   A Docker image is like a shipping container for software.
#   It includes everything: the app, the libraries, the Python version.
#   Run it on any machine → same result. No "works on my machine" problems.
#
# HOW TO BUILD AND RUN:
#   docker build -t mlzero .
#   docker run -p 8501:8501 mlzero
#   → open http://localhost:8501
#
#   Or with docker-compose (easier):
#   docker-compose up
# ================================================================================

# ── Base image ─────────────────────────────────────────────────────────────────
# python:3.11-slim = official Python 3.11 on minimal Debian Linux
# "slim" = stripped to essentials (no extra tools) → smaller image (~50MB vs ~300MB)
FROM python:3.11-slim

# ── Metadata labels ────────────────────────────────────────────────────────────
LABEL maintainer="mlzero learner"
LABEL description="ML Training Dashboard — mlzero project"
LABEL version="0.1.0"

# ── System dependencies ────────────────────────────────────────────────────────
# Some Python packages (numpy, matplotlib) need C libraries to compile.
# We install them once here so pip install works cleanly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
#   ^ clean up apt cache to keep image small

# ── Set working directory ──────────────────────────────────────────────────────
# All subsequent commands run from /workspace inside the container.
# This is where we'll copy our project files.
WORKDIR /workspace

# ── Install Python dependencies (cached layer) ────────────────────────────────
# IMPORTANT: Copy requirements.txt FIRST (before copying all code).
# Why? Docker caches layers. If we copy all code first, every code change
# invalidates the pip install cache → slow rebuilds.
# By copying only requirements.txt first, pip install is re-run ONLY when
# dependencies change (rare), not on every code change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir streamlit>=1.28

# ── Copy project files ─────────────────────────────────────────────────────────
COPY setup.py .
COPY src/ src/
COPY app/ app/
COPY scripts/ scripts/
COPY configs/ configs/
COPY docs/ docs/

# ── Install mlzero package ─────────────────────────────────────────────────────
# pip install -e . = editable install = imports like `from mlzero...` work
RUN pip install --no-cache-dir -e .

# ── Create output directories ──────────────────────────────────────────────────
RUN mkdir -p outputs/plots outputs/ci_logs

# ── Expose port ────────────────────────────────────────────────────────────────
# Streamlit runs on port 8501 by default.
# EXPOSE tells Docker "this container listens on port 8501".
# The actual port mapping (host:container) is done at runtime with -p.
EXPOSE 8501

# ── Streamlit config ───────────────────────────────────────────────────────────
# Disable browser auto-open (doesn't work in containers)
# Allow connections from outside the container (0.0.0.0 = all network interfaces)
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ── Default command ────────────────────────────────────────────────────────────
# When someone runs the container, start the Streamlit app.
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
