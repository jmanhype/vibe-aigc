# Vibe AIGC Docker Image
FROM python:3.12-slim

LABEL maintainer="Vibe AIGC Contributors"
LABEL description="A New Paradigm for Content Generation via Agentic Orchestration"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN useradd --create-home --shell /bin/bash vibe

# Set working directory
WORKDIR /app

# Install dependencies
COPY pyproject.toml README.md LICENSE ./
COPY vibe_aigc/ ./vibe_aigc/

RUN pip install --no-cache-dir -e .

# Switch to non-root user
USER vibe

# Default command
ENTRYPOINT ["vibe-aigc"]
CMD ["--help"]
