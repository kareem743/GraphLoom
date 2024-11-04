# GraphLoom Demo

## Llama Demo

### Running the Docker Container

To start the `open-webui` container with specific configurations, use the following `docker run` command:

```bash
docker run \
  --hostname=0dd6d76d1824 \                    # Set the hostname for the container
  --user=0:0 \                                 # Run as root user
  --mac-address=02:42:ac:11:00:02 \            # Set MAC address
  --env PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \  # System paths
  --env LANG=C.UTF-8 \                         # Set language to UTF-8
  --env GPG_KEY=A035C8C19219BA821ECEA86B64E628F8D684696D \  # GPG key for package signing
  --env PYTHON_VERSION=3.11.10 \               # Python version used
  --env PYTHON_SHA256=07a4356e912900e61a15cb0949a06c4a05012e213ecd6b4e84d0f67aabbee372 \  # Python checksum
  --env ENV=prod \                             # Environment mode (production)
  --env PORT=8080 \                            # App port inside the container
  --env USE_OLLAMA_DOCKER=false \              # Configure Ollama Docker support
  --env USE_CUDA_DOCKER=false \                # Use CUDA support (false by default)
  --env USE_CUDA_DOCKER_VER=cu121 \            # CUDA version, if enabled
  --env USE_EMBEDDING_MODEL_DOCKER=sentence-transformers/all-MiniLM-L6-v2 \  # Embedding model
  --env USE_RERANKING_MODEL_DOCKER= \          # Reranking model (not set here)
  --env OLLAMA_BASE_URL=/ollama \              # Base URL for Ollama
  --env OPENAI_API_BASE_URL= \                 # Base URL for OpenAI API (if set)
  --env OPENAI_API_KEY= \                      # OpenAI API key (if needed)
  --env WEBUI_SECRET_KEY= \                    # Secret key for web UI security
  --env SCARF_NO_ANALYTICS=true \              # Disable analytics tracking
  --env DO_NOT_TRACK=true \                    # Further disable tracking
  --env ANONYMIZED_TELEMETRY=false \           # Disable anonymous telemetry
  --env WHISPER_MODEL=base \                   # Model for Whisper (speech recognition)
  --env WHISPER_MODEL_DIR=/app/backend/data/cache/whisper/models \  # Whisper model directory
  --env RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \  # Retrieval Augmented Generation embedding model
  --env RAG_RERANKING_MODEL= \                 # RAG reranking model (not set here)
  --env SENTENCE_TRANSFORMERS_HOME=/app/backend/data/cache/embedding/models \  # Sentence transformers cache
  --env TIKTOKEN_ENCODING_NAME=cl100k_base \   # Encoding name for tokenization
  --env TIKTOKEN_CACHE_DIR=/app/backend/data/cache/tiktoken \  # Cache directory for tokenization
  --env HF_HOME=/app/backend/data/cache/embedding/models \    # Hugging Face cache
  --env HOME=/root \                           # Set home directory
  --env WEBUI_BUILD_VERSION=7228b39064ac28e1240bf8998f2a35535c6f7ef5 \  # Build version of web UI
  --env DOCKER=true \                          # Run inside Docker
  --volume=open-webui:/app/backend/data \      # Mount volume for persistent data
  --workdir=/app/backend \                     # Set working directory in the container
  -p 3000:8080 \                               # Map port 3000 (host) to port 8080 (container)
  --restart=always \                           # Automatically restart the container if it stops
  --label 'org.opencontainers.image.created=2024-10-30T17:16:48.499Z' \   # Creation date label
  --label 'org.opencontainers.image.description=User-friendly AI Interface (Supports Ollama, OpenAI API, ...)' \  # Description label
  --label 'org.opencontainers.image.licenses=MIT' \                       # License label
  --label 'org.opencontainers.image.revision=7228b39064ac28e1240bf8998f2a35535c6f7ef5' \  # Revision label
  --label 'org.opencontainers.image.source=https://github.com/open-webui/open-webui' \    # Source label
  --label 'org.opencontainers.image.title=open-webui' \                                   # Title label
  --label 'org.opencontainers.image.url=https://github.com/open-webui/open-webui' \       # URL label
  --label 'org.opencontainers.image.version=main' \                                       # Version label
  --add-host host.docker.internal:host-gateway \                                          # Host gateway for Docker
  --runtime=runc \                           # Runtime environment
  -d ghcr.io/open-webui/open-webui:main      # Run container in detached mode from GitHub Container Registry
n
