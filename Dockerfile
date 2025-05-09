# ---- Builder Stage ----
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    UV_SYSTEM_PYTHON=true

# Install build essentials
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install uv (Python package installer)
RUN pip install --no-cache-dir uv

# Install Python dependencies
COPY pyproject.toml requirements.txt ./
RUN uv pip install -r requirements.txt

# Copy application code and install the project
COPY . .
RUN uv pip install .

# Setup NLTK data and Playwright browsers
ENV NLTK_DATA=/usr/local/share/nltk_data
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright
RUN mkdir -p ${NLTK_DATA} && chmod -R 777 ${NLTK_DATA}
RUN mkdir -p ${PLAYWRIGHT_BROWSERS_PATH} && chmod -R 777 ${PLAYWRIGHT_BROWSERS_PATH}
RUN crawl4ai-setup

# ---- Runtime Stage ----
FROM python:3.12-slim AS runtime

ARG APP_USER=appuser
ARG APP_GROUP=appgroup
ARG UID=1000
ARG GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PATH="/home/${APP_USER}/.local/bin:${PATH}" \
    PORT=8051 \
    NLTK_DATA=/usr/local/share/nltk_data \
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright

# Create non-root user
RUN groupadd -g ${GID} ${APP_GROUP} && \
    useradd -u ${UID} -g ${APP_GROUP} -s /bin/sh -m ${APP_USER}

# Install runtime dependencies, including those for Playwright browsers
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # For Playwright (Chromium)
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdbus-1-3 libdrm2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2 libpangocairo-1.0-0 \
    libx11-xcb1 libxcb-dri3-0 libxcb-glx0 libxcb-shm0 libxshmfence1 libxxf86vm1 \
    # General headless browser support
    xvfb \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages, executables, NLTK data, and Playwright browsers from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY --from=builder ${NLTK_DATA} ${NLTK_DATA}
COPY --from=builder --chown=${APP_USER}:${APP_GROUP} ${PLAYWRIGHT_BROWSERS_PATH} ${PLAYWRIGHT_BROWSERS_PATH}

# Copy application code
COPY . .

# Set ownership for app directory and user cache
RUN chown -R ${APP_USER}:${APP_GROUP} /app \
    && mkdir -p /home/${APP_USER}/.cache/ms-playwright && chown -R ${APP_USER}:${APP_GROUP} /home/${APP_USER}/.cache/ms-playwright

USER ${APP_USER}

EXPOSE ${PORT}

CMD ["python", "-m", "src.crawl4ai_mcp"]
