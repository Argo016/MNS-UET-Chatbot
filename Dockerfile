FROM python:3.12-slim AS build
WORKDIR /app

RUN apt-get update && \
    apt-get install --no-install-suggests --no-install-recommends -y \
    build-essential \
    libpq-dev \
    libatlas-base-dev \
    clang \
    libclang-dev \
    cmake \
    pkg-config \
    git && \
    python3 -m venv /venv && \
    /venv/bin/pip install --disable-pip-version-check --upgrade pip setuptools wheel && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN /venv/bin/pip install --disable-pip-version-check --no-cache-dir --no-build-isolation \
    -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

# Final stage
FROM python:3.12-slim
WORKDIR /app

COPY --from=build /venv /venv
COPY --from=build /app /app

ENV PATH="/venv/bin:$PATH"

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false"]
