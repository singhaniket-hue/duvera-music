FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
ARG INSTALL_VOCALS=0
RUN if [ "$INSTALL_VOCALS" = "1" ]; then pip install --no-cache-dir ".[vocals]"; else pip install --no-cache-dir .; fi
RUN useradd --create-home director
USER director
WORKDIR /work
ENTRYPOINT ["duvera-music"]
