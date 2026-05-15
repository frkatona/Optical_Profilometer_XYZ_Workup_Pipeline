FROM python:3.13-slim AS analysis-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY analyze_heightmap.py interpolation_study.py README.md ./

RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

FROM python:3.13-slim AS blender-renderer

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends blender && \
    rm -rf /var/lib/apt/lists/*

COPY blender/render_obj.py /app/blender/render_obj.py

RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["blender", "--background", "--python-exit-code", "1", "--python", "/app/blender/render_obj.py", "--"]
CMD ["--help"]

FROM analysis-base AS runtime

USER appuser

ENTRYPOINT ["python", "analyze_heightmap.py"]
CMD ["--help"]
