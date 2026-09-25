# Foresight, in a container. Four decisions, each of which has a reason.
FROM python:3.12-slim AS base

# 1. A non-root user. A container that runs as root is a container where a code
#    execution bug becomes a host problem.
RUN useradd --create-home --uid 10001 foresight
WORKDIR /app

# 2. Dependencies in their own layer, before the source. The requirements change
#    rarely and the source changes constantly; this ordering is the difference
#    between a four-second rebuild and a four-minute one. PyTorch comes from its CPU
#    index first, so the image does not carry gigabytes of GPU libraries it never uses.
# LightGBM needs the GNU OpenMP runtime, which the slim image leaves out.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=foresight:foresight code/ code/
USER foresight

# 3. No secrets. Not in an ARG, not in an ENV, not in a file. They arrive at runtime
#    from the platform's secret store, because a secret baked into a layer is in every
#    registry that pulled it and deleting the layer does not remove it.
#
# 4. The same determinism settings the book's runner uses, so a score computed in the
#    container matches a score computed on the page to the last digit.
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/code PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TZ=UTC

# The container runs Foresight's API (Chapter 22). Data, registry and
# scores arrive as volumes (docker-compose.yml), never in the image;
# the same image runs the monthly job with another command.
EXPOSE 8000

# Healthy means the process answers. /health also says "degraded" when
# a model is missing: a reason to alert somebody, not to restart, which
# would load the same registry and miss the same model. No curl here.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request as u; \
u.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "foresight.serve.api:app", \
     "--host", "0.0.0.0", "--port", "8000"]
