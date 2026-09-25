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

# Until Chapter 22 gives Foresight an API, the container checks its own environment and
# generates the dataset. Chapter 22 replaces this line with the service, adds a health
# check, and exposes port 8000.
CMD ["sh", "-c", "python code/meridian/generate.py && python code/meridian/generate_ml.py && python code/_preflight.py"]
