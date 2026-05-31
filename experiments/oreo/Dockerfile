# Oreo evaluation container
# Provides the exact original environment: Java 8 (for SourcererCC/EProperties),
# Python 3.6, TensorFlow 1.5.0, Keras 2.1.3 (for the Siamese model).
#
# Build:  docker build --platform=linux/amd64 -t oreo-runner experiments/oreo/
# Run via oreo_runner.py --docker  (see runner for volume layout)

FROM --platform=linux/amd64 python:3.6-slim

# Java 8 + Ant (required by SourcererCC)
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-11-jdk-headless \
        ant \
    && rm -rf /var/lib/apt/lists/*

# Python deps — exact versions from oreo-artifact/oreo/python_scripts/dependencies/
RUN pip install --no-cache-dir \
    tensorflow==1.5.0 \
    keras==2.1.3 \
    h5py==2.7.1 \
    pandas==0.22.0 \
    numpy==1.14.0 \
    scipy==1.0.0 \
    scikit-learn==0.19.1

COPY docker_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Volume layout expected at runtime:
#   /oreo-artifact/   ← oreo-artifact repo (read-write: SourcererCC writes state here)
#   /data/blocks.file ← Phase 1 output (read-only)
#   /data/output/     ← predictions written here

ENTRYPOINT ["/entrypoint.sh"]
