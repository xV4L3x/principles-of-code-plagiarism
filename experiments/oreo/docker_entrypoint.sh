#!/bin/bash
# Oreo Phase 2 + Phase 3 inside Docker.
# Expects the container to be launched with:
#   -v <oreo-artifact>:/oreo-artifact
#   -v <blocks.file>:/data/blocks.file:ro
#   -v <output-dir>:/data/output

set -e

OREO=/oreo-artifact/oreo
BLOCKS=/data/blocks.file
OUTPUT=/data/output

echo "=== Oreo Docker entrypoint ==="
echo "Java: $(java -version 2>&1 | head -1)"
echo "Python: $(python3 --version)"
echo ""

# ── Sanity checks ─────────────────────────────────────────────────────────────
[ -f "$BLOCKS" ]             || { echo "ERROR: $BLOCKS not found"; exit 1; }
[ -f "$OREO/ml_model/oreo_model_fse.h5" ] || { echo "ERROR: model not found"; exit 1; }
mkdir -p "$OUTPUT"

# ── Build the SourcererCC JAR (Java 11 is compatible with source 1.8) ─────────
echo "=== Phase 2: building SourcererCC JAR ==="
cd "$OREO/clone-detector"

# Fix runnodes.sh: reduce heap from 10g to 2g so the container doesn't OOM
sed -i 's/-Xms10g/-Xms512m/g; s/-Xmx10g/-Xmx2g/g' runnodes.sh

# Fix the OUTPUT_DIR variable reference that EProperties fails to resolve
sed -i 's/OUTPUT_DIR=\${NODE_PREFIX}\/output/OUTPUT_DIR=NODE_1\/output/' sourcerer-cc.properties

cd "$OREO/clone-detector"
ant -buildfile build.xml cdi -q
echo "JAR built."

# ── Set up SourcererCC input ───────────────────────────────────────────────────
echo ""
echo "=== Phase 2: running SourcererCC ==="
mkdir -p "$OREO/clone-detector/input/dataset"
cp "$BLOCKS" "$OREO/clone-detector/input/dataset/blocks.file"

# Remove any stale candidate artifacts (e.g. from a prior Python Phase 2 run
# on the host that left macOS absolute paths in candidatesList.txt)
rm -rf "$OREO/results/candidates/"
rm -rf "$OREO/results/predictions/"

# Clean previous run state if present
[ -f "$OREO/clone-detector/scriptinator_metadata.scc" ] && \
    rm "$OREO/clone-detector/scriptinator_metadata.scc"
bash "$OREO/clone-detector/cleanup.sh" 2>/dev/null || true

cd "$OREO/clone-detector"
python3 controller.py 1
echo "SourcererCC done."

# ── Phase 3: ML prediction (all ports) ───────────────────────────────────────
echo ""
echo "=== Phase 3: Siamese model prediction ==="
mkdir -p "$OREO/results/predictions"
cd "$OREO/python_scripts"

# Run Predictor for each port that has candidate files
for port in 9900 9901 9902 9903; do
    list="$OREO/results/candidates/$port/candidatesList.txt"
    if [ -f "$list" ]; then
        echo "Running Predictor for port $port..."
        python3 Predictor.py $port
    fi
done
echo "Prediction done."

# ── Collect output ────────────────────────────────────────────────────────────
echo ""
echo "=== Collecting output ==="
find "$OREO/results/predictions" -name "*.txt" -exec cp {} "$OUTPUT/" \;
n=$(ls "$OUTPUT"/*.txt 2>/dev/null | wc -l | tr -d ' ')
echo "Copied $n prediction file(s) to $OUTPUT"
