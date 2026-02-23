#!/usr/bin/env bash
# Download TII drone-racing-dataset using curl (macOS has curl, not wget).
# Run from anywhere; pass the path to your cloned drone-racing-dataset repo.
#
# Usage:
#   ./gate_detection/tests/scripts/download_tii_dataset_curl.sh /path/to/drone-racing-dataset
#   ./gate_detection/tests/scripts/download_tii_dataset_curl.sh ../external_data/drone-racing-dataset

set -e
DATASET_ROOT="${1:?Usage: $0 /path/to/drone-racing-dataset}"
cd "$DATASET_ROOT"
mkdir -p data && cd data

BASE="https://github.com/Drone-Racing/drone-racing-dataset/releases/download/v3.0.0"

# echo "Downloading autonomous chunks..."
# curl -L -f -o autonomous_zipchunk01 "$BASE/autonomous_zipchunk01"
# curl -L -f -o autonomous_zipchunk02 "$BASE/autonomous_zipchunk02"
# curl -L -f -o autonomous_zipchunk03 "$BASE/autonomous_zipchunk03"
# cat autonomous_zipchunk* > autonomous.zip
# rm -f autonomous_zipchunk*

echo "Downloading piloted chunks..."
# for i in 01 02 03 04 05 06 07; do
#   curl -L -f -o "piloted_zipchunk$i" "$BASE/piloted_zipchunk$i"
# done
# cat piloted_zipchunk* > piloted.zip
# rm -f piloted_zipchunk*

echo "Unzipping..."
unzip -o autonomous.zip
# unzip -o piloted.zip
find autonomous piloted -type f -name '*.zip' -exec sh -c 'unzip -o -d "$(dirname "$1")" "$1"' _ {} \;

echo "Done. data/autonomous and data/piloted are ready."
