#!/usr/bin/env bash

# Creates all requisite Docker images and runs tests for the package
# Customize for your package by adding to PREQUISITE_IMAGES as needed
# Will create all Docker images with the latest tag

# Set the package name from the directory of this file
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PACKAGE_NAME=$(basename "$SCRIPT_DIR")

# Build prerequisite Docker images if needed (for private dependencies)
PREREQUISITE_IMAGES=()
for IMAGE in "${PREREQUISITE_IMAGES[@]}"; do
  # Run their make_docker.sh scripts, assuming that they're in the same directory
  bash "$SCRIPT_DIR"/../"$IMAGE"/make_docker.sh
done

# Build the Docker image
docker build -t "$PACKAGE_NAME":latest "$SCRIPT_DIR"

# Run the Docker container, install requirements.txt, and run pytest
docker run \
  -it --rm \
  -v /mnt/HDSCA_Development:/mnt/HDSCA_Development \
  -v /mnt/csidata:/mnt/csidata \
  --entrypoint="" \
  "$PACKAGE_NAME":latest \
  /bin/bash -c "pip install -r requirements.txt && pytest"
