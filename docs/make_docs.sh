#!/usr/bin/env bash

# Set the package name from the directory above this file (should be in ./docs/)
PACKAGE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && cd .. && pwd )"
PACKAGE_NAME=$(basename "$PACKAGE_DIR")

# Creates the documentation for the package
MODULES="$PACKAGE_NAME ./tests"

# We assume the appropriate virtual environment has already been activated
pip install pdoc -q --disable-pip-version-check && \
pdoc -t "$PACKAGE_DIR/docs/theme" -o "$PACKAGE_DIR/docs" $MODULES && \
echo "Successfully generated documentation at $PACKAGE_DIR/docs."
