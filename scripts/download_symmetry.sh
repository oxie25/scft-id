#!/bin/bash

# Script to download symmetry group data from GitHub repository
# Downloads directories 1, 2, and 3 from https://github.com/dmorse/pscfpp/tree/master/data/groups

set -e  # Exit on any error

# Configuration
REPO_URL="https://github.com/dmorse/pscfpp.git"
TARGET_DIRS=("1" "2" "3")
OUTPUT_DIR="structures/symmetry"
TEMP_DIR="temp_repo"

echo "Starting download of symmetry group data..."

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Clone the repository with sparse checkout
echo "Cloning repository..."
git clone --no-checkout --depth 1 "$REPO_URL" "$TEMP_DIR"
cd "$TEMP_DIR"

# Enable sparse checkout
git sparse-checkout init --cone

# Configure sparse checkout to only download the target directories
echo "Configuring sparse checkout for directories: ${TARGET_DIRS[*]}"
for dir in "${TARGET_DIRS[@]}"; do
    echo "data/groups/$dir" >> .git/info/sparse-checkout
done

# Checkout the files
echo "Downloading files..."
git checkout

# Copy the downloaded directories to output location
echo "Copying files to output directory..."
for dir in "${TARGET_DIRS[@]}"; do
    if [ -d "data/groups/$dir" ]; then
        cp -r "data/groups/$dir" "../$OUTPUT_DIR/"
        echo "✓ Downloaded directory: $dir"
    else
        echo "✗ Directory not found: $dir"
    fi
done

# Clean up
cd ..
rm -rf "$TEMP_DIR"

echo "Download complete! Files saved to: $OUTPUT_DIR"
echo "Downloaded directories:"
ls -la "$OUTPUT_DIR"
