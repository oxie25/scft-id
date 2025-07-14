#!/bin/bash

# Script to download data from Zenodo repository
# Usage: ./download_data.sh [ZENODO_RECORD_ID] [DESTINATION_DIR]

set -e  # Exit immediately if a command exits with a non-zero status

# Default values
ZENODO_RECORD_ID="15756493"
DESTINATION_DIR=${1:-"structures"}

# Function to display usage instructions
usage() {
    echo "Usage: $0 [DESTINATION_DIR]"
    echo "  DESTINATION_DIR: Directory to save downloaded files (default: structures)"
    exit 1
}

# Create destination directory if it doesn't exist
mkdir -p "$DESTINATION_DIR"
echo "Downloading files to $DESTINATION_DIR"

# Get the Zenodo record metadata to extract file links
echo "Fetching metadata for Zenodo record $ZENODO_RECORD_ID..."
METADATA_URL="https://zenodo.org/api/records/$ZENODO_RECORD_ID"

# Check if curl or wget is available
if command -v curl &> /dev/null; then
    METADATA=$(curl -s "$METADATA_URL")
elif command -v wget &> /dev/null; then
    METADATA=$(wget -q -O - "$METADATA_URL")
else
    echo "Error: Neither curl nor wget is installed. Please install one of them."
    exit 1
fi

# Check if jq is available for JSON parsing
if ! command -v jq &> /dev/null; then
    echo "Warning: jq is not installed. Will attempt a basic URL extraction."
    # Simple extraction using grep and sed - less reliable but a fallback
    echo "$METADATA" | grep -o 'https://zenodo.org/api/files/[^"]*' | while read -r URL; do
        FILENAME=$(basename "$URL")
        echo "Downloading $FILENAME..."
        if command -v curl &> /dev/null; then
            curl -L "$URL" -o "$DESTINATION_DIR/$FILENAME"
        else
            wget -O "$DESTINATION_DIR/$FILENAME" "$URL"
        fi
    done
else
    # Use jq for proper JSON parsing
    echo "$METADATA" | jq -r '.files[] | .links.self' | while read -r URL; do
        FILENAME=$(echo "$METADATA" | jq -r ".files[] | select(.links.self == \"$URL\") | .key")
        echo "Downloading $FILENAME..."
        if command -v curl &> /dev/null; then
            curl -L "$URL" -o "$DESTINATION_DIR/$FILENAME"
        else
            wget -O "$DESTINATION_DIR/$FILENAME" "$URL"
        fi
    done
fi

echo "Download complete! Files saved to $DESTINATION_DIR"