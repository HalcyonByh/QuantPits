#!/bin/bash
# Script to build and run the Dockerized test environment

set -e

# Default variables
IMAGE_NAME="quantpits-test-env"
CONTAINER_NAME="quantpits-test-run"
DOCKER_BUILD_FLAGS=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --rebuild) DOCKER_BUILD_FLAGS="--no-cache"; shift ;;
        -h|--help) 
            echo "Usage: ./run_tests_docker.sh [OPTIONS]"
            echo "Options:"
            echo "  --rebuild    Force a clean rebuild of the Docker image without using cache"
            echo "  -h, --help   Show this help message"
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

cd "$(dirname "$0")"

echo "============================================================"
if [ -n "$DOCKER_BUILD_FLAGS" ]; then
    echo "Building Docker image FROM SCRATCH ($IMAGE_NAME)..."
else
    echo "Building Docker image ($IMAGE_NAME) using layer cache..."
    echo "(Use --rebuild if you want to install new dependencies from scratch)"
fi
echo "============================================================"

# Build the test image
docker build $DOCKER_BUILD_FLAGS -f Dockerfile.test -t $IMAGE_NAME .

echo "============================================================"
echo "Running tests in isolated environment..."
echo "============================================================"

# Run the container and automatically remove it when done
docker run --rm --name $CONTAINER_NAME $IMAGE_NAME
