#!/bin/bash
# Script to stop the backup inference server Docker container

IMAGE_NAME="cuzco-backup-inference-server"

# Find the running container ID for the image
CONTAINER_ID=$(docker ps -q --filter ancestor=$IMAGE_NAME)

if [ -z "$CONTAINER_ID" ]; then
  echo "No running container found for image: $IMAGE_NAME."
  exit 0
fi

echo "Stopping container: $CONTAINER_ID..."
docker stop $CONTAINER_ID
