# Script to build and run the backup inference server Docker container

IMAGE_NAME="cuzco-backup-inference-server"
PORT=7860

# Build the Docker image
echo "Building Docker image: $IMAGE_NAME..."
docker build -t $IMAGE_NAME .

# Run the Docker container
echo "Running Docker container on port $PORT..."
docker run -p $PORT:$PORT $IMAGE_NAME
