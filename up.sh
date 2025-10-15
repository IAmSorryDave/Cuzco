# Script to build and run the backup inference server Docker container

IMAGE_NAME="cuzco"
PORT=7860

read -p "Enter read-only Hugging Face Token: " kung_foo

export MY_SECRET=kung_foo

# Build the Docker image
echo "Building Docker image: $IMAGE_NAME..."
docker build --secret id=my_secret_id, env=MY_SECRET -t $IMAGE_NAME .

# Run the Docker container
echo "Running Docker container on port $PORT..."
docker run -p $PORT:$PORT $IMAGE_NAME
