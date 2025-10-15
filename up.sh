# Script to build and run the backup inference server Docker container

IMAGE_NAME="cuzco"
PORT=7860

read -s -p "Hugging Face Read Access Token: " TOKEN

export MY_SECRET=TOKEN

# Build the Docker image
echo "Building Docker image: $IMAGE_NAME..."
docker build --secret id=env,env=MY_SECRET -t $IMAGE_NAME .

# Run the Docker container
echo "Running Docker container on port $PORT..."
docker run -p $PORT:$PORT $IMAGE_NAME
