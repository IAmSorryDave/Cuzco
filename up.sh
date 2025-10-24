# Script to build and run the backup inference server Docker container

IMAGE_NAME="cuzco"
PORT=7860

read -s -p "Hugging Face Read Access Token: " TOKEN

export NONE_OF_YOUR_BEES_WAX=TOKEN

# Build the Docker image
# echo "Building Docker image: $IMAGE_NAME..."

echo "Building Docker image: $IMAGE_NAME..."
DOCKER_BUILDKIT=1 docker build --secret id=mums_the_word,env=NONE_OF_YOUR_BEES_WAX -t $IMAGE_NAME .

# Run the Docker container
echo "Running Docker container on port $PORT..."
docker run -d -p $PORT:$PORT -e "HF_ACCESS_TOKEN=$NONE_OF_YOUR_BEES_WAX" $IMAGE_NAME
