docker build --rm -f "Dockerfile" -t homevision:latest "."

docker run -it --gpus all --network host -v /home/yue/homevision_data:/app/data homevision