# Use an appropriate base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git

# Clone the ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git

# Change working directory to ComfyUI
WORKDIR /app/ComfyUI

# Install ComfyUI dependencies
RUN pip install -r requirements.txt

# Create custom_nodes directory if it doesn't exist
RUN mkdir -p /app/ComfyUI/custom_nodes

# Clone the MimicMotionWrapper repository into custom_nodes
RUN git clone https://github.com/kijai/ComfyUI-MimicMotionWrapper.git /app/ComfyUI/custom_nodes/ComfyUI-MimicMotionWrapper

# Install MimicMotionWrapper dependencies
RUN pip install -r /app/ComfyUI/custom_nodes/ComfyUI-MimicMotionWrapper/requirements.txt

# Set the entrypoint to run ComfyUI
CMD ["python", "main.py"]
