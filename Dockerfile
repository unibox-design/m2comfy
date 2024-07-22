# Use a basic Ubuntu image
FROM ubuntu:22.04

# Install necessary packages
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    wget \
    unzip

# Clone the ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git

# Clone the Mimic Motion Wrapper into custom_nodes
RUN git clone https://github.com/kijai/ComfyUI-MimicMotionWrapper.git ComfyUI/custom_nodes/ComfyUI-MimicMotionWrapper

# Copy the installation script into the container
COPY install_nvidia.sh /install_nvidia.sh

# Make the script executable
RUN chmod +x /install_nvidia.sh

# Expose port 8188 for ComfyUI
EXPOSE 8188

# Set the entrypoint to the installation script
ENTRYPOINT ["/install_nvidia.sh"]