# Use the latest nvidia/cuda image as base
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install necessary packages
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    wget \
    unzip

# Install PyTorch for CUDA
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Clone the ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git

# Set working directory to ComfyUI
WORKDIR /ComfyUI

# Install ComfyUI dependencies
RUN pip3 install -r requirements.txt

# Clone the Mimic Motion Wrapper into custom_nodes
RUN git clone https://github.com/kijai/ComfyUI-MimicMotionWrapper.git custom_nodes/ComfyUI-MimicMotionWrapper

# Install Mimic Motion Wrapper dependencies
RUN pip3 install -r custom_nodes/ComfyUI-MimicMotionWrapper/requirements.txt

# Create necessary directories
RUN mkdir -p models/checkpoints \
    && mkdir -p models/vae \
    && mkdir -p models/mimicmotion \
    && mkdir -p models/diffusers/stable-video-diffusion-img2vid-xt-1-1

# Download MimicMotion model
RUN wget -O models/mimicmotion/mimicmotion-model.bin https://huggingface.co/Kijai/MimicMotion_pruned/resolve/main/mimicmotion-model.bin

# Download SVD XT model
RUN wget -O models/diffusers/stable-video-diffusion-img2vid-xt-1-1/model_index.json https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/model_index.json \
    && wget -O models/diffusers/stable-video-diffusion-img2vid-xt-1-1/feature_extractor/preprocessor_config.json https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/feature_extractor/preprocessor_config.json \
    && wget -O models/diffusers/stable-video-diffusion-img2vid-xt-1-1/image_encoder/config.json https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/image_encoder/config.json \
    && wget -O models/diffusers/stable-video-diffusion-img2vid-xt-1-1/image_encoder/model.fp16.safetensors https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/image_encoder/model.fp16.safetensors \
    && wget -O models/diffusers/stable-video-diffusion-img2vid-xt-1-1/scheduler/scheduler_config.json https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/scheduler/scheduler_config.json \
    && wget -O models/diffusers/stable-video-diffusion-img2vid-xt-1-1/unet/config.json https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/unet/config.json \
    && wget -O models/diffusers/stable-video-diffusion-img2vid-xt-1-1/unet/diffusion_pytorch_model.fp16.safetensors https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors \
    && wget -O models/diffusers/stable-video-diffusion-img2vid-xt-1-1/vae/config.json https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/vae/config.json \
    && wget -O models/diffusers/stable-video-diffusion-img2vid-xt-1-1/vae/diffusion_pytorch_model.fp16.safetensors https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors

# Expose port 8188 for ComfyUI
EXPOSE 8188

# Command to run ComfyUI
CMD ["python3", "main.py"]
