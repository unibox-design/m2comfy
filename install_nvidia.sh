#!/bin/bash

# Update package list and install dependencies
apt-get update && apt-get install -y \
    wget \
    build-essential \
    gcc \
    make \
    dkms \
    linux-headers-$(uname -r)

# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.1-1_amd64.deb
dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.1-1_amd64.deb
cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*.key /usr/share/keyrings/
apt-get update
apt-get -y install cuda

# Install PyTorch for CUDA
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip3 install -r /ComfyUI/requirements.txt
pip3 install -r /ComfyUI/custom_nodes/ComfyUI-MimicMotionWrapper/requirements.txt

# Create necessary directories
mkdir -p /ComfyUI/models/checkpoints \
    /ComfyUI/models/vae \
    /ComfyUI/models/mimicmotion \
    /ComfyUI/models/diffusers/stable-video-diffusion-img2vid-xt-1-1

# Download MimicMotion model
wget -O /ComfyUI/models/mimicmotion/mimicmotion-model.bin https://huggingface.co/Kijai/MimicMotion_pruned/resolve/main/mimicmotion-model.bin

# Download SVD XT model
wget -O /ComfyUI/models/diffusers/stable-video-diffusion-img2vid-xt-1-1/model_index.json https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/model_index.json \
    && wget -O /ComfyUI/models/diffusers/stable-video-diffusion-img2vid-xt-1-1/feature_extractor/preprocessor_config.json https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/feature_extractor/preprocessor_config.json \
    && wget -O /ComfyUI/models/diffusers/stable-video-diffusion-img2vid-xt-1-1/image_encoder/config.json https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/image_encoder/config.json \
    && wget -O /ComfyUI/models/diffusers/stable-video-diffusion-img2vid-xt-1-1/image_encoder/model.fp16.safetensors https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/image_encoder/model.fp16.safetensors \
    && wget -O /ComfyUI/models/diffusers/stable-video-diffusion-img2vid-xt-1-1/scheduler/scheduler_config.json https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/scheduler/scheduler_config.json \
    && wget -O /ComfyUI/models/diffusers/stable-video-diffusion-img2vid-xt-1-1/unet/config.json https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/unet/config.json \
    && wget -O /ComfyUI/models/diffusers/stable-video-diffusion-img2vid-xt-1-1/unet/diffusion_pytorch_model.fp16.safetensors https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors \
    && wget -O /ComfyUI/models/diffusers/stable-video-diffusion-img2vid-xt-1-1/vae/config.json https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/vae/config.json \
    && wget -O /ComfyUI/models/diffusers/stable-video-diffusion-img2vid-xt-1-1/vae/diffusion_pytorch_model.fp16.safetensors https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors

# Run ComfyUI
cd /ComfyUI
python3 main.py
