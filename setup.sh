#!/bin/sh

# Navigate to the workspace directory
cd /workspace

# Clone the ComfyUI repository
git clone https://github.com/comfyanonymous/ComfyUI.git

# Navigate to the ComfyUI directory
cd ComfyUI

# Clone the Mimic Motion Wrapper into the custom_nodes folder
git clone https://github.com/kijai/ComfyUI-MimicMotionWrapper.git custom_nodes/ComfyUI-MimicMotionWrapper

# Install ComfyUI dependencies
pip install -r requirements.txt

# Install Mimic Motion Wrapper dependencies
pip install -r custom_nodes/ComfyUI-MimicMotionWrapper/requirements.txt

# Run ComfyUI
python main.py
