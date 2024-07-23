# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /workspace

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the setup script into the container
COPY setup.sh /workspace/setup.sh

# Make the setup script executable
RUN chmod +x /workspace/setup.sh

# Expose the port the app runs on
EXPOSE 8000

# Set the command to run the setup script
CMD ["sh", "/workspace/setup.sh"]
