# Base image from devcontainers for Python 3.12 on Debian Bullseye
FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# Set a working directory for the container
WORKDIR /workspace

# Update system packages and install build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpq-dev \
    libopencv-dev \
    libpng-dev \
    libtiff-dev \
    libjasper-dev \
    libdcmtk-dev \
    pkg-config \
    libatlas-base-dev \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file into the container
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies globally
RUN pip3 install --upgrade pip setuptools wheel \
    && pip3 install -r /workspace/requirements.txt \
    && rm -rf /root/.cache/pip  # Clean up temporary files to reduce image size

# Set default command
CMD ["bash"]
