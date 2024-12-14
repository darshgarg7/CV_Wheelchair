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
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    libopencv-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjasper-dev \
    libdcmtk-dev \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    libatlas-base-dev \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies globally
COPY requirements.txt /requirements.txt
RUN pip3 install --upgrade pip setuptools wheel \
    && pip3 install -r /requirements.txt \
    && rm -rf /root/.cache/pip  # Clean up temporary files to reduce image size

CMD ["bash"]
