# Stage 1: Build dependencies
FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye AS build

WORKDIR /workspace

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
    libdcmtk-dev \
    pkg-config \
    libatlas-base-dev \
    libturbojpeg-dev \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/

RUN pip3 install -r /workspace/requirements.txt \
    && rm -rf /workspace/.cache/pip

# Stage 2: Final image for running the application
FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

WORKDIR /workspace

# Copy dependencies from the build stage
COPY --from=build /workspace /workspace

EXPOSE 8000

CMD ["bash"]
