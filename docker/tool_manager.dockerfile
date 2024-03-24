FROM ubuntu:22.04

# install basic packages
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    vim \
    nano \
    unzip \
    zip \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# docker build -f tool_service/docker/tool_node.dockerfile -t modelscope-agent/toolnode:v0.1 .
# docker push modelscope-agent/toolnode:v0.1
