FROM ubuntu:22.04

EXPOSE 31513

WORKDIR /app

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


# file ready
RUN rm -rf /tmp/* /var/tmp/*
RUN mkdir -p assets
RUN mkdir -p workspace

# install modelscope_agent
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY modelscope_agent .
ENV PYTHONPATH $PYTHONPATH:/app/modelscope_agent
ENV BASE_TOOL_DIR /app/assets

# install tool_node
COPY tool_node .

#
# docker build -f tool_service/docker/tool_node.dockerfile -t modelscope-agent/tool-node:v0.1 .
# docker push modelscope-agent/tool-node:v0.1
