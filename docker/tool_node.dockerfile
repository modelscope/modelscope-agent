FROM ubuntu:22.04

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
RUN pip install fastapi uvicorn
RUN pip install torch
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY modelscope_agent /app/modelscope_agent
ENV PYTHONPATH $PYTHONPATH:/app/modelscope_agent:/app/tool_service
ENV BASE_TOOL_DIR /app/assets

# install tool_node
COPY tool_service /app/tool_service


#ENTRYPOINT exec uvicorn tool_service.tool_node.api:app --host 0.0.0.0 --port $PORT


#ENTRYPOINT [ "uvicorn", "tool_service.main:app", "--host", "0.0.0.0","--port","31513" ]
#
# docker build -f tool_service/docker/tool_node.dockerfile -t modelscope-agent/tool-node:v0.1 .
# docker push modelscope-agent/tool-node:v0.1
