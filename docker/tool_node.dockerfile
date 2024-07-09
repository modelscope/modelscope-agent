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

## install modelscope_agent
#RUN pip install torch
#RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn

# install ffmpeg
RUN  wget -O ffmpeg.tar.xz https://modelscope-agent.oss-cn-hangzhou.aliyuncs.com/resources/ffmpeg.tar.xz && \
     tar xvf ffmpeg.tar.xz


ENV PYTHONPATH $PYTHONPATH:/app/modelscope_agent:/app/modelscope_agent_servers
ENV BASE_TOOL_DIR /app/assets
ENV PATH=/app/ffmpeg-git-20240629-amd64-static:$PATH

# install tool_node
COPY modelscope_agent_servers /app/modelscope_agent_servers
COPY modelscope_agent /app/modelscope_agent

# start up script file
COPY scripts/run_tool_node.sh /app/run_tool_node.sh
RUN chmod +x /app/run_tool_node.sh
#ENTRYPOINT exec uvicorn tool_service.tool_node.api:app --host 0.0.0.0 --port $PORT


#ENTRYPOINT [ "uvicorn", "tool_service.main:app", "--host", "0.0.0.0","--port","31513" ]
#
# docker build -f tool_service/docker/tool_node.dockerfile -t modelscope-agent/tool-node:v0.1 .
# docker push modelscope-agent/tool-node:v0.1
