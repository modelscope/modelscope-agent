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

# install dependency
ENV PYTHONPATH $PYTHONPATH:/app/tool_service
RUN pip install fastapi pydantic uvicorn docker sqlmodel

COPY tool_service /app/tool_service

#ENTRYPOINT exec uvicorn tool_service.tool_manager.api:app --host 0.0.0.0 --port 31511
