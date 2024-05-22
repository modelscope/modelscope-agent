#!/bin/bash

# install dependencies might be done in venv, not much dependencies here
echo "Installing dependencies from requirements.txt..."
pip3 install -r modelscope_agent_servers/requirements.txt


# running
echo "Running fastapi assistant server at port 31512."
export PYTHONPATH=$PYTHONPATH:modelscope_agent_servers

if [ -z "$1" ]; then
    export MODEL_SERVER=dashscope
else
    export MODEL_SERVER=$1
fi

if [ "$MODEL_SERVER" == "ollama" ]; then
    ollama serve

uvicorn modelscope_agent_servers.assistant_server.api:app --host 0.0.0.0 --port 31512
