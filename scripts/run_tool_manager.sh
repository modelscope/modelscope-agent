#!/bin/bash

# check if `docker` cmd exists
if ! command -v docker &> /dev/null
then
    echo "Docker could not be found"
    exit 1
else
    echo "Docker is installed"
    # check if docker dameon is running
    if ! docker info &> /dev/null; then
        echo "Docker daemon is not running"
        exit 1
    else
        echo "Docker daemon is running"
    fi
fi


# use venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# build tool node image
function build_docker_image {
  echo "Building tool node image, the first time might be token 10 mins."
  docker build -f docker/tool_node.dockerfile -t modelscope-agent/tool-node .
}

# install dependencies might be done in venv, not much dependencies here
echo "Installing dependencies from requirements.txt..."
pip3 install -r modelscope_agent_servers/requirements.txt

# Check if the first argument is "build", if so, build the Docker image
if [ "$1" == "build" ]; then
    build_docker_image
else
    echo "Skipping Docker build as per the input argument."
fi

# running
echo "Running fastapi tool manager server at port 31511."
export PYTHONPATH=$PYTHONPATH:modelscope_agent_servers
uvicorn modelscope_agent_servers.tool_manager_server.api:app --host 0.0.0.0 --port 31511
