#!/bin/bash

# install dependencies might be done in venv, not much dependencies here
echo "Installing dependencies from requirements.txt..."
pip3 install -r modelscope_agent_servers/requirements.txt

# Initialize optional variables with empty strings as default values
MODEL_DIR=""
MODEL_SERVER=""
MODEL_NAME=""

# Save all command line arguments
ALL_ARGS="$@"

# Loop through arguments and process them
while [[ $# -gt 0 ]]; do
    case $1 in
        --served-model-name)
        MODEL_NAME="$2"
        shift # past argument
        shift # past value
        ;;
        --model)
        MODEL_DIR="$2"
        shift # past argument
        shift # past value
        ;;
        --model-server)
        MODEL_SERVER="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        shift # past argument
        ;;
    esac
done

# Optionally, echo variables for debugging or confirmation
echo "Model name: $MODEL_NAME"
echo "Model directory: $MODEL_DIR"
echo "Model server: $MODEL_SERVER"

# running
echo "Running fastapi assistant server at port 31512."
export PYTHONPATH=$PYTHONPATH:modelscope_agent_servers

if [ "$MODEL_DIR" != "" ]; then
    echo "Running vllm server, please make sure install vllm"
    # Start the first server in the background on port 8000
    python -m vllm.entrypoints.openai.api_server $ALL_ARGS & SERVER_1_PID=$!
    export MODEL_SERVER=vllm-server
    export OPENAI_API_BASE=http://localhost:8000/v1
    export VLLM_USE_MODELSCOPE=false
    echo "Model server: $MODEL_SERVER"
    echo "OPENAI_API_BASE: $OPENAI_API_BASE"

    # Function to check if the first server is up
    check_first_server() {
        echo "Checking if Server 1 is up..."
        for i in {1..100000000}; do # try up to 100000000 times
            curl -s http://localhost:8000 > /dev/null
            if [ $? -eq 0 ]; then
                echo "Server 1 is up and running."
                return 0
            else
                echo "Server 1 is not ready yet. Retrying..."
                sleep 4
            fi
        done
        return 1
    }

    # Wait for the first server to be up
    if check_first_server; then
        # Start the second server on port 31512
        echo "Starting Server 2..."
        uvicorn modelscope_agent_servers.assistant_server.api:app --host 0.0.0.0 --port 31512 & SERVER_2_PID=$!
    else
        echo "Failed to start Server 1."
        exit 1
    fi
    # Kill the first server when the second server is stopped
    wait $SERVER_1_PID
    wait $SERVER_2_PID

elif [ -n "$MODEL_SERVER" ]; then
    echo "Running specified model server: $MODEL_SERVER..."
    if [ "$MODEL_SERVER" == "ollama" ]; then
      ollama serve
    fi
    uvicorn modelscope_agent_servers.assistant_server.api:app --host 0.0.0.0 --port 31512
else
    MODEL_SERVER=dashscope
    echo "Running FastAPI assistant server at port 31512 as default."
    uvicorn modelscope_agent_servers.assistant_server.api:app --host 0.0.0.0 --port 31512
fi
