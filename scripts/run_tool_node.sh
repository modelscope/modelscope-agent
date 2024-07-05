#!/bin/bash

# get OSS_URL from ENV
OSS_URL=${TOOL_OSS_URL}
ZIP_FILE_NAME="new_tool.zip"
DESTINATION_FOLDER="/app/modelscope_agent/tools/contrib/new_tool"

mkdir -p /app/assets
echo "{\"name\": \"${TOOL_NAME}\"}" > /app/assets/configuration.json

# check if OSS_URL is empty, if empty them run a normal tool node server.
if [ -z "${OSS_URL}" ]; then
  uvicorn modelscope_agent_servers.tool_node_server.api:app --host 0.0.0.0 --port "$1"
fi

# Make sure the destination folder exists
mkdir -p "${DESTINATION_FOLDER}"

# download the zip file
wget -O "${ZIP_FILE_NAME}" "${OSS_URL}"

# check if download is successful
if [ $? -ne 0 ]; then
  echo "Download failed."
  exit 1
else
  echo "Downloaded ${ZIP_FILE_NAME} successfully."

  # unzip the downloaded file
  unzip -o "${ZIP_FILE_NAME}" -d "${DESTINATION_FOLDER}"
  for subfolder in "${DESTINATION_FOLDER}"/*; do
    if [ -d "$subfolder" ]; then # Check if it's a directory
        find "$subfolder" -type f -exec mv {} "${DESTINATION_FOLDER}"/ \;
        # Optionally, remove the now-empty subdirectory
        rmdir "$subfolder"
    fi
  done
  echo "from .new_tool import *" >> /app/modelscope_agent/tools/contrib/__init__.py

  # check if extraction is successful
  if [ $? -ne 0 ]; then
    echo "Extraction failed."
    exit 1
  else
    echo "Extracted ${ZIP_FILE_NAME} into ${DESTINATION_FOLDER}."

    # clean up the downloaded zip file
    rm "${ZIP_FILE_NAME}"
    echo "Removed the downloaded zip file."
  fi
fi

# get config from ENV
TOOL_NAME=${TOOL_NAME}

uvicorn modelscope_agent_servers.tool_node_server.api:app --host 0.0.0.0 --port "$1"
#sleep 90m
