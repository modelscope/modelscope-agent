#!/bin/bash
MODELSCOPE_CACHE_DIR_IN_CONTAINER=/modelscope_cache
IMAGE_NAME=registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope
IMAGE_VERSION=ubuntu22.04-cuda11.8.0-py310-torch2.1.0-tf2.14.0-1.9.5-ci
MODELSCOPE_DOMAIN=www.modelscope.cn

CODE_DIR_IN_CONTAINER=/Maas-lib
CI_COMMAND=pytest
echo "$USER"
echo "ci command: $CI_COMMAND"

CONTAINER_NAME="modelscope-agent-ci"
# pull image if there are update
docker pull ${IMAGE_NAME}:${IMAGE_VERSION}
docker run --rm --name $CONTAINER_NAME --shm-size=16gb \
            -e CI_TEST=True \
            -e MODELSCOPE_CACHE=$MODELSCOPE_CACHE_DIR_IN_CONTAINER \
            -e MODELSCOPE_DOMAIN=$MODELSCOPE_DOMAIN \
            -e MODELSCOPE_ENVIRONMENT='ci' \
            --workdir=$CODE_DIR_IN_CONTAINER \
            ${IMAGE_NAME}:${IMAGE_VERSION} \
            $CI_COMMAND

if [ $? -ne 0 ]; then
  echo "Running test case failed, please check the log!"
  exit -1
fi
