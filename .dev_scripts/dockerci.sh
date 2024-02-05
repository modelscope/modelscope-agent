#!/bin/bash

# install dependencies for ci
apt install ffmpeg
pip install torch
export CODE_INTERPRETER_WORK_DIR=${GITHUB_WORKSPACE}
echo "${CODE_INTERPRETER_WORK_DIR}"

# cp file
cp tests/samples/* "${CODE_INTERPRETER_WORK_DIR}/"
ls  "${CODE_INTERPRETER_WORK_DIR}"

# run ci
pytest
