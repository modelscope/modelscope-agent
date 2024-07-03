#!/bin/bash

# install dependencies for ci
#wget -O ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz
#tar xvf ffmpeg.tar.xz
export export PATH=`pwd`:$PATH
pip install torch
export CODE_INTERPRETER_WORK_DIR=${GITHUB_WORKSPACE}
echo "${CODE_INTERPRETER_WORK_DIR}"

# cp file
cp tests/samples/* "${CODE_INTERPRETER_WORK_DIR}/"
ls  "${CODE_INTERPRETER_WORK_DIR}"
pip install playwright
playwright install --with-deps chromium

# install package
pip install fastapi pydantic uvicorn docker sqlmodel transformers ray

# run ci
pytest tests
