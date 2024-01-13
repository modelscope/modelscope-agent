#!/bin/bash


# install dependencies for ci
pip install torch
export CODE_INTERPRETER_WORK_DIR=${GITHUB_WORKSPACE}
echo "${CODE_INTERPRETER_WORK_DIR}"

# cp file
cp tests/samples/luoli15.jpg "${CODE_INTERPRETER_WORK_DIR}/luoli15.jpg"
ls  "${CODE_INTERPRETER_WORK_DIR}"

# run ci
pytest
