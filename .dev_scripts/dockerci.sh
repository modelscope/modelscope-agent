#!/bin/bash

# install dependencies for ci
pip install torch

# run ci
pytest
