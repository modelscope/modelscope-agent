#!/bin/bash

pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set install.trusted-host mirrors.aliyun.com
git config --global --add safe.directory /Maas-lib
git config --global user.email tmp
git config --global user.name tmp.com

pip install torch

pytest
