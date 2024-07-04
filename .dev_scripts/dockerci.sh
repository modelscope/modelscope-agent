#!/bin/bash

# install dependencies for ci
#wget -O ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz
#tar xvf ffmpeg.tar.xz

sudo apt-get install libcurl4 openssl
wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2004-7.0.11.tgz
tar -zxvf mongodb-linux-x86_64-ubuntu2004-7.0.11.tgz
export PATH=`pwd`/mongodb-linux-x86_64-ubuntu2004-7.0.11/bin:$PATH

mkdir mongodb
mongod --dbpath ./mongodb --logpath ./mongo.log --fork

export PATH=`pwd`:$PATH
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
pip install pymongo motor llama-index-storage-docstore-mongodb llama-index-storage-index-store-mongodb llama-index-readers-mongodb

# run ci
pytest tests
