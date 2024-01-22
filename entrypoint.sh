#!/bin/bash

CPU_CORES=$(nproc)
WORKERS=$((CPU_CORES))

exec gunicorn -w $WORKERS server:app -b 0.0.0.0:5000

