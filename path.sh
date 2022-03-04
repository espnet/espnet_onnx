#!/bin/bash

export PATH=$PATH:${PWD}/tools/commands
export PYTHONPATH=$PYTHONPATH:${PWD}/src/net:${PWD}/src/utils:${PWD}/src/writer

export UPLOAD_DIR=${PWD}/upload
export CSV_DIR=${PWD}/resources

source tools/venv/bin/activate
