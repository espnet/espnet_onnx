#!/bin/bash

set -euo pipefail

# Step 0: Hyper parameters

s3prl_path="/home/circleci/s3prl"
home_dir="/home/circleci"

# Step 1: Setup environments
## Create virtual environment
cd ${home_dir}
python3 -m venv venv
source ${home_dir}/venv/bin/activate

## install dependencies
python -m pip install --upgrade pip wheel
python -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r "${home_dir}/espnet_onnx/tools/requirements_test.txt" -e "${home_dir}/espnet_onnx"

# Step 2: install s3prl
if [ ! -d ${s3prl_path} ]; then
    cd ${home_dir}
    git clone https://github.com/s3prl/s3prl
fi


# Step 3: install
cd ${s3prl_path}
pip install -e .

# Step 4: Run tests
cd ${home_dir}/espnet_onnx
pytest tests \
    --config_dir tests/test_config \
    --wav_dir tests/test_wavs
