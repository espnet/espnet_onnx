#!/bin/bash

# Step 0: Hyper parameters
custom_ort_url="https://github.com/espnet/espnet_onnx/releases/download/custom_ort_v1.11.1.espnet.3/onnxruntime-1.11.1.espnet-cp38-cp38-linux_x86_64.whl"
custom_ort_file="onnxruntime-1.11.1.espnet-cp38-cp38-linux_x86_64.whl"
s3prl_path="/home/circleci/s3prl"
home_dir="/home/circleci"

# Step 1: Setup environments
## Create virtual environment
cd ${home_dir}
python3 -m venv venv
source ${home_dir}/venv/bin/activate

## install dependencies
pip install --upgrade pip
rm -rf ${home_dir}/.local/lib/python*
pip install wheel
wget ${custom_ort_url}
pip install ${custom_ort_file}

pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0
pip install -r ${home_dir}/espnet_onnx/tools/requirements_export.txt
pip install pytest


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
