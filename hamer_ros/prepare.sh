#!/usr/bin/bash

# python3 -m pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
# python3 -m pip install -e hamer/.[all]

git submodule update --init --recursive
python3 -m pip install -U setuptools pip gdown
python3 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# install hand_object_detector
cd ../third_party/frankmocap && python3 -m pip install -r docs/requirements.txt
export FORCE_CUDA=1
bash scripts/install_frankmocap.sh

# install hamer
cd ../hamer
python3 -m pip install -e .[all]
python3 -m pip install mmcv==1.3.9
python3 -m pip install -v -e third-party/ViTPose
bash fetch_demo_data.sh
