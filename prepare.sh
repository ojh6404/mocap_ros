#!/usr/bin/bash

git submodule update --init --recursive
python3 -m pip install -U setuptools pip gdown
python3 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# install hand_object_detector and frankmocap
cd frankmocap && python3 -m pip install -r docs/requirements.txt
bash scripts/install_frankmocap.sh
mkdir -p extra_data/smpl
gdown https://drive.google.com/uc\?id\=1Vx4tRkTpi0M4awzb1AiTTaosOMEuK2O5 -O extra_data/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
gdown https://drive.google.com/uc\?id\=1zG9X15BGX3ywxn4ZgBUCJSGyNi7m2VWC -O extra_data/smpl/SMPLX_NEUTRAL.pkl
