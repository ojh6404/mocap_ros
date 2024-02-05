#!/usr/bin/bash

git submodule update --init --recursive
python3 -m pip install -U setuptools pip gdown
python3 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
cd hand_object_detector && python3 -m pip install -r requirements.txt
mkdir -p models/res101_handobj_100K/pascal_voc && gdown https://drive.google.com/uc\?id\=1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE -O models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth
cd lib && python3 setup.py build develop --user
# install frnakmocap
cd ../../frankmocap && python3 -m pip install -r docs/requirements.txt
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
python3 -m pip install pytorch3d
bash scripts/install_frankmocap.sh
mkdir -p extra_data/smpl
gdown https://drive.google.com/uc\?id\=1Vx4tRkTpi0M4awzb1AiTTaosOMEuK2O5 -O extra_data/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
gdown https://drive.google.com/uc\?id\=1zG9X15BGX3ywxn4ZgBUCJSGyNi7m2VWC -O extra_data/smpl/SMPLX_NEUTRAL.pkl
