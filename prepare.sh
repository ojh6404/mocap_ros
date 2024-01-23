#!/usr/bin/bash

git submodule update --init --recursive
python3 -m pip install -U setuptools pip gdown
python3 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
cd hand_object_detector && python3 -m pip install -r requirements.txt
mkdir -p models/res101_handobj_100K/pascal_voc && gdown https://drive.google.com/uc\?id\=1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE -O models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth
cd lib && python3 setup.py build develop --user
