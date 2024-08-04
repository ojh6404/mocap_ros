#!/usr/bin/bash

git submodule update --init --recursive
python3 -m pip install -U pip gdown pyvirtualdisplay
python3 -m pip install -U "setuptools<70"
python3 -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
export FORCE_CUDA=1
python3 -m pip install git+https://github.com/facebookresearch/pytorch3d.git

# install hand_object_detector and frankmocap
cd frankmocap && python3 -m pip install -r docs/requirements.txt
rm -rf detectors/hand_object_detector/lib/pycocotools && rm -rf detectors/hand_object_detector/lib/datasets
bash scripts/install_frankmocap.sh
mkdir -p extra_data/smpl
gdown https://drive.google.com/uc\?id\=1LBRm4pZzB7gp5aSPr_M-kTI2mrKMi483 -O extra_data/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
gdown https://drive.google.com/uc\?id\=1zG9X15BGX3ywxn4ZgBUCJSGyNi7m2VWC -O extra_data/smpl/SMPLX_NEUTRAL.pkl

# install Arbitrary-Hands-3D-Reconstruction
cd ../Arbitrary-Hands-3D-Reconstruction && python3 -m pip install -r requirements.txt
gdown https://drive.google.com/uc\?id\=1sgZ9dF0FH5z9wSXm9dNXuSyN3ZaTX28U -O mano/MANO_RIGHT.pkl
gdown https://drive.google.com/uc\?id\=17GjLggQpHoJKaZsvG2kSS9Zn4Fp3lW3w -O mano/MANO_LEFT.pkl
mkdir -p checkpoints && gdown https://drive.google.com/uc\?id\=1aCeKMVgIPqYjafMyUJsYzc0h6qeuveG9 -O checkpoints/wild.pkl

# install hamer
cd ../hamer && python3 -m pip install -e .[all]
bash fetch_demo_data.sh
mkdir _DATA/data/mano -p
gdown https://drive.google.com/uc\?id\=1sgZ9dF0FH5z9wSXm9dNXuSyN3ZaTX28U -O _DATA/data/mano/MANO_RIGHT.pkl
# remove unnecessary files
rm -rf hamer_demo_data.tar.gz
rm -rf _DATA/hamer_demo_data.tar.gz
rm -rf _DATA/vitpose_ckpts

# install 4D-Humans
cd ../4D-Humans && python3 -m pip install -e .[all]
mkdir data && gdown https://drive.google.com/uc\?id\=1LBRm4pZzB7gp5aSPr_M-kTI2mrKMi483 -O data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
python3 -m pip install -U PyOpenGL PyOpenGL_accelerate numpy mediapipe supervision ultralytics

# Ensure package install for ROS
python3 -m pip install rospkg gnupg pycryptodomex
