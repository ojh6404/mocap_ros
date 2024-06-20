#!/usr/bin/bash

git submodule update --init --recursive
python3 -m pip install -U pip gdown pyvirtualdisplay
python3 -m pip install "setuptools<70"
python3 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_HOME=$PWD/cub-1.10.0
export FORCE_CUDA=1
python3 -m pip install git+https://github.com/facebookresearch/pytorch3d.git

# install hand_object_detector and frankmocap
cd frankmocap && python3 -m pip install -r docs/requirements.txt
bash scripts/install_frankmocap.sh
mkdir -p extra_data/smpl
gdown https://drive.google.com/uc\?id\=1Vx4tRkTpi0M4awzb1AiTTaosOMEuK2O5 -O extra_data/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
gdown https://drive.google.com/uc\?id\=1zG9X15BGX3ywxn4ZgBUCJSGyNi7m2VWC -O extra_data/smpl/SMPLX_NEUTRAL.pkl

# install Arbitrary-Hands-3D-Reconstruction
cd ../Arbitrary-Hands-3D-Reconstruction && python3 -m pip install -r requirements.txt
gdown https://drive.google.com/uc\?id\=1sgZ9dF0FH5z9wSXm9dNXuSyN3ZaTX28U -O mano/MANO_RIGHT.pkl
gdown https://drive.google.com/uc\?id\=17GjLggQpHoJKaZsvG2kSS9Zn4Fp3lW3w -O mano/MANO_LEFT.pkl
mkdir -p checkpoints && gdown https://drive.google.com/uc\?id\=1aCeKMVgIPqYjafMyUJsYzc0h6qeuveG9 -O checkpoints/wild.pkl
python3 -m pip install PyOpenGL==3.1.7 PyOpenGL_accelerate==3.1.7 numpy==1.23.1

# install hamer
python3 -m pip install pytorch-lightning==2.1
cd ../hamer && python3 -m pip install -e .[all]
bash fetch_demo_data.sh
mkdir _DATA/data/mano -p
gdown https://drive.google.com/uc\?id\=1sgZ9dF0FH5z9wSXm9dNXuSyN3ZaTX28U -O _DATA/data/mano/MANO_RIGHT.pkl
# remove unnecessary files
rm -rf hamer_demo_data.tar.gz
rm -rf _DATA/hamer_demo_data.tar.gz
rm -rf _DATA/vitpose_ckpts
