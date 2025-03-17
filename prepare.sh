#!/usr/bin/bash

git submodule update --init --recursive
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e motion_capture
pip install -U "setuptools<70"
export FORCE_CUDA=1
pip install git+https://github.com/facebookresearch/pytorch3d.git

# install hand_object_detector and frankmocap
cd motion_capture/third_party/frankmocap && pip install -r docs/requirements.txt
rm -rf detectors/hand_object_detector/lib/pycocotools && rm -rf detectors/hand_object_detector/lib/datasets # remove pycocotools and datasets cause they may cause conflicts with existing pycocotools
bash scripts/install_frankmocap.sh
mkdir -p extra_data/smpl
gdown https://drive.google.com/uc\?id\=1LBRm4pZzB7gp5aSPr_M-kTI2mrKMi483 -O extra_data/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl -c
gdown https://drive.google.com/uc\?id\=1zG9X15BGX3ywxn4ZgBUCJSGyNi7m2VWC -O extra_data/smpl/SMPLX_NEUTRAL.pkl -c

# install Arbitrary-Hands-3D-Reconstruction
cd ../Arbitrary-Hands-3D-Reconstruction && pip install -r requirements.txt
gdown https://drive.google.com/uc\?id\=1sgZ9dF0FH5z9wSXm9dNXuSyN3ZaTX28U -O mano/MANO_RIGHT.pkl -c
gdown https://drive.google.com/uc\?id\=17GjLggQpHoJKaZsvG2kSS9Zn4Fp3lW3w -O mano/MANO_LEFT.pkl -c
mkdir -p checkpoints && gdown https://drive.google.com/uc\?id\=1aCeKMVgIPqYjafMyUJsYzc0h6qeuveG9 -O checkpoints/wild.pkl -c

# install hamer
cd ../hamer && pip install -e .[all]
bash fetch_demo_data.sh
mkdir _DATA/data/mano -p
gdown https://drive.google.com/uc\?id\=1sgZ9dF0FH5z9wSXm9dNXuSyN3ZaTX28U -O _DATA/data/mano/MANO_RIGHT.pkl -c
# remove unnecessary files
rm -rf hamer_demo_data.tar.gz
rm -rf _DATA/hamer_demo_data.tar.gz
rm -rf _DATA/vitpose_ckpts

# install wilor
cd ../WiLoR && pip install -e .
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./pretrained_models/
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt -P ./pretrained_models/
gdown https://drive.google.com/uc\?id\=1sgZ9dF0FH5z9wSXm9dNXuSyN3ZaTX28U -O mano_data/MANO_RIGHT.pkl -c

# install 4D-Humans
cd ../4D-Humans && pip install -e .[all]
mkdir data && gdown https://drive.google.com/uc\?id\=1LBRm4pZzB7gp5aSPr_M-kTI2mrKMi483 -O data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl -c
pip install -U PyOpenGL PyOpenGL_accelerate git+https://github.com/ojh6404/chumpy.git@patch-python3.11

# for ros build
pip install psutil empy==3.3.2 rospkg gnupg pycryptodomex catkin-tools wheel cython ninja importlib_metadata netifaces
