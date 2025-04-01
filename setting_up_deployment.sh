#!/bin/bash

# ########## Install prerequisites for deep-start
echo "Updating system..."
apt-get update
apt-get upgrade -y
apt-get install -y git
apt-get install -y libgl1
apt-get install -y wget
rm -rf /var/lib/apt/lists/*
echo "-----------------------"

export LANG="C.UTF-8"

# ########## Install RCLONE
echo "Installing rclone..."
curl -O https://downloads.rclone.org/rclone-current-linux-amd64.deb
dpkg -i rclone-current-linux-amd64.deb
apt install -f
mkdir /srv/.rclone/
touch /srv/.rclone/rclone.conf
rm rclone-current-linux-amd64.deb
rm -rf /var/lib/apt/lists/*
echo "-----------------------"

# # ########## Set required environment variables
# # Training with Tensorflow requires XLA_FLAGS to be set, which
# # in turn requires CUDA_HOME to be set as an environment variable.
# # Define relevant path as CUDA_HOME here to prevent KeyError.
# echo "Setting CUDA_HOME environment variable..."
# cuda_path="/usr/local/cuda/"
# if [ -d "$cuda_path" ]; then
#     export CUDA_HOME="$cuda_path"
#     echo "CUDA_HOME is set to $CUDA_HOME"
# else
#     echo "Path $cuda_path does not exist! CUDA_HOME cannot be set."
#     exit 1
# fi
# echo "-----------------------"

# ########## Setup up API and submodule repositories
echo "Setting up thermal-anomaly-segmenter repository..."
# upgrade pip
pip install --upgrade pip

# clone API repository
git clone --depth 1 https://github.com/ai4os-hub/thermal-anomaly-segmenter.git /srv/thermal-anomaly-segmenter

# get submodule
cd thermal-anomaly-segmenter

# install pre-requisites for repo installations
pip install torch==2.0.1+cu117 --extra-index-url "https://download.pytorch.org/whl/cu117"
# install repos
pip install -e .

echo "================================================================"
echo "DEPLOYMENT SETUP COMPLETE"
echo "================================================================"

echo "Starting deepaas"
deep-start