FROM nvidia/cuda:11.4.3-base-ubuntu20.04

WORKDIR /project

# Build with some basic utilities
RUN apt-get update && apt-get install -y \
    python3-pip \
    apt-utils \
    vim \
    git 

# alias python='python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install \
    pyyaml \
    wandb \
    matplotlib \
    numpy \
    pandas \
    rtpt \
    scipy \
    scikit-learn \
    seaborn \
    torch==1.9.0+cu111 \
    torchvision==0.10.0+cu111  -f https://download.pytorch.org/whl/torch_stable.html \
    tensorboard \
    Pillow \
