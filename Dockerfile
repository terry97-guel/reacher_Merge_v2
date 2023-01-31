# Use nvidia/cuda version matches your server
FROM nvidia/cuda:11.4.1-base-ubuntu20.04

# Install ubuntu apt packages. Do not remove default packages.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    apt-utils \
    build-essential \
    ca-certificates \
    curl \
    git \
    htop \
    sudo \
    vim \
    wget \
    python3-dev \
    python3-pip \
    tmux \
    patchelf \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    xpra \
    xserver-xorg-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip3 --no-cache-dir install --upgrade \
    pip \
    setuptools \
    ipython \
    ipdb \
    matplotlib \
    pandas \
    scipy \
    torch \
    jupyter \
    torchvision \
    torchtext \
    torchsummary \
    slacker \
    tqdm

ARG UNAME
ARG UID
ARG GID

RUN addgroup --gid ${GID} ${UNAME}
RUN useradd -m -u ${UID} -g ${GID} -s /bin/bash -p ${UNAME} ${UNAME}
RUN echo "${UNAME}:${UNAME}" | chpasswd
RUN adduser ${UNAME} sudo
RUN usermod -aG sudo ${UNAME}
RUN gpasswd -a ${UNAME} sudo

USER ${UNAME}
WORKDIR /home/${UNAME}
