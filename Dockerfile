###############################################################
###############################################################
# Dockerfile for creating a container for running the
# path classification and path prediction machine learning
# algorithms with an Anaconda environment, GCC, and all
# dependencies included.
################################################################
###############################################################

FROM debian:stable-slim

# Install essential dependencies
RUN apt-get update \
    && apt-get install -y gcc g++ make autoconf libtool pkg-config \
        tar python3 wget

# Ensure a fixed root directory
ENV HOME=/root
ENV ANACONDA_PATH=$HOME/.anaconda3

# Install Anaconda3
WORKDIR /tmp
ARG anaconda_url="https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh"
RUN wget ${anaconda_url} -O anaconda3-installer.sh
RUN bash ./anaconda3-installer.sh -b -p $ANACONDA_PATH
ENV PATH="${PATH}:${ANACONDA_PATH}/bin"

# Create Anaconda environment for ML
WORKDIR $HOME/jlab-ml

COPY ./conda_environment.yml ./conda_environment.yml
RUN conda env create -f conda_environment.yml

# Initialize conda for bash
RUN conda init bash

# Copy over code
COPY ./src/ ./src/
COPY ./README.md ./README.md
