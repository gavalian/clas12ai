# JLab Machine Learning Project
This repository contains code pertaining to the ML project for JLab.

# Requirements
* Anaconda (tested with Python 3.7)

# Installation
Create a new Anaconda environment using the YML file provided:

`conda env create -f conda_environment.yml`

# Usage
**Make sure to activate the conda environment.**

Execute `ml-cli.py` under `src/` with the necessary CLI arguments.

To see the available CLI parameters execute `./ml-cli.py -h`

The script makes use of subcommands.

To get help for the `train` subcommand for example, execute `./ml-cli.py train -h`.

# Docker
A Dockerfile is provided in order to create a Docker image containing all of the code and dependencies.

## Building the Docker Image
You can use the below command to build the Docker image:

`docker build . -t jlabml:1.0.0`

## Create a Docker Container
You can use the below command to create a Docker container. It includes a mount directory so you can share data with the container and the host operating system.

`docker run -it --mount type=bind,source=$(pwd)/data,target=/data jlabml:1.0.0`

Before running any ML script activate the respective Anaconda environment:

`conda activate ml`