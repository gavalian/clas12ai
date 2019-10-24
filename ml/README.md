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