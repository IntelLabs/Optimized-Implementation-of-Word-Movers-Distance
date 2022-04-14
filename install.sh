#!/bin/bash

# sinkhorn_wmd/install.sh

conda create -n sinkhorn_env python=3.6 pip -y
source activate sinkhorn_env

pip install numba==0.43.1
pip install scipy==1.2.1
pip install numpy==1.16.3
pip install pandas==0.24.2
pip install tqdm==4.32.1