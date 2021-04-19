#!/bin/bash

#SBATCH -n 1
#SBATCH -w gpu-node004
#SBATCH -c 16
#SBATCH -p defq
#SBATCH -J cscTrain

cd ..

python3 -u pretrain_2d.py --device 2 --checkPoint ../../train/recon/csc_2d/pretrain/kernel_10_channel_32_lmbda_0.005

