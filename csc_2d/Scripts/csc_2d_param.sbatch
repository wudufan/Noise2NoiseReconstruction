#!/bin/bash

#SBATCH -n 1
#SBATCH -w gpu-node005
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J cscParam


CHECKPOINT=../../train/recon/csc_2d/pretrain/kernel_10_channel_32_lmbda_0.005.npy

for BETA in 0.005 0.01 0.02 0.04 0.06
do
    OUTDIR=../../train/recon/csc_2d/${BETA}
    for i in 2 16 26 54 55 73 75 86 93 95
    do
        ./csc_2d.sh 2 $i $BETA $CHECKPOINT $OUTDIR
    done
done
