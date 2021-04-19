#!/bin/bash

#SBATCH -n 1
#SBATCH -w gpu-node005
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J csc2d

BETA=0.02
CHECKPOINT=../../train/recon/csc_2d/pretrain/kernel_10_channel_32_lmbda_0.005.npy
OUTDIR=../../train/recon/csc_2d/${BETA}

for ((i=0; i<100; i+=1))
do
    ./csc_2d.sh 2 $i $BETA $CHECKPOINT $OUTDIR
    #./csc_2d.sh 0 $((i+0)) $BETA $CHECKPOINT $OUTDIR &
    #./csc_2d.sh 1 $((i+1)) $BETA $CHECKPOINT $OUTDIR &
    #./csc_2d.sh 2 $((i+2)) $BETA $CHECKPOINT $OUTDIR &
    #./csc_2d.sh 3 $((i+3)) $BETA $CHECKPOINT $OUTDIR &
    #wait
done
