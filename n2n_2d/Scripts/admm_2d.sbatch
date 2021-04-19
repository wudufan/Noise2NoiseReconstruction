#!/bin/bash

#SBATCH -n 1
#SBATCH -w gpu-node004
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J admm_2d

GAMMA=7.5e-4
BETA=1
CHECKPOINT=../../train/recon/n2n_2d/pretrain/beta_0_filter_2_quarter_encoder_decoder_depth_4
OUTDIR=../../train/recon/n2n_2d/admm/encoder_decoder_all/${GAMMA}

for ((i=0; i<100; i+=1))
do
    ./admm_2d.sh 3 $i $GAMMA $CHECKPOINT $OUTDIR
done
