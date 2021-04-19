#!/bin/bash

#SBATCH -n 1
#SBATCH -w gpu-node005
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J admmParam


CHECKPOINT=../../train/recon/n2n_2d/pretrain/beta_0_filter_2_quarter_encoder_decoder_depth_4
MODEL=encoder_decoder
DEPTH=4

for GAMMA in 2.5e-4 5e-4 7.5e-4 1e-3 1.25e-3
do
    for BETA in 0.25 0.5 1 5 10
    do
        OUTDIR=../../train/recon/n2n_2d/admm/${MODEL}/${GAMMA}_${BETA}
        for i in 2 16 26 54 55 73 75 86 93 95
        do
            ./admm_2d.sh 3 $i $GAMMA $CHECKPOINT $OUTDIR $MODEL $DEPTH $BETA
        done
    done
done
