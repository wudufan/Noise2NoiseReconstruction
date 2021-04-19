#!/bin/bash

#SBATCH -n 1
#SBATCH -w gpu-node005
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J n2nDenoiseGauss

DEVICE=1
MODEL=encoder_decoder

cd ..

# denoising 2d
for BETA in 0 1 2 3 4
do 
    CHECKPOINT=../../train/recon/n2n_denoising_2d/pretrain/beta_${BETA}_gaussian
    for i in 2 16 26 54 55 73 75 86 93 95
    do
        OUTDIR=../../train/recon/n2n_denoising_2d/denoising/${BETA}/${i}
        python3 denoising.py --device $DEVICE --slices $i $((i+1)) --checkPoint $CHECKPOINT --model $MODEL --outDir $OUTDIR
    done
done
