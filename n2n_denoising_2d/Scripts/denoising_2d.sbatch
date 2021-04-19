#!/bin/bash

#SBATCH -n 1
#SBATCH -w gpu-node005
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J n2nDenoiseGauss

DEVICE=2
MODEL=encoder_decoder
BETA=2

cd ..

# denoising 2d
CHECKPOINT=../../train/recon/n2n_denoising_2d/pretrain/beta_${BETA}_gaussian
for ((i=0; i<100; i+=1))
do
    OUTDIR=../../train/recon/n2n_denoising_2d/denoising/${BETA}/${i}
    python3 denoising.py --device $DEVICE --slices $i $((i+1)) --checkPoint $CHECKPOINT --model $MODEL --outDir $OUTDIR
done
