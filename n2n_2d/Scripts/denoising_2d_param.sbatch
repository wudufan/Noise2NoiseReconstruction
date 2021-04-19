#!/bin/bash

#SBATCH -n 1
#SBATCH -w gpu-node005
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J n2nDenoisingParam

CHECKPOINT=../../train/recon/n2n_2d/pretrain
OUTDIR=../../train/recon/n2n_2d/denoising

for BETA in 0 1 2 3 4 5
do
    for i in 2 16 26 54 55 73 75 86 93 95
    do
        ./denoising_2d.sh 0 $i ${CHECKPOINT}/beta_${BETA}_filter_2_quarter_encoder_decoder_depth_4 ${OUTDIR}/beta_${BETA}
    done
done

CHECKPOINT=../../train/recon/n2n_fp/pretrain
OUTDIR=../../train/recon/n2n_fp/denoising

for DOSE in 0.5 0.333 0.25 0.166 0.125
do
    for BETA in 0 1 2 3
    do
        for i in 2 16 26 54 55 73 75 86 93 95
        do
            ./denoising_fp.sh 0 $i $DOSE ${CHECKPOINT}/beta_${BETA}_filter_2_dose_${DOSE} ${OUTDIR}/${DOSE}/beta_${BETA}
        done
    done
done
