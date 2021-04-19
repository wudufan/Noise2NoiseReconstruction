#!/bin/bash

#SBATCH -n 4
#SBATCH -w gpu-node005
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J n2n2dModel

GAMMA=7.5e-4
BETA=5
CHECKPOINT=../../train/recon/n2n_2d/pretrain/beta_0_filter_2_quarter
OUTDIR=../../train/recon/n2n_2d/model

for i in 2 16 26 54 55 73 75 86 93 95
do
    ./n2n_2d.sh 0 $i $GAMMA ${CHECKPOINT}_unet_depth_4 ${OUTDIR}/unet_depth_4 unet 4 &
    ./n2n_2d.sh 1 $i $GAMMA ${CHECKPOINT}_encoder_decoder_depth_4 ${OUTDIR}/encoder_decoder_depth_4 encoder_decoder 4 &
    ./n2n_2d.sh 2 $i $GAMMA ${CHECKPOINT}_resnet_depth_9 ${OUTDIR}/resnet_depth_9 resnet 9 &
    wait
done

