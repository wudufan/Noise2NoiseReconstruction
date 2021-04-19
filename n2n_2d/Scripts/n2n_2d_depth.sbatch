#!/bin/bash

#SBATCH -n 4
#SBATCH -w gpu-node004
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J n2n2dDepth

GAMMA=7.5e-4
BETA=5
CHECKPOINT=../../train/recon/n2n_2d/pretrain/beta_0_filter_2_quarter_encoder_decoder_depth
OUTDIR=../../train/recon/n2n_2d/encoder_decoder_depth_study

for i in 2 16 26 54 55 73 75 86 93 95
do
    ./n2n_2d.sh 0 $i $GAMMA ${CHECKPOINT}_1 ${OUTDIR}/1 encoder_decoder 1 $BETA &
    ./n2n_2d.sh 1 $i $GAMMA ${CHECKPOINT}_2 ${OUTDIR}/2 encoder_decoder 2 $BETA &
    ./n2n_2d.sh 2 $i $GAMMA ${CHECKPOINT}_3 ${OUTDIR}/3 encoder_decoder 3 $BETA &
    ./n2n_2d.sh 3 $i $GAMMA ${CHECKPOINT}_5 ${OUTDIR}/5 encoder_decoder 5 $BETA &
    wait
done

