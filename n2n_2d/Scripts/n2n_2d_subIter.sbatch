#!/bin/bash

#SBATCH -n 4
#SBATCH -w gpu-node004
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J n2n2dSubIter

GAMMA=7.5e-4
BETA=5
DEPTH=4
CHECKPOINT=../../train/recon/n2n_2d/pretrain/beta_0_filter_2_quarter_encoder_decoder_depth_${DEPTH}
OUTDIR=../../train/recon/n2n_2d/encoder_decoder_subIter_study

for i in 2 16 26 54 55 73 75 86 93 95
do
    ./n2n_2d.sh 0 $i $GAMMA ${CHECKPOINT} ${OUTDIR}/1 encoder_decoder $DEPTH $BETA 1 &
    ./n2n_2d.sh 1 $i $GAMMA ${CHECKPOINT} ${OUTDIR}/3 encoder_decoder $DEPTH $BETA 3 &
    ./n2n_2d.sh 2 $i $GAMMA ${CHECKPOINT} ${OUTDIR}/7 encoder_decoder $DEPTH $BETA 7 &
    ./n2n_2d.sh 3 $i $GAMMA ${CHECKPOINT} ${OUTDIR}/10 encoder_decoder $DEPTH $BETA 10 &
    wait
done

