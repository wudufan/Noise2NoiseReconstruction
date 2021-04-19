#!/bin/bash

#SBATCH -n 4
#SBATCH -w gpu-node004
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J n2nParam


CHECKPOINT=../../train/recon/n2n_2d/pretrain/beta_0_filter_2_quarter_encoder_decoder_depth_4
MODEL=encoder_decoder
DEPTH=4

#for GAMMA in 2.5e-4 5e-4 7.5e-4 1e-3 1.25e-3
for GAMMA in 1.5e-3 1.75e-3 2e-3
do
    OUTDIR=../../train/recon/n2n_2d/${MODEL}/${GAMMA}
    for i in 2 16 26 54 55 73 75 86 93 95
    do
        ./n2n_2d.sh 1 $i $GAMMA $CHECKPOINT ${OUTDIR}_1 $MODEL $DEPTH 1 &
        ./n2n_2d.sh 2 $i $GAMMA $CHECKPOINT ${OUTDIR}_2.5 $MODEL $DEPTH 2.5 &
        ./n2n_2d.sh 3 $i $GAMMA $CHECKPOINT ${OUTDIR}_5 $MODEL $DEPTH 5 &
        wait
    done
done
