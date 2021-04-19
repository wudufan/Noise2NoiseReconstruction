#!/bin/bash

#SBATCH -n 4
#SBATCH -w gpu-node004
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J n2n_2d

GAMMA=7.5e-4
CHECKPOINT=../../train/recon/n2n_2d/pretrain/beta_0_filter_2_quarter_encoder_decoder_depth_4
OUTDIR=../../train/recon/n2n_2d/encoder_decoder_all/${GAMMA}

for ((i=0; i<100; i+=4))
do
    ./n2n_2d.sh 0 $((i+0)) $GAMMA $CHECKPOINT $OUTDIR &
    ./n2n_2d.sh 1 $((i+1)) $GAMMA $CHECKPOINT $OUTDIR &
    ./n2n_2d.sh 2 $((i+2)) $GAMMA $CHECKPOINT $OUTDIR &
    ./n2n_2d.sh 3 $((i+3)) $GAMMA $CHECKPOINT $OUTDIR &
    wait
done
