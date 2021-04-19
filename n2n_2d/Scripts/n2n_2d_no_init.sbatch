#!/bin/bash

#SBATCH -n 1
#SBATCH -w gpu-node005
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J n2n_2d_no_init

GAMMA=7.5e-4
CHECKPOINT=none
OUTDIR=../../train/recon/n2n_2d/encoder_decoder_all_no_init/${GAMMA}

for ((i=0; i<100; i+=1))
do
    ./n2n_2d_no_init.sh 3 $i $GAMMA $CHECKPOINT $OUTDIR
done
