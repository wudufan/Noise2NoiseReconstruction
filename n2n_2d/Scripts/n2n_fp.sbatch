#!/bin/bash

#SBATCH -n 1
#SBATCH -w gpu-node004
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J n2nFP


MODEL=encoder_decoder
DEPTH=4
BETA=5

for DOSE in 0.5 0.333 0.25 0.166 0.125
do 
    CHECKPOINT=../../train/recon/n2n_fp/pretrain/beta_0_filter_2_dose_${DOSE}
    for GAMMA in 2.5e-4 5e-4 7.5e-4 1e-3 1.25e-3
    do
        OUTDIR=../../train/recon/n2n_fp/${MODEL}/${DOSE}/${GAMMA}
        for i in 2 16 26 54 55 73 75 86 93 95
        do
            ./n2n_fp.sh 1 $i $DOSE $GAMMA $CHECKPOINT $OUTDIR $MODEL $DEPTH $BETA
        done
    done
done

