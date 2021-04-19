#!/bin/bash

#SBATCH -n 1
#SBATCH -w gpu-node005
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J cscFP


CHECKPOINT=../../train/recon/csc_fp/pretrain/kernel_10_channel_32_lmbda_0.005_dose

for DOSE in 0.5 0.333 0.25 0.166 0.125
do
    for BETA in 0.005 0.01 0.02 0.04 0.06
    do
        OUTDIR=../../train/recon/csc_fp/${DOSE}/${BETA}
        for i in 2 16 26 54 55 73 75 86 93 95
        do
            ./csc_fp.sh 0 $i $DOSE $BETA ${CHECKPOINT}_${DOSE}.npy $OUTDIR
        done
    done
done

