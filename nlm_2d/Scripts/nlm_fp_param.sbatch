#!/bin/bash

#SBATCH -n 1
#SBATCH -w gpu-node005
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J nlmfp

BETA=7.5e-4
SIGMA=0.075
OUTDIR=../../train/recon/nlm_fp/

for dose in 0.5 0.333 0.25 0.166 0.125
do
    for BETA in 2.5e-4 5e-4 7.5e-4 1e-3 1.25e-3
    do
        for i in 2 16 26 54 55 73 75 86 93 95
        do
            ./nlm_fp.sh 2 $i $dose $BETA $SIGMA ${OUTDIR}/${dose}/${BETA}
        done
    done
done