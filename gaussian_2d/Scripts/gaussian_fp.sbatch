#!/bin/bash

#SBATCH -n 1
#SBATCH -w gpu-node005
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J gaussfp

BETA=1.25e-4
OUTDIR=../../train/recon/gaussian_fp/${BETA}

for dose in 0.5 0.333 0.25 0.166 0.125
do
    OUTDIR=../../train/recon/gaussian_fp/${dose}/${BETA}
    for i in 2 16 26 54 55 73 75 86 93 95
    do
        ./gaussian_fp.sh 2 $i $dose $BETA $OUTDIR
    done
done
