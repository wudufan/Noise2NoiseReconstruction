#!/bin/bash

#SBATCH -n 1
#SBATCH -w gpu-node005
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J tvParam

BETA=2.5e-5

for BETA in 1e-5 2e-5 3e-5 4e-5 5e-5 6e-5 
do
    OUTDIR=../../train/recon/tv_2d/${BETA}
    for i in 2 16 26 54 55 73 75 86 93 95
    do
        ./tv_2d.sh 1 $i $BETA $OUTDIR
    done
done
