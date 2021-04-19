#!/bin/bash

#SBATCH -n 1
#SBATCH -w gpu-node005
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J tvFPParam


for dose in 0.5 0.333 0.25 0.166 0.125
do
    for BETA in 2e-5 3e-5 4e-5 5e-5 6e-5 
    do
        OUTDIR=../../train/recon/tv_fp/${dose}/${BETA}
        for i in 2 16 26 54 55 73 75 86 93 95
        do
            ./tv_fp.sh 3 $i $dose $BETA $OUTDIR
        done
    done
done

