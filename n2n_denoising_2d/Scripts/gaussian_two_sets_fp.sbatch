#!/bin/bash

#SBATCH -n 2
#SBATCH -w gpu-node005
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J gauss2fp

BETA=1.25e-4
OUTDIR=../../train/recon/n2n_denoising_fp/gaussian_two_sets/${BETA}

for dose in 0.5 0.333 0.25 0.166 0.125
do
    OUTDIR=../../train/recon/n2n_denoising_fp/gaussian_two_sets/${dose}/${BETA}
    for i in 2 16 26 54 55 73 75 86 93 95
    do
        ./gaussian_two_sets_fp.sh 2 $i 0 $dose $BETA $OUTDIR &
        ./gaussian_two_sets_fp.sh 3 $i 1 $dose $BETA $OUTDIR &
        wait
    done
done
