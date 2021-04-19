#!/bin/bash

#SBATCH -n 2
#SBATCH -w gpu-node005
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J gauss2d

BETA=1.25e-4
OUTDIR=../../train/recon/n2n_denoising_2d/gaussian_two_sets/${BETA}

for ((i=0; i<100; i+=1))
do
    ./gaussian_two_sets_2d.sh 2 $i 0 $BETA $OUTDIR &
    ./gaussian_two_sets_2d.sh 3 $i 1 $BETA $OUTDIR &
    wait
done
