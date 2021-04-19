#!/bin/bash

#SBATCH -n 4
#SBATCH -w gpu-node005
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J gauss2d

BETA=1.25e-4
OUTDIR=../../train/recon/gaussian_2d/${BETA}

for i in 2 16 26 54 55 73 75 86 93 95
do
    ./gaussian_2d_ref.sh 0 $i 1e-4 ../../train/recon/gaussian_2d_ref/1e-4 &
    ./gaussian_2d_ref.sh 1 $i 1.25e-4 ../../train/recon/gaussian_2d_ref/1.25e-4 &
    ./gaussian_2d_ref.sh 2 $i 1.5e-4 ../../train/recon/gaussian_2d_ref/1.5e-4 &
    wait
done
