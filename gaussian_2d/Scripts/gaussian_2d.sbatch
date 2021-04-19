#!/bin/bash

#SBATCH -n 4
#SBATCH -w gpu-node005
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J gauss2d

BETA=1.25e-4
OUTDIR=../../train/recon/gaussian_2d/${BETA}

for ((i=0; i<100; i+=4))
do
    ./gaussian_2d.sh 0 $((i+0)) $BETA $OUTDIR &
    ./gaussian_2d.sh 1 $((i+1)) $BETA $OUTDIR &
    ./gaussian_2d.sh 2 $((i+2)) $BETA $OUTDIR &
    ./gaussian_2d.sh 3 $((i+3)) $BETA $OUTDIR &
    wait
done
