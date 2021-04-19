#!/bin/bash

#SBATCH -n 4
#SBATCH -w gpu-node005
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J nlm2d

BETA=5e-4
SIGMA=0.075
OUTDIR=../../train/recon/nlm_2d/${BETA}_sigma_${SIGMA}

for ((i=0; i<100; i+=4))
do
    ./nlm_2d.sh 0 $((i+0)) $BETA $SIGMA $OUTDIR &
    ./nlm_2d.sh 1 $((i+1)) $BETA $SIGMA $OUTDIR &
    ./nlm_2d.sh 2 $((i+2)) $BETA $SIGMA $OUTDIR &
    ./nlm_2d.sh 3 $((i+3)) $BETA $SIGMA $OUTDIR &
    wait
done
