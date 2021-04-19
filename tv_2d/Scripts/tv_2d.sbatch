#!/bin/bash

#SBATCH -n 4
#SBATCH -w gpu-node004
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J tv2d

BETA=4e-5
OUTDIR=../../train/recon/tv_2d/${BETA}

for ((i=0; i<100; i+=4))
do
    ./tv_2d.sh 0 $((i+0)) $BETA $OUTDIR &
    ./tv_2d.sh 1 $((i+1)) $BETA $OUTDIR &
    ./tv_2d.sh 2 $((i+2)) $BETA $OUTDIR &
    ./tv_2d.sh 3 $((i+3)) $BETA $OUTDIR &
    wait
done
