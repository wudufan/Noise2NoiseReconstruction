#!/bin/bash

#SBATCH -n 4
#SBATCH -w gpu-node004
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J gauss2d_origin

BETA=1.25e-4
OUTDIR=../../train/recon/gaussian_2d_all_slices/${BETA}

for name in L067 L096 L109 L143 L192 L286 L291 L310 L333 L506
do
    for ((i=0; i<48; i+=4))
    do
        ./gaussian_2d_from_origin.sh 0 $name $((i+0)) $BETA $OUTDIR &
        ./gaussian_2d_from_origin.sh 1 $name $((i+1)) $BETA $OUTDIR &
        ./gaussian_2d_from_origin.sh 2 $name $((i+2)) $BETA $OUTDIR &
        ./gaussian_2d_from_origin.sh 3 $name $((i+3)) $BETA $OUTDIR &
        wait
    done
    ./gaussian_2d_from_origin.sh 0 $name 48 $BETA $OUTDIR &
    ./gaussian_2d_from_origin.sh 1 $name 49 $BETA $OUTDIR &
    wait
done


