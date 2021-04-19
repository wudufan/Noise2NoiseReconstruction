#!/bin/bash

#SBATCH -n 4
#SBATCH -w gpu-node004
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J nlm2d

BETA=7.5e-4
SIGMA=0.05
OUTDIR=../../train/recon/nlm_2d/

#for SIGMA in 0.01 0.025 0.05 0.075 0.1
#do
#    for i in 2 16 26 54 55 73 75 86 93 95
#    do
#        ./nlm_2d.sh 0 $i 2.5e-4 $SIGMA ${OUTDIR}2.5e-4_sigma_${SIGMA} &
#        ./nlm_2d.sh 1 $i 5e-4 $SIGMA ${OUTDIR}5e-4_sigma_${SIGMA} &
#        ./nlm_2d.sh 2 $i 7.5e-4 $SIGMA ${OUTDIR}7.5e-4_sigma_${SIGMA} &
#        ./nlm_2d.sh 3 $i 1e-4 $SIGMA ${OUTDIR}1e-4_sigma_${SIGMA} &
#        wait
#    done
#done


for SIGMA in 0.125
do
    for i in 2 16 26 54 55 73 75 86 93 95
    do
        ./nlm_2d.sh 0 $i 2.5e-4 $SIGMA ${OUTDIR}2.5e-4_sigma_${SIGMA} &
        ./nlm_2d.sh 1 $i 5e-4 $SIGMA ${OUTDIR}5e-4_sigma_${SIGMA} &
        ./nlm_2d.sh 2 $i 7.5e-4 $SIGMA ${OUTDIR}7.5e-4_sigma_${SIGMA} &
        ./nlm_2d.sh 3 $i 1e-3 $SIGMA ${OUTDIR}1e-3_sigma_${SIGMA} &
        wait
    done
done