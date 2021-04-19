#!/bin/bash

#SBATCH -n 3
#SBATCH -w gpu-node004
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J n2c_fp

OUTDIR=../../train/recon/n2c_fp/encoder_decoder

for DOSE in 0.5 0.333 0.25 0.166 0.125
do
    ./n2c_fp.sh 0 L067 L096 $DOSE ${OUTDIR}/${DOSE} &
    ./n2c_fp.sh 1 L109 L143 $DOSE ${OUTDIR}/${DOSE} &
    ./n2c_fp.sh 2 L192 L286 $DOSE ${OUTDIR}/${DOSE} &
    wait

    ./n2c_fp.sh 1 L291 L310 $DOSE ${OUTDIR}/${DOSE} &
    ./n2c_fp.sh 2 L333 L506 $DOSE ${OUTDIR}/${DOSE} &
    wait
done

