#!/bin/bash

#SBATCH -n 3
#SBATCH -w gpu-node005
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J n2c_2d

OUTDIR=../../train/recon/n2c_2d/encoder_decoder

./n2c_2d.sh 0 L067 L096 $OUTDIR &
./n2c_2d.sh 1 L109 L143 $OUTDIR &
./n2c_2d.sh 2 L192 L286 $OUTDIR &
wait

./n2c_2d.sh 1 L291 L310 $OUTDIR &
./n2c_2d.sh 2 L333 L506 $OUTDIR &
wait
