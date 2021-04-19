#!/bin/bash

#SBATCH -n 1
#SBATCH -w gpu-node005
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J fbp2d_origin

INDIR=/home/dwu/DeepRecon/train/data/real_2D_3_layer_mean
OUTDIR=/home/dwu/DeepRecon/train/data/real_2D_3_layer_mean_resolution_0.8

cd ..
for name in L067 L096 L109 L143 L192 L286 L291 L310 L333 L506
do
    python3 fbp_2d_from_origin.py --device 0 --slices 0 50 --sino ${INDIR}/${name}_full_sino.npy --outDir $OUTDIR
done


