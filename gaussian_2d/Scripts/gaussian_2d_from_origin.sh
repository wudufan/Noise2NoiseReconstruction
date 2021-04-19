#!/bin/bash

DEVICE=$1
NAME=$2
SLICE=$3
BETA=$4
OUTDIR=$5

cd ..

python3 gaussian_2d.py --device $DEVICE \
--slices $SLICE $((SLICE + 1)) --betaRecon $BETA \
--sino /home/dwu/DeepRecon/train/data/real_2D_3_layer_mean/${NAME}_full_sino.npy \
--ref /home/dwu/DeepRecon/train/data/real_2D_3_layer_mean/${NAME}_full_hann.npy \
--outDir ${OUTDIR}/${NAME}/${SLICE} >> Outputs/gaussian_2d_from_origin/${NAME}.txt

