#!/bin/bash

DEVICE=$1
SLICE=$2
BETA=$3
OUTDIR=$4

cd ..

python3 gaussian_2d.py --device $DEVICE \
--slices $SLICE $((SLICE + 1)) --betaRecon $BETA \
--sino ../../train/recon/data/sino/full_sino.npy --ref ../../train/recon/data/sino/full_hann.npy \
--outDir ${OUTDIR}/${SLICE}

