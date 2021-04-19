#!/bin/bash

DEVICE=$1
SLICE=$2
CHECKPOINT=$3
OUTDIR=$4
MODEL=${5:-encoder_decoder}
DEPTH=${6:-4}

cd ..

python3 denoising_2d.py --device $DEVICE --model $MODEL --depth $DEPTH \
--slices $SLICE $((SLICE + 1)) \
--checkPoint $CHECKPOINT --outDir ${OUTDIR}/${SLICE}

