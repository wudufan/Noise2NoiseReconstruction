#!/bin/bash

DEVICE=$1
SLICE=$2
DOSE=$3
CHECKPOINT=$4
OUTDIR=$5
MODEL=${6:-encoder_decoder}
DEPTH=${7:-4}

cd ..

python3 denoising_fp.py --device $DEVICE --model $MODEL --depth $DEPTH --doseRate $DOSE \
--slices $SLICE $((SLICE + 1)) \
--checkPoint $CHECKPOINT --outDir ${OUTDIR}/${SLICE}

