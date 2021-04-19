#!/bin/bash

DEVICE=$1
SLICE=$2
BETA=$3
CHECKPOINT=$4
OUTDIR=$5

cd ..

python3 csc_2d.py --device $DEVICE \
--slices $SLICE $((SLICE + 1)) --betaRecon $BETA \
--checkPoint $CHECKPOINT --outDir ${OUTDIR}/${SLICE}

