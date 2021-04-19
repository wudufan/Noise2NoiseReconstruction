#!/bin/bash

DEVICE=$1
SLICE=$2
DOSE=$3
BETA=$4
CHECKPOINT=$5
OUTDIR=$6

cd ..

python3 csc_fp.py --device $DEVICE --doseRate $DOSE \
--slices $SLICE $((SLICE + 1)) --betaRecon $BETA \
--checkPoint $CHECKPOINT --outDir ${OUTDIR}/${SLICE}

