#!/bin/bash

DEVICE=$1
SLICE=$2
BETA=$3
SIGMA=$4
OUTDIR=$5

cd ..

python3 nlm_2d.py --device $DEVICE \
--slices $SLICE $((SLICE + 1)) --betaRecon $BETA --sigma $SIGMA \
--outDir ${OUTDIR}/${SLICE}

