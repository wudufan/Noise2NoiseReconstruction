#!/bin/bash

DEVICE=$1
SLICE=$2
DOSE=$3
BETA=$4
SIGMA=$5
OUTDIR=$6

cd ..

python3 nlm_fp.py --device $DEVICE --dose $DOSE \
--slices $SLICE $((SLICE + 1)) --betaRecon $BETA --sigma $SIGMA \
--outDir ${OUTDIR}/${SLICE}

