#!/bin/bash

DEVICE=$1
SLICE=$2
DOSE=$3
BETA=$4
OUTDIR=$5

cd ..

python3 tv_fp.py --device $DEVICE --doseRate $DOSE \
--slices $SLICE $((SLICE + 1)) --betaRecon $BETA \
--outDir ${OUTDIR}/${SLICE}

