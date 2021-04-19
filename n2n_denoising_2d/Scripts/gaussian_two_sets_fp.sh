#!/bin/bash

DEVICE=$1
SLICE=$2
SET=$3
DOSE=$4
BETA=$5
OUTDIR=$6

cd ..

python3 gaussian_two_sets_fp.py --device $DEVICE --setToUse $SET --doseRate $DOSE \
--slices $SLICE $((SLICE + 1)) --betaRecon $BETA \
--outDir ${OUTDIR}/${SLICE}

