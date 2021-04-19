#!/bin/bash

DEVICE=$1
SLICE=$2
SET=$3
BETA=$4
OUTDIR=$5

cd ..

python3 gaussian_two_sets_2d.py --device $DEVICE --setToUse $SET \
--slices $SLICE $((SLICE + 1)) --betaRecon $BETA \
--outDir ${OUTDIR}/${SLICE}

