#!/bin/bash

DEVICE=$1
SLICE=$2
DOSE=$3
GAMMA=$4
CHECKPOINT=$5
OUTDIR=$6
MODEL=${7:-encoder_decoder}
DEPTH=${8:-4}
BETA=${9:-5}
SUBITER=${10:-5}

cd ..

python3 recon_alter_fp.py --device $DEVICE --imgshape 96 96 1 --model $MODEL --depth $DEPTH --doseRate $DOSE \
--slices $SLICE $((SLICE + 1)) --gamma $GAMMA --betaRecon $BETA --nSubIters $SUBITER \
--checkPoint $CHECKPOINT --outDir ${OUTDIR}/${SLICE}

