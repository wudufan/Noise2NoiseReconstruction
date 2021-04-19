#!/bin/bash

DEVICE=$1
SLICE=$2
GAMMA=$3
CHECKPOINT=$4
OUTDIR=$5
MODEL=${6:-encoder_decoder}
DEPTH=${7:-4}
BETA=${8:-5}
SUBITER=${9:-5}

cd ..

python3 recon_alter_2d.py --device $DEVICE --imgshape 96 96 1 --model $MODEL --depth $DEPTH \
--slices $SLICE $((SLICE + 1)) --gamma $GAMMA --betaRecon $BETA --nSubIters $SUBITER \
--checkPoint $CHECKPOINT --outDir ${OUTDIR}/${SLICE}

