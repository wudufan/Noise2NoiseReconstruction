#!/bin/bash

DEVICE=$1
TEST1=$2
TEST2=$3
OUTDIR=$4

cd ..

python3 n2c_2d.py --device $DEVICE --imgshape 96 96 1 --model encoder_decoder \
--testingName $TEST1 $TEST2 --outDir ${OUTDIR}/${TEST1}_${TEST2} > Outputs/${TEST1}_${TEST2}.txt

