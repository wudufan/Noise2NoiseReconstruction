#!/bin/bash

DEVICE=$1
TEST1=$2
TEST2=$3
DOSE=$4
OUTDIR=$5

cd ..

python3 n2c_fp.py --device $DEVICE --imgshape 96 96 1 --model encoder_decoder --doseRate $DOSE \
--testingName $TEST1 $TEST2 --outDir ${OUTDIR}/${TEST1}_${TEST2} > Outputs/${TEST1}_${TEST2}_${DOSE}.txt

