#!/bin/bash

#SBATCH -n 1
#SBATCH -w gpu-node004
#SBATCH -c 16
#SBATCH -p defq
#SBATCH -J cscFPTrain

cd ..

for DOSE in 0.5 0.333 0.25 0.166 0.125
do
    python3 -u pretrain_fp.py --device 3 --doseRate $DOSE \
    --checkPoint ../../train/recon/csc_fp/pretrain/kernel_10_channel_32_lmbda_0.005_dose_${DOSE}
done
    

