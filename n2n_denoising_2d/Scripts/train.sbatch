#!/bin/bash

#SBATCH -n 1
#SBATCH -w gpu-node004
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J n2nTrainGauss

DEVICE=3
MODEL=encoder_decoder

cd ..

# train 2d
for BETA in 0 1 2 3 4
do 
    CHECKPOINT=../../train/recon/n2n_denoising_2d/pretrain/beta_${BETA}_gaussian
    python3 train.py --device $DEVICE --beta $BETA --checkPoint $CHECKPOINT --model $MODEL --imgshape 96 96 1
done

# train fp
for DOSE in 0.5 0.333 0.25 0.166 0.125
do
    SRC=../../train/recon/n2n_denoising_fp/gaussian_two_sets/${DOSE}/recon
    REF=../../train/recon/data/fp/full_hann_10.npy
    
    for BETA in 0 1 2 3 4
    do 
        CHECKPOINT=../../train/recon/n2n_denoising_fp/pretrain/beta_${BETA}_gaussian_dose_${DOSE}
        python3 train.py --device $DEVICE --beta $BETA --checkPoint $CHECKPOINT --model $MODEL --imgshape 96 96 1 --src $SRC --ref $REF
    done
done

