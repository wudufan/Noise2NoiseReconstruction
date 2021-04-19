#!/bin/bash

#SBATCH -n 1
#SBATCH -w gpu-node005
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J n2nDFP

DEVICE=3
MODEL=encoder_decoder
SLICE=(2 16 26 54 55 73 75 86 93 95) 

cd ..

# denoising 2d
for DOSE in 0.5 0.333 0.25 0.166 0.125
do
    SRC=../../train/recon/n2n_denoising_fp/gaussian_two_sets/${DOSE}/recon
    REF=../../train/recon/data/fp/full_hann_10.npy
        
    for BETA in 0 1 2 3 4
    do 
        CHECKPOINT=../../train/recon/n2n_denoising_fp/pretrain/beta_${BETA}_gaussian_dose_${DOSE}
        for ((i=0; i<10; i+=1))
        do
            iSlice=${SLICE[$i]}
            OUTDIR=../../train/recon/n2n_denoising_fp/denoising/${DOSE}/${BETA}/${iSlice}
            #echo $OUTDIR
            python3 denoising.py --device $DEVICE --slices $i $((i+1)) --checkPoint $CHECKPOINT --model $MODEL --outDir $OUTDIR \
            --src $SRC --ref $REF
        done
    done
done

