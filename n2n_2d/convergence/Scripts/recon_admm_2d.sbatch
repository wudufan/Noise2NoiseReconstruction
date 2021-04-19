#!/bin/bash

#SBATCH -n 2
#SBATCH -w gpu-node005
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J n2nAdmm

CHECKPOINT=../../../train/recon/n2n_2d/pretrain/beta_0_filter_2_quarter_encoder_decoder_depth_4
OUTDIR=../../../train/recon/n2n_2d/converge/admm

cd ..

for i in 2 16 26 54 55 73 75 86 93 95
do
    # whole image wtih pre training
    python3 recon_admm_2d.py --model encoder_decoder --device 0 --slices $((i+0)) $((i+1)) \
    --imgshape 640 640 1 --nPatchesPerImg 1 --nIters 1000 \
    --checkPoint $CHECKPOINT --outDir ${OUTDIR}/whole_img_pre_train/$i &
    
    # patch image with pre training
    python3 recon_admm_2d.py --model encoder_decoder --device 1 --slices $((i+0)) $((i+1)) \
    --imgshape 96 96 1 --nPatchesPerImg 40 --nIters 1000 \
    --checkPoint $CHECKPOINT --outDir ${OUTDIR}/patch_img_pre_train/$i &
    
    wait
done
