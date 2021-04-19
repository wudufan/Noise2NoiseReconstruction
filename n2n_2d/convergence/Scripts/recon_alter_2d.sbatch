#!/bin/bash

#SBATCH -n 4
#SBATCH -w gpu-node005
#SBATCH -c 4
#SBATCH -p defq
#SBATCH -J n2nConverge

DEPTH=4
CHECKPOINT=../../../train/recon/n2n_2d/pretrain/beta_0_filter_2_quarter_encoder_decoder_depth_${DEPTH}
OUTDIR=../../../train/recon/n2n_2d/converge/alter_os

cd ..

for i in 16
do
    # whole image wtih pre training
    python3 recon_alter_2d.py --model encoder_decoder --device 0 --slices $((i+0)) $((i+1)) --depth $DEPTH \
    --imgshape 640 640 1 --nPatchesPerImg 1 --outputInterval 50 --nIters 1000 --nSubsets 12 --nesterov 0.5 \
    --checkPoint $CHECKPOINT --outDir ${OUTDIR}/whole_img_pre_train/$i &
    
    # patch image with pre training
    python3 recon_alter_2d.py --model encoder_decoder --device 1 --slices $((i+0)) $((i+1)) --depth $DEPTH \
    --imgshape 96 96 1 --nPatchesPerImg 40 --outputInterval 50 --nIters 1000 --nSubsets 12 --nesterov 0.5 \
    --checkPoint $CHECKPOINT --outDir ${OUTDIR}/patch_img_pre_train/$i &
    
    # patch image without pre training
    python3 recon_alter_2d.py --model encoder_decoder --device 2 --slices $((i+0)) $((i+1)) --depth $DEPTH \
    --imgshape 96 96 1 --nPatchesPerImg 40 --outputInterval 50 --nIters 1000 --nSubsets 12 --nesterov 0.5 \
    --outDir ${OUTDIR}/patch_img_no_pre_train/$i &
    
    # patch image lbfgs
    python3 recon_alter_2d.py --model encoder_decoder --device 3 --slices $((i+0)) $((i+1)) --depth $DEPTH \
    --imgshape 96 96 1 --nPatchesPerImg 40 --outputInterval 50 --optimizer lbfgs --nIters 1000 --nSubsets 12 --nesterov 0.5 \
    --checkPoint $CHECKPOINT --outDir ${OUTDIR}/patch_img_lbfgs/$i &
    
    wait
done
