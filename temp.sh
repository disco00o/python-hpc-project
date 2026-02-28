#!/usr/bin/env bash
#BSUB -J nvidia-smi
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -oo temp_%J.out
#BSUB -eo temp_%J.err

nvidia-smi