#!/bin/bash
#SBATCH -o arc1-run-%j.out # STDOUT
#SBATCH -e arc1-run-%j.err # STDERR

source /home/mllorens/.bashrc
cd "/scratch/mllorens/Glitch Denoising via Sparse Dictionary Learning/code"
conda activate tesi

printf "RUNNING PYTHON SCRIPT\n"
python arc1-run.py
printf "DONE\n"
