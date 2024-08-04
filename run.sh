#!/bin/bash

#SBATCH --gres gpu:1
#SBATCH --mem 10G
#SBATCH --cpus-per-task 1
#SBATCH --time 12:00:00
#SBATCH -p exercise-eml
#SBATCH -o slurm_output.log

# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate aca_walli

python test_lenet.py


