#!/bin/bash

#SBATCH -J BRATS CNN
#SBATCH -A JAMNIK-SL3
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=03:00:00

#! Do not change:
#SBATCH -p tesla

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load dot
module load scheduler
module load java/jdk1.8.0_45
module load vgl/2.3.1/64
module load intel/impi/4.1.3.045
module load global
module load intel/cce/12.1.10.319
module load intel/fce/12.1.10.319
module load intel/mkl/10.3.10.319
module load default-impi
module load python/3.5.1
module load cudnn/5.0_cuda-8.0
module load cuda/8.0-RC
module load intel/cce/11.0.081
module load intel/fce/11.0.081
module load lapackpp

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.

echo -e "Sourcing venv...\n"
source ~/venv/bin/activate
python --version

echo -e "Start running cifar example\n"
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python3 ~/partIIproject/cnn.py
