#!/bin/bash
#SBATCH -A desi -q debug -C cpu -N 1 -t 00:30:00 -J folpsfkpt_validation
#SBATCH -o /pscratch/sd/r/rohlf/folpsfkpt_validation/job_%j.log
source /global/cfs/cdirs/desi/users/rohlf/folpsfkpt_env/activate.sh
cd /global/homes/r/rohlf/folps_fkpt_jr/folpsfkpt
XLA_THREADS=10 validation/run_all.sh /pscratch/sd/r/rohlf/folpsfkpt_validation/FolpsD_upstream /pscratch/sd/r/rohlf/folpsfkpt_validation/fkptjax_upstream ${N:-100}
python validation/compare.py
