#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=8         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=2-00:00            # Runtime in D-HH:MM
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=40G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=slurm_logs/job_%j.out  # File to which STDOUT will be written
#SBATCH --error=slurm_logs/job_%j.out   # File to which STDERR will be written
#SBATCH --mail-type=FAIL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=<your-email>  # Email to which notifications will be sent

# print info about current job
echo "---------- JOB INFOS ------------"
scontrol show job $SLURM_JOB_ID 
echo -e "---------------------------------\n"

# Due to a potential bug, we need to manually load our bash configurations first
source $HOME/.bashrc
# export LD_LIBRARY_PATH="${HOME}/.conda/envs/pytorch-env/lib"

# Next activate the conda environment 
conda activate pytorch

# Run our code
echo "-------- PYTHON OUTPUT ----------"
python train.py --experiment_name="chaksu-novertflip" --use_data_augmentation --use_rois --base_model="resnet50"
echo "---------------------------------"

# Deactivate environment again
conda deactivate