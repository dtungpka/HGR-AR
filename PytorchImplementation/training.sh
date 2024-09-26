#!/bin/bash
#SBATCH --job-name=HG-TRAIN
#SBATCH --account=ddt_acc23
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=71:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=end          
#SBATCH --mail-type=fail
#SBATCH --mail-user=21010294@st.phenikaa-uni.edu.vn
#SBATCH --output=logs/%x_%j_%D.out
#SBATCH --error=logs/%x_%j_%D.err
#SBATCH --nodelist=hpc23

module purge
module load cuda
module load python
source /home/21010294/VSR/VSREnv/bin/activate
module list
python -c "import sys; print(sys.path)"

which python
python --version
python /home/21010294/VSR/cudacheck.py
squeue --me
cd /work/21010294/HandGesture/


# Run the job with the calculated start and end values

python train.py
python fusion_model.py



