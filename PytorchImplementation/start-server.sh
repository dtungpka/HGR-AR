#!/bin/bash
#SBATCH --job-name=ADL-jupyter
#SBATCH --account=ddt_acc23
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=end          
#SBATCH --mail-type=fail
#SBATCH --mail-user=21010294@st.phenikaa-uni.edu.vn
#SBATCH --output=logs/%x_%j_%D.out
#SBATCH --error=logs/%x_%j_%D.err
#SBATCH --nodelist=hpc08

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

jupyter notebook --no-browser --port=8880 



