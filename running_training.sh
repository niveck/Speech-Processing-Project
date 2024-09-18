#!/bin/bash
#SBATCH --mem=32gb
#SBATCH -c2
#SBATCH --time=2:0:0
#SBATCH --gres=gpu:1,vmem:32g
#SBATCH --error=180924alldata/error_log_job%A.txt
#SBATCH --output=180924alldata/output_log_job%A.txt
#SBATCH --job-name=ctc_llm_training_180924alldata
#SBATCH --mail-user=niv.eckhaus@mail.huji.ac.il
#SBATCH --mail-type=ALL

cd /cs/snapless/gabis/nive/speech/Speech-Processing-Project/
module load cuda/11.7
module load nvidia
source ../../venvs/async_env/bin/activate
python final_project.py
