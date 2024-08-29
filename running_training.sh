#!/bin/bash
#SBATCH --mem=32gb
#SBATCH -c2
#SBATCH --time=23:0:0
#SBATCH --gres=gpu:1,vmem:32g
#SBATCH --error=error_log_job%A.txt
#SBATCH --output=output_log_job%A.txt
#SBATCH --job-name=ctc_llm_training_290824

cd /cs/snapless/gabis/nive/speech/Speech-Processing-Project/
module load cuda/11.7
../../venvs/async_env/bin/python final_project.py
