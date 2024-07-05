#!/bin/bash
#SBATCH --job-name=PatternRecognitionGodMode
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

source ~/pytorch-2.1.0/bin/activate
python3 search_nn.py