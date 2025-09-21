#!/bin/bash

#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00

# python env
source $HOME/.character/bin/activate
# tokens
source $HOME/OpenCharacterTraining/.env

cd $HOME/OpenCharacterTraining/finetuning/introspection
./$1.sh $2