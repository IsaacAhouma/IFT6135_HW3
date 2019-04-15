#!/bin/bash

#PBS -A colosse-users
#PBS -l feature=k80
#PBS -l nodes=1:gpus=1
#PBS -l walltime=10:00:00

export ROOT_DIR=$HOME'/HW3'


s_exec python $ROOT_DIR'/train.py'
