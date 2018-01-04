#!/bin/bash
source /mnt/work/env-tf/bin/activate
export PYTHONPATH=$PYTHONPATH:$(readlink -f $(dirname ${BASH_SOURCE[@]})/../../dlframework)
export CUDA_VISIBLE_DEVICES=
pushd . > /dev/null
