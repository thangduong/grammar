#!/bin/bash
source /mnt/work/env-tf/bin/activate
export PYTHONPATH=$PYTHONPATH:$(readlink -f $(dirname ${BASH_SOURCE[@]})/../../dlframework):$(readlink -f $(dirname ${BASH_SOURCE[@]}))
if [ "$#" -eq 1 ]; then
	export CUDA_VISIBLE_DEVICES=$1
else
	export CUDA_VISIBLE_DEVICES=
fi
pushd . > /dev/null
