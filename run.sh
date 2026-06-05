#!/bin/bash

DEBUG=False

# install dependencies
# TODO: only create new venv if this dir does not exist
#python3 -m venv venv
source venv/bin/activate
#pip install -r requirements.txt

# clear output dirs
rm -r *_out 

# run training scripts
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py --arch 'mobilenet' --debug $DEBUG > out.log
mv out.log mobilenet_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py --arch 'bert' --debug $DEBUG > out.log
mv out.log bert_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	--arch 'latefusion' \
	--path-mobilenet mobilenet_out/model_state_dict.pt \
	--path-bert bert_out/model_state_dict.pt \
	--debug $DEBUG > out.log
mv out.log latefusionmobilenet_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py --arch 'vit' --debug $DEBUG > out.log
mv out.log vit_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	--arch 'latefusion' \
	--path-vit vit_out/model_state_dict.pt \
	--path-bert bert_out/model_state_dict.pt \
	--debug $DEBUG > out.log
mv out.log latefusionvit_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py --arch 'midfusion' --vision-model 'mobilenet' --debug $DEBUG > out.log
mv out.log midfusionmobilenet_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py --arch 'midfusion' --vision-model 'vit' --debug $DEBUG > out.log
mv out.log midfusionvit_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py --arch 'earlyfusion' --vision-model 'mobilenet' --debug $DEBUG > out.log
mv out.log earlyfusionmobilenet_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py --arch 'earlyfusion' --vision-model 'vit' --debug $DEBUG > out.log
mv out.log earlyfusionvit_out
