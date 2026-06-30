#!/bin/bash

export PYTHONWARNINGS="ignore"
export TRANSFORMERS_VERBOSITY=error

GPU=1
DEBUG= 
#--debug
CROSS_ATTN_FUSION= 
#--cross-attn-fusion
# use attention fusion layers for mid fusion
ATTN_FUSION=--attn-fusion

# install dependencies
# TODO: only create new venv if this dir does not exist
#python3 -m venv venv2
source venv2/bin/activate
#pip install -r requirements.txt

WEIGHTS_DIR='./trial2'

# run evaluations scripts
CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py --arch 'mobilenet' $DEBUG --weights $WEIGHTS_DIR/mobilenet_out/model_state_dict.pt
CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py --arch 'bert' $DEBUG --weights $WEIGHTS_DIR/bert_out/model_state_dict.pt
CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
	--arch 'latefusion' \
	--vision-model 'mobilenet' \
	$CROSS_ATTN_FUSION \
	$DEBUG \
	--weights $WEIGHTS_DIR/latefusionmobilenet_out/model_state_dict.pt
CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py --arch 'vit' $DEBUG --weights $WEIGHTS_DIR/vit_out/model_state_dict.pt
CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
	--arch 'latefusion' \
	--vision-model 'vit' \
	$CROSS_ATTN_FUSION \
	$DEBUG \
	--weights $WEIGHTS_DIR/latefusionvit_out/model_state_dict.pt
CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
	--arch 'midfusion' \
	--vision-model 'mobilenet' \
	$CROSS_ATTN_FUSION \
	$ATTN_FUSION \
	$DEBUG \
	--weights $WEIGHTS_DIR/midfusionmobilenet_out/model_state_dict.pt
CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
	--arch 'midfusion' \
	--vision-model 'vit' \
	$CROSS_ATTN_FUSION \
	$ATTN_FUSION \
	$DEBUG \
	--weights $WEIGHTS_DIR/midfusionvit_out/model_state_dict.pt
WEIGHTS_DIR='./latency_test_models'
CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
	--arch 'earlyfusion' \
	--vision-model 'mobilenet' \
	$CROSS_ATTN_FUSION \
	$DEBUG \
	--weights $WEIGHTS_DIR/earlyfusionmobilenet_out/model_state_dict.pt
CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
	--arch 'earlyfusion' \
	--vision-model 'vit' \
	$CROSS_ATTN_FUSION \
	$DEBUG \
	--weights $WEIGHTS_DIR/earlyfusionvit_out/model_state_dict.pt
exit
